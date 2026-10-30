#!/usr/bin/env python3
"""
scripts/verify_table_readiness.py

Independent, strictly read-only pre-flight verification script for the
Planner Sensitivity Experiment.
Validates table readiness, column storage policies, HNSW index presence,
and 100% chunk ID equivalence across all experimental catalogs before running benchmarks.

Guarantees:
  1. Strictly read-only: Executes zero DDL/DML; never creates, alters, or drops any object.
  2. Protects production baseline databases from access.
  3. Verifies STORAGE PLAIN on fp32_plain relations.
  4. Confirms HNSW index existence and valid status.
  5. Verifies 100% chunk ID parity across the three representation tables.
"""

import os
import sys
import argparse
import psycopg
from psycopg import sql
from pgvector.psycopg import register_vector

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import get_db_connection_string

SCHEMA_NAME = "planner_sensitivity"

ALLOWED_DATABASES = {
    "compression_aware_rag_exp",
    "compression_aware_rag_scale",
    "compression_aware_rag_test"
}

DISALLOWED_DATABASES = {
    "postgres",
    "compression_aware_rag",
    "smart_document_chatbot"
}

SCALE_SPECS = {
    "1001": {
        "scale": 1001,
        "database": "compression_aware_rag_exp",
        "expected_count": 1001
    },
    "5000": {
        "scale": 5000,
        "database": "compression_aware_rag_scale",
        "expected_count": 5000
    },
    "10000": {
        "scale": 10000,
        "database": "compression_aware_rag_scale",
        "expected_count": 10000
    },
    "test": {
        "scale": "test",
        "database": "compression_aware_rag_test",
        "expected_count": 180
    }
}

REPRESENTATIONS = ["fp32_extended", "fp32_plain", "fp16_extended"]


def verify_scale_readiness(scale_tag: str, spec: dict) -> dict:
    """Verifies table presence, row counts, storage policies, and indexes for a corpus scale."""
    target_db = spec["database"]
    if target_db in DISALLOWED_DATABASES:
        raise PermissionError(f"Access to database '{target_db}' is strictly forbidden.")
    if target_db not in ALLOWED_DATABASES:
        raise ValueError(f"Database '{target_db}' is not an approved experimental catalog.")

    expected_rows = spec["expected_count"]
    results = {
        "scale": scale_tag,
        "database": target_db,
        "ready": True,
        "checks": {}
    }

    conn = psycopg.connect(get_db_connection_string(target_db))
    conn.read_only = True
    register_vector(conn)

    try:
        with conn.cursor() as cur:
            # 1. Verify schema
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.schemata
                WHERE schema_name = %s;
            """, (SCHEMA_NAME,))
            schema_exists = (cur.fetchone()[0] == 1)
            results["checks"]["schema_exists"] = schema_exists
            if not schema_exists:
                results["ready"] = False
                return results

            # 2. Check each representation table
            for rep in REPRESENTATIONS:
                tbl_name = f"chunks_{scale_tag}_{rep}"
                idx_name = f"{tbl_name}_hnsw_idx"

                # Check table exists
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s;
                """, (SCHEMA_NAME, tbl_name))
                tbl_exists = (cur.fetchone()[0] == 1)

                if not tbl_exists:
                    results["checks"][tbl_name] = {"exists": False}
                    results["ready"] = False
                    continue

                # Check row count
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
                    schema=sql.Identifier(SCHEMA_NAME),
                    table=sql.Identifier(tbl_name)
                ))
                actual_rows = cur.fetchone()[0]

                # Check storage policy
                cur.execute("""
                    SELECT attstorage FROM pg_attribute
                    WHERE attrelid = %s::regclass AND attname = 'embedding';
                """, (f"{SCHEMA_NAME}.{tbl_name}",))
                att_row = cur.fetchone()
                storage_char = att_row[0] if att_row else None
                storage_type = "PLAIN" if storage_char == "p" else "EXTENDED" if storage_char == "x" else "OTHER"

                expected_storage = "PLAIN" if rep == "fp32_plain" else "EXTENDED"
                storage_ok = (storage_type == expected_storage)

                # Check HNSW index
                cur.execute("""
                    SELECT c.relname, i.indisvalid
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indexrelid
                    JOIN pg_class t ON t.oid = i.indrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    WHERE n.nspname = %s AND t.relname = %s AND c.relname = %s;
                """, (SCHEMA_NAME, tbl_name, idx_name))
                idx_row = cur.fetchone()
                idx_exists = (idx_row is not None)
                idx_valid = bool(idx_row[1]) if idx_row else False

                rep_ok = (actual_rows == expected_rows and storage_ok and idx_exists and idx_valid)
                if not rep_ok:
                    results["ready"] = False

                results["checks"][tbl_name] = {
                    "exists": True,
                    "rows": actual_rows,
                    "expected_rows": expected_rows,
                    "rows_ok": (actual_rows == expected_rows),
                    "storage_policy": storage_type,
                    "storage_ok": storage_ok,
                    "hnsw_index": idx_name,
                    "index_exists": idx_exists,
                    "index_valid": idx_valid
                }

            # 3. Check chunk ID parity across the three tables
            if results["ready"]:
                t1 = f"chunks_{scale_tag}_fp32_extended"
                t2 = f"chunks_{scale_tag}_fp32_plain"
                t3 = f"chunks_{scale_tag}_fp16_extended"
                parity_query = sql.SQL("""
                    SELECT COUNT(*) FROM (
                        (SELECT chunk_id FROM {schema}.{t1} EXCEPT SELECT chunk_id FROM {schema}.{t2})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t1} EXCEPT SELECT chunk_id FROM {schema}.{t3})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t2} EXCEPT SELECT chunk_id FROM {schema}.{t1})
                    ) diff;
                """).format(
                    schema=sql.Identifier(SCHEMA_NAME),
                    t1=sql.Identifier(t1),
                    t2=sql.Identifier(t2),
                    t3=sql.Identifier(t3)
                )
                cur.execute(parity_query)
                mismatches = cur.fetchone()[0]
                parity_ok = (mismatches == 0)
                results["checks"]["parity_mismatches"] = mismatches
                results["checks"]["parity_ok"] = parity_ok
                if not parity_ok:
                    results["ready"] = False
    finally:
        conn.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Read-only verification of planner sensitivity experimental tables.")
    parser.add_argument("--scale", choices=["1001", "5000", "10000", "test", "all"], default="all",
                        help="Scale to verify (default: all).")
    args = parser.parse_args()

    scales_to_check = ["1001", "5000", "10000", "test"] if args.scale == "all" else [args.scale]

    print("================================================================================")
    print("PLANNER SENSITIVITY TABLE READINESS VERIFICATION (READ-ONLY)")
    print("================================================================================")

    all_ready = True
    for s_tag in scales_to_check:
        spec = SCALE_SPECS[s_tag]
        res = verify_scale_readiness(s_tag, spec)
        status_str = "READY" if res["ready"] else "NOT READY"
        print(f"\n[Scale {s_tag}] Database: {res['database']} => {status_str}")
        for k, v in res["checks"].items():
            if isinstance(v, dict):
                print(f"  - {k}: rows={v.get('rows')} ({'OK' if v.get('rows_ok') else 'MISMATCH'}), "
                      f"storage={v.get('storage_policy')} ({'OK' if v.get('storage_ok') else 'INVALID'}), "
                      f"index={v.get('hnsw_index')} ({'VALID' if v.get('index_valid') else 'MISSING'})")
            else:
                print(f"  - {k}: {v}")
        if not res["ready"]:
            all_ready = False

    print("\n================================================================================")
    if all_ready:
        print("[VERIFICATION PASSED] All experimental tables, storage policies, and HNSW indexes are verified ready.")
        sys.exit(0)
    else:
        print("[VERIFICATION FAILED] One or more experimental scales are missing or misconfigured.")
        sys.exit(1)


if __name__ == "__main__":
    main()
