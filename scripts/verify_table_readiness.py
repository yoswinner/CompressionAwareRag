#!/usr/bin/env python3
"""
scripts/verify_table_readiness.py

Independent, strictly read-only pre-flight verification script for the
Planner Sensitivity Experiment.
Validates table readiness, exact row counts, column types and dimensions,
storage policies, HNSW index presence, validity, operator classes, index parameters,
physical storage metadata, and 100% chunk ID equivalence across all experimental catalogs
before running benchmarks.

Guarantees:
  1. Strictly read-only: Executes zero DDL/DML; sets conn.read_only = True.
  2. Protects production baseline databases from access.
  3. Verifies STORAGE PLAIN on fp32_plain relations and EXTENDED on others.
  4. Confirms HNSW index existence, valid status (indisvalid), and expected cosine operator class.
  5. Inspects HNSW index options (m, ef_construction) where available.
  6. Verifies exact expected row counts and 100% chunk ID parity across the three representations.
  7. Reports comprehensive storage footprint metadata.
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

EXPECTED_HNSW_M = 16
EXPECTED_HNSW_EF_CONSTRUCTION = 64

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

# PostgreSQL pg_attribute.attstorage specification:
# 'p' = PLAIN (inline, uncompressed)
# 'e' = EXTERNAL (secondary TOAST, uncompressed)
# 'm' = MAIN (inline if fits, compressed if necessary before out-of-line)
# 'x' = EXTENDED (secondary TOAST, compressed if compressible)
ATTSTORAGE_MAP = {
    'p': 'PLAIN',
    'e': 'EXTERNAL',
    'm': 'MAIN',
    'x': 'EXTENDED',
}



def parse_reloptions(options_list: list[str] | None) -> dict:
    """Parses pg_class.reloptions list like ['m=16', 'ef_construction=64'] into a dict."""
    if not options_list:
        return {}
    res = {}
    for opt in options_list:
        if "=" in opt:
            k, v = opt.split("=", 1)
            try:
                res[k.strip()] = int(v.strip())
            except ValueError:
                res[k.strip()] = v.strip()
    return res


def verify_scale_readiness(scale_tag: str, spec: dict) -> dict:
    """
    Verifies table presence, row counts, storage policies, column types,
    HNSW index operator classes, index parameters, and chunk parity for a scale.
    """
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
            # 1. Verify schema existence
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
                is_halfvec = "fp16" in rep
                expected_type = "halfvec(768)" if is_halfvec else "vector(768)"
                expected_storage = "PLAIN" if rep == "fp32_plain" else "EXTENDED"
                expected_opclass = "halfvec_cosine_ops" if is_halfvec else "vector_cosine_ops"

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

                # Check row count & distinct chunk_id count
                cur.execute(sql.SQL("SELECT COUNT(*), COUNT(DISTINCT chunk_id) FROM {schema}.{table};").format(
                    schema=sql.Identifier(SCHEMA_NAME),
                    table=sql.Identifier(tbl_name)
                ))
                actual_rows, distinct_rows = cur.fetchone()
                rows_ok = (actual_rows == expected_rows)
                distinct_ok = (distinct_rows == actual_rows)

                # Check primary key on chunk_id
                cur.execute("""
                    SELECT c.conname, cardinality(c.conkey)
                    FROM pg_constraint c
                    JOIN pg_class t ON c.conrelid = t.oid
                    JOIN pg_namespace n ON t.relnamespace = n.oid
                    JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(c.conkey)
                    WHERE n.nspname = %s
                      AND t.relname = %s
                      AND c.contype = 'p'
                      AND a.attname = 'chunk_id';
                """, (SCHEMA_NAME, tbl_name))
                pk_row = cur.fetchone()
                pk_name = pk_row[0] if pk_row else None
                pk_col_count = pk_row[1] if pk_row else 0
                pk_ok = (pk_row is not None and pk_col_count == 1)

                # Check column type & dimension
                cur.execute("""
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class c ON a.attrelid = c.oid
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding';
                """, (SCHEMA_NAME, tbl_name))
                col_type_row = cur.fetchone()
                actual_type = col_type_row[0] if col_type_row else None
                type_ok = (actual_type == expected_type)

                cur.execute(sql.SQL("SELECT DISTINCT vector_dims(embedding) FROM {schema}.{table};").format(
                    schema=sql.Identifier(SCHEMA_NAME),
                    table=sql.Identifier(tbl_name)
                ))
                dims = [r[0] for r in cur.fetchall()]
                dims_ok = (dims == [768])

                # Check storage policy
                cur.execute("""
                    SELECT attstorage FROM pg_attribute
                    WHERE attrelid = %s::regclass AND attname = 'embedding';
                """, (f"{SCHEMA_NAME}.{tbl_name}",))
                att_row = cur.fetchone()
                storage_char = att_row[0] if att_row else None
                storage_type = ATTSTORAGE_MAP.get(storage_char, "OTHER")
                storage_ok = (storage_type == expected_storage)

                # Check HNSW index, operator class, and reloptions
                cur.execute("""
                    SELECT c.relname, i.indisvalid, opc.opcname, am.amname, c.reloptions
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indexrelid
                    JOIN pg_class t ON t.oid = i.indrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    JOIN pg_am am ON am.oid = c.relam
                    JOIN pg_opclass opc ON opc.oid = i.indclass[0]
                    WHERE n.nspname = %s AND t.relname = %s AND c.relname = %s;
                """, (SCHEMA_NAME, tbl_name, idx_name))
                idx_row = cur.fetchone()
                idx_exists = (idx_row is not None)
                idx_valid = bool(idx_row[1]) if idx_row else False
                actual_opclass = idx_row[2] if idx_row else None
                am_name = idx_row[3] if idx_row else None
                raw_options = idx_row[4] if idx_row else []
                parsed_opts = parse_reloptions(raw_options)

                opclass_ok = (actual_opclass == expected_opclass and am_name == "hnsw")

                actual_m = parsed_opts.get("m")
                actual_ef_construction = parsed_opts.get("ef_construction")
                params_ok = (
                    actual_m == EXPECTED_HNSW_M and
                    actual_ef_construction == EXPECTED_HNSW_EF_CONSTRUCTION
                )

                # Inspect relation storage metadata
                cur.execute("""
                    SELECT pg_relation_size(c.oid), pg_total_relation_size(c.oid), c.reltoastrelid
                    FROM pg_class c
                    JOIN pg_namespace n ON c.relnamespace = n.oid
                    WHERE n.nspname = %s AND c.relname = %s;
                """, (SCHEMA_NAME, tbl_name))
                size_row = cur.fetchone()
                heap_bytes = size_row[0] if size_row else 0
                total_bytes = size_row[1] if size_row else 0
                toast_oid = size_row[2] if size_row else 0

                toast_bytes = 0
                if toast_oid and toast_oid != 0:
                    cur.execute("SELECT pg_total_relation_size(%s);", (toast_oid,))
                    toast_bytes = cur.fetchone()[0] or 0

                idx_bytes = 0
                if idx_exists:
                    cur.execute("""
                        SELECT pg_relation_size(c.oid)
                        FROM pg_class c
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = %s AND c.relname = %s;
                    """, (SCHEMA_NAME, idx_name))
                    idx_size_row = cur.fetchone()
                    idx_bytes = idx_size_row[0] if idx_size_row else 0

                rep_ok = (
                    rows_ok and
                    distinct_ok and
                    pk_ok and
                    type_ok and
                    dims_ok and
                    storage_ok and
                    idx_exists and
                    idx_valid and
                    opclass_ok and
                    params_ok
                )
                if not rep_ok:
                    results["ready"] = False

                results["checks"][tbl_name] = {
                    "exists": True,
                    "rows": actual_rows,
                    "expected_rows": expected_rows,
                    "rows_ok": rows_ok,
                    "distinct_rows": distinct_rows,
                    "distinct_ok": distinct_ok,
                    "primary_key": pk_name,
                    "pk_ok": pk_ok,
                    "column_type": actual_type,
                    "expected_type": expected_type,
                    "type_ok": type_ok,
                    "dimensions": dims,
                    "dims_ok": dims_ok,
                    "storage_policy": storage_type,
                    "expected_storage": expected_storage,
                    "storage_ok": storage_ok,
                    "hnsw_index": idx_name,
                    "index_exists": idx_exists,
                    "index_valid": idx_valid,
                    "operator_class": actual_opclass,
                    "expected_opclass": expected_opclass,
                    "opclass_ok": opclass_ok,
                    "index_parameters": parsed_opts,
                    "expected_parameters": {
                        "m": EXPECTED_HNSW_M,
                        "ef_construction": EXPECTED_HNSW_EF_CONSTRUCTION
                    },
                    "params_ok": params_ok,
                    "heap_bytes": heap_bytes,
                    "toast_bytes": toast_bytes,
                    "hnsw_index_bytes": idx_bytes,
                    "total_bytes": total_bytes
                }

            # 3. Check chunk ID parity across the three tables (explicitly bidirectional across all pairs)
            if results["ready"]:
                t1 = f"chunks_{scale_tag}_fp32_extended"
                t2 = f"chunks_{scale_tag}_fp32_plain"
                t3 = f"chunks_{scale_tag}_fp16_extended"
                parity_query = sql.SQL("""
                    SELECT COUNT(*) FROM (
                        (SELECT chunk_id FROM {schema}.{t1} EXCEPT SELECT chunk_id FROM {schema}.{t2})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t2} EXCEPT SELECT chunk_id FROM {schema}.{t1})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t1} EXCEPT SELECT chunk_id FROM {schema}.{t3})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t3} EXCEPT SELECT chunk_id FROM {schema}.{t1})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t2} EXCEPT SELECT chunk_id FROM {schema}.{t3})
                        UNION ALL
                        (SELECT chunk_id FROM {schema}.{t3} EXCEPT SELECT chunk_id FROM {schema}.{t2})
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
                act_params = v.get("index_parameters", {})
                exp_params = v.get("expected_parameters", {"m": EXPECTED_HNSW_M, "ef_construction": EXPECTED_HNSW_EF_CONSTRUCTION})
                params_status = "OK" if v.get("params_ok") else "MISMATCH"
                print(f"  - {k}: rows={v.get('rows')} ({'OK' if v.get('rows_ok') else 'MISMATCH'}), "
                      f"distinct={'OK' if v.get('distinct_ok') else 'DUPLICATES'}, "
                      f"pk={v.get('primary_key')} ({'OK' if v.get('pk_ok') else 'MISSING'}), "
                      f"type={v.get('column_type')} ({'OK' if v.get('type_ok') else 'MISMATCH'}), "
                      f"storage={v.get('storage_policy')} ({'OK' if v.get('storage_ok') else 'INVALID'}), "
                      f"opclass={v.get('operator_class')} ({'OK' if v.get('opclass_ok') else 'INVALID'}), "
                      f"index={v.get('hnsw_index')} ({'VALID' if v.get('index_valid') else 'MISSING'}), "
                      f"params={act_params} (expected {exp_params}) ({params_status})")
            else:
                print(f"  - {k}: {v}")
        if not res["ready"]:
            all_ready = False

    print("\n================================================================================")
    if all_ready:
        print("[VERIFICATION PASSED] All experimental tables, storage policies, types, opclasses, and HNSW indexes are verified ready.")
        sys.exit(0)
    else:
        print("[VERIFICATION FAILED] One or more experimental scales are missing or misconfigured.")
        sys.exit(1)


if __name__ == "__main__":
    main()
