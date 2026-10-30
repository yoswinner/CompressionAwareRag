#!/usr/bin/env python3
"""
scripts/provision_planner_sensitivity.py

One-time database provisioning script for the Planner Sensitivity Experiment.
Creates isolated experimental tables and HNSW indexes within the dedicated
schema 'planner_sensitivity'.

Guarantees:
1. Strict fail-closed isolation: Refuses to touch production baseline catalogs ('compression_aware_rag', 'postgres', 'smart_document_chatbot').
2. Non-interference: Baseline tables in 'public' are strictly untouched.
3. Storage policy isolation: Sets STORAGE PLAIN for fp32_plain before row insertion.
4. Parity verification: Verifies 100% chunk ID equivalence across all three representations.
5. Dry-run mode by default; requires explicit '--apply' flag to execute DDL/DML.
"""

import os
import sys
import argparse
import csv
import json
import time
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
    "compression_aware_rag",  # Production baseline: strictly protected
    "smart_document_chatbot"
}

SCALE_SPECS = {
    "1001": {
        "scale": 1001,
        "database": "compression_aware_rag_exp",
        "source_schema": "public",
        "source_table": "chunks_embedding_fp32",
        "expected_count": 1001
    },
    "5000": {
        "scale": 5000,
        "database": "compression_aware_rag_scale",
        "source_schema": "public",
        "source_table": "chunks_5k_fp32",
        "expected_count": 5000
    },
    "10000": {
        "scale": 10000,
        "database": "compression_aware_rag_scale",
        "source_schema": "public",
        "source_table": "chunks_10k_fp32",
        "expected_count": 10000
    },
    "test": {
        "scale": 180,
        "database": "compression_aware_rag_test",
        "source_database": "compression_aware_rag_exp",
        "source_schema": "public",
        "source_table": "chunks_embedding_fp32",
        "expected_count": 180
    }
}

REPRESENTATIONS = ["fp32_extended", "fp32_plain", "fp16_extended"]


def assert_approved_database(conn: psycopg.Connection, expected_db: str) -> str:
    """Fail-closed assertion ensuring target database is strictly allowed."""
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        active_db = cur.fetchone()[0]
        if active_db in DISALLOWED_DATABASES:
            raise RuntimeError(
                f"FATAL SAFETY VIOLATION: Refusing to operate on protected database '{active_db}'!"
            )
        if active_db not in ALLOWED_DATABASES:
            raise RuntimeError(
                f"FATAL SAFETY VIOLATION: Database '{active_db}' is not in approved list {ALLOWED_DATABASES}!"
            )
        if expected_db and active_db != expected_db:
            raise RuntimeError(
                f"DATABASE MISMATCH: Connected to '{active_db}', expected '{expected_db}'."
            )
        return active_db


def check_table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    """Returns True if table exists in the specified schema."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s;
        """, (schema, table))
        return cur.fetchone() is not None


def ensure_schema_exists(conn: psycopg.Connection, schema: str = SCHEMA_NAME, apply: bool = False):
    """Ensures dedicated experimental schema exists."""
    if not apply:
        print(f"[DRY RUN] Would verify/create schema '{schema}'.")
        return
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(schema)))
        print(f"[APPLIED] Schema '{schema}' ensured.")


def create_representation_table(
    conn: psycopg.Connection,
    scale_tag: str,
    rep: str,
    apply: bool = False
) -> str:
    """
    Creates an isolated table in planner_sensitivity schema with the appropriate storage policy.
    CRITICAL: For fp32_plain, STORAGE PLAIN must be set before inserting any rows!
    """
    table_name = f"chunks_{scale_tag}_{rep}"
    if check_table_exists(conn, SCHEMA_NAME, table_name):
        print(f"[-] Table {SCHEMA_NAME}.{table_name} already exists. Refusing to overwrite.")
        return table_name

    is_halfvec = "fp16" in rep
    col_type = "halfvec(768)" if is_halfvec else "vector(768)"
    storage_policy = "PLAIN" if rep == "fp32_plain" else "EXTENDED"

    if not apply:
        print(f"[DRY RUN] Would create table {SCHEMA_NAME}.{table_name} ({col_type}, STORAGE {storage_policy}).")
        return table_name

    with conn.cursor() as cur:
        # 1. Create table structure
        create_sql = sql.SQL("""
            CREATE TABLE {schema}.{table} (
                chunk_id INT PRIMARY KEY,
                embedding {type} NOT NULL
            );
        """).format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name),
            type=sql.SQL(col_type)
        )
        cur.execute(create_sql)

        # 2. Set storage policy BEFORE inserting rows
        alter_storage_sql = sql.SQL("""
            ALTER TABLE {schema}.{table} ALTER COLUMN embedding SET STORAGE {policy};
        """).format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name),
            policy=sql.SQL(storage_policy)
        )
        cur.execute(alter_storage_sql)
        print(f"[APPLIED] Created {SCHEMA_NAME}.{table_name} with column embedding {col_type} (STORAGE {storage_policy}).")

    return table_name


def populate_representation_table(
    conn: psycopg.Connection,
    scale_tag: str,
    rep: str,
    source_schema: str,
    source_table: str,
    limit: int | None = None,
    source_conn: psycopg.Connection | None = None,
    apply: bool = False
):
    """
    Populates the experimental table from the verified FP32 source table.
    Preserves identical chunk IDs and casts to halfvec(768) if FP16.
    Supports in-database insert or streaming across connections.
    """
    table_name = f"chunks_{scale_tag}_{rep}"
    if not apply:
        print(f"[DRY RUN] Would populate {SCHEMA_NAME}.{table_name} from {source_schema}.{source_table} (limit={limit}).")
        return

    with conn.cursor() as cur:
        # Check current row count
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        ))
        cnt = cur.fetchone()[0]
        if cnt > 0:
            print(f"[-] Table {SCHEMA_NAME}.{table_name} already contains {cnt} rows. Skipping insert.")
            return

        is_halfvec = "fp16" in rep
        cast_expr = sql.SQL("embedding::halfvec(768)") if is_halfvec else sql.SQL("embedding")

        if source_conn is not None and source_conn != conn:
            # Stream from cross-database source
            with source_conn.cursor() as src_cur:
                q = sql.SQL("""
                    SELECT chunk_id, embedding
                    FROM {src_schema}.{src_table}
                    ORDER BY chunk_id ASC
                """).format(
                    src_schema=sql.Identifier(source_schema),
                    src_table=sql.Identifier(source_table)
                )
                if limit is not None:
                    q = q + sql.SQL(" LIMIT {}").format(sql.Literal(limit))
                src_cur.execute(q)
                src_rows = src_cur.fetchall()

            insert_sql = sql.SQL("""
                INSERT INTO {schema}.{table} (chunk_id, embedding)
                VALUES (%s, %s::{cast});
            """).format(
                schema=sql.Identifier(SCHEMA_NAME),
                table=sql.Identifier(table_name),
                cast=sql.SQL("halfvec(768)" if is_halfvec else "vector(768)")
            )
            cur.executemany(insert_sql, src_rows)
        else:
            # In-database insert
            if limit is not None:
                insert_sql = sql.SQL("""
                    INSERT INTO {schema}.{table} (chunk_id, embedding)
                    SELECT chunk_id, {cast}
                    FROM {src_schema}.{src_table}
                    ORDER BY chunk_id ASC
                    LIMIT %s;
                """).format(
                    schema=sql.Identifier(SCHEMA_NAME),
                    table=sql.Identifier(table_name),
                    cast=cast_expr,
                    src_schema=sql.Identifier(source_schema),
                    src_table=sql.Identifier(source_table)
                )
                cur.execute(insert_sql, (limit,))
            else:
                insert_sql = sql.SQL("""
                    INSERT INTO {schema}.{table} (chunk_id, embedding)
                    SELECT chunk_id, {cast}
                    FROM {src_schema}.{src_table}
                    ORDER BY chunk_id ASC;
                """).format(
                    schema=sql.Identifier(SCHEMA_NAME),
                    table=sql.Identifier(table_name),
                    cast=cast_expr,
                    src_schema=sql.Identifier(source_schema),
                    src_table=sql.Identifier(source_table)
                )
                cur.execute(insert_sql)

        cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        ))
        inserted = cur.fetchone()[0]
        print(f"[APPLIED] Populated {SCHEMA_NAME}.{table_name}: {inserted} rows inserted.")


def build_representation_hnsw_index(
    conn: psycopg.Connection,
    scale_tag: str,
    rep: str,
    m: int = 16,
    ef_construction: int = 64,
    apply: bool = False
) -> str:
    """
    Builds the HNSW vector index using the appropriate cosine distance operator class.
    """
    table_name = f"chunks_{scale_tag}_{rep}"
    index_name = f"{table_name}_hnsw_idx"

    if not apply:
        print(f"[DRY RUN] Would build HNSW index {SCHEMA_NAME}.{index_name} on {table_name} (m={m}, ef_construction={ef_construction}).")
        return index_name

    with conn.cursor() as cur:
        # Check if index already exists
        cur.execute("""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = %s AND tablename = %s AND indexname = %s;
        """, (SCHEMA_NAME, table_name, index_name))
        if cur.fetchone():
            print(f"[-] Index {SCHEMA_NAME}.{index_name} already exists. Skipping build.")
            return index_name

        is_halfvec = "fp16" in rep
        opclass = "halfvec_cosine_ops" if is_halfvec else "vector_cosine_ops"

        print(f"[APPLIED] Building HNSW index {SCHEMA_NAME}.{index_name} ({opclass}, m={m}, ef_construction={ef_construction})...")
        t0 = time.perf_counter()
        idx_sql = sql.SQL("""
            CREATE INDEX {idx}
            ON {schema}.{table}
            USING hnsw (embedding {opclass})
            WITH (m = {m}, ef_construction = {ef_c});
        """).format(
            idx=sql.Identifier(index_name),
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name),
            opclass=sql.SQL(opclass),
            m=sql.Literal(m),
            ef_c=sql.Literal(ef_construction)
        )
        cur.execute(idx_sql)
        build_time = time.perf_counter() - t0
        print(f"[APPLIED] HNSW index {SCHEMA_NAME}.{index_name} built successfully in {build_time:.2f}s.")

    return index_name


def analyze_experimental_table(
    conn: psycopg.Connection,
    scale_tag: str,
    rep: str,
    apply: bool = False
):
    """
    Executes ANALYZE strictly on the newly provisioned table in planner_sensitivity schema.
    DO NOT execute on baseline or public tables.
    """
    table_name = f"chunks_{scale_tag}_{rep}"
    if not apply:
        print(f"[DRY RUN] Would run ANALYZE {SCHEMA_NAME}.{table_name}.")
        return

    with conn.cursor() as cur:
        analyze_sql = sql.SQL("ANALYZE {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        )
        cur.execute(analyze_sql)
        print(f"[APPLIED] ANALYZE completed for {SCHEMA_NAME}.{table_name}.")


def inspect_table_metrics(
    conn: psycopg.Connection,
    scale_tag: str,
    rep: str
) -> dict:
    """
    Inspects physical footprint, relation pages, reltuples, and storage layout of an experimental table.
    """
    table_name = f"chunks_{scale_tag}_{rep}"
    with conn.cursor() as cur:
        # Heap bytes and page count
        cur.execute("""
            SELECT pg_relation_size(c.oid), c.relpages, c.reltuples
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, table_name))
        row = cur.fetchone()
        if not row:
            return {"error": f"Table {SCHEMA_NAME}.{table_name} not found"}
        heap_b, relpages, reltuples = row

        # Total inclusive bytes
        cur.execute("""
            SELECT pg_total_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, table_name))
        total_inc_b = cur.fetchone()[0]

        # TOAST size
        cur.execute("""
            SELECT c.reltoastrelid
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, table_name))
        toast_oid_row = cur.fetchone()
        toast_oid = toast_oid_row[0] if toast_oid_row else 0

        toast_tbl_b = 0
        toast_idx_b = 0
        if toast_oid and toast_oid != 0:
            cur.execute("SELECT pg_relation_size(%s);", (toast_oid,))
            toast_tbl_b = cur.fetchone()[0] or 0
            cur.execute("SELECT pg_indexes_size(%s);", (toast_oid,))
            toast_idx_b = cur.fetchone()[0] or 0

        # HNSW Index size
        index_name = f"{table_name}_hnsw_idx"
        cur.execute("""
            SELECT pg_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, index_name))
        idx_row = cur.fetchone()
        idx_bytes = idx_row[0] if idx_row else 0

        # PK Index size
        cur.execute("""
            SELECT pg_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, f"{table_name}_pkey"))
        pk_row = cur.fetchone()
        pk_bytes = pk_row[0] if pk_row else 0

        # Row count
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        ))
        actual_rows = cur.fetchone()[0]

        # Vector dimensions
        cur.execute(sql.SQL("SELECT DISTINCT vector_dims(embedding) FROM {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        ))
        dims = [r[0] for r in cur.fetchall()]

        # Storage policy
        cur.execute("""
            SELECT a.attstorage
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding';
        """, (SCHEMA_NAME, table_name))
        storage_policy_code = cur.fetchone()[0]
        policy_map = {'p': 'PLAIN', 'e': 'EXTENDED', 'm': 'MAIN', 'x': 'EXTERNAL'}
        storage_policy = policy_map.get(storage_policy_code, storage_policy_code)

        # Average column size
        cur.execute(sql.SQL("""
            SELECT min(pg_column_size(embedding)), max(pg_column_size(embedding)), avg(pg_column_size(embedding))
            FROM {schema}.{table};
        """).format(schema=sql.Identifier(SCHEMA_NAME), table=sql.Identifier(table_name)))
        min_col, max_col, avg_col = cur.fetchone()

        return {
            "schema": SCHEMA_NAME,
            "table_name": table_name,
            "scale": scale_tag,
            "representation": rep,
            "status": "provisioned",
            "storage_policy": storage_policy,
            "row_count": actual_rows,
            "reltuples": float(reltuples),
            "relpages": int(relpages),
            "vector_dims": dims,
            "column_size_min_bytes": min_col,
            "column_size_max_bytes": max_col,
            "column_size_avg_bytes": float(avg_col) if avg_col else 0.0,
            "heap_bytes": heap_b,
            "toast_table_bytes": toast_tbl_b,
            "toast_index_bytes": toast_idx_b,
            "toast_total_bytes": toast_tbl_b + toast_idx_b,
            "pk_index_bytes": pk_bytes,
            "hnsw_index_bytes": idx_bytes,
            "total_inclusive_bytes": total_inc_b
        }


def validate_chunk_id_equivalence(conn: psycopg.Connection, scale_tag: str) -> dict:
    """
    Validates that chunk_ids across the 3 representation tables for this scale match 100% bi-directionally.
    """
    tbl_fp32_ext = f"chunks_{scale_tag}_fp32_extended"
    tbl_fp32_pln = f"chunks_{scale_tag}_fp32_plain"
    tbl_fp16_ext = f"chunks_{scale_tag}_fp16_extended"

    for tbl in [tbl_fp32_ext, tbl_fp32_pln, tbl_fp16_ext]:
        if not check_table_exists(conn, SCHEMA_NAME, tbl):
            return {"scale": scale_tag, "valid": False, "error": f"Table {tbl} missing"}

    with conn.cursor() as cur:
        query = sql.SQL("""
            SELECT COUNT(*)
            FROM (
                (SELECT chunk_id FROM {schema}.{t1} EXCEPT SELECT chunk_id FROM {schema}.{t2})
                UNION ALL
                (SELECT chunk_id FROM {schema}.{t2} EXCEPT SELECT chunk_id FROM {schema}.{t1})
                UNION ALL
                (SELECT chunk_id FROM {schema}.{t1} EXCEPT SELECT chunk_id FROM {schema}.{t3})
                UNION ALL
                (SELECT chunk_id FROM {schema}.{t3} EXCEPT SELECT chunk_id FROM {schema}.{t1})
            ) diff;
        """).format(
            schema=sql.Identifier(SCHEMA_NAME),
            t1=sql.Identifier(tbl_fp32_ext),
            t2=sql.Identifier(tbl_fp32_pln),
            t3=sql.Identifier(tbl_fp16_ext)
        )
        cur.execute(query)
        mismatched = cur.fetchone()[0]

        cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{t1};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            t1=sql.Identifier(tbl_fp32_ext)
        ))
        total_rows = cur.fetchone()[0]

        is_valid = (mismatched == 0 and total_rows > 0)
        return {
            "scale": scale_tag,
            "valid": is_valid,
            "total_rows": total_rows,
            "mismatched_ids": mismatched
        }


def provision_scale(
    scale_tag: str,
    spec: dict,
    apply: bool = False,
    m: int = 16,
    ef_construction: int = 64
) -> list[dict]:
    """Provisions all 3 representations for a single corpus scale."""
    target_db = spec["database"]
    source_db = spec.get("source_database", target_db)
    source_schema = spec["source_schema"]
    source_table = spec["source_table"]
    expected_count = spec.get("expected_count")

    print(f"\n=======================================================")
    print(f"PROVISIONING SCALE {scale_tag} (Target Database: '{target_db}', Apply: {apply})")
    print(f"=======================================================")

    conn = psycopg.connect(get_db_connection_string(target_db), autocommit=True)
    register_vector(conn)
    metrics_list = []

    source_conn = None
    if source_db != target_db:
        source_conn = psycopg.connect(get_db_connection_string(source_db), autocommit=True)
        register_vector(source_conn)

    try:
        assert_approved_database(conn, target_db)
        if source_conn:
            assert_approved_database(source_conn, source_db)

        # 1. Ensure extension and schema
        if apply:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        ensure_schema_exists(conn, SCHEMA_NAME, apply=apply)

        # 2. Check source table
        check_conn = source_conn if source_conn else conn
        with check_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s;
            """, (source_schema, source_table))
            if cur.fetchone()[0] == 0:
                raise RuntimeError(f"Source table '{source_schema}.{source_table}' missing in '{source_db}'!")

            cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
                schema=sql.Identifier(source_schema),
                table=sql.Identifier(source_table)
            ))
            source_rows = cur.fetchone()[0]
            print(f"Verified source table '{source_schema}.{source_table}' in '{source_db}': {source_rows} rows.")

        limit_to_use = expected_count if expected_count else source_rows

        # 3. Provision each of the 3 representations
        for rep in REPRESENTATIONS:
            print(f"\n--- Processing Representation: {rep} ---")
            create_representation_table(conn, scale_tag, rep, apply=apply)
            populate_representation_table(
                conn, scale_tag, rep, source_schema, source_table,
                limit=limit_to_use, source_conn=source_conn, apply=apply
            )
            build_representation_hnsw_index(conn, scale_tag, rep, m=m, ef_construction=ef_construction, apply=apply)
            analyze_experimental_table(conn, scale_tag, rep, apply=apply)

            if apply:
                m_info = inspect_table_metrics(conn, scale_tag, rep)
                metrics_list.append(m_info)
                print(f"  Storage footprint: Heap={m_info['heap_bytes']/1024:.1f} KiB, TOAST={m_info['toast_total_bytes']/1024:.1f} KiB, Index={m_info['hnsw_index_bytes']/1024:.1f} KiB")

        # 4. Validate chunk ID parity
        if apply:
            print(f"\n--- Validating Chunk ID Equivalence ---")
            val = validate_chunk_id_equivalence(conn, scale_tag)
            if not val["valid"]:
                raise RuntimeError(f"PARITY VALIDATION FAILED for scale {scale_tag}: {val}")
            print(f"[VERIFIED] 100% Chunk ID equivalence confirmed across all 3 tables ({val['total_rows']} rows, 0 mismatches).")

        return metrics_list
    finally:
        conn.close()
        if source_conn:
            source_conn.close()


def export_storage_metrics_csv(metrics_records: list[dict], output_csv_path: str):
    """Exports recorded storage footprints to CSV."""
    if not metrics_records:
        return
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    fieldnames = [
        "schema", "table_name", "scale", "representation", "storage_policy",
        "row_count", "reltuples", "relpages", "vector_dims",
        "column_size_min_bytes", "column_size_max_bytes", "column_size_avg_bytes",
        "heap_bytes", "toast_table_bytes", "toast_index_bytes", "toast_total_bytes",
        "pk_index_bytes", "hnsw_index_bytes", "total_inclusive_bytes"
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in metrics_records:
            writer.writerow(r)
    print(f"[EXPORTED] Storage metrics written to {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Provision isolated experimental tables for planner sensitivity benchmark.")
    parser.add_argument("--scale", choices=["1001", "5000", "10000", "test", "all"], default="all",
                        help="Corpus scale to provision (default: all).")
    parser.add_argument("--apply", action="store_true", default=False,
                        help="Execute table creation, population, and indexing. Default is dry-run.")
    parser.add_argument("--m", type=int, default=16, help="HNSW m parameter (default: 16).")
    parser.add_argument("--ef-construction", type=int, default=64, help="HNSW ef_construction (default: 64).")
    parser.add_argument("--csv-output", default="results/planner_sensitivity/storage_results.csv",
                        help="Path to output storage metrics CSV.")
    args = parser.parse_args()

    scales_to_run = ["1001", "5000", "10000"] if args.scale == "all" else [args.scale]

    print("================================================================================")
    print("PLANNER SENSITIVITY EXPERIMENTAL TABLE PROVISIONING PROTOCOL")
    print(f"Mode: {'APPLY (Live DDL/DML)' if args.apply else 'DRY RUN (Read-Only Inspection)'}")
    print(f"Target Scales: {scales_to_run}")
    print("================================================================================")

    all_metrics = []
    for s_tag in scales_to_run:
        spec = SCALE_SPECS[s_tag]
        res = provision_scale(s_tag, spec, apply=args.apply, m=args.m, ef_construction=args.ef_construction)
        all_metrics.extend(res)

    if args.apply and all_metrics:
        export_storage_metrics_csv(all_metrics, args.csv_output)
        print("\n[SUCCESS] Provisioning and validation successfully completed.")
    elif not args.apply:
        print("\n[INFO] Dry-run complete. Re-run with '--apply' to execute provisioning.")


if __name__ == "__main__":
    main()
