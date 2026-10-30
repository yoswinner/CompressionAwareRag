#!/usr/bin/env python3
"""
scripts/provision_planner_sensitivity.py

One-time database provisioning script for the Planner Sensitivity Experiment.
Creates isolated experimental tables and HNSW indexes within the dedicated
schema 'planner_sensitivity'.

Guarantees & Scientific Integrity:
1. Strict fail-closed isolation: Refuses to touch production baseline catalogs
   ('compression_aware_rag', 'postgres', 'smart_document_chatbot').
2. Non-interference: Baseline tables in 'public' are strictly untouched.
3. Pre-flight source verification: Asserts source table exists, has rows >= expected_count,
   and embedding dimension == 768 before any DDL/DML.
4. Pre-flight target state check: Aborts immediately if partial target tables/indexes exist.
   Never automatically drops or overwrites pre-existing data.
5. Transactional execution per scale: Uses autocommit=False. Table creation, STORAGE PLAIN,
   row insertion, HNSW indexing, validation, and ANALYZE execute within a single transaction per scale.
   Rolls back completely on any error (including ANALYZE failure), leaving zero partial data.
6. Storage policy isolation: Sets STORAGE PLAIN for fp32_plain before inserting rows.
7. Post-population validation: Verifies exact expected row counts, storage policies,
   HNSW validity, cosine operator classes, and 100% chunk ID equivalence.
8. In-transaction statistics: Executes ANALYZE explicitly on all three new experimental tables
   inside the transaction before commit.
9. Dry-run mode by default; requires explicit '--apply' flag to execute DDL/DML.
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

EXPECTED_HNSW_M = 16
EXPECTED_HNSW_EF_CONSTRUCTION = 64
TARGET_MAINTENANCE_WORK_MEM = "512MB"

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


def check_index_exists(conn: psycopg.Connection, schema: str, index: str) -> bool:
    """Returns True if index exists in the specified schema."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM pg_indexes
            WHERE schemaname = %s AND indexname = %s;
        """, (schema, index))
        return cur.fetchone() is not None


def check_source_table_readiness(
    conn: psycopg.Connection,
    source_schema: str,
    source_table: str,
    expected_count: int
) -> tuple[int, int]:
    """
    Validates source table existence, row count >= expected_count, and vector dimension == 768.
    Returns (source_rows, vector_dim).
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s;
        """, (source_schema, source_table))
        if not cur.fetchone():
            raise RuntimeError(f"Source table '{source_schema}.{source_table}' does not exist!")

        cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
            schema=sql.Identifier(source_schema),
            table=sql.Identifier(source_table)
        ))
        source_rows = cur.fetchone()[0]
        if source_rows < expected_count:
            raise RuntimeError(
                f"Source table '{source_schema}.{source_table}' contains {source_rows} rows, "
                f"which is less than the required {expected_count} rows!"
            )

        cur.execute(sql.SQL("SELECT DISTINCT vector_dims(embedding) FROM {schema}.{table};").format(
            schema=sql.Identifier(source_schema),
            table=sql.Identifier(source_table)
        ))
        dims = [r[0] for r in cur.fetchall()]
        if dims != [768]:
            raise RuntimeError(
                f"Source table '{source_schema}.{source_table}' has invalid vector dimensions: {dims}. "
                f"Expected [768]."
            )
        return source_rows, 768


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
                UNION ALL
                (SELECT chunk_id FROM {schema}.{t2} EXCEPT SELECT chunk_id FROM {schema}.{t3})
                UNION ALL
                (SELECT chunk_id FROM {schema}.{t3} EXCEPT SELECT chunk_id FROM {schema}.{t2})
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


def check_target_scale_state(conn: psycopg.Connection, scale_tag: str, expected_count: int | None = None) -> str:
    """
    Checks the existing state of target tables and indexes for a scale.
    Fails closed if any invariant is violated.
    Returns:
      - 'clean': No tables or indexes exist. Ready for provisioning.
      - 'fully_provisioned': All 3 tables and 3 indexes exist and meet all invariants:
          1. Exact expected row count in all three tables.
          2. Column types vector(768) / halfvec(768) and 768-dimensional embeddings.
          3. Correct storage policies (PLAIN for fp32_plain, EXTENDED for others).
          4. Correct HNSW access method ('hnsw').
          5. Correct operator classes (vector_cosine_ops / halfvec_cosine_ops).
          6. Valid index status (indisvalid = true).
          7. HNSW parameters m=16 and ef_construction=64.
          8. 100% chunk ID parity across all three tables.
      - 'partial': Inconsistent or partial state. Provisioning must abort before writing.
    """
    tbls = [f"chunks_{scale_tag}_{rep}" for rep in REPRESENTATIONS]
    idxs = [f"chunks_{scale_tag}_{rep}_hnsw_idx" for rep in REPRESENTATIONS]

    existing_tbls = [t for t in tbls if check_table_exists(conn, SCHEMA_NAME, t)]
    existing_idxs = [i for i in idxs if check_index_exists(conn, SCHEMA_NAME, i)]

    # Clean state: zero tables and zero indexes exist
    if len(existing_tbls) == 0 and len(existing_idxs) == 0:
        return "clean"

    # If some objects exist but not all 6, it is partial
    if len(existing_tbls) != 3 or len(existing_idxs) != 3:
        return "partial"

    # Resolve expected row count
    if expected_count is None:
        spec = SCALE_SPECS.get(str(scale_tag))
        if spec and "expected_count" in spec:
            expected_count = spec["expected_count"]
        else:
            try:
                expected_count = int(scale_tag)
            except ValueError:
                return "partial"

    if expected_count is None or expected_count <= 0:
        return "partial"

    with conn.cursor() as cur:
        for rep in REPRESENTATIONS:
            tbl_name = f"chunks_{scale_tag}_{rep}"
            idx_name = f"{tbl_name}_hnsw_idx"
            is_halfvec = "fp16" in rep
            expected_type = "halfvec(768)" if is_halfvec else "vector(768)"
            expected_storage_char = "p" if rep == "fp32_plain" else "x"
            expected_opclass = "halfvec_cosine_ops" if is_halfvec else "vector_cosine_ops"

            # 1. Exact row count & distinct chunk_id count
            cur.execute(sql.SQL("SELECT COUNT(*), COUNT(DISTINCT chunk_id) FROM {schema}.{table};").format(
                schema=sql.Identifier(SCHEMA_NAME),
                table=sql.Identifier(tbl_name)
            ))
            actual_rows, distinct_rows = cur.fetchone()
            if actual_rows != expected_count or distinct_rows != actual_rows:
                return "partial"

            # 2. Valid primary key on chunk_id
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
            if not pk_row or pk_row[1] != 1:
                return "partial"

            # 3. Column type and 768-dim embeddings
            cur.execute("""
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding';
            """, (SCHEMA_NAME, tbl_name))
            col_row = cur.fetchone()
            if not col_row or col_row[0] != expected_type:
                return "partial"

            cur.execute(sql.SQL("SELECT DISTINCT vector_dims(embedding) FROM {schema}.{table};").format(
                schema=sql.Identifier(SCHEMA_NAME),
                table=sql.Identifier(tbl_name)
            ))
            dims = [r[0] for r in cur.fetchall()]
            if dims != [768]:
                return "partial"

            # 3. Storage policy
            cur.execute("""
                SELECT attstorage FROM pg_attribute
                WHERE attrelid = %s::regclass AND attname = 'embedding';
            """, (f"{SCHEMA_NAME}.{tbl_name}",))
            att_row = cur.fetchone()
            if not att_row or att_row[0] != expected_storage_char:
                return "partial"

            # 4. HNSW access method, operator class, valid index status, m=16, ef_construction=64
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
            if not idx_row:
                return "partial"
            idx_valid = bool(idx_row[1])
            actual_opclass = idx_row[2]
            am_name = idx_row[3]
            raw_options = idx_row[4] or []

            if not idx_valid or am_name != "hnsw" or actual_opclass != expected_opclass:
                return "partial"

            parsed_opts = parse_reloptions(raw_options)
            if parsed_opts.get("m") != EXPECTED_HNSW_M or parsed_opts.get("ef_construction") != EXPECTED_HNSW_EF_CONSTRUCTION:
                return "partial"

    # 5. Chunk-ID parity
    parity_check = validate_chunk_id_equivalence(conn, scale_tag)
    if not parity_check.get("valid") or parity_check.get("total_rows") != expected_count:
        return "partial"

    return "fully_provisioned"


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

        cur.execute("""
            SELECT pg_total_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, table_name))
        total_inc_b = cur.fetchone()[0]

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

        index_name = f"{table_name}_hnsw_idx"
        cur.execute("""
            SELECT pg_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, index_name))
        idx_row = cur.fetchone()
        idx_bytes = idx_row[0] if idx_row else 0

        cur.execute("""
            SELECT pg_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s;
        """, (SCHEMA_NAME, f"{table_name}_pkey"))
        pk_row = cur.fetchone()
        pk_bytes = pk_row[0] if pk_row else 0

        cur.execute(sql.SQL("SELECT COUNT(*) FROM {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        ))
        actual_rows = cur.fetchone()[0]

        cur.execute(sql.SQL("SELECT DISTINCT vector_dims(embedding) FROM {schema}.{table};").format(
            schema=sql.Identifier(SCHEMA_NAME),
            table=sql.Identifier(table_name)
        ))
        dims = [r[0] for r in cur.fetchall()]

        cur.execute("""
            SELECT a.attstorage
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding';
        """, (SCHEMA_NAME, table_name))
        storage_policy_code = cur.fetchone()[0]
        storage_policy = ATTSTORAGE_MAP.get(storage_policy_code, storage_policy_code)

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
            "min_col_size_bytes": min_col,
            "max_col_size_bytes": max_col,
            "avg_col_size_bytes": float(avg_col) if avg_col else 0.0,
            "column_size_min_bytes": min_col,
            "column_size_max_bytes": max_col,
            "column_size_avg_bytes": float(avg_col) if avg_col else 0.0,
            "heap_bytes": heap_b,
            "toast_table_bytes": toast_tbl_b,
            "toast_index_bytes": toast_idx_b,
            "toast_total_bytes": toast_tbl_b + toast_idx_b,
            "pk_index_bytes": pk_bytes,
            "hnsw_index_bytes": idx_bytes,
            "all_indexes_bytes": pk_bytes + idx_bytes,
            "total_inclusive_bytes": total_inc_b
        }


def provision_scale(
    scale_tag: str,
    spec: dict,
    apply: bool = False,
    m: int = 16,
    ef_construction: int = 64
) -> list[dict]:
    """
    Provisions all 3 representations for a single corpus scale.
    Strictly transactional per scale: uses autocommit=False and commits only
    upon 100% verified creation, population, index build, and parity validation.
    Rolls back completely if any failure occurs.
    """
    target_db = spec["database"]
    source_db = spec.get("source_database", target_db)
    source_schema = spec["source_schema"]
    source_table = spec["source_table"]
    expected_count = spec["expected_count"]

    print(f"\n=======================================================")
    print(f"PROVISIONING SCALE {scale_tag} (Target Database: '{target_db}', Apply: {apply})")
    print(f"=======================================================")

    conn = psycopg.connect(get_db_connection_string(target_db), autocommit=False)
    register_vector(conn)

    def on_notice(diag):
        print(f"    [POSTGRES NOTICE] {diag.message_primary}")

    conn.add_notice_handler(on_notice)
    metrics_list = []

    source_conn = None
    if source_db != target_db:
        source_conn = psycopg.connect(get_db_connection_string(source_db), autocommit=True)
        source_conn.read_only = True
        register_vector(source_conn)

    try:
        assert_approved_database(conn, target_db)
        if source_conn:
            assert_approved_database(source_conn, source_db)

        # 1. Pre-flight source readiness inspection
        check_conn = source_conn if source_conn else conn
        src_rows, src_dims = check_source_table_readiness(
            check_conn, source_schema, source_table, expected_count
        )
        print(f"[PRE-FLIGHT] Source table '{source_schema}.{source_table}' verified in '{source_db}': {src_rows} rows (required >= {expected_count}), dimension {src_dims}.")

        # 2. Pre-flight target state inspection
        target_state = check_target_scale_state(conn, scale_tag, expected_count=expected_count)
        if target_state == "partial":
            raise RuntimeError(
                f"PARTIAL OR INCONSISTENT PRE-EXISTING STATE DETECTED for scale '{scale_tag}' in database '{target_db}'! "
                f"Inconsistent tables, indexes, types, parameters, or row counts exist. To prevent data corruption, "
                f"provisioning has aborted before any write. The provisioner will never automatically DROP existing objects."
            )
        elif target_state == "fully_provisioned":
            print(f"[ALREADY PROVISIONED] Scale '{scale_tag}' in '{target_db}' already exists and is verified valid. Skipping.")
            return [inspect_table_metrics(conn, scale_tag, rep) for rep in REPRESENTATIONS]

        if not apply:
            print(f"[DRY RUN] Scale '{scale_tag}' is clean and ready to provision {expected_count} rows across {REPRESENTATIONS}.")
            return []

        # 3. Transactional Execution Phase
        print(f"[TRANSACTION BEGIN] Starting atomic provisioning transaction for scale '{scale_tag}'...")
        try:
            with conn.cursor() as cur:
                # Explicitly configure transaction-scoped maintenance_work_mem for HNSW index builds
                cur.execute(sql.SQL("SET LOCAL maintenance_work_mem = {};").format(sql.Literal(TARGET_MAINTENANCE_WORK_MEM)))
                cur.execute("SHOW maintenance_work_mem;")
                effective_mwm = cur.fetchone()[0]
                print(f"  [CONFIG] Transaction maintenance_work_mem configured: {effective_mwm}")

                # Ensure extension and schema
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(sql.Identifier(SCHEMA_NAME)))

                for rep in REPRESENTATIONS:
                    tbl_name = f"chunks_{scale_tag}_{rep}"
                    idx_name = f"{tbl_name}_hnsw_idx"
                    is_halfvec = "fp16" in rep
                    col_type = "halfvec(768)" if is_halfvec else "vector(768)"
                    storage_policy = "PLAIN" if rep == "fp32_plain" else "EXTENDED"
                    opclass = "halfvec_cosine_ops" if is_halfvec else "vector_cosine_ops"

                    print(f"  - Creating {SCHEMA_NAME}.{tbl_name} ({col_type}, STORAGE {storage_policy})...")
                    cur.execute(sql.SQL("""
                        CREATE TABLE {schema}.{table} (
                            chunk_id INT PRIMARY KEY,
                            embedding {type} NOT NULL
                        );
                    """).format(
                        schema=sql.Identifier(SCHEMA_NAME),
                        table=sql.Identifier(tbl_name),
                        type=sql.SQL(col_type)
                    ))

                    cur.execute(sql.SQL("""
                        ALTER TABLE {schema}.{table} ALTER COLUMN embedding SET STORAGE {policy};
                    """).format(
                        schema=sql.Identifier(SCHEMA_NAME),
                        table=sql.Identifier(tbl_name),
                        policy=sql.SQL(storage_policy)
                    ))

                    print(f"  - Populating {SCHEMA_NAME}.{tbl_name} ({expected_count} rows)...")
                    if source_conn is not None and source_conn != conn:
                        with source_conn.cursor() as src_cur:
                            src_cur.execute(sql.SQL("""
                                SELECT chunk_id, embedding
                                FROM {src_schema}.{src_table}
                                ORDER BY chunk_id ASC
                                LIMIT %s;
                            """).format(
                                src_schema=sql.Identifier(source_schema),
                                src_table=sql.Identifier(source_table)
                            ), (expected_count,))
                            src_rows_data = src_cur.fetchall()
                        ins_sql = sql.SQL("""
                            INSERT INTO {schema}.{table} (chunk_id, embedding)
                            VALUES (%s, %s::{cast});
                        """).format(
                            schema=sql.Identifier(SCHEMA_NAME),
                            table=sql.Identifier(tbl_name),
                            cast=sql.SQL(col_type)
                        )
                        cur.executemany(ins_sql, src_rows_data)
                    else:
                        ins_sql = sql.SQL("""
                            INSERT INTO {schema}.{table} (chunk_id, embedding)
                            SELECT chunk_id, {cast_expr}
                            FROM {src_schema}.{src_table}
                            ORDER BY chunk_id ASC
                            LIMIT %s;
                        """).format(
                            schema=sql.Identifier(SCHEMA_NAME),
                            table=sql.Identifier(tbl_name),
                            cast_expr=sql.SQL("embedding::halfvec(768)" if is_halfvec else "embedding"),
                            src_schema=sql.Identifier(source_schema),
                            src_table=sql.Identifier(source_table)
                        )
                        cur.execute(ins_sql, (expected_count,))

                    print(f"  - Building HNSW index {SCHEMA_NAME}.{idx_name} ({opclass}, m={m}, ef_construction={ef_construction})...")
                    t_idx0 = time.perf_counter()
                    cur.execute(sql.SQL("""
                        CREATE INDEX {idx}
                        ON {schema}.{table}
                        USING hnsw (embedding {opclass})
                        WITH (m = {m}, ef_construction = {ef_c});
                    """).format(
                        idx=sql.Identifier(idx_name),
                        schema=sql.Identifier(SCHEMA_NAME),
                        table=sql.Identifier(tbl_name),
                        opclass=sql.SQL(opclass),
                        m=sql.Literal(m),
                        ef_c=sql.Literal(ef_construction)
                    ))
                    print(f"    Index built in {time.perf_counter() - t_idx0:.2f}s.")

                # In-transaction verification
                print("  - Verifying in-transaction post-population integrity...")
                for rep in REPRESENTATIONS:
                    tbl_name = f"chunks_{scale_tag}_{rep}"
                    cur.execute(sql.SQL("SELECT COUNT(*), COUNT(DISTINCT chunk_id) FROM {schema}.{table};").format(
                        schema=sql.Identifier(SCHEMA_NAME),
                        table=sql.Identifier(tbl_name)
                    ))
                    act_cnt, dist_cnt = cur.fetchone()
                    if act_cnt != expected_count:
                        raise RuntimeError(f"Row count assertion failed on {tbl_name}: expected {expected_count}, got {act_cnt}")
                    if dist_cnt != act_cnt:
                        raise RuntimeError(f"Duplicate chunk IDs detected in {tbl_name}: distinct={dist_cnt}, total={act_cnt}")

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
                    pk_info = cur.fetchone()
                    if not pk_info or pk_info[1] != 1:
                        raise RuntimeError(f"Primary key assertion failed on {tbl_name}: expected single-column primary key on chunk_id")

                val = validate_chunk_id_equivalence(conn, scale_tag)
                if not val["valid"]:
                    raise RuntimeError(f"Chunk ID parity validation failed: {val}")

                for rep in REPRESENTATIONS:
                    tbl_name = f"chunks_{scale_tag}_{rep}"
                    expected_storage = "PLAIN" if rep == "fp32_plain" else "EXTENDED"
                    cur.execute("""
                        SELECT a.attstorage
                        FROM pg_attribute a
                        JOIN pg_class c ON a.attrelid = c.oid
                        JOIN pg_namespace n ON c.relnamespace = n.oid
                        WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding';
                    """, (SCHEMA_NAME, tbl_name))
                    st_code = cur.fetchone()[0]
                    st_policy = ATTSTORAGE_MAP.get(st_code, "UNKNOWN")
                    if st_policy != expected_storage:
                        raise RuntimeError(f"Storage policy check failed on {tbl_name}: expected {expected_storage}, got {st_policy}")

                for rep in REPRESENTATIONS:
                    tbl_name = f"chunks_{scale_tag}_{rep}"
                    idx_name = f"{tbl_name}_hnsw_idx"
                    expected_opc = "halfvec_cosine_ops" if "fp16" in rep else "vector_cosine_ops"
                    cur.execute("""
                        SELECT opc.opcname, i.indisvalid
                        FROM pg_index i
                        JOIN pg_class c ON c.oid = i.indexrelid
                        JOIN pg_class t ON t.oid = i.indrelid
                        JOIN pg_namespace n ON n.oid = t.relnamespace
                        JOIN pg_opclass opc ON opc.oid = i.indclass[0]
                        WHERE n.nspname = %s AND t.relname = %s AND c.relname = %s;
                    """, (SCHEMA_NAME, tbl_name, idx_name))
                    idx_info = cur.fetchone()
                    if not idx_info:
                        raise RuntimeError(f"Index {idx_name} not found after creation!")
                    if not idx_info[1]:
                        raise RuntimeError(f"Index {idx_name} is marked INVALID!")
                    if idx_info[0] != expected_opc:
                        raise RuntimeError(f"Index {idx_name} operator class mismatch: expected {expected_opc}, got {idx_info[0]}")

                # Step 8: Explicit in-transaction ANALYZE on all three experimental tables before commit
                print("  - Executing in-transaction ANALYZE on experimental tables...")
                for rep in REPRESENTATIONS:
                    tbl_name = f"chunks_{scale_tag}_{rep}"
                    cur.execute(sql.SQL("ANALYZE {schema}.{table};").format(
                        schema=sql.Identifier(SCHEMA_NAME),
                        table=sql.Identifier(tbl_name)
                    ))
                    print(f"    ANALYZE {SCHEMA_NAME}.{tbl_name} completed.")

            # Step 9: Commit the entire scale ONLY after all operations, validations, and ANALYZE succeed
            conn.commit()
            print(f"[COMMITTED] Scale '{scale_tag}' transaction COMMITTED successfully.")

        except Exception as e:
            conn.rollback()
            print(f"[ROLLED BACK] Provisioning scale '{scale_tag}' failed. Transaction ROLLED BACK completely: {e}")
            raise

        # 4. Post-commit physical inspection (read-only)
        for rep in REPRESENTATIONS:
            m_info = inspect_table_metrics(conn, scale_tag, rep)
            metrics_list.append(m_info)
            print(f"  Storage footprint ({rep}): Heap={m_info['heap_bytes']/1024:.1f} KiB, TOAST={m_info['toast_total_bytes']/1024:.1f} KiB, Index={m_info['hnsw_index_bytes']/1024:.1f} KiB")

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
        "schema", "table_name", "scale", "representation", "status", "storage_policy",
        "row_count", "reltuples", "relpages", "vector_dims",
        "min_col_size_bytes", "max_col_size_bytes", "avg_col_size_bytes",
        "column_size_min_bytes", "column_size_max_bytes", "column_size_avg_bytes",
        "heap_bytes", "toast_table_bytes", "toast_index_bytes", "toast_total_bytes",
        "pk_index_bytes", "hnsw_index_bytes", "all_indexes_bytes", "total_inclusive_bytes"
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
