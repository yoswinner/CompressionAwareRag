import os
import sys

# CRITICAL TEST SUITE ISOLATION:
# Force DB_NAME to compression_aware_rag_test BEFORE any project modules read config
os.environ["DB_NAME"] = "compression_aware_rag_test"

import pytest
import psycopg
from src.config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, get_db_name, get_db_connection_string
from src.database import get_connection, initialize_schema, clear_database, assert_test_database, TEST_DB_NAME

@pytest.fixture(scope="session", autouse=True)
def ensure_test_database_ready():
    """
    Session-level fixture:
    1. Verifies that DB_NAME is strictly compression_aware_rag_test.
    2. Ensures compression_aware_rag_test database exists in PostgreSQL.
    3. Initializes the required schema within compression_aware_rag_test.
    Never touches or falls back to compression_aware_rag.
    """
    if get_db_name() != TEST_DB_NAME:
        raise RuntimeError(
            f"FATAL: get_db_name() is '{get_db_name()}', expected '{TEST_DB_NAME}'. "
            f"Test execution aborted to prevent accidental production database mutation."
        )

    # Check if database exists, create if missing
    admin_conn_str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/postgres"
    with psycopg.connect(admin_conn_str, autocommit=True) as admin_conn:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (TEST_DB_NAME,))
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {TEST_DB_NAME};")

    # Connect to test DB and ensure schema is initialized
    conn = get_connection()
    try:
        assert_test_database(conn)
        initialize_schema(conn=conn)
    finally:
        conn.close()

@pytest.fixture(autouse=True)
def fail_closed_test_db_guard():
    """
    Autouse fixture executed before each test:
    Fail-closed guard ensuring the active database is strictly compression_aware_rag_test.
    Aborts test execution immediately if targeting any other database.
    """
    active_env_db = os.environ.get("DB_NAME")
    if active_env_db != TEST_DB_NAME:
        pytest.fail(
            f"FAIL-CLOSED SAFETY GUARD TRIGGERED: os.environ['DB_NAME'] is '{active_env_db}', "
            f"not '{TEST_DB_NAME}'. Aborting to prevent production database mutation."
        )

    conn = get_connection()
    try:
        assert_test_database(conn)
    finally:
        conn.close()

@pytest.fixture
def clean_test_db():
    """
    Fixture providing a clean test database.
    Confines truncation strictly to compression_aware_rag_test.
    """
    assert_test_database()
    clear_database()
    yield
    assert_test_database()
    clear_database()
