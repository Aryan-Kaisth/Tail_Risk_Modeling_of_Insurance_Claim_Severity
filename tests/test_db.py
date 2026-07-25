from unittest.mock import MagicMock, patch
import pandas as pd
import psycopg
import pytest

from src.exception import CustomException
from src.utils.db import get_connection, fetch_dataframe


import os


@patch("src.utils.db.psycopg.connect")
def test_get_connection_success(mock_connect, database_env_vars):
    """Verifies successful connection creation with environment parameters."""
    fake_connection = MagicMock()
    mock_connect.return_value = fake_connection

    conn = get_connection()

    assert conn is fake_connection
    # Assert using environment variables injected by database_env_vars fixture
    mock_connect.assert_called_once_with(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
    )


@pytest.mark.parametrize(
    "error_scenario", ["missing_env_var", "operational_error", "unexpected_error"]
)
@patch("src.utils.db.psycopg.connect")
def test_get_connection_failure_modes(
    mock_connect, database_env_vars, monkeypatch, error_scenario: str
):
    """Tests missing environment variables, DB down, and unexpected errors."""
    if error_scenario == "missing_env_var":
        monkeypatch.delenv("DB_HOST", raising=False)
    elif error_scenario == "operational_error":
        mock_connect.side_effect = psycopg.OperationalError("Server unreachable")
    elif error_scenario == "unexpected_error":
        mock_connect.side_effect = RuntimeError("Unexpected connection failure")

    with pytest.raises(CustomException):
        get_connection()

    if error_scenario == "missing_env_var":
        mock_connect.assert_not_called()


@patch("src.utils.db.get_connection")
@patch("src.utils.db.pd.read_sql_query")
def test_fetch_dataframe_success(mock_read_sql, mock_get_connection):
    """Verifies fetch_dataframe manages DB connection context and returns DataFrame."""
    # Setup mock connection context manager: with get_connection() as conn
    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn

    # Setup expected DataFrame
    expected_df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    mock_read_sql.return_value = expected_df

    query = "SELECT * FROM users;"
    result_df = fetch_dataframe(query)

    # Verify context opened, SQL executed, and DataFrame returned correctly
    pd.testing.assert_frame_equal(result_df, expected_df)
    mock_read_sql.assert_called_once_with(query, mock_conn)


@pytest.mark.parametrize(
    "query_error",
    [
        psycopg.errors.UndefinedTable("Table 'users' does not exist"),
        psycopg.errors.UndefinedColumn("Column 'invalid_col' does not exist"),
        psycopg.ProgrammingError("Syntax error in SQL query"),
        RuntimeError("Network socket closed unexpectedly"),
    ],
)
@patch("src.utils.db.get_connection")
@patch("src.utils.db.pd.read_sql_query")
def test_fetch_dataframe_failure_modes(mock_read_sql, mock_get_connection, query_error):
    """Verifies that invalid SQL or DB execution failures map to CustomException."""
    mock_conn = MagicMock()
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    mock_read_sql.side_effect = query_error

    with pytest.raises(CustomException):
        fetch_dataframe("SELECT * FROM invalid_table;")
