from unittest.mock import MagicMock, patch

import psycopg
import pytest

from src.exception import CustomException
from src.utils.db import get_connection

# Test cases for get_connection()
@patch("src.utils.db.psycopg.connect")
def test_get_connection_success(mock_connect, database_env_vars):
    # Return a fake connection.
    fake_connection = MagicMock()
    mock_connect.return_value = fake_connection

    conn = get_connection()

    assert conn is fake_connection
    mock_connect.assert_called_once()


@patch("src.utils.db.psycopg.connect")
def test_get_connection_operational_error(mock_connect, database_env_vars):
    # Simulate a connection failure.
    mock_connect.side_effect = psycopg.OperationalError("Server down")

    with pytest.raises(CustomException):
        get_connection()

    mock_connect.assert_called_once()


@patch("src.utils.db.psycopg.connect")
def test_get_connection_missing_env_variable(
    mock_connect,
    database_env_vars,
    monkeypatch,
):
    # Remove one required environment variable.
    monkeypatch.delenv("DB_HOST", raising=False)

    with pytest.raises(CustomException):
        get_connection()

    mock_connect.assert_not_called()


@patch("src.utils.db.psycopg.connect")
def test_get_connection_unexpected_error(mock_connect, database_env_vars):
    # Simulate an unexpected failure.
    mock_connect.side_effect = RuntimeError("Something unexpected happened")

    with pytest.raises(CustomException):
        get_connection()

    mock_connect.assert_called_once()