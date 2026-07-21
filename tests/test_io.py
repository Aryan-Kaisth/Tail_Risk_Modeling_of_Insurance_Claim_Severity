import joblib
import pandas as pd
import pytest
from unittest.mock import patch

from src.exception import CustomException
from src.utils.io import (
    save_csv_file,
    read_csv_file,
    save_object,
    load_object,
)


def test_save_csv_file_success(tmp_path):
    # Save into a nested path to verify missing directories are created.
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
        }
    )

    file_path = tmp_path / "level1" / "output.csv"

    save_csv_file(df, file_path)

    assert file_path.exists()

    # Verify the saved contents match the original DataFrame.
    saved_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(saved_df, df)


@patch("src.utils.io.pd.DataFrame.to_csv")
def test_save_csv_file_failure(mock_to_csv, tmp_path):
    # Simulate an unexpected failure while writing the CSV.
    mock_to_csv.side_effect = RuntimeError("Disk full")

    df = pd.DataFrame({"A": [1]})

    with pytest.raises(CustomException):
        save_csv_file(df, tmp_path / "output.csv")

    mock_to_csv.assert_called_once()


def test_read_csv_file_success(tmp_path):
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
        }
    )

    file_path = tmp_path / "output.csv"
    df.to_csv(file_path, index=False)

    result = read_csv_file(file_path)

    pd.testing.assert_frame_equal(result, df)


def test_read_csv_file_not_found():
    # Reading a non-existent file should raise our custom exception.
    with pytest.raises(CustomException):
        read_csv_file("this_file_does_not_exist.csv")


def test_read_csv_file_empty_csv(tmp_path):
    # Create an empty CSV file.
    file_path = tmp_path / "empty.csv"
    file_path.touch()

    with pytest.raises(CustomException):
        read_csv_file(file_path)


@patch("src.utils.io.pd.read_csv")
def test_read_csv_file_parser_error(mock_read_csv):
    # Force pandas to raise a ParserError.
    mock_read_csv.side_effect = pd.errors.ParserError("Invalid CSV")

    with pytest.raises(CustomException):
        read_csv_file("corrupted.csv")

    mock_read_csv.assert_called_once_with("corrupted.csv")


@patch("src.utils.io.pd.read_csv")
def test_read_csv_file_unexpected_error(mock_read_csv):
    # Simulate an unexpected error from pandas.
    mock_read_csv.side_effect = RuntimeError("Unexpected error")

    with pytest.raises(CustomException):
        read_csv_file("dummy.csv")

    mock_read_csv.assert_called_once_with("dummy.csv")


def test_load_object_success(tmp_path):
    obj = {"a": 1}

    file_path = tmp_path / "model.pkl"

    joblib.dump(obj, file_path)

    loaded = load_object(file_path)

    # Loaded object should have the same contents.
    assert loaded == obj


def test_load_object_file_not_found():
    with pytest.raises(CustomException):
        load_object("does_not_exist.pkl")


@patch("src.utils.io.joblib.load")
def test_load_object_eof_error(mock_load, tmp_path):
    # Simulate loading a corrupted/truncated object file.
    mock_load.side_effect = EOFError("Unexpected end of file")

    file_path = tmp_path / "corrupted.pkl"
    file_path.touch()

    with pytest.raises(CustomException):
        load_object(file_path)

    mock_load.assert_called_once_with(file_path)


@patch("src.utils.io.joblib.load")
def test_load_object_unexpected_error(mock_load):
    # Simulate an unexpected error while loading the object.
    mock_load.side_effect = RuntimeError("Unexpected error")

    with pytest.raises(CustomException):
        load_object("dummy.pkl")

    mock_load.assert_called_once_with("dummy.pkl")


def test_save_object_success(tmp_path):
    # Save into a nested path to verify directory creation.
    obj = {
        "name": "Alice",
        "age": 25,
    }

    file_path = tmp_path / "models" / "v1" / "model.pkl"

    save_object(file_path, obj)

    assert file_path.exists()

    loaded = joblib.load(file_path)

    # Verify the object was serialized and restored correctly.
    assert loaded == obj


@patch("src.utils.io.joblib.dump")
def test_save_object_dump_failure(mock_dump):
    # Simulate a failure while serializing the object.
    mock_dump.side_effect = RuntimeError("Disk full")

    obj = {"x": 1}

    with pytest.raises(CustomException):
        save_object("dummy.pkl", obj)

    mock_dump.assert_called_once()
