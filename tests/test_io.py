import joblib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression

from src.exception import CustomException
from src.utils.io import (
    save_csv_file,
    read_csv_file,
    save_object,
    load_object,
)

# CSV WRITE TESTS
def test_save_csv_file_creates_nested_directories(tmp_path: Path):
    """Behavior: save_csv_file automatically creates missing parent directories."""
    df = pd.DataFrame({"a": [1]})
    nested_path = tmp_path / "deep" / "nested" / "dir" / "data.csv"

    save_csv_file(df, nested_path)

    assert nested_path.exists()


def test_save_csv_file_preserves_dataframe_content(tmp_path: Path):
    """Behavior: save_csv_file accurately writes DataFrame values to disk."""
    df = pd.DataFrame({"id": [1, 2], "value": [10.5, 20.0]})
    file_path = tmp_path / "output.csv"

    save_csv_file(df, file_path)

    saved_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(saved_df, df)


@patch("src.utils.io.pd.DataFrame.to_csv")
def test_save_csv_file_raises_custom_exception_on_disk_error(mock_to_csv, tmp_path: Path):
    """Behavior: System/disk errors during CSV save map to CustomException."""
    mock_to_csv.side_effect = PermissionError("Permission denied")
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(CustomException):
        save_csv_file(df, tmp_path / "output.csv")


# CSV READ TESTS
def test_read_csv_file_returns_correct_dataframe(tmp_path: Path):
    """Behavior: read_csv_file successfully parses a valid CSV into a DataFrame."""
    df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    result_df = read_csv_file(file_path)

    pd.testing.assert_frame_equal(result_df, df)


@pytest.mark.parametrize(
    "file_scenario",
    ["missing", "empty", "corrupted_syntax"]
)
def test_read_csv_file_raises_custom_exception_on_invalid_file(tmp_path: Path, file_scenario: str):
    """Behavior: File reading failures consistently map to CustomException."""
    file_path = tmp_path / f"{file_scenario}.csv"

    if file_scenario == "empty":
        file_path.touch()
    elif file_scenario == "corrupted_syntax":
        file_path.write_text('col1,col2\n"unclosed multiline string')
    elif file_scenario == "missing":
        file_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(CustomException):
        read_csv_file(file_path)


# OBJECT SAVE TESTS
def test_save_object_creates_file_and_parent_directories(tmp_path: Path):
    """Behavior: save_object creates missing directories and serializes model."""
    model = LogisticRegression()
    model_path = tmp_path / "artifacts" / "models" / "model.pkl"

    save_object(model_path, model)

    assert model_path.exists()


@patch("src.utils.io.joblib.dump")
def test_save_object_raises_custom_exception_on_failure(mock_dump, tmp_path: Path):
    """Behavior: Serialization failures in save_object map to CustomException."""
    mock_dump.side_effect = RuntimeError("Serialization failure")

    with pytest.raises(CustomException):
        save_object(tmp_path / "model.pkl", obj={"test": 1})


# OBJECT LOAD TESTS
def test_load_object_restores_scikit_model_state(tmp_path: Path):
    """Behavior: load_object independently restores model object and attributes."""
    # Setup fixture independently using raw joblib.dump
    model = LogisticRegression(C=0.5)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)

    file_path = tmp_path / "model.pkl"
    joblib.dump(model, file_path)

    # Test load_object independently
    loaded_model = load_object(file_path)

    assert isinstance(loaded_model, LogisticRegression)
    assert loaded_model.C == 0.5
    np.testing.assert_array_equal(loaded_model.predict(X), model.predict(X))


@pytest.mark.parametrize(
    "object_scenario",
    ["missing", "corrupted_bytes"]
)
def test_load_object_raises_custom_exception_on_invalid_file(tmp_path: Path, object_scenario: str):
    """Behavior: Object loading failures consistently map to CustomException."""
    file_path = tmp_path / f"{object_scenario}.pkl"

    if object_scenario == "corrupted_bytes":
        file_path.write_bytes(b"INVALID_PICKLE_HEADER")
    elif object_scenario == "missing":
        file_path = tmp_path / "non_existent.pkl"

    with pytest.raises(CustomException):
        load_object(file_path)