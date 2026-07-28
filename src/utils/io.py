# src/utils/io.py
import sys
import joblib
import pandas as pd
from pathlib import Path
from src.logger import get_logger
from src.exception import CustomException
import yaml
import json

logger = get_logger(__name__)


def read_csv_file(file_path: Path) -> pd.DataFrame:
    try:
        logger.debug("Reading CSV from: %s", file_path)
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def save_csv_file(data: pd.DataFrame, file_path: Path) -> None:
    try:
        logger.debug("Writing DataFrame shape %s to: %s", data.shape, file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(file_path, index=False)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: Path) -> object:
    try:
        logger.debug("Loading object from: %s", file_path)
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path: Path, obj: object) -> None:
    try:
        logger.debug("Saving object to: %s", file_path)
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)


def read_yaml_file(file_path: Path) -> dict:
    try:
        logger.debug("Reading YAML from: %s", file_path)

        with open(file_path, "r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise CustomException(e, sys)

def write_json_file(file_path: Path, data: dict) -> None:
    try:
        logger.debug("Writing JSON file to: %s", file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4)

        logger.info("JSON file written successfully: %s", file_path)

    except Exception as e:
        raise CustomException(e, sys)

def write_text_file(file_path: Path, content: str) -> None:
    try:
        logger.debug("Writing text file to: %s", file_path)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as text_file:
            text_file.write(content)

        logger.info("Text file written successfully: %s", file_path)

    except Exception as e:
        raise CustomException(e, sys)