import os
import sys
import joblib
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)


def read_csv_file(file_path: str) -> pd.DataFrame:
    try:
        logger.info(f"Reading CSV file from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info("Successfully read CSV file")
        return df

    # Catch specific pandas/file errors first before custom exception
    except FileNotFoundError as e:
        logger.error(f"CSV file not found: {e}")
        raise CustomException(e, sys)

    except pd.errors.EmptyDataError as e:
        logger.error(f"CSV file is empty: {e}")
        raise CustomException(e, sys)

    except pd.errors.ParserError as e:
        logger.error(f"Corrupted CSV structure: {e}")
        raise CustomException(e, sys)

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        raise CustomException(e, sys)


def save_csv_file(data: pd.DataFrame, file_path: str) -> None:
    try:
        logger.info(f"Saving DataFrame to: {file_path}")

        # Make sure target dir exists
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        data.to_csv(file_path, index=False)
        logger.info("Successfully saved CSV")

    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> object:
    try:
        logger.info(f"Loading object from: {file_path}")
        obj = joblib.load(file_path)
        logger.info("Successfully loaded object")
        return obj

    except FileNotFoundError as e:
        logger.error(f"Object file not found: {e}")
        raise CustomException(e, sys)

    except EOFError as e:
        logger.error(f"Invalid object file: {e}")
        raise CustomException(e, sys)

    except Exception as e:
        logger.error(f"Failed to load object: {e}")
        raise CustomException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    try:
        logger.info(f"Saving object to: {file_path}")

        # Create dir if missing
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)
        logger.info("Successfully saved object")

    except Exception as e:
        logger.error(f"Failed to save object: {e}")
        raise CustomException(e, sys)
