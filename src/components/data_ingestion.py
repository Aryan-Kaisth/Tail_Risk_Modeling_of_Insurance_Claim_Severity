# src/components/data_ingestion.py

import sys
from pathlib import Path
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import get_logger
from src.utils.db import fetch_dataframe
from src.utils.io import save_csv_file

logger = get_logger(__name__)


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path = Path("artifacts") / "data_ingestion" / "raw.csv"
    train_data_path: Path = Path("artifacts") / "data_ingestion" / "train.csv"
    test_data_path: Path = Path("artifacts") / "data_ingestion" / "test.csv"
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


@dataclass(frozen=True)
class DataIngestionArtifact:
    train_data_path: Path
    test_data_path: Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self, query: str) -> DataIngestionArtifact:
        try:
            logger.info("--- Starting Data Ingestion Stage ---")

            # Fetch & Save Raw Data
            data = fetch_dataframe(query)
            logger.info("Ingested raw dataset from database. Shape: %s", data.shape)

            save_csv_file(data, self.config.raw_data_path)

            # Split Dataset
            train_data, test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle,
            )
            logger.info(
                "Split dataset (test_size=%.2f) -> Train: %d rows | Test: %d rows",
                self.config.test_size,
                len(train_data),
                len(test_data),
            )

            # Save Splitted Artifacts
            save_csv_file(train_data, self.config.train_data_path)
            save_csv_file(test_data, self.config.test_data_path)

            logger.info(
                "Saved artifacts successfully to directory: '%s'",
                self.config.raw_data_path.parent,
            )
            logger.info("--- Data Ingestion Stage Completed ---")

            return DataIngestionArtifact(
                train_data_path=self.config.train_data_path,
                test_data_path=self.config.test_data_path,
            )

        except Exception as e:
            logger.exception("Data Ingestion stage failed.")
            raise CustomException(e, sys)
