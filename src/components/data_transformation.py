import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import CustomException
from src.utils.main_utils import (
    save_object,
    read_yaml_file,
    read_csv_file,
)

from feature_engine.encoding import RareLabelEncoder
from feature_engine.selection import DropCorrelatedFeatures, DropConstantFeatures

import warnings
warnings.filterwarnings("ignore")

# Config
@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join(
        "artifacts", "data_transformation", "preprocessor.pkl"
    )
    schema_path: str = os.path.join("config", "schema.yaml")


# Data Transformation
class DataTransformation:
    def __init__(self):
        try:
            self.config = DataTransformationConfig()
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)

            # Load schema
            self.schema = read_yaml_file(self.config.schema_path)
            self.cat_cols = self.schema["categorical_cols"]
            self.num_cols = self.schema["numerical_cols"]
            self.target_col = self.schema["target_col"]

            logging.info("[TRANSFORMATION] Schema loaded successfully")

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def _drop_null_rows(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        Drop rows with nulls only in required columns
        """
        required_cols = self.cat_cols + self.num_cols + [self.target_col]

        before = df.shape[0]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        after = df.shape[0]

        logging.info(
            f"[NULL HANDLING] {name}: dropped {before - after} rows "
            f"(from {before} → {after})"
        )

        return df

    # --------------------------------------------------
    def get_preprocessor(self) -> Pipeline:
        """
        Pipeline order:
        1. Drop constant categorical features
        2. Drop correlated numerical features
        3. Rare label encoding (dynamic categorical detection)
        """
        try:
            return Pipeline(
                steps=[
                    (
                        "drop_constant_features",
                        DropConstantFeatures(
                            tol=0.99,
                            variables=self.cat_cols,
                        ),
                    ),
                    (
                        "drop_correlated_features",
                        DropCorrelatedFeatures(
                            variables=self.num_cols,
                        ),
                    ),
                    (
                        "rare_label_encoding",
                        RareLabelEncoder(
                            max_n_categories=5,
                            replace_with="Other",
                            variables=None,
                            missing_values="ignore",
                        ),
                    ),
                ]
            )

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: str,
    ):
        """
        Read CSVs internally, fit on train, transform train & test
        """
        try:
            logging.info("===== DATA TRANSFORMATION STARTED =====")

            # Read data
            train_df = read_csv_file(train_path)
            test_df = read_csv_file(test_path)

            # Drop null rows
            train_df = self._drop_null_rows(train_df, "TRAIN")
            test_df = self._drop_null_rows(test_df, "TEST")

            # Split features & target
            X_train = train_df.drop(columns=[self.target_col])
            y_train = train_df[self.target_col]

            X_test = test_df.drop(columns=[self.target_col])
            y_test = test_df[self.target_col]

            # Build & fit preprocessor
            preprocessor = self.get_preprocessor()

            logging.info("[TRANSFORMATION] Fitting preprocessor on training data")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Save preprocessor
            save_object(
                file_path=self.config.preprocessor_path,
                obj=preprocessor,
            )

            logging.info("[TRANSFORMATION] Preprocessor saved successfully")
            logging.info("===== DATA TRANSFORMATION COMPLETED =====")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("[TRANSFORMATION] Error in Data Transformation")
            raise CustomException(e, sys)


# Testing
if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion, DataIngestionConfig

    # 1️⃣ Ingestion
    ingestion = DataIngestion(DataIngestionConfig())
    train_path, test_path = ingestion.initiate_data_ingestion()

    # 2️⃣ Transformation
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
    )

    print("✅ X_train shape:", X_train.shape)
    print("✅ X_test shape:", X_test.shape)
    print("✅ y_train shape:", y_train.shape)
    print("✅ y_test shape:", y_test.shape)
