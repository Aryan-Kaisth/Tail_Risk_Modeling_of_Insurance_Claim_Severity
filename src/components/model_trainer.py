import os
import sys
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils.model_utils import (
    get_catboost_model,
    get_metrics,
)
from src.utils.main_utils import save_object

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation

# Config
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join(
        "artifacts", "model_trainer", "catboost_quantile.pkl"
    )


# Model Trainer
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        try:
            self.config = config
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

            logging.info(
                f"[INIT] ModelTrainer initialized | Model path: {self.config.model_path}"
            )

        except Exception as e:
            raise CustomException(e, sys)

    # --------------------------------------------------
    def initiate_model_trainer(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        quantile: float = 0.9,
    ):
        """
        Train CatBoost quantile regression model and evaluate performance.
        """
        try:
            logging.info("[TRAINER] Creating CatBoost quantile model")
            model = get_catboost_model(quantile=quantile)

            # Detect categorical features from TRANSFORMED data
            cat_features = X_train.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            logging.info(
                f"[TRAINER] Detected {len(cat_features)} categorical features"
            )

            logging.info("[TRAINER] Training started")
            model.fit(
                X_train,
                y_train,
                cat_features=cat_features,
            )

            logging.info("[TRAINER] Training completed")

            logging.info("[TRAINER] Evaluating model")
            metrics = get_metrics(
                model=model,
                X_test=X_test,
                y_test=y_test,
                quantile=quantile,
            )

            logging.info(f"[TRAINER] Metrics: {metrics}")

            logging.info("[TRAINER] Saving trained model")
            save_object(
                file_path=self.config.model_path,
                obj=model,
            )

            logging.info("[TRAINER] ModelTrainer pipeline completed successfully")

            return model, metrics

        except Exception as e:
            logging.exception("[TRAINER] Model training failed")
            raise CustomException(e, sys)


# Testing
if __name__ == "__main__":

    # Data Ingestion
    ingestion = DataIngestion(DataIngestionConfig())
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Data Transformation (path-based)
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(
        train_path=train_path,
        test_path=test_path,
    )

    # Model Training
    trainer = ModelTrainer(ModelTrainerConfig())
    model, metrics = trainer.initiate_model_trainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        quantile=0.9,
    )

    print("✅ Training completed successfully")
    print("📊 Metrics:", metrics)
