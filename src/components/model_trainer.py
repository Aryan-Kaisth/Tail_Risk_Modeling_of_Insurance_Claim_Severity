import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.utils.model_utils import (
    get_catboost_model,
    get_metrics,
)

from src.utils.main_utils import (
    save_object,
    read_yaml_file
)

# CONFIG
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join(
        "artifacts", "model_trainer", "catboost.pkl"
    )

# TRAINER
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        logging.info(
            f"[INIT] ModelTrainer initialized | Model path: {self.config.model_path}"
        )

    # -----------------------------------------------------
    def initiate_model_trainer(self, X_train, X_test, y_train, y_test, cat_features: list | None = None, quantile: float = 0.9):
        """
        Train CatBoost quantile regression model and evaluate performance.

        Parameters
        ----------
        X_train, X_test : pd.DataFrame
            Training and testing features.
        y_train, y_test : pd.Series
            Training and testing targets.
        cat_features : list, optional
            List of categorical feature names.
        quantile : float, default=0.9
            Quantile level for regression.

        Returns
        -------
        model : CatBoostRegressor
            Trained CatBoost model.
        metrics : dict
            Dictionary containing D² pinball score and coverage.
        """
        try:
            logging.info("[TRAINER] Fetching CatBoost model from utils")
            model = get_catboost_model(quantile=quantile)

            logging.info("[TRAINER] Model training started")

            if cat_features:
                model.fit(
                    X_train,
                    y_train,
                    cat_features=cat_features
                )
            else:
                model.fit(X_train, y_train)

            logging.info("[TRAINER] Model training completed")

            logging.info("[TRAINER] Evaluating model performance")
            metrics = get_metrics(
                model=model,
                X_test=X_test,
                y_test=y_test,
                quantile=quantile
            )

            logging.info("[TRAINER] Saving trained model")
            save_object(
                file_path=self.config.model_path,
                obj=model
            )

            logging.info("[TRAINER] ModelTrainer pipeline completed successfully")

            return model, metrics

        except Exception as e:
            logging.exception("[TRAINER] Model training pipeline failed")
            raise CustomException(e, sys)


# ---- Testing ----
cat_cols = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14', 'cat15', 'cat16', 'cat17', 'cat18', 'cat19', 'cat20', 'cat21', 'cat22', 'cat23', 'cat24', 'cat25', 'cat26', 'cat27', 'cat28', 'cat29', 'cat30', 'cat31', 'cat32', 'cat33', 'cat34', 'cat35', 'cat36', 'cat37', 'cat38', 'cat39', 'cat40', 'cat41', 'cat42', 'cat43', 'cat44', 'cat45', 'cat46', 'cat47', 'cat48', 'cat49', 'cat50', 'cat51', 'cat52', 'cat53', 'cat54', 'cat55', 'cat56', 'cat57', 'cat58', 'cat59', 'cat60', 'cat61', 'cat62', 'cat63', 'cat64', 'cat65', 'cat66', 'cat67', 'cat68', 'cat69', 'cat70', 'cat71', 'cat72', 'cat73', 'cat74', 'cat75', 'cat76', 'cat77', 'cat78', 'cat79', 'cat80', 'cat81', 'cat82', 'cat83', 'cat84', 'cat85', 'cat86', 'cat87', 'cat88', 'cat89', 'cat90', 'cat91', 'cat92', 'cat93', 'cat94', 'cat95', 'cat96', 'cat97', 'cat98', 'cat99', 'cat100', 'cat101', 'cat102', 'cat103', 'cat104', 'cat105', 'cat106', 'cat107', 'cat108', 'cat109', 'cat110', 'cat111', 'cat112', 'cat113', 'cat114', 'cat115', 'cat116']

ingest_config = DataIngestionConfig()
data_ngestion = DataIngestion(config=ingest_config)
train_data_path, test_data_path = data_ngestion.initiate_data_ingestion()
train_data = pd.read_csv(train_data_path).dropna(axis=0).reset_index()
test_data = pd.read_csv(test_data_path).dropna(axis=0).reset_index()
X_train = train_data.drop('loss', axis=1)
y_train = train_data.loss
X_test = test_data.drop('loss', axis=1)
y_test = test_data.loss

model_config = ModelTrainerConfig()
model_trainer = ModelTrainer(config=model_config)
model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test, cat_cols, 0.9)
