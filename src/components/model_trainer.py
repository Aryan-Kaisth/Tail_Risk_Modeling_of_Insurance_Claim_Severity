import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
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