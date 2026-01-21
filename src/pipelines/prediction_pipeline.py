import os
import sys
import joblib
import pandas as pd

from src.logger import logging
from src.exception import CustomException

class PredictionPipeline:
    def __init__(self):
        try:
            self.preprocessor_path = os.path.join(
                "artifacts", "data_transformation", "preprocessor.pkl"
            )
            self.model_path = os.path.join(
                "artifacts", "model_trainer", "catboost_quantile.pkl"
            )

            logging.info("[INIT] Loading preprocessor")
            self.preprocessor = joblib.load(self.preprocessor_path)

            logging.info("[INIT] Loading trained model")
            self.model = joblib.load(self.model_path)

            logging.info("[INIT] PredictionPipeline initialized successfully")

        except Exception as e:
            logging.exception("[INIT] Failed to load model or preprocessor")
            raise CustomException(e, sys)

    # --------------------------------------------------
    def predict(self, features: pd.DataFrame):
        """
        Generate predictions using trained preprocessor + CatBoost model.

        Parameters
        ----------
        features : pd.DataFrame
            Raw input features (same format as training data, without target)

        Returns
        -------
        np.ndarray
            Predicted quantile values
        """
        try:
            logging.info("[PREDICT] Starting inference pipeline")

            features = features.copy()

            # Apply preprocessing
            logging.info("[PREDICT] Applying preprocessing")
            features_transformed = self.preprocessor.transform(features)

            # Model prediction
            logging.info("[PREDICT] Generating predictions")
            predictions = self.model.predict(features_transformed)

            logging.info("[PREDICT] Inference completed successfully")

            return predictions

        except Exception as e:
            logging.exception("[PREDICT] Inference failed")
            raise CustomException(e, sys)
