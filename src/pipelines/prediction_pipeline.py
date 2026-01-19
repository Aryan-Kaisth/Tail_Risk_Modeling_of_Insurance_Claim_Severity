import os
import sys
import joblib
from src.logger import logging
from src.exception import CustomException

class PredictionPipeline:
    def __init__(self):
        try:
            model_path = os.path.join(
                "artifacts", "model_trainer", "catboost.pkl"
            )

            logging.info(f"[INIT] Loading model from {model_path}")
            self.model = joblib.load(model_path)

            logging.info("[INIT] PredictionPipeline ready")

        except Exception as e:
            logging.exception("[INIT] Model loading failed")
            raise CustomException(e, sys)

    def predict(self, features):
        """
        Generate predictions using the trained CatBoost model.

        Parameters
        ----------
        features : pandas.DataFrame or array-like
            Input features for prediction.

        Returns
        -------
        float or np.ndarray
            Predicted quantile value(s).

        Raises
        ------
        CustomException
            If prediction fails.
        """
        try:
            logging.info("[PREDICT] Starting inference")

            predictions = self.model.predict(features)

            logging.info("[PREDICT] Inference completed successfully")

            return predictions

        except Exception as e:
            logging.exception("[PREDICT] Inference failed")
            raise CustomException(e, sys)
