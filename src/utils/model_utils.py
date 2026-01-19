from src.logger import logging
from src.exception import CustomException
from catboost import CatBoostRegressor
import sys

from sklearn.metrics import (
    mean_pinball_loss,
    d2_pinball_score
)

def get_catboost_model(quantile: float = 0.9):
    """
    Create and return a CatBoost quantile regression model.

    Parameters
    ----------
    quantile : float, default=0.9
        Quantile level (e.g., 0.5, 0.75, 0.9, 0.95).

    Returns
    -------
    CatBoostRegressor
        Configured CatBoost quantile regression model.

    Raises
    ------
    CustomException
        If model initialization fails.
    """
    try:
        logging.info(
            f"[MODEL INIT] Initializing CatBoost Quantile model "
            f"(alpha={quantile})"
        )

        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={quantile}",
            eval_metric=f"Quantile:alpha={quantile}",
            iterations=3000,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=5,
            min_data_in_leaf=50,
            random_seed=42,
            verbose=False
        )

        logging.info("[MODEL INIT] CatBoost model initialized successfully.")
        return model

    except Exception as e:
        logging.error("[MODEL INIT] Failed to initialize CatBoost model.")
        raise CustomException(e, sys)


def get_metrics(model, X_test, y_test, quantile: float = 0.9):
    """
    Evaluate a quantile regression model using pinball loss,
    D² pinball score, and coverage.

    Parameters
    ----------
    model : trained model
        Trained quantile regression model.
    X_test : pandas.DataFrame or array-like
        Test features.
    y_test : pandas.Series or array-like
        True target values.
    quantile : float, default=0.9
        Quantile level used during model training.

    Returns
    -------
    dict
        Dictionary containing:
        - pinball_loss : float
        - d2_pinball : float
        - coverage : float

    Raises
    ------
    CustomException
        If evaluation fails.
    """
    try:
        logging.info(
            f"[MODEL EVAL] Evaluating model for quantile={quantile}"
        )

        # Predictions
        y_pred = model.predict(X_test)

        # Mean pinball loss
        pinball = mean_pinball_loss(
            y_test,
            y_pred,
            alpha=quantile
        )

        # D² pinball score
        d2 = d2_pinball_score(
            y_test,
            y_pred,
            alpha=quantile
        )

        # Coverage
        coverage = (y_test <= y_pred).mean()

        metrics = {
            "pinball_loss": pinball,
            "d2_pinball": d2,
            "coverage": coverage
        }

        logging.info(
            f"[MODEL EVAL] Evaluation completed | "
            f"Pinball={pinball:.4f}, "
            f"D²={d2:.4f}, "
            f"Coverage={coverage:.4f}"
        )

        return metrics

    except Exception as e:
        logging.error("[MODEL EVAL] Model evaluation failed.")
        raise CustomException(e, sys)