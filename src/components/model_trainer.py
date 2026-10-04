import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        # IMPORTANT:
        # Need () because we need an object/instance.

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Split training and test input data"
            )

            # ---------------------------------
            # Split X and y
            # ---------------------------------

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # ---------------------------------
            # Define models
            # ---------------------------------

            models = {

                "Random Forest":
                    RandomForestRegressor(),

                "Decision Tree":
                    DecisionTreeRegressor(),

                "Gradient Boosting":
                    GradientBoostingRegressor(),

                "Linear Regression":
                    LinearRegression(),

                "XGBRegressor":
                    XGBRegressor(),

                "CatBoosting Regressor":
                    CatBoostRegressor(
                        verbose=False
                    ),

                "AdaBoost Regressor":
                    AdaBoostRegressor()
            }

            # ---------------------------------
            # Hyperparameters
            # ---------------------------------

            params = {

                # -----------------------------
                # Decision Tree
                # -----------------------------
                "Decision Tree": {
                     "criterion": [
                        "squared_error",
                        "absolute_error",
                         "poisson"
                        ]
                },

                # -----------------------------
                # Random Forest
                # -----------------------------

                "Random Forest": {

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                # -----------------------------
                # Gradient Boosting
                # -----------------------------

                "Gradient Boosting": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],

                    "subsample": [
                        0.6,
                        0.7,
                        0.75,
                        0.8,
                        0.85,
                        0.9
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                # -----------------------------
                # Linear Regression
                # -----------------------------

                "Linear Regression": {},

                # -----------------------------
                # XGBoost
                # -----------------------------

                "XGBRegressor": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                # -----------------------------
                # CatBoost
                # -----------------------------

                "CatBoosting Regressor": {

                    "depth": [
                        6,
                        8,
                        10
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],

                    "iterations": [
                        30,
                        50,
                        100
                    ]
                },

                # -----------------------------
                # AdaBoost
                # -----------------------------

                "AdaBoost Regressor": {

                    "learning_rate": [
                        0.1,
                        0.01,
                        0.5,
                        0.001
                    ],

                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                }
            }

            # ---------------------------------
            # Evaluate models
            # ---------------------------------

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            # ---------------------------------
            # Get best model score
            # ---------------------------------

            best_model_score = max(
                sorted(model_report.values())
            )

            # ---------------------------------
            # Get best model name
            # ---------------------------------

            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(best_model_score)
            ]

            # ---------------------------------
            # Get best model
            # ---------------------------------

            best_model = models[
                best_model_name
            ]

            # ---------------------------------
            # Check model performance
            # ---------------------------------

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found"
                )

            logging.info(
                "Best found model on both training "
                "and testing dataset"
            )

            logging.info(
                f"Best Model: {best_model_name}"
            )

            logging.info(
                f"Best Model Score: {best_model_score}"
            )

            # ---------------------------------
            # Save best model
            # ---------------------------------

            save_object(
                file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                obj=best_model
            )

            # ---------------------------------
            # Prediction
            # ---------------------------------

            predicted = best_model.predict(
                X_test
            )

            # ---------------------------------
            # R2 Score
            # ---------------------------------

            r2_square = r2_score(
                y_test,
                predicted
            )

            logging.info(
                f"R2 Score: {r2_square}"
            )

            return (
                best_model,
                r2_square
            )

        except Exception as e:

            raise CustomException(e, sys)