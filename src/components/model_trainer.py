import sys
import os

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        'artifacts',
        'model.pkl'
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_path
    ):

        try:

            logging.info(
                "Split training and test input data"
            )


            # Separate input features and target

            X_train = train_array.iloc[:, :-1]
            y_train = train_array.iloc[:, -1]

            X_test = test_array.iloc[:, :-1]
            y_test = test_array.iloc[:, -1]


            logging.info(
                f"X_train shape: {X_train.shape}"
            )

            logging.info(
                f"X_test shape: {X_test.shape}"
            )

            logging.info(
                f"y_train shape: {y_train.shape}"
            )

            logging.info(
                f"y_test shape: {y_test.shape}"
            )


            # Define models

            models = {

                "Random Forest":
                    RandomForestRegressor(),

                "Decision Tree":
                    DecisionTreeRegressor(),

                "Gradient Boosting":
                    GradientBoostingRegressor(),

                "Linear Regression":
                    LinearRegression(),

                "K-Neighbors Regressor":
                    KNeighborsRegressor(),

                "XGB Regressor":
                    XGBRegressor(),

                "CatBoosting Regressor":
                    CatBoostRegressor(
                        verbose=False
                    ),

                "AdaBoost Regressor":
                    AdaBoostRegressor()
            }


            # Evaluate all models

            model_report: dict = evaluate_model(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models
            )


            logging.info(
                f"Model Report: {model_report}"
            )


            # Find best model score

            best_model_score = max(
                model_report.values()
            )


            # Find best model name

            best_model_name = list(
                model_report.keys()
            )[
                list(
                    model_report.values()
                ).index(
                    best_model_score
                )
            ]


            # Get best model

            best_model = models[
                best_model_name
            ]


            logging.info(
                f"Best model: {best_model_name}"
            )

            logging.info(
                f"Best model score: {best_model_score}"
            )


            # Check minimum score

            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found with score greater than 0.6",
                    sys
                )


            logging.info(
                f"Best model found: [{best_model_name}]"
            )


            # Save best model

            save_object(

                file_path=(
                    self
                    .model_trainer_config
                    .trained_model_file_path
                ),

                obj=best_model
            )


            logging.info(
                "Best model saved successfully"
            )


            # Calculate R2 score

            predictions = best_model.predict(
                X_test
            )

            r2_square = r2_score(
                y_test,
                predictions
            )


            logging.info(
                f"R2 Square: {r2_square}"
            )


            return r2_square


        except Exception as e:

            raise CustomException(
                e,
                sys
            )