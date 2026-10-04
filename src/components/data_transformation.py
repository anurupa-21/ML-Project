import sys
import os
import pandas as pd

from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path: str = os.path.join(
        'artifacts',
        'preprocessor.pkl'
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = (
            DataTransformationConfig()
        )


    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation.
        '''

        try:

            numerical_columns = [
                'math_score',
                'reading_score',
                'writing_score'
            ]

            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]


            # Numerical Pipeline

            num_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='median'
                        )
                    ),
                    (
                        'scaler',
                        StandardScaler()
                    )
                ]
            )


            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='most_frequent'
                        )
                    ),
                    (
                        'one_hot_encoder',
                        OneHotEncoder(
                            handle_unknown='ignore'
                        )
                    ),
                    (
                        'scaler',
                        StandardScaler(
                            with_mean=False
                        )
                    )
                ]
            )


            logging.info(
                f"Categorical columns: "
                f"{categorical_columns}"
            )

            logging.info(
                f"Numerical columns: "
                f"{numerical_columns}"
            )


            # Column Transformer

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        'num_pipeline',
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        'cat_pipeline',
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor


        except Exception as e:

            raise CustomException(e, sys)


    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            # Read train and test data

            train_df = pd.read_csv(
                train_path
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(
                "Read train and test data completed"
            )

            logging.info(
                f"Train Dataframe Head:\n"
                f"{train_df.head().to_string()}"
            )

            logging.info(
                f"Test Dataframe Head:\n"
                f"{test_df.head().to_string()}"
            )


            # Get preprocessing object

            logging.info(
                "Obtaining preprocessor object"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )


            # Target column

            target_column_name = 'average'


            # Separate input features and target

            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]


            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]


            logging.info(
                "Applying preprocessing object on "
                "training dataframe and testing dataframe"
            )


            # Fit and transform training data

            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )


            # Transform testing data

            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )


            logging.info(
                "Preprocessing completed"
            )


            # Get transformed feature names

            feature_names = (
                preprocessing_obj
                .get_feature_names_out()
            )


            # Convert sparse matrix to array

            if hasattr(
                input_feature_train_arr,
                "toarray"
            ):

                input_feature_train_arr = (
                    input_feature_train_arr.toarray()
                )


            if hasattr(
                input_feature_test_arr,
                "toarray"
            ):

                input_feature_test_arr = (
                    input_feature_test_arr.toarray()
                )


            # Create transformed DataFrames

            train_arr = pd.DataFrame(
                input_feature_train_arr,
                columns=feature_names
            )

            test_arr = pd.DataFrame(
                input_feature_test_arr,
                columns=feature_names
            )


            # Reset index before adding target

            target_feature_train_df = (
                target_feature_train_df
                .reset_index(drop=True)
            )

            target_feature_test_df = (
                target_feature_test_df
                .reset_index(drop=True)
            )


            # Add target column

            train_arr[target_column_name] = (
                target_feature_train_df
            )

            test_arr[target_column_name] = (
                target_feature_test_df
            )


            logging.info(
                "Saving preprocessing object"
            )


            # Save preprocessing object

            save_object(
                file_path=(
                    self
                    .data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )


            logging.info(
                "Preprocessing object saved successfully"
            )


            return (
                train_arr,
                test_arr,
                (
                    self
                    .data_transformation_config
                    .preprocessor_obj_file_path
                )
            )


        except Exception as e:

            raise CustomException(e, sys)