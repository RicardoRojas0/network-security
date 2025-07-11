import os
import sys
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from src.constants.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.utils.utils import (
    save_numpy_array_data,
    save_preprocessor,
)


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(error_message=e)

    @staticmethod
    def _read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(error_message=e)

    def knn_imputer(cls) -> Pipeline:
        """
        Initiates KNN Imputer with specified parameters in training_pipeline
        and returns a Pipeline Object with KNN Imputer as first step.
        """
        try:
            knn_imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(
                f"Intializing KNN Imputer with params: {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            preprocessor = Pipeline([("imputer", knn_imputer)])
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(error_message=e)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            df_train = DataTransformation._read_data(
                self.data_validation_artifact.validated_train_file_path
            )
            df_test = DataTransformation._read_data(
                self.data_validation_artifact.validated_test_file_path
            )

            # Splitting train and test sets into features and target
            df_train_features = df_train.drop(columns=[TARGET_COLUMN])
            df_train_target = df_train[TARGET_COLUMN]
            df_train_target = df_train_target.replace(-1, 0)

            df_test_features = df_test.drop(columns=[TARGET_COLUMN])
            df_test_target = df_test[TARGET_COLUMN]
            df_test_target = df_test_target.replace(-1, 0)

            # KNN Imputer
            imputer = self.knn_imputer()
            preprocessor = imputer.fit(df_train_features)
            df_train_features_transformed = preprocessor.transform(df_train_features)
            df_test_features_transformed = preprocessor.transform(df_test_features)

            train_array_transformed = np.c_[
                df_train_features_transformed, np.array(df_train_target)
            ]
            test_array_transformed = np.c_[
                df_test_features_transformed, np.array(df_test_target)
            ]

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_array_transformed,
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_array_transformed,
            )

            save_preprocessor(
                file_path=self.data_transformation_config.preprocessor_file_path,
                preprocessor=preprocessor,
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_file_path=self.data_transformation_config.preprocessor_file_path,
            )
            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(error_message=e)
