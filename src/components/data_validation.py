import os
import sys
import pandas as pd
from scipy.stats import ks_2samp
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from src.utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            logging.error("Couldn't define the variables in DataValidation Class")
            raise NetworkSecurityException(error_message=e)

    # Only use in data validation, no object will be created
    @staticmethod
    def _read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)

        except Exception as e:
            logging.error("Unable to read data from file path")
            raise NetworkSecurityException(error_message=e)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            actual_columns = len(dataframe.columns)

            logging.info(f"DataFrame has {actual_columns} columns")
            logging.info(f"Required number of columns {number_of_columns}")

            if actual_columns == number_of_columns:
                return True
            else:
                logging.warning(
                    f"Column validation failed: Expected {number_of_columns}, got {actual_columns}"
                )
                return False

        except Exception as e:
            logging.error("Unable to perform number of columns validation")
            raise NetworkSecurityException(error_message=e)

    def detect_data_drift(
        self, df_base: pd.DataFrame, df_current: pd.DataFrame, threshold=0.05
    ) -> bool:
        try:
            status = True
            report = {}
            for column in df_base.columns:
                df_1 = df_base[column]
                df_2 = df_current[column]

                # Compare distribution of two samples
                same_distance = ks_2samp(data1=df_1, data2=df_2)
                if same_distance.pvalue >= threshold:
                    data_drift = False
                else:
                    data_drift = True
                    status = False

                report.update(
                    {
                        column: {
                            "p_value": float(same_distance.pvalue),
                            "drift_status": data_drift,
                        }
                    }
                )

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            directory_path = os.path.dirname(drift_report_file_path)
            if directory_path:
                os.makedirs(directory_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status

        except Exception as e:
            logging.error("Unable to perform data drift detection")
            raise NetworkSecurityException(error_message=e)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            df_train = DataValidation._read_data(file_path=train_file_path)
            df_test = DataValidation._read_data(file_path=test_file_path)

            # Validate number of columns
            train_status = self.validate_number_of_columns(dataframe=df_train)
            if not train_status:
                error_message = (
                    "Train DataFrame does not contain all the required columns."
                )
                logging.error(error_message)
                raise NetworkSecurityException(error_message=error_message)
            test_status = self.validate_number_of_columns(dataframe=df_test)
            if not test_status:
                error_message = (
                    "Test DataFrame does not contain all the required columns."
                )
                logging.error(error_message)
                raise NetworkSecurityException(error_message=error_message)

            # Detect data drift
            drift_status = self.detect_data_drift(df_base=df_train, df_current=df_test)

            validation_status = train_status and test_status and drift_status

            validated_train_file_path = None
            validated_test_file_path = None
            invalidated_train_file_path = None
            invalidated_test_file_path = None

            if validation_status:
                logging.info("Data validation successfull. Saving validated files.")
                validated_train_file_path = (
                    self.data_validation_config.validated_train_file_path
                )
                validated_test_file_path = (
                    self.data_validation_config.validated_test_file_path
                )

                os.makedirs(os.path.dirname(validated_train_file_path), exist_ok=True)

                df_train.to_csv(
                    validated_train_file_path,
                    index=False,
                    header=True,
                )

                df_test.to_csv(
                    validated_test_file_path,
                    index=False,
                    header=True,
                )
            else:
                logging.warning(
                    "Data validation failed. Saving files in invalidated data path."
                )
                invalidated_train_file_path = (
                    self.data_validation_config.invalidated_train_file_path
                )
                invalidated_test_file_path = (
                    self.data_validation_config.invalidated_test_file_path
                )

                os.makedirs(os.path.dirname(invalidated_train_file_path), exist_ok=True)

                df_train.to_csv(
                    invalidated_train_file_path,
                    index=False,
                    header=True,
                )

                df_test.to_csv(
                    invalidated_test_file_path,
                    index=False,
                    header=True,
                )

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                validated_train_file_path=validated_train_file_path
                if validation_status
                else None,
                validated_test_file_path=validated_test_file_path
                if validation_status
                else None,
                invalidated_train_file_path=invalidated_train_file_path,
                invalidated_test_file_path=invalidated_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact

        except Exception as e:
            logging.error("Unable to perform data validation.")
            raise NetworkSecurityException(error_message=e)
