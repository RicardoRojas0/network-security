import os
from datetime import datetime
from src.constants import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%Y_%m_%d-%H:%M:%S")
        self.timestamp: str = timestamp
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_directory: str = training_pipeline.ARTIFACT_DIRECTORY
        self.artifact_path: str = os.path.join(self.artifact_directory, timestamp)


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_directory: str = os.path.join(
            training_pipeline_config.artifact_path,
            training_pipeline.DATA_INGESTION_DIRECTORY_NAME,
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_directory,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIRECTORY,
            training_pipeline.FILE_NAME,
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_directory,
            training_pipeline.DATA_INGESTION_INGESTED_DIRECTORY,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_directory,
            training_pipeline.DATA_INGESTION_INGESTED_DIRECTORY,
            training_pipeline.TEST_FILE_NAME,
        )
        self.train_test_split_ratio: float = (
            training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        )
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_directory: str = os.path.join(
            training_pipeline_config.artifact_path,
            training_pipeline.DATA_VALIDATION_DIRECTORY_NAME,
        )
        self.validated_data_directory: str = os.path.join(
            self.data_validation_directory,
            training_pipeline.DATA_VALIDATION_VALIDATED_DIRECTORY,
        )
        self.validated_train_file_path: str = os.path.join(
            self.validated_data_directory,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.validated_test_file_path: str = os.path.join(
            self.validated_data_directory,
            training_pipeline.TEST_FILE_NAME,
        )
        self.invalidated_data_directory: str = os.path.join(
            self.data_validation_directory,
            training_pipeline.DATA_VALIDATION_INVALIDATED_DIRECTORY,
        )
        self.invalidated_train_file_path: str = os.path.join(
            self.invalidated_data_directory,
            training_pipeline.TRAIN_FILE_NAME,
        )
        self.invalidated_test_file_path: str = os.path.join(
            self.invalidated_data_directory,
            training_pipeline.TEST_FILE_NAME,
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_directory,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIRECTORY,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_NAME,
        )
