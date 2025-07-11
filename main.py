from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
)
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        # Configurations
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_validation_config = DataValidationConfig(
            training_pipeline_config=training_pipeline_config
        )
        data_transformation_config = DataTransformationConfig(
            training_pipeline_config=training_pipeline_config
        )

        # Data Ingestion
        logging.info("=== INITIATING DATA INGESTION PROCESS ===")
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("=== DATA INGESTION PROCESS COMPLETED ===")

        # Data Validation
        logging.info("=== INITIATING DATA VALIDATION PROCESS ===")
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config,
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        print(data_validation_artifact)
        logging.info("=== DATA VALIDATION PROCESS COMPLETED ===")

        # Data Transformation
        logging.info("=== INITIATING DATA TRANSFORMATION PROCESS ===")
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config,
        )
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        print(data_transformation_artifact)
        logging.info("=== DATA TRANSFORMATION PROCESS COMPLETED ===")

    except Exception as e:
        logging.error("Something failed in the Data Ingestion Process")
        raise NetworkSecurityException(error_message=e)
