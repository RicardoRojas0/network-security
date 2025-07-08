from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(
            training_pipeline_config=training_pipeline_config
        )

        logging.info("--- INITIATING DATA INGESTION PROCESS ---")
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
        logging.info("--- DATA INGESTION PROCESS COMPLETED ---")

    except Exception as e:
        logging.error("Something failed in the Data Ingestion Process")
        raise NetworkSecurityException(error_message=e, error_details=sys)
