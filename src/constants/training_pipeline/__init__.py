import os

"""
COMMON CONSTANT VARIABLES FOR TRAINING PIPELINE
"""
ARTIFACT_DIRECTORY: str = "artifacts"
PIPELINE_NAME: str = "network-security"
FILE_NAME: str = "phisingData.csv"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
TARGET_COLUMN: str = "Result"


"""
DATA INGESTION RELATED CONSTANTS
"""
DATA_INGESTION_DATABASE_NAME: str = "machine_learning_db"
DATA_INGESTION_COLLECTION_NAME: str = "phishing_data"
DATA_INGESTION_DIRECTORY_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIRECTORY: str = "feature_store"
DATA_INGESTION_INGESTED_DIRECTORY: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


"""
DATA VALIDATION RELATED CONSTANTS
"""
DATA_VALIDATION_DIRECTORY_NAME: str = "data_validation"
DATA_VALIDATION_VALIDATED_DIRECTORY: str = "validated"
DATA_VALIDATION_INVALIDATED_DIRECTORY: str = "invalidated"
DATA_VALIDATION_DRIFT_REPORT_DIRECTORY: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_NAME: str = "report.yaml"