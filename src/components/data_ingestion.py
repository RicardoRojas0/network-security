from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
import os
import sys
import pandas as pd
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Configurations for data ingestion
from src.entity.config_entity import DataIngestionConfig

# Artifacts for data ingestion
from src.entity.artifact_entity import DataIngestionArtifact

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(error_message=e, error_details=sys)

    def import_collection_as_dataframe(self):
        """
        Read data from MongoDB and format it as a DataFrame
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGODB_URI)
            collection = self.mongo_client[database_name][collection_name]

            dataframe = pd.DataFrame(list(collection.find()))
            if "_id" in dataframe.columns.to_list():
                dataframe = dataframe.drop(columns=["_id"])

            dataframe = dataframe.replace({"na": np.nan})
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(error_message=e, error_details=sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Saves data into the feature store
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            directory_path = os.path.dirname(feature_store_file_path)
            os.makedirs(directory_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(error_message=e, error_details=sys)

    def split_data_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
            )
            logging.info("Performed train test split on the dataframe successfully.")

            directory_path = os.path.dirname(
                self.data_ingestion_config.training_file_path
            )
            os.makedirs(directory_path, exist_ok=True)
            logging.info("Exporting train and test sets.")

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True,
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True,
            )
            logging.info("Successfully exported train and test sets to path")

        except Exception as e:
            raise NetworkSecurityException(error_message=e, error_details=sys)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.import_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe=dataframe)
            dataframe = self.split_data_train_test(dataframe=dataframe)
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(error_message=e, error_details=sys)
