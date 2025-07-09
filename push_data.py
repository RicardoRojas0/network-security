import os
import sys
import json
from dotenv import load_dotenv
import certifi
import pandas as pd
import numpy as np
import pymongo
import pymongo.mongo_client
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
print(MONGODB_URI)

certificate_authorities = certifi.where()


class MongoETL:
    def __init__(self, database, collection):
        self.database_name = database
        self.collection_name = collection
        self.records = None
        self.client = None

        try:
            self.client = pymongo.MongoClient(MONGODB_URI)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
        except Exception as e:
            if self.client:
                self.client.close()
            raise NetworkSecurityException(error_message=e, error_details=sys)

    def csv_to_json(self, file_path):
        """
        Convert CSV format into JSON format, to be able to upload to MongoDB.
        """
        logging.info("Initiating format transformation from CSV to JSON for MongoDB")
        try:
            data = pd.read_csv(file_path)
            self.records = data.to_dict(orient="records")
            return self.records
        except Exception as e:
            raise NetworkSecurityException(error_message=e, error_details=sys)

    def push_data_to_mongodb(self):
        if self.records is None:
            logging.error("No data to upload to MongoDB.")
            raise ValueError(
                "No data to insert. Make sure to first call csv_to_json function."
            )
        try:
            result = self.collection.insert_many(self.records)
            logging.info("Data successfully uploaded to MongoDB.")
            return len(result.inserted_ids)
        except Exception as e:
            logging.error("Something wen't wrong when trying to upload data to MongoDB")
            raise NetworkSecurityException(error_message=e)


if __name__ == "__main__":
    FILE_PATH = "data/phisingData.csv"
    DATABASE = "machine_learning_db"
    COLLECTION = "phishing_data"
    mongo_etl = MongoETL(database=DATABASE, collection=COLLECTION)
    records = mongo_etl.csv_to_json(file_path=FILE_PATH)
    print(records)
    num_of_records = mongo_etl.push_data_to_mongodb()
    print(num_of_records)
