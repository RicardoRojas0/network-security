import os
import sys
import yaml
import dill
import pickle
import numpy as np
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file=file_path, mode="rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logging.error("Unable to perform read yaml file.")
        raise NetworkSecurityException(error_message=e)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if not replace and os.path.exists(file_path):
            logging.info(f"The file {file_path} already exists. Writing is omitted.")
            return

        parent_directory = os.path.dirname(file_path)
        if parent_directory:
            os.makedirs(parent_directory, exist_ok=True)

        with open(file=file_path, mode="w") as file:
            yaml.dump(content, file)
            
    except Exception as e:
        logging.error(f"Unable to write yaml file in path: {file_path}")
        raise NetworkSecurityException(error_message=e)
