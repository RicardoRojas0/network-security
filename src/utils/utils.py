import os
import sys
import yaml
import dill
import pickle
import numpy as np
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a yaml file from specific path
    """
    try:
        with open(file=file_path, mode="rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        logging.error("Unable to perform read yaml file.")
        raise NetworkSecurityException(error_message=e)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes yaml file to specific path
    """
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


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array to specific path

    Args:
        file_path (str): Location of file to save
        array (np.array): Array of data to save

    Raises:
        NetworkSecurityException: If array can't be saved
    """
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)
        with open(file=file_path, mode="wb") as file:
            np.save(file, array)

    except Exception as e:
        logging.error(f"Unable to save numpy array in path: {file_path}")
        raise NetworkSecurityException(error_message=e)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array from specific path
    """
    try:
        with open(file=file_path, mode="rb") as file:
            return np.load(file)
    except Exception as e:
        raise NetworkSecurityException(error_message=e)


def save_preprocessor(file_path: str, preprocessor: object) -> None:
    """
    Save preprocessor in specific path

    Args:
        file_path (str): Location of file to save
        preprocessor (object): Preprocessor to save

    Raises:
        NetworkSecurityException: If preprocessor can't be saved
    """
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)
        with open(file=file_path, mode="wb") as file:
            pickle.dump(preprocessor, file)
    except Exception as e:
        logging.error(f"Unable to save preprocessor in path: {file_path}")
        raise NetworkSecurityException(error_message=e)


def load_preprocessor(file_path: str) -> object:
    """
    Load preprocessor from specific path
    """
    try:
        if not os.path.exists(file_path):
            logging.warning(f"The file: {file_path} does not exists")
            raise Exception(f"The file: {file_path} does not exists")
        with open(file=file_path, mode="rb") as file:
            print(file)
            return pickle.load(file)
    except Exception as e:
        raise NetworkSecurityException(error_message=e)


def evaluate_model(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    models: dict,
    param_grid: dict,
):
    report = {}
    try:
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param_grid[list(models.keys())[i]]

            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=3,
                n_jobs=-1,
                refit=True,
            )
            gs.fit(X=X_train, y=y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            test_score = f1_score(y_true=y_test, y_pred=y_pred)

            report[list(models.keys())[i]] = test_score
            return report
    except Exception as e:
        raise NetworkSecurityException(error_message=e)
