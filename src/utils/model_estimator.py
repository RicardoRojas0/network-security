import os
import sys
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.constants.training_pipeline import (
    MODEL_TRAINER_TRAINED_MODEL_DIRECTORY,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
)


class ModelEstimator:
    def __init__(self, preprocessor: object, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(error_message=e)

    def predict(self, X):
        try:
            X_transformed = self.preprocessor.transform(X)
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(error_message=e)
