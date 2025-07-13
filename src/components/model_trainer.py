import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.utils.utils import (
    save_preprocessor,
    load_preprocessor,
    load_numpy_array_data,
    evaluate_model,
)
from src.utils.classification_metrics import classification_scores
from src.utils.model_estimator import ModelEstimator


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(error_message=e)

    def train_model(
        self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array
    ):
        try:
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, n_jobs=-1),
                "KNN": KNeighborsClassifier(n_jobs=-1),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
            }

            params = {
                "Logistic Regression": {"C": [0.1], "max_iter": [1000]},
                "KNN": {"n_neighbors": [3, 5]},
                "Decision Tree": {"max_depth": [5, 10], "min_samples_split": [2, 4]},
                "AdaBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.01, 0.001],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.1, 0.01, 0.001],
                    "max_depth": [3],
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10],
                    "min_samples_split": [2],
                },
            }

            model_report: dict = evaluate_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                models=models,
                param_grid=params,
            )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            classification_train_metric = classification_scores(
                y_true=y_train, y_pred=y_train_pred
            )
            classification_test_metric = classification_scores(
                y_true=y_test, y_pred=y_test_pred
            )

            preprocessor = load_preprocessor(
                file_path=self.data_transformation_artifact.preprocessor_file_path
            )

            model_directory = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_directory, exist_ok=True)

            model = ModelEstimator(preprocessor=preprocessor, model=best_model)

            save_preprocessor(
                file_path=self.model_trainer_config.trained_model_file_path,
                preprocessor=model,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model trainer artficat: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(error_message=e)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            train_array = load_numpy_array_data(file_path=train_file_path)
            test_array = load_numpy_array_data(file_path=test_file_path)

            # Splitting sets into features and target
            X_train, X_test, y_train, y_test = (
                train_array[:, :-1],
                test_array[:, :-1],
                train_array[:, -1],
                test_array[:, -1],
            )

            model_trainer_artifact = self.train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(error_message=e)
