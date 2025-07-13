import pandas as pd
from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from sklearn.metrics import precision_score, recall_score, f1_score


def classification_scores(
    y_true: pd.Series, y_pred: pd.Series
) -> ClassificationMetricArtifact:
    try:
        model_precision_score = precision_score(y_true=y_true, y_pred=y_pred)
        model_recall_score = recall_score(y_true=y_true, y_pred=y_pred)
        model_f1_score = f1_score(y_true=y_true, y_pred=y_pred)

        classification_metric_artifact = ClassificationMetricArtifact(
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            f1_score=model_f1_score,
        )
        return classification_metric_artifact

    except Exception as e:
        logging.warning("Couldn't obtain the classification metrics for evaluation")
        raise NetworkSecurityException(error_message=e)
