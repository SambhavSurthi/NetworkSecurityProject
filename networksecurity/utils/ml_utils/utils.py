from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import CustomException
from sklearn.metrics import f1_score, precision_score, recall_score
from networksecurity.constant.training_constants import SAVED_MODEL_DIR, MODEL_FILE_NAME
from networksecurity.logging.logger import logging
import os
import sys
from networksecurity.logging.logger import logging


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:

        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        classification_metric = ClassificationMetricArtifact(f1_score=model_f1_score,
                                                             precision_score=model_precision_score,
                                                             recall_score=model_recall_score)
        return classification_metric
    except Exception as e:
        raise CustomException(e, sys)


class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise CustomException(e, sys)
