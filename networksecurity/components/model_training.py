import os
import sys
import mlflow
import dagshub
from urllib.parse import urlparse

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.utils import NetworkModel, get_classification_score
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

# =========================
# DAGSHUB + MLFLOW SETUP
# =========================
dagshub.init(
    repo_owner="SambhavSurthi",
    repo_name="NetworkSecurityProject",
    mlflow=True
)

mlflow.set_experiment("NetworkSecurity-Experiment")
mlflow.sklearn.autolog()


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    # =========================
    # TRAIN MODEL
    # =========================
    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(verbose=0),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=0),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"]
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128]
            },
            "Gradient Boosting": {
                "learning_rate": [.1, .01, .05],
                "subsample": [0.7, 0.8, 0.9],
                "n_estimators": [16, 32, 64]
            },
            "Logistic Regression": {},
            "AdaBoost": {
                "learning_rate": [.1, .01],
                "n_estimators": [16, 32, 64]
            }
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params
        )

        # Get best model
        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        logging.info(f"Best model selected: {best_model_name}")

        # =========================
        # MLFLOW TRACKING
        # =========================
        with mlflow.start_run(run_name=best_model_name):

            # Train Predictions
            y_train_pred = best_model.predict(X_train)
            train_metric = get_classification_score(y_train, y_train_pred)

            # Test Predictions
            y_test_pred = best_model.predict(X_test)
            test_metric = get_classification_score(y_test, y_test_pred)

            # Log metrics manually (autolog handles params)
            mlflow.log_param("best_model", best_model_name)

            mlflow.log_metric("train_f1", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)

            mlflow.log_metric("test_f1", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)

            mlflow.sklearn.log_model(best_model, "model")

        # =========================
        # SAVE FINAL MODEL LOCALLY
        # =========================
        preprocessor = load_object(
            file_path=self.data_transformation_artifact.processing_model_file_path
        )

        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )

        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        # FIXED: saving object not class
        save_object(
            self.model_trainer_config.trained_model_file_path,
            obj=network_model
        )

        # Also optional local copy
        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

        return model_trainer_artifact

    # =========================
    # INITIATE TRAINING
    # =========================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.tranformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomException(e, sys)
