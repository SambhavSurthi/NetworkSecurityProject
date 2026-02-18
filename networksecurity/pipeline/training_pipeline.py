import os
import sys
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logger
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.model_training import ModelTrainer
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidtionArtifact,DataTransformationArtifact,ModelTrainerArtifact,ClassificationMetricArtifact
from networksecurity.entity.config_entity import Training_Path,DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig

class TrainingPipeline:
    def __init__(self):
        self.training_path=Training_Path()

    def start_data_ingestion(self):
        try:
            data_ingestion_config=DataIngestionConfig(self.training_path)
            ingestion = DataIngestion(data_ingestion_config)
            ingestion_artifact = ingestion.ingest()

            return ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_validation(self,ingestion_artifact):
        try:
            validation = DataValidation(
                dataingestion_artifact=ingestion_artifact,
                datavalidation_config=DataValidationConfig()
            )

            validation_artifact = validation.validate()
            return validation_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def start_data_transformation(self,validation_artifact):
        try:
            tranformation = DataTransformation(
                datatransformation_config=DataTransformationConfig(),
                datavalidation_artifact=validation_artifact
            )

            transformation_artifact = tranformation.transform()

            return transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)

    def start_model_training(self,transformation_artifact):
        try:
            model = ModelTrainer(model_trainer_config=ModelTrainerConfig(self.training_path),
                                 data_transformation_artifact=transformation_artifact)
            res = model.initiate_model_trainer()

            return res
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self):
        try:
            ingested = self.start_data_ingestion()
            validated = self.start_data_validation(ingested)
            transformed = self.start_data_transformation(validated)
            trained = self.start_model_training(transformed)

            return trained
        except Exception as e:
            raise CustomException(e,sys)

