from datetime import datetime
import os
from networksecurity.constant import training_constants
from dataclasses import dataclass

@dataclass
class Training_Path:
    timestamp=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    artifact_dir=training_constants.ARTIFACTS_DIR
    training_path=os.path.join(artifact_dir,timestamp)
    
    
    
class DataIngestionConfig:
    def __init__(self,training_paths= Training_Path):
        self.data_ingestion_dir=os.path.join(
            training_paths.training_path,training_constants.DATA_INGESTION_DIR
        )
        self.raw_data_dir=os.path.join(
            self.data_ingestion_dir,training_constants.RAW_DATA_DIR
        )
        self.raw_data_path=os.path.join(
            self.raw_data_dir,training_constants.RAW_DATA
        )
        self.ingested_data_dir=os.path.join(
            self.data_ingestion_dir,training_constants.INGESTED_DATA_DIR
        )
        self.train_data_path=os.path.join(
            self.ingested_data_dir,training_constants.TRAIN_DATA
        )
        self.test_data_path=os.path.join(
            self.ingested_data_dir,training_constants.TEST_DATA
        )


class DataValidationConfig:
    def __init__(self,train_paths=Training_Path):
        self.data_validation_dir=os.path.join(
            train_paths.training_path,training_constants.DATA_VALIDATION_DIR
        )

        self.valid_data_dir=os.path.join(
            self.data_validation_dir,training_constants.VALID_DATA_DIR
        )
        self.valid_train_data_file=os.path.join(
            self.valid_data_dir,training_constants.TRAIN_DATA
        )
        self.valid_test_data_file=os.path.join(
            self.valid_data_dir,training_constants.TEST_DATA
        )
        self.invalid_data_dir=os.path.join(
            self.data_validation_dir,training_constants.INVALID_DATA_DIR
        )
        self.invalid_train_data_file=os.path.join(
            self.invalid_data_dir,training_constants.TRAIN_DATA
        )
        self.invalid_test_data_file=os.path.join(
            self.invalid_data_dir,training_constants.TEST_DATA
        )

        self.drift_report_dir=os.path.join(
            self.data_validation_dir,training_constants.DRIFT_REPORT_DIR
        )
        self.drift_report_path=os.path.join(
            self.drift_report_dir,training_constants.DRIFT_REPORT_NAME
        )

class DataTransformationConfig:
    def __init__(self,traning_path=Training_Path):
        self.data_transformed_dir=os.path.join(
            traning_path.training_path,training_constants.DATA_TRANSFORMATION_DIR
        )



        self.transformed_dir=os.path.join(
            self.data_transformed_dir,training_constants.TRANSFORMED_DIR
        )

        self.transformed_train_file=os.path.join(
            self.transformed_dir,training_constants.TRANSFORMED_TRAIN_FILE
        )

        self.transformed_test_file=os.path.join(
            self.transformed_dir,training_constants.TRANSFORMED_TEST_FILE
        )



        self.transformed_obj_dir=os.path.join(
            self.data_transformed_dir,training_constants.TRANSFORMED_OBJ_DIR
        )

        self.transformed_obj_file=os.path.join(
            self.transformed_obj_dir,training_constants.PREPROCESSING_OBJ
        )


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: Training_Path):

        self.model_trainer_dir = os.path.join(
            training_pipeline_config.training_path,
            training_constants.MODEL_TRAINER_DIR_NAME
        )

        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            training_constants.MODEL_TRAINER_TRAINED_MODEL_DIR,
            training_constants.MODEL_FILE_NAME
        )

        self.expected_accuracy = training_constants.MODEL_TRAINER_EXPECTED_SCORE

        self.overfitting_underfitting_threshold = (
            training_constants.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
        )
