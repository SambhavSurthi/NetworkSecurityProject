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