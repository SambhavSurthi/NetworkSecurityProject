import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ARTIFACTS_DIR:str='artifacts'
RAW_DATA:str='phisingData.csv'
TRAIN_DATA:str='train.csv'
TEST_DATA:str='test.csv'

DATA_INGESTION_DIR:str='data_ingestion'
RAW_DATA_DIR:str='raw_data'
INGESTED_DATA_DIR:str='ingested_data'

TRAIN_TEST_SPLIT_RATIO:float=0.2


load_dotenv()
MONGODB_URL=os.getenv("MONGODB_URL")

DATASET_PATH:str=os.path.join('dataset','phisingData.csv')
MONGODB_URL:str=MONGODB_URL
DATABASE:str='NetworkSecurity'
COLLECTIONS:str='NetworkSecurityDataset'

DATA_VALIDATION_DIR:str='data_validation'
VALID_DATA_DIR:str='valid_data'
INVALID_DATA_DIR:str='invalid_data'
DRIFT_REPORT_DIR:str='drift_report'
DRIFT_REPORT_NAME:str='drift_report.yaml'
SCHEMA_FILE:str=os.path.join('data_schema','schema.yaml')


DATA_TRANSFORMATION_DIR:str='data_transformation'
TRANSFORMED_DIR:str='transformed'
TRANSFORMED_OBJ_DIR:str='transformed_obj'
PREPROCESSING_OBJ:str='preprocessing.pkl'
TRANSFORMED_TRAIN_FILE:str='train.npy'
TRANSFORMED_TEST_FILE:str='test.npy'
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
TARGET_COLUMN='Result'