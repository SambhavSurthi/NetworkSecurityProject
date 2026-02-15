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