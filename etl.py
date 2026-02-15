import pandas as pd
import numpy as np
import os
import sys
import json

from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logger

from dotenv import load_dotenv
from dataclasses import dataclass

from networksecurity.constant import etl_constants

import certifi
cv=certifi.where()

import pymongo

load_dotenv()
MONGODB_URL=os.getenv("MONGODB_URL")

@dataclass
class ETLConfig:
    dataset_path:str=os.path.join('dataset','phisingData.csv')
    mongodb_url:str=MONGODB_URL
    database:str='NetworkSecurity'
    collection:str='NetworkSecurityDataset'
    
class ETL:
    def __init__(self):
        self.etl_config=ETLConfig()
    
    def extract(self):
        try:
            dataset=pd.read_csv(self.etl_config.dataset_path)
            dataset.reset_index(drop=True,inplace=True)
            return dataset
        except Exception as e:
            raise CustomException(e,sys)
    
    def transform(self):
        try:
            data=self.extract()
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
    
    def load(self,records):
        try:
            if not self.etl_config.mongodb_url:
                raise ValueError("MongoDB URL is not set in environment variables.")
            else:
                self.mongo_client=pymongo.MongoClient(self.etl_config.mongodb_url)
                self.database=self.mongo_client[self.etl_config.database]
                self.collections=self.database[self.etl_config.collection]
                self.collections.insert_many(records)
                return len(records)
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=='__main__':
    etl=ETL()
    data=etl.transform()
    res=etl.load(data)
    print(res)