import pandas as pd
import numpy as np
import os
import sys
import pymongo

from networksecurity.constant import training_constants
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.exception.exception import CustomException
from networksecurity.logging.logger import logger
from typing import  List
from sklearn.model_selection import train_test_split
from networksecurity.entity.artifact_entity import DataIngestionArtifact


from dotenv import load_dotenv


class DataIngestion:
    def __init__(self,dataingestionconfig=DataIngestionConfig()):
        self.dataingestionconfig=dataingestionconfig
        
        
    def export_collections_as_dataframe(self):
        try:
            database_name=training_constants.DATABASE
            collections_name=training_constants.COLLECTIONS
            mongodb_url=training_constants.MONGODB_URL
            
            self.mongodb_client=pymongo.MongoClient(mongodb_url)
            
            collections=self.mongodb_client[database_name][collections_name]
            
            dataset=pd.DataFrame(list(collections.find()))
            
            if '_id' in dataset.columns.to_list():
                dataset=dataset.drop(columns=['_id'],axis=1)
                
            dataset.replace(['na', 'NA', 'missing'], np.nan, inplace=True)
            
            return dataset
        except Exception as e:
            raise CustomException(e,sys)
        
    def save_raw_data(self):
        try:
            dataset=self.export_collections_as_dataframe()
            raw_data_dir=self.dataingestionconfig.raw_data_dir
            raw_data_path=self.dataingestionconfig.raw_data_path
            os.makedirs(raw_data_dir,exist_ok=True)
            dataset.to_csv(raw_data_path,header=True,index=False)
            return dataset
        except Exception as e:
            raise CustomException(e,sys)
        
    def implement_train_test_split(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=training_constants.TRAIN_TEST_SPLIT_RATIO,random_state=42)
            ingested_data_dir=self.dataingestionconfig.ingested_data_dir
            train_path=self.dataingestionconfig.train_data_path
            test_path=self.dataingestionconfig.test_data_path
            
            os.makedirs(ingested_data_dir,exist_ok=True)
            
            train_set.to_csv(train_path,index=False,header=True)
            test_set.to_csv(test_path,index=False,header=True)
        except Exception as e:
            raise CustomException(e,sys)
        
    def ingest(self):
        try:
            dataset=self.save_raw_data()
            self.implement_train_test_split(dataframe=dataset)
            dataingestionartifact=DataIngestionArtifact(train_data_path=self.dataingestionconfig.train_data_path,
                                                        test_data_path=self.dataingestionconfig.test_data_path)
            return dataingestionartifact
            
        except Exception as e:
            raise CustomException(e,sys)