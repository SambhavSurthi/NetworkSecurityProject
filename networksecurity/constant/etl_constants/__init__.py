from dotenv import load_dotenv
import os


load_dotenv(dotenv_path='D:\NetworkSecurityProject\.env')
MONGODB_URL=os.getenv("MONGODB_URL")

DATASET_PATH:str=os.path.join('dataset','phisingData.csv')
MONGODB_URL:str=MONGODB_URL
DATABASE:str='NetworkSecurity'
COLLECTION:str='NetworkSecurityDataset'

