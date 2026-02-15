from networksecurity.components.data_ingestion import DataIngestion

if __name__=='__main__':
    print('Initiating Data Ingestion')
    obj=DataIngestion()
    dataset=obj.ingest()
    print('Data Ingestion Successful')