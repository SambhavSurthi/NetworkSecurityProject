from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataValidationConfig




if __name__ == '__main__':
    print('Initiating Data Ingestion')

    ingestion = DataIngestion()
    ingestion_artifact = ingestion.ingest()

    print('Data Ingestion Successful')

    validation = DataValidation(
        dataingestion_artifact=ingestion_artifact,
        datavalidation_config=DataValidationConfig()
    )

    validation_artifact = validation.validate()

    print(validation_artifact)
