from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.components.model_training import ModelTrainer
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.config_entity import Training_Path

if __name__ == '__main__':
    print('Initiating Data Ingestion')
    training_path = Training_Path()

    ingestion = DataIngestion()
    ingestion_artifact = ingestion.ingest()

    print('Data Ingestion Successful')

    validation = DataValidation(
        dataingestion_artifact=ingestion_artifact,
        datavalidation_config=DataValidationConfig()
    )

    validation_artifact = validation.validate()

    tranformation=DataTransformation(
        datatransformation_config=DataTransformationConfig(),
        datavalidation_artifact=validation_artifact
    )

    transformation_artifact=tranformation.transform()

    model=ModelTrainer(model_trainer_config=ModelTrainerConfig(training_path),data_transformation_artifact=transformation_artifact)
    res=model.initiate_model_trainer()
    print(res)


