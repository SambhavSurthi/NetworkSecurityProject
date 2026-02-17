from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_data_path:str
    test_data_path:str


@dataclass
class DataValidtionArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str


@dataclass
class DataTransformationArtifact:
    tranformed_train_file_path:str
    transformed_test_file_path:str
    processing_model_file_path:str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact