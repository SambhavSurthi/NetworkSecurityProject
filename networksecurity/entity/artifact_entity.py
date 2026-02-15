from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_data_path:str
    test_data_path:str