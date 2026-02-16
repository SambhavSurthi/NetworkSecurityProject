import os
import sys

import pandas as pd

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import CustomException

from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidtionArtifact
from networksecurity.constant import training_constants
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp
class DataValidation:
    def __init__(self,dataingestion_artifact:DataIngestionArtifact,
                 datavalidation_config:DataValidationConfig):
        try:
            self.dataingestion_artifact = dataingestion_artifact
            self.datavalidation_config = datavalidation_config
            self.schema_config = read_yaml_file(training_constants.SCHEMA_FILE)
        except Exception as e:
            raise CustomException(e,sys)

    @staticmethod
    def read_data(filepath)->pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CustomException(e,sys)

    def validate_no_of_cols(self, dataframe: pd.DataFrame) -> bool:
        try:
            schema_columns = self.schema_config["columns"]
            expected_len = len(schema_columns)
            dataframe_len = len(dataframe.columns)

            return expected_len == dataframe_len

        except Exception as e:
            raise CustomException(e, sys)

    def validate_numeric_cols(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_config["numerical_columns"]

            missing_cols = []
            wrong_dtype_cols = []

            for col in numerical_columns:
                if col not in dataframe.columns:
                    missing_cols.append(col)
                else:
                    if not pd.api.types.is_numeric_dtype(dataframe[col]):
                        wrong_dtype_cols.append(
                            f"{col} (found {dataframe[col].dtype})"
                        )

            if len(missing_cols) > 0:
                raise Exception(f"Missing numerical columns: {missing_cols}")

            if len(wrong_dtype_cols) > 0:
                raise Exception(f"Columns with non-numeric dtype: {wrong_dtype_cols}")

            logger.info("All numerical columns validated successfully.")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def data_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column: {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_status": is_found

                }})
            drift_report_file_path = self.datavalidation_config.drift_report_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise CustomException(e,sys)

    def validate(self):
        try:
            train_data_path = self.dataingestion_artifact.train_data_path
            test_data_path = self.dataingestion_artifact.test_data_path

            train_dataframe = self.read_data(train_data_path)
            test_dataframe = self.read_data(test_data_path)

            status_validate_no_of_cols_train=self.validate_no_of_cols(train_dataframe)
            status_validate_no_of_cols_test=self.validate_no_of_cols(test_dataframe)
            status_validate_numeric_cols_train=self.validate_numeric_cols(train_dataframe)
            status_validate_numeric_cols_test=self.validate_numeric_cols(test_dataframe)
            # Validate number of columns
            if not status_validate_no_of_cols_train:
                raise Exception("Train dataframe does not contain all columns.")

            if not status_validate_no_of_cols_test:
                raise Exception("Test dataframe does not contain all columns.")

            # Validate numeric columns
            if not status_validate_numeric_cols_train:
                raise Exception("Train dataframe does not contain all columns numeric.")
            if not status_validate_numeric_cols_test:
                raise Exception("Test dataframe does not contain all columns numeric.")

            # Data drift
            status = self.data_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            os.makedirs(self.datavalidation_config.valid_data_dir, exist_ok=True)

            if (status_validate_no_of_cols_train and status_validate_no_of_cols_test) and (status_validate_numeric_cols_train and status_validate_numeric_cols_test):
                train_dataframe.to_csv(
                    self.datavalidation_config.valid_train_data_file,
                    index=False,
                    header=True
                )

                test_dataframe.to_csv(
                    self.datavalidation_config.valid_test_data_file,
                    index=False,
                    header=True
                )
            else:
                train_dataframe.to_csv(
                    self.datavalidation_config.invalid_train_data_file,
                    index=False,
                    header=True
                )

                test_dataframe.to_csv(
                    self.datavalidation_config.invalid_test_data_file,
                    index=False,
                    header=True
                )

            data_validation_artifact = DataValidtionArtifact(
                validation_status=status,
                valid_train_file_path=self.datavalidation_config.valid_train_data_file,
                valid_test_file_path=self.datavalidation_config.valid_test_data_file,
                invalid_train_file_path=self.datavalidation_config.invalid_train_data_file,
                invalid_test_file_path=self.datavalidation_config.invalid_test_data_file,
                drift_report_file_path=self.datavalidation_config.drift_report_path,
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)
