import os
import sys

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.logging.logger import logger
from networksecurity.exception.exception import CustomException

from networksecurity.constant import training_constants
from networksecurity.constant.training_constants import DATA_TRANSFORMATION_IMPUTER_PARAMS,TARGET_COLUMN
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidtionArtifact,DataTransformationArtifact
from networksecurity.utils.main_utils.utils import save_numpy_array_data,save_object


class DataTransformation:
    def __init__(self,datatransformation_config=DataTransformationConfig,
                 datavalidation_artifact=DataValidtionArtifact):
        try:
            self.datavalidation_artifact = datavalidation_artifact
            self.datatransformation_config = datatransformation_config
        except Exception as e:
            raise CustomException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
          cls: DataTransformation

        Returns:
          A Pipeline object
        """
        logger.info(
            "Entered get_data_trnasformer_object method of Trnasformation class"
        )
        try:
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logger.info(
                f"Initialise KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )
            processor: Pipeline = Pipeline([("imputer", imputer)])
            return processor
        except Exception as e:
            raise CustomException(e, sys)

    def transform(self):
        try:
            train_dataset=DataTransformation.read_data(self.datavalidation_artifact.valid_train_file_path)
            test_dataset=DataTransformation.read_data(self.datavalidation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df = train_dataset.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_dataset[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            # testing dataframe
            input_feature_test_df = test_dataset.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_dataset[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # save numpy array data
            save_numpy_array_data(self.datatransformation_config.transformed_train_file, array=train_arr, )
            save_numpy_array_data(self.datatransformation_config.transformed_test_file, array=test_arr, )
            save_object(self.datatransformation_config.transformed_obj_file, preprocessor_object, )

            save_object("final_model/preprocessor.pkl", preprocessor_object, )

            # preparing artifacts

            data_transformation_artifact = DataTransformationArtifact(
                tranformed_train_file_path=self.datatransformation_config.transformed_train_file,
                transformed_test_file_path=self.datatransformation_config.transformed_test_file,
                processing_model_file_path=self.datatransformation_config.transformed_obj_file
            )
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)