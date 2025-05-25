import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = [
                "I_beta-HCG(mIU/mL)", 
                "II_beta-HCG(mIU/mL)", 
                "AMH(ng/mL)"
            ]

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_features)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("✅ Loaded train and test datasets for transformation")

            input_features_train = train_df.drop(columns=["PCOS"])
            target_feature_train = train_df["PCOS"]

            input_features_test = test_df.drop(columns=["PCOS"])
            target_feature_test = test_df["PCOS"]

            preprocessing_obj = self.get_data_transformer_object()
            input_features_train_scaled = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_scaled = preprocessing_obj.transform(input_features_test)

            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info("✅ Preprocessor saved to file")

            return (
                input_features_train_scaled,
                input_features_test_scaled,
                target_feature_train,
                target_feature_test,
                self.config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
