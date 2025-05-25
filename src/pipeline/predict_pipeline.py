import os
import sys
import pandas as pd
import joblib
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, input_data: dict):
        try:
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)

            logging.info("✅ Model and preprocessor loaded successfully")

            input_df = pd.DataFrame([input_data])
            input_transformed = preprocessor.transform(input_df)
            prediction = model.predict(input_transformed)[0]

            return prediction
        except Exception as e:
            raise CustomException(e, sys)
