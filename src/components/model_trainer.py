import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logging.info(f"✅ Model Accuracy: {acc}")
            logging.info(f"\n{report}")

            save_object(self.config.trained_model_file_path, model)
            logging.info(f"📦 Model saved to {self.config.trained_model_file_path}")

            return acc, report
        except Exception as e:
            raise CustomException(e, sys)
