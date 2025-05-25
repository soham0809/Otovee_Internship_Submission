import os
import sys
import joblib
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            joblib.dump(obj, f)
        logging.info(f"✅ Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return joblib.load(f)
    except Exception as e:
        raise CustomException(e, sys)
