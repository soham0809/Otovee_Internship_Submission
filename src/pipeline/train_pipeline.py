import sys
import os
import traceback
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def start_training_pipeline():
    try:
        # Step 1: Ingest Data
        logging.info("Starting data ingestion")
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")

        # Step 2: Transform Data
        logging.info("Starting data transformation")
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = transformer.initiate_data_transformation(train_path, test_path)
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")

        # Step 3: Train Model
        logging.info("Starting model training")
        trainer = ModelTrainer()
        accuracy, report = trainer.initiate_model_training(X_train, X_test, y_train, y_test)

        print("\n[SUCCESS] Training Complete. Accuracy: {:.2f}%".format(accuracy * 100))
        print(report)
        return accuracy, report
    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        traceback.print_exc()
        raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Make sure the required directories exist
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs(os.path.join("artifacts", "explanations"), exist_ok=True)
        os.makedirs(os.path.join("logs", "predictions"), exist_ok=True)
        
        # Run the pipeline
        start_training_pipeline()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)
