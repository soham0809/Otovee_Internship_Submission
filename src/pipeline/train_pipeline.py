from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def start_training_pipeline():
    # Step 1: Ingest Data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Step 2: Transform Data
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test, _ = transformer.initiate_data_transformation(train_path, test_path)

    # Step 3: Train Model
    trainer = ModelTrainer()
    accuracy, report = trainer.initiate_model_training(X_train, X_test, y_train, y_test)

    print(f"\n✅ Training Complete. Accuracy: {accuracy * 100:.2f}%")
    print(report)

if __name__ == "__main__":
    start_training_pipeline()
