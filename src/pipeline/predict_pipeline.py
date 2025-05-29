import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import json
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        self.prediction_logs_path = os.path.join("logs", "predictions")
        self.explanation_path = os.path.join("artifacts", "explanations")
        
        # Create directories if they don't exist
        os.makedirs(self.prediction_logs_path, exist_ok=True)
        os.makedirs(self.explanation_path, exist_ok=True)
    
    def _add_engineered_features(self, input_df):
        """Add the same engineered features used during training"""
        try:
            # Create a copy to avoid modifying the original
            df = input_df.copy()
            
            # Add HCG ratio
            if "I_beta-HCG(mIU/mL)" in df.columns and "II_beta-HCG(mIU/mL)" in df.columns:
                df["HCG_ratio"] = df["II_beta-HCG(mIU/mL)"] / (df["I_beta-HCG(mIU/mL)"] + 1e-6)  # Avoid division by zero
            
            # Add log transforms
            for col in ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]:
                if col in df.columns:
                    df[f"{col}_log"] = np.log1p(df[col])
            
            return df
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")
            raise CustomException(f"Error in feature engineering: {str(e)}", sys)
    
    def _validate_input(self, input_data):
        """Validate input data format and values"""
        required_fields = ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check if values are numeric
        for field in required_fields:
            try:
                value = float(input_data[field])
                # Check for negative values
                if value < 0:
                    raise ValueError(f"Field {field} cannot have negative value: {value}")
            except (ValueError, TypeError):
                raise ValueError(f"Field {field} must be a valid number")
        
        return True
    
    def _log_prediction(self, input_data, prediction, confidence=None):
        """Log prediction details for monitoring"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "input": input_data,
                "prediction": int(prediction),
                "confidence": float(confidence) if confidence is not None else None
            }
            
            # Save to JSON file
            log_file = os.path.join(self.prediction_logs_path, f"prediction_{timestamp}.json")
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=4, default=str)
            
            logging.info(f"Prediction logged to {log_file}")
        except Exception as e:
            logging.warning(f"Failed to log prediction: {str(e)}")
    
    def _generate_explanation(self, model, preprocessor, input_df, prediction):
        """Generate SHAP explanation for the prediction"""
        try:
            # Skip if the model doesn't support predict_proba
            if not hasattr(model, 'predict_proba'):
                logging.warning("Model doesn't support probability predictions, skipping explanation")
                return None
            
            # Create explainer
            explainer = shap.Explainer(model)
            
            # Transform input data
            input_transformed = preprocessor.transform(input_df)
            
            # Get SHAP values
            shap_values = explainer(input_transformed)
            
            # Generate and save explanation plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.figure(figsize=(10, 6))
            
            # Feature names - must match the order in the preprocessor
            feature_names = [
                "I_beta-HCG", "II_beta-HCG", "AMH", 
                "I_beta-HCG_log", "II_beta-HCG_log", "AMH_log",
                "HCG_ratio"
            ]
            
            # Force plot for individual prediction
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            explanation_file = os.path.join(self.explanation_path, f"explanation_{timestamp}.png")
            plt.savefig(explanation_file)
            plt.close()
            
            logging.info(f"Explanation saved to {explanation_file}")
            return explanation_file
        except Exception as e:
            logging.warning(f"Failed to generate explanation: {str(e)}")
            return None

    def predict(self, input_data: dict):
        """Make prediction with confidence score and explanation"""
        try:
            # Validate input data
            self._validate_input(input_data)
            logging.info("✅ Input data validated")
            
            # Load model and preprocessor
            try:
                model = joblib.load(self.model_path)
                preprocessor = joblib.load(self.preprocessor_path)
                logging.info("✅ Model and preprocessor loaded successfully")
            except FileNotFoundError as e:
                logging.error(f"Model or preprocessor file not found: {str(e)}")
                raise CustomException("Model files not found. Please train the model first.", sys)
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise CustomException(f"Failed to load model: {str(e)}", sys)
            
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Add engineered features (same as in training)
            input_df = self._add_engineered_features(input_df)
            logging.info("✅ Engineered features added")
            
            # Transform input
            input_transformed = preprocessor.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_transformed)[0]
            logging.info(f"✅ Prediction generated: {prediction}")
            
            # Get confidence score if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_transformed)[0]
                confidence = probabilities[1] if prediction == 1 else probabilities[0]
                logging.info(f"✅ Confidence score: {confidence:.4f}")
            
            # Generate explanation
            explanation_path = self._generate_explanation(model, preprocessor, input_df, prediction)
            
            # Log prediction
            self._log_prediction(input_data, prediction, confidence)
            
            # Return prediction with confidence if available
            result = {
                "prediction": int(prediction),
                "confidence": float(confidence) if confidence is not None else None,
                "explanation_path": explanation_path
            }
            
            return result
        except ValueError as e:
            # Handle validation errors
            logging.error(f"Input validation error: {str(e)}")
            raise CustomException(f"Input validation error: {str(e)}", sys)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise CustomException(e, sys)
