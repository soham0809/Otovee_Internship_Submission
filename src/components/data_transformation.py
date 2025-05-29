import os
import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTENC
from imblearn.combine import SMOTETomek

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    feature_engineering_obj_file_path = os.path.join("artifacts", "feature_engineering.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def detect_outliers(self, df, columns, threshold=1.5):
        """Detect and log outliers using IQR method"""
        outliers_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if not outliers.empty:
                outliers_info[col] = len(outliers)
                logging.info(f"Detected {len(outliers)} outliers in column {col}")
        return outliers_info
    
    def handle_outliers(self, df, columns, method='winsorize', threshold=1.5):
        """Handle outliers using specified method"""
        for col in columns:
            if method == 'winsorize':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logging.info(f"Winsorized outliers in column {col}")
        return df
    
    def add_engineered_features(self, df):
        """Add engineered features based on domain knowledge"""
        # Calculate ratio between hormones (often more informative than absolute values)
        if "I_beta-HCG(mIU/mL)" in df.columns and "II_beta-HCG(mIU/mL)" in df.columns:
            df["HCG_ratio"] = df["II_beta-HCG(mIU/mL)"] / (df["I_beta-HCG(mIU/mL)"] + 1e-6)  # Avoid division by zero
            logging.info("✅ Added HCG ratio feature")
            
        # Log transform for skewed hormone distributions
        for col in ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]:
            if col in df.columns:
                # Add small constant to handle zeros
                df[f"{col}_log"] = np.log1p(df[col])
                logging.info(f"✅ Added log transform for {col}")
                
        return df

    def get_data_transformer_object(self):
        try:
            # Define feature groups
            base_numeric_features = [
                "I_beta-HCG(mIU/mL)", 
                "II_beta-HCG(mIU/mL)", 
                "AMH(ng/mL)"
            ]
            
            # Features that might be skewed and benefit from robust scaling
            hormone_features = [
                "I_beta-HCG(mIU/mL)", 
                "II_beta-HCG(mIU/mL)"
            ]
            
            # Features that might benefit from standard scaling
            other_numeric = [
                "AMH(ng/mL)"
            ]
            
            # Log-transformed features
            log_features = [
                "I_beta-HCG(mIU/mL)_log", 
                "II_beta-HCG(mIU/mL)_log", 
                "AMH(ng/mL)_log"
            ]
            
            # Ratio features
            ratio_features = [
                "HCG_ratio"
            ]
            
            # Create specialized pipelines for different feature types
            hormone_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", RobustScaler())
            ])
            
            standard_num_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler())
            ])
            
            log_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", StandardScaler())
            ])
            
            ratio_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler", PowerTransformer(method='yeo-johnson'))
            ])
            
            # Combine all transformers
            preprocessor = ColumnTransformer([
                ("hormone_features", hormone_pipeline, hormone_features),
                ("other_numeric", standard_num_pipeline, other_numeric),
                ("log_features", log_pipeline, log_features),
                ("ratio_features", ratio_pipeline, ratio_features)
            ])

            logging.info("✅ Created preprocessing pipeline with specialized transformers")
            return preprocessor
        except Exception as e:
            logging.error(f"Error in creating data transformer: {str(e)}")
            raise CustomException(e, sys)

    def apply_smote(self, X, y):
        """Apply SMOTE-Tomek to handle class imbalance"""
        try:
            logging.info("Applying SMOTE-Tomek for handling class imbalance")
            logging.info(f"Class distribution before resampling: {pd.Series(y).value_counts()}")
            
            # Use SMOTETomek which combines over and undersampling
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
            
            logging.info(f"Class distribution after resampling: {pd.Series(y_resampled).value_counts()}")
            return X_resampled, y_resampled
        except Exception as e:
            logging.error(f"Error in applying SMOTE: {str(e)}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("✅ Loaded train and test datasets for transformation")
            logging.info(f"Train dataset shape: {train_df.shape}, Test dataset shape: {test_df.shape}")
            
            # Detect outliers in training data
            numeric_cols = ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]
            outliers_info = self.detect_outliers(train_df, numeric_cols)
            if outliers_info:
                logging.info(f"Detected outliers: {outliers_info}")
                train_df = self.handle_outliers(train_df, numeric_cols)
            
            # Feature engineering on both train and test sets
            train_df = self.add_engineered_features(train_df)
            test_df = self.add_engineered_features(test_df)
            logging.info("✅ Added engineered features to datasets")
            
            # Separate features and target
            input_features_train = train_df.drop(columns=["PCOS"])
            target_feature_train = train_df["PCOS"]

            input_features_test = test_df.drop(columns=["PCOS"])
            target_feature_test = test_df["PCOS"]
            
            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Transform the features
            input_features_train_scaled = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_scaled = preprocessing_obj.transform(input_features_test)
            
            # Apply SMOTE for handling class imbalance (only on training data)
            X_train_resampled, y_train_resampled = self.apply_smote(
                input_features_train_scaled, target_feature_train
            )
            
            # Save the preprocessor
            save_object(self.config.preprocessor_obj_file_path, preprocessing_obj)
            logging.info(f"✅ Preprocessor saved to {self.config.preprocessor_obj_file_path}")

            return (
                X_train_resampled,
                input_features_test_scaled,
                y_train_resampled,
                target_feature_test,
                self.config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e, sys)
