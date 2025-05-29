import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    validation_data_path: str = os.path.join("artifacts", "validation.csv")
    data_quality_report_path: str = os.path.join("artifacts", "data_quality_report.html")
    data_profile_path: str = os.path.join("artifacts", "data_profile")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def validate_data(self, df):
        """Validate data quality and log issues"""
        validation_results = {}
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            validation_results["missing_values"] = missing_values[missing_values > 0].to_dict()
            logging.warning(f"Missing values detected: {validation_results['missing_values']}")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["duplicate_count"] = duplicate_count
            logging.warning(f"Found {duplicate_count} duplicate rows")
        
        # Check for data types
        validation_results["dtypes"] = df.dtypes.astype(str).to_dict()
        
        # Check for value ranges in numeric columns
        numeric_cols = ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]
        range_info = {}
        for col in numeric_cols:
            if col in df.columns:
                # First check for non-numeric values
                non_numeric_mask = pd.to_numeric(df[col], errors='coerce').isna() & ~df[col].isna()
                non_numeric_values = df.loc[non_numeric_mask, col].unique().tolist() if any(non_numeric_mask) else []
                
                if non_numeric_values:
                    logging.warning(f"Column {col} contains non-numeric values: {non_numeric_values}")
                    range_info[col] = {
                        "non_numeric_values": non_numeric_values,
                        "non_numeric_count": non_numeric_mask.sum()
                    }
                    continue
                
                # Convert to numeric for stats calculation
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                range_info[col] = {
                    "min": float(numeric_values.min()) if not numeric_values.isna().all() else None,
                    "max": float(numeric_values.max()) if not numeric_values.isna().all() else None,
                    "mean": float(numeric_values.mean()) if not numeric_values.isna().all() else None,
                    "median": float(numeric_values.median()) if not numeric_values.isna().all() else None,
                    "std": float(numeric_values.std()) if not numeric_values.isna().all() else None
                }
        validation_results["numeric_ranges"] = range_info
        
        # Check class distribution for target variable
        if "PCOS" in df.columns:
            class_counts = df["PCOS"].value_counts().to_dict()
            validation_results["class_distribution"] = class_counts
            
            # Log class imbalance if present
            total = sum(class_counts.values())
            for cls, count in class_counts.items():
                percentage = (count / total) * 100
                logging.info(f"Class {cls}: {count} samples ({percentage:.2f}%)")
            
            # Check for severe imbalance
            min_class = min(class_counts.values())
            max_class = max(class_counts.values())
            imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
            validation_results["imbalance_ratio"] = imbalance_ratio
            
            if imbalance_ratio > 1.5:
                logging.warning(f"Class imbalance detected with ratio {imbalance_ratio:.2f}")
        
        return validation_results
    
    def generate_data_visualizations(self, df, output_dir):
        """Generate and save exploratory visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Distribution of target variable
        if "PCOS" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.countplot(x="PCOS", data=df)
            plt.title("Distribution of PCOS Classes")
            plt.savefig(os.path.join(output_dir, "target_distribution.png"))
            plt.close()
        
        # Distribution of numeric features
        numeric_cols = ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]
        for col in numeric_cols:
            if col in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Histogram with KDE
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribution of {col}")
                plt.savefig(os.path.join(output_dir, f"{col.replace('/', '_')}_dist.png"))
                plt.close()
                
                # Boxplot by target class
                if "PCOS" in df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(x="PCOS", y=col, data=df)
                    plt.title(f"{col} by PCOS Status")
                    plt.savefig(os.path.join(output_dir, f"{col.replace('/', '_')}_by_target.png"))
                    plt.close()
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
            plt.close()
        
        logging.info(f"✅ Data visualizations saved to {output_dir}")

    def initiate_data_ingestion(self):
        logging.info("▶ Starting data ingestion")
        try:
            # Load data with better error handling
            try:
                df = pd.read_csv("notebook/data/stud.csv")
                logging.info(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            except Exception as e:
                logging.error(f"Error loading dataset: {str(e)}")
                raise CustomException(f"Failed to load dataset: {str(e)}", sys)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Validate data before processing
            logging.info("Validating data quality")
            validation_results = self.validate_data(df)
            
            # Convert numeric columns with better error handling
            numeric_cols = ["I_beta-HCG(mIU/mL)", "II_beta-HCG(mIU/mL)", "AMH(ng/mL)"]
            for col in numeric_cols:
                if col in df.columns:
                    # Check for non-numeric values before conversion
                    non_numeric = df[col][pd.to_numeric(df[col], errors='coerce').isna()]
                    if len(non_numeric) > 0:
                        logging.warning(f"Found {len(non_numeric)} non-numeric values in {col}: {non_numeric.unique()}")
                    
                    # Convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logging.info(f"✅ Converted {col} to numeric type")
                else:
                    logging.warning(f"Column {col} not found in dataset")
            
            # Handle missing values
            missing_before = df.isnull().sum().sum()
            if missing_before > 0:
                logging.info(f"Found {missing_before} missing values before cleaning")
            
            # Drop rows with missing values in key columns
            df.dropna(subset=numeric_cols + ["PCOS"], inplace=True)
            missing_after = df.isnull().sum().sum()
            logging.info(f"✅ Removed rows with missing values. Remaining rows: {df.shape[0]}")
            
            # Generate data visualizations
            self.generate_data_visualizations(df, self.ingestion_config.data_profile_path)
            
            # Save processed data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"✅ Raw data saved to {self.ingestion_config.raw_data_path}")
            
            # Use stratified split to maintain class distribution
            if "PCOS" in df.columns:
                # First split: train+validation vs test (80/20)
                train_val_set, test_set = train_test_split(
                    df, 
                    test_size=0.2, 
                    random_state=42,
                    stratify=df["PCOS"]
                )
                
                # Second split: train vs validation (80/20 of the 80% = 64/16)
                train_set, val_set = train_test_split(
                    train_val_set,
                    test_size=0.2,
                    random_state=42,
                    stratify=train_val_set["PCOS"]
                )
                
                # Save all splits
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                val_set.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                
                logging.info(f"✅ Data split using stratified sampling:")
                logging.info(f"   - Training set: {train_set.shape[0]} samples")
                logging.info(f"   - Validation set: {val_set.shape[0]} samples")
                logging.info(f"   - Test set: {test_set.shape[0]} samples")
                
                # Log class distribution in splits
                logging.info(f"Class distribution in training set: {train_set['PCOS'].value_counts().to_dict()}")
                logging.info(f"Class distribution in validation set: {val_set['PCOS'].value_counts().to_dict()}")
                logging.info(f"Class distribution in test set: {test_set['PCOS'].value_counts().to_dict()}")
            else:
                # Regular split if no target column
                train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
                train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
                test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
                logging.info("✅ Data split into train and test sets")
            
            # Save data quality report as JSON
            import json
            with open(os.path.join(self.ingestion_config.data_profile_path, "data_quality_report.json"), 'w') as f:
                json.dump(validation_results, f, indent=4, default=str)
            
            logging.info("✅ Data ingestion completed successfully")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            logging.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)
