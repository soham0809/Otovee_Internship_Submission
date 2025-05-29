import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from dataclasses import dataclass

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef

# Cross-validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Model explanation
import shap

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_evaluation_file_path = os.path.join("artifacts", "model_evaluation.json")
    feature_importance_plot_path = os.path.join("artifacts", "feature_importance.png")
    confusion_matrix_plot_path = os.path.join("artifacts", "confusion_matrix.png")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def get_models(self):
        """Return a dictionary of models to evaluate"""
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "XGBoost": XGBClassifier(scale_pos_weight=2, random_state=42),
            "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
            "SVC": SVC(probability=True, class_weight='balanced', random_state=42)
        }
        return models
    
    def create_ensemble_model(self, base_models):
        """Create ensemble models from base models"""
        # Voting Classifier
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft'  # Use probability predictions
        )
        
        # Stacking Classifier
        stacking_clf = StackingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        
        return {
            "Voting Ensemble": voting_clf,
            "Stacking Ensemble": stacking_clf
        }
    
    def evaluate_models_with_cv(self, models, X, y, cv=5):
        """Evaluate models using cross-validation"""
        cv_results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, model in models.items():
            try:
                # Accuracy
                accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                # F1 Score
                f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
                # ROC AUC
                roc_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
                
                cv_results[name] = {
                    'accuracy': accuracy_scores.mean(),
                    'f1': f1_scores.mean(),
                    'roc_auc': roc_auc_scores.mean()
                }
                
                logging.info(f"CV Results for {name}: Accuracy={accuracy_scores.mean():.4f}, "
                           f"F1={f1_scores.mean():.4f}, ROC AUC={roc_auc_scores.mean():.4f}")
            except Exception as e:
                logging.error(f"Error evaluating {name} with CV: {str(e)}")
        
        return cv_results
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        """Plot and save confusion matrix"""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            classes = ['No PCOS', 'PCOS']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig(save_path)
            logging.info(f"Confusion matrix saved to {save_path}")
        except Exception as e:
            logging.error(f"Error plotting confusion matrix: {str(e)}")
    
    def generate_shap_explanations(self, model, X_test, feature_names):
        """Generate SHAP explanations for model predictions"""
        try:
            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test)
                
                # Summary plot
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join("artifacts", "shap_summary.png"))
                plt.close()
                
                # Dependence plots for top features
                for i in range(min(3, len(feature_names))):
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(i, shap_values.values, X_test, feature_names=feature_names, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join("artifacts", f"shap_dependence_{feature_names[i]}.png"))
                    plt.close()
                
                logging.info("✅ SHAP explanations generated and saved")
                return shap_values
            else:
                logging.warning("Model doesn't support predict_proba, skipping SHAP explanations")
                return None
        except Exception as e:
            logging.error(f"Error generating SHAP explanations: {str(e)}")
            return None

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting model training process")
            logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            logging.info(f"y_train distribution: {pd.Series(y_train).value_counts()}")
            
            # Get base models
            base_models = self.get_models()
            
            # Evaluate base models with cross-validation
            logging.info("Evaluating base models with cross-validation")
            cv_results = self.evaluate_models_with_cv(base_models, X_train, y_train)
            
            # Create ensemble models
            logging.info("Creating ensemble models")
            ensemble_models = self.create_ensemble_model({
                name: model for name, model in base_models.items() 
                if cv_results.get(name, {}).get('accuracy', 0) > 0.6
            })
            
            # Evaluate ensemble models
            logging.info("Evaluating ensemble models with cross-validation")
            ensemble_cv_results = self.evaluate_models_with_cv(ensemble_models, X_train, y_train)
            
            # Combine all results
            all_results = {**cv_results, **ensemble_cv_results}
            
            # Find best model based on ROC AUC
            best_model_name = max(all_results, key=lambda x: all_results[x]['roc_auc'])
            
            # Get the best model (either from base or ensemble)
            if best_model_name in base_models:
                best_model = base_models[best_model_name]
            else:
                best_model = ensemble_models[best_model_name]
            
            # Train best model on full training data
            logging.info(f"Training best model: {best_model_name}")
            best_model.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Generate classification report
            report = classification_report(y_test, y_pred)
            
            # Log results
            logging.info(f"✅ Best Model: {best_model_name}")
            logging.info(f"✅ Accuracy: {accuracy:.4f}")
            logging.info(f"✅ F1 Score: {f1:.4f}")
            logging.info(f"✅ Precision: {precision:.4f}")
            logging.info(f"✅ Recall: {recall:.4f}")
            logging.info(f"✅ Matthews Correlation Coefficient: {mcc:.4f}")
            if roc_auc is not None:
                logging.info(f"✅ ROC AUC: {roc_auc:.4f}")
            logging.info(f"\n{report}")
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_test, y_pred, self.config.confusion_matrix_plot_path)
            
            # Generate feature importance plot for tree-based models
            if hasattr(best_model, 'feature_importances_'):
                plt.figure(figsize=(10, 6))
                
                # Get feature names - assuming X_train is a numpy array
                feature_names = [
                    "I_beta-HCG", "II_beta-HCG", "AMH", 
                    "I_beta-HCG_log", "II_beta-HCG_log", "AMH_log",
                    "HCG_ratio"
                ]
                
                # Sort features by importance
                indices = np.argsort(best_model.feature_importances_)[::-1]
                plt.title('Feature Importance')
                plt.bar(range(X_train.shape[1]), best_model.feature_importances_[indices])
                plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=90)
                plt.tight_layout()
                plt.savefig(self.config.feature_importance_plot_path)
                plt.close()
                logging.info(f"Feature importance plot saved to {self.config.feature_importance_plot_path}")
            
            # Generate SHAP explanations
            feature_names = [
                "I_beta-HCG", "II_beta-HCG", "AMH", 
                "I_beta-HCG_log", "II_beta-HCG_log", "AMH_log",
                "HCG_ratio"
            ]
            self.generate_shap_explanations(best_model, X_test, feature_names)
            
            # Save best model
            save_object(self.config.trained_model_file_path, best_model)
            logging.info(f"📦 Best model saved to {self.config.trained_model_file_path}")
            
            # Save model evaluation results
            evaluation_results = {
                "best_model": best_model_name,
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "mcc": float(mcc),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "cv_results": {k: {m: float(v) for m, v in metrics.items()} for k, metrics in all_results.items()}
            }
            
            # Save as JSON
            import json
            with open(self.config.model_evaluation_file_path, 'w') as f:
                json.dump(evaluation_results, f, indent=4)
            
            logging.info(f"📊 Model evaluation results saved to {self.config.model_evaluation_file_path}")
            
            return accuracy, report
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)
