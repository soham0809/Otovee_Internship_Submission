# Advanced PCOS Prediction System

An industry-grade machine learning system for predicting Polycystic Ovary Syndrome (PCOS) based on hormonal biomarkers. This project implements a robust ML pipeline with advanced preprocessing, ensemble models, and explainable AI features.

## 🔬 Project Overview

This system analyzes three key hormonal biomarkers (I_beta-HCG, II_beta-HCG, and AMH) to predict the likelihood of PCOS, providing confidence scores and visual explanations for each prediction. The enhanced model achieves improved accuracy through advanced feature engineering, ensemble methods, and proper handling of class imbalance.

## 🔧 Key Features

### Advanced Data Processing
- Robust outlier detection and handling using IQR method
- Feature engineering (hormone ratios, log transformations)
- Data quality validation and reporting
- Stratified sampling for better class distribution

### Enhanced Machine Learning
- Ensemble models (Voting and Stacking classifiers)
- Cross-validation for reliable performance evaluation
- Class imbalance correction with SMOTE-Tomek
- Comprehensive evaluation metrics (F1, ROC-AUC, MCC)

### Explainable AI
- SHAP explanations for individual predictions
- Feature importance visualization
- Confidence scores with predictions

### Production-Ready Features
- Modern web interface with responsive design
- Model performance information page
- Input validation and error handling
- API endpoint for programmatic access
- Prediction logging for monitoring

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/soham0809/PCOS_Data_Pipeline
cd NEW_MAIN

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Training the Model

```bash
python src/pipeline/train_pipeline.py
```

### Running the Web Application

```bash
python app.py
```

Access the application at http://localhost:5000

## 📊 Model Performance

The current model achieves improved performance through ensemble methods and advanced feature engineering:

- Accuracy: ~75-80% (depending on the ensemble method selected)
- F1 Score: ~0.70-0.75
- ROC-AUC: ~0.80-0.85

## 🔄 API Usage

The system provides a REST API endpoint for programmatic access:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"I_beta-HCG(mIU/mL)": 15.0, "II_beta-HCG(mIU/mL)": 15.0, "AMH(ng/mL)": 3.56}'
```

## 📁 Project Structure

```
NEW_MAIN/
├── app.py                    # Flask web application
├── src/                      # Source code
│   ├── components/           # ML pipeline components
│   │   ├── data_ingestion.py # Data loading and preprocessing
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py  # Model training and evaluation
│   ├── pipeline/             # Pipeline orchestration
│   │   ├── predict_pipeline.py # Prediction pipeline
│   │   └── train_pipeline.py # Training pipeline
│   ├── exception.py          # Custom exception handling
│   └── logger.py             # Logging configuration
├── templates/                # Web UI templates
└── artifacts/                # Generated model artifacts
```

## 🛠️ Future Improvements

- Integration with electronic health records
- Additional clinical features beyond hormonal markers
- Deployment to cloud platforms (AWS, Azure, GCP)
- Mobile application interface
- Longitudinal tracking of predictions

## 📄 License

MIT

## 👥 Contributors

- Soham Joshi
