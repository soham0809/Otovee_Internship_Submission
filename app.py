from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import json
from datetime import datetime
from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = Flask(__name__)

# Configure static folders
app.config['EXPLANATION_FOLDER'] = os.path.join('artifacts', 'explanations')

@app.route('/')
def home():
    return render_template('index.html', input_data={})

@app.route('/explanations/<path:filename>')
def serve_explanation(filename):
    """Serve SHAP explanation images"""
    return send_from_directory(app.config['EXPLANATION_FOLDER'], filename)

@app.route('/model-info')
def model_info():
    """Display model evaluation information"""
    try:
        model_eval_path = os.path.join('artifacts', 'model_evaluation.json')
        if os.path.exists(model_eval_path):
            with open(model_eval_path, 'r') as f:
                model_data = json.load(f)
            return render_template('model_info.html', model_data=model_data)
        else:
            return render_template('model_info.html', error="Model evaluation data not found. Please train the model first.")
    except Exception as e:
        logging.error(f"Error loading model info: {str(e)}")
        return render_template('model_info.html', error=f"Error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract input data from form
            input_data = {
                "I_beta-HCG(mIU/mL)": float(request.form["I_beta-HCG(mIU/mL)"]),
                "II_beta-HCG(mIU/mL)": float(request.form["II_beta-HCG(mIU/mL)"]),
                "AMH(ng/mL)": float(request.form["AMH(ng/mL)"])
            }
            
            # Log the input
            logging.info(f"Received prediction request with input: {input_data}")
            
            # Make prediction
            pipeline = PredictPipeline()
            result = pipeline.predict(input_data)
            
            # Extract prediction details
            prediction = result["prediction"]
            confidence = result["confidence"]
            explanation_path = result["explanation_path"]
            
            # Format confidence as percentage if available
            confidence_text = f"{confidence*100:.2f}%" if confidence is not None else "Not available"
            
            # Get explanation filename if available
            explanation_filename = os.path.basename(explanation_path) if explanation_path else None
            
            # Prepare result message
            prediction_text = f"Predicted PCOS status: {'Yes' if prediction == 1 else 'No'}"
            
            return render_template(
                'index.html', 
                prediction_text=prediction_text,
                prediction=prediction,
                confidence=confidence_text,
                explanation_filename=explanation_filename,
                input_data=input_data
            )
        except CustomException as e:
            logging.error(f"CustomException in prediction: {str(e)}")
            return render_template('index.html', error=str(e))
        except Exception as e:
            logging.error(f"Unexpected error in prediction: {str(e)}")
            return render_template('index.html', error=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Get JSON data
        input_data = request.json
        
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Make prediction
        pipeline = PredictPipeline()
        result = pipeline.predict(input_data)
        
        # Return JSON response
        return jsonify({
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
