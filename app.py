from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_data = {
                "I_beta-HCG(mIU/mL)": float(request.form["I_beta-HCG(mIU/mL)"]),
                "II_beta-HCG(mIU/mL)": float(request.form["II_beta-HCG(mIU/mL)"]),
                "AMH(ng/mL)": float(request.form["AMH(ng/mL)"])
            }

            pipeline = PredictPipeline()
            result = pipeline.predict(input_data)

            return render_template('index.html', prediction_text=f"Predicted PCOS status: {int(result)} (0=No, 1=Yes)")
        except Exception as e:
            return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
