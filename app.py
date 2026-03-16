from flask import Flask, render_template, request, Response, jsonify
import joblib
import pandas as pd
import os
import json
from groq import Groq

app = Flask(__name__)
# Export app for Vercel
app_instance = app

# Load the trained machine learning model
MODEL_PATH = "heart_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Groq API client
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not found. Please train the model first by running `python model_train.py`.", 500

    try:
        # Get data from form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])

        # Create a dataframe for the model input
        features = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]], 
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak'])
        
        # Make prediction
        prob = model.predict_proba(features)[0]
        disease_prob = prob[1] * 100
        
        # Determine risk level based on probability thresholds
        if disease_prob >= 65:
            risk_level = "High Risk"
        elif disease_prob >= 35:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        # Human readable labels for the template
        sex_label = "Male" if sex == 1 else "Female"
        cp_labels = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}
        fbs_label = "Yes" if fbs == 1 else "No"
        restecg_labels = {0: "Normal", 1: "ST-T Abnormal", 2: "LV Hypertrophy"}
        exang_label = "Yes" if exang == 1 else "No"

        patient_data = {
            "age": age,
            "sex": sex_label,
            "chest_pain": cp_labels.get(int(cp), str(int(cp))),
            "resting_bp": trestbps,
            "cholesterol": chol,
            "fasting_blood_sugar_above_120": fbs_label,
            "resting_ecg": restecg_labels.get(int(restecg), str(int(restecg))),
            "max_heart_rate": thalach,
            "exercise_angina": exang_label,
            "st_depression": oldpeak
        }

        # Radar data normalization (0-100 scale)
        def normalize(val, min_val, max_val):
            return max(0, min(100, ((val - min_val) / (max_val - min_val)) * 100))

        # Main Patient Dataset
        radar_patient = {
            "Age": normalize(age, 20, 80),
            "BP": normalize(trestbps, 90, 200),
            "Chol": normalize(chol, 120, 400),
            "HeartRate": normalize(thalach, 60, 200),
            "Stress": normalize(oldpeak, 0, 6),
            "Pain": normalize(cp, 1, 4)
        }

        # Healthy Benchmark (Optimized heart health)
        radar_healthy = {
            "Age": 50,  # Middle ground for comparison
            "BP": normalize(115, 90, 200),
            "Chol": normalize(180, 120, 400),
            "HeartRate": normalize(170, 60, 200),
            "Stress": normalize(0.5, 0, 6),
            "Pain": normalize(1, 1, 4)
        }

        # Clinical Average (Typical baseline)
        radar_average = {
            "Age": normalize(45, 20, 80),
            "BP": normalize(140, 90, 200),
            "Chol": normalize(240, 120, 400),
            "HeartRate": normalize(145, 60, 200),
            "Stress": normalize(2.1, 0, 6),
            "Pain": normalize(2.5, 1, 4)
        }

        radar_data = {
            "patient": radar_patient,
            "healthy": radar_healthy,
            "average": radar_average
        }

        return render_template('result.html', 
                               risk_level=risk_level, 
                               probability=round(disease_prob, 2),
                               patient_data=json.dumps(patient_data),
                               radar_data=json.dumps(radar_data))

    except Exception as e:
        return str(e), 400


@app.route('/api/advice', methods=['POST'])
def get_advice():
    """Stream AI health advice from Groq based on patient data and prediction."""
    try:
        data = request.get_json()
        patient = data.get('patient_data', {})
        risk_level = data.get('risk_level', 'Unknown')
        probability = data.get('probability', 0)

        prompt = f"""You are a kind, experienced cardiologist explaining results to a patient in simple, caring language. 
Based on the following health data and heart disease risk prediction, give personalized health advice.

Patient Data:
- Age: {patient.get('age', 'N/A')}
- Sex: {patient.get('sex', 'N/A')}
- Chest Pain Type: {patient.get('chest_pain', 'N/A')}
- Resting Blood Pressure: {patient.get('resting_bp', 'N/A')} mmHg
- Cholesterol: {patient.get('cholesterol', 'N/A')} mg/dl
- Fasting Blood Sugar > 120 mg/dl: {patient.get('fasting_blood_sugar_above_120', 'N/A')}
- Resting ECG: {patient.get('resting_ecg', 'N/A')}
- Maximum Heart Rate Achieved: {patient.get('max_heart_rate', 'N/A')}
- Exercise Induced Angina: {patient.get('exercise_angina', 'N/A')}
- ST Depression (Oldpeak): {patient.get('st_depression', 'N/A')}

Prediction Result: {risk_level} ({probability}% disease probability)

Instructions:
- Explain what their results mean in plain, warm language (like talking to a friend who is worried)
- Give 3-5 specific, actionable health tips personalized to their data
- If high risk, be reassuring but firm about seeing a doctor
- If low risk, encourage them to keep up good habits
- Keep it concise — around 150-200 words
- Do NOT use markdown formatting, headers, or bullet points with asterisks. Use plain text with line breaks.
- Use simple numbered lists (1. 2. 3.) for tips"""

        def generate():
            stream = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'text': content})}\n\n"
            yield "data: [DONE]\n\n"

        return Response(generate(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
