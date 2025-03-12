from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle

# ✅ Load the trained model
MODEL_PATH = "survey_model.h5"
PREPROCESSING_PATH = "preprocessing.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load preprocessing objects (LabelEncoders & Scaler)
with open(PREPROCESSING_PATH, "rb") as f:
    preprocessors = pickle.load(f)

label_encoders = preprocessors["label_encoders"]
scaler = preprocessors["scaler"]
target_encoder = preprocessors["target_encoder"]

# ✅ FastAPI app instance
app = FastAPI()

# ✅ Define request model
class SurveyAnalysisRequest(BaseModel):
    userId: str = None  # userId is null in request
    recentResponses: list
    selectedHealthIssue: str
    selectedSymptom: str
    dynamicUserDetails: dict  # Only changing details

# ✅ Define response model
class SurveyAnalysisResponse(BaseModel):
    question: str
    triggerSeverityAnalysis: bool

# ✅ Home route
@app.get("/")
def home():
    return {"message": "Survey Analysis API is running!"}

# ✅ Predict route
@app.post("/survey_question")
def predict_survey_question(request: SurveyAnalysisRequest):
    try:
        # 🔹 Convert request data into feature array
        features = []

        # 🔸 Encode categorical features
        for col in ["Diet", "Exercise", "Symptom Trend"]:
            if col in request.dynamicUserDetails:
                features.append(label_encoders[col].transform([request.dynamicUserDetails[col]])[0])
            else:
                features.append(0)  # Default if missing

        # 🔸 Encode binary & numeric features
        binary_numeric_keys = ["Weight Change", "Smoke/Alcohol", "Medications", "Stress", "Sleep Issues", "Energy Level", "Symptom Worsening", "Consulted Doctor"]
        for key in binary_numeric_keys:
            features.append(request.dynamicUserDetails.get(key, 0))  # Default to 0 if missing

        # 🔸 Scale features
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        # 🔹 Make prediction
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_health_issue = target_encoder.inverse_transform([predicted_index])[0]

        # 🔥 Generate AI-selected survey question
        question = f"How has your {request.selectedSymptom} changed recently?"
        trigger_severity_analysis = predicted_health_issue in ["Diabetes", "Hypertension", "Asthma"]

        return SurveyAnalysisResponse(question=question, triggerSeverityAnalysis=trigger_severity_analysis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
