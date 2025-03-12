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

label_encoders = preprocessors.get("label_encoders", {})  # Default to empty dict
scaler = preprocessors.get("scaler", None)
target_encoder = preprocessors.get("target_encoder", None)

# ✅ FastAPI app instance
app = FastAPI()

# ✅ Define request model
class SurveyAnalysisRequest(BaseModel):
    userId: str = None
    recentResponses: list
    selectedHealthIssue: str
    selectedSymptom: str
    dynamicUserDetails: dict

# ✅ Define response model
class SurveyAnalysisResponse(BaseModel):
    question: str
    triggerSeverityAnalysis: bool

# ✅ Home route
@app.get("/")
def home():
    return {"message": "Survey Analysis API is running!"}

# ✅ Process input data function
def process_input_data(request):
    try:
        features = []
        # 🔸 Encode categorical features (if present)
        for col in ["Diet", "Exercise", "Symptom Trend"]:
            if col in request.dynamicUserDetails and col in label_encoders:
                features.append(label_encoders[col].transform([request.dynamicUserDetails[col]])[0])
            else:
                features.append(0)  # Default if missing

        # 🔸 Encode binary & numeric features
        for key in ["Weight Change", "Smoke/Alcohol", "Medications", "Stress", "Sleep Issues", "Energy Level", "Symptom Worsening", "Consulted Doctor"]:
            features.append(request.dynamicUserDetails.get(key, 0))

        # 🔸 Scale features
        if scaler:
            features = scaler.transform([features])

        return np.array(features)

    except Exception as e:
        print("⚠️ Error in feature processing:", str(e))
        return None

# ✅ Predict route
@app.post("/survey_question")
def predict_survey_question(request: SurveyAnalysisRequest):
    try:
        processed_data = process_input_data(request)
        if processed_data is None:
            raise HTTPException(status_code=400, detail="Failed to preprocess input data")

        # ✅ Make prediction
        prediction = model.predict(processed_data)
        predicted_index = np.argmax(prediction)
        predicted_health_issue = target_encoder.inverse_transform([predicted_index])[0] if target_encoder else "Unknown"

        # ✅ Generate AI-selected survey question
        question = f"How has your {request.selectedSymptom} changed recently?"
        trigger_severity_analysis = predicted_health_issue in ["Diabetes", "Hypertension", "Asthma"]

        return SurveyAnalysisResponse(question=question, triggerSeverityAnalysis=trigger_severity_analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
