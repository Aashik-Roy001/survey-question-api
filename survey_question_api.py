from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
import json
import logging


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

        # 🔹 One-hot encode symptoms (38 total features)
        all_symptoms = sum(HEALTH_ISSUES.values(), [])  # Get full symptom list
        symptom_dict = {symptom: 0 for symptom in all_symptoms}  # Default all to 0
        
        # Set selected symptom to 1 if it's in the list
        if request.selectedSymptom in symptom_dict:
            symptom_dict[request.selectedSymptom] = 1
        
        features.extend(symptom_dict.values())  # Append one-hot symptom encoding

        # 🔹 Encode categorical features safely
        for col in ["Diet", "Exercise", "Symptom Trend"]:
            value = request.dynamicUserDetails.get(col, "Unknown")  # Default "Unknown"
            if value in label_encoders[col].classes_:
                features.append(label_encoders[col].transform([value])[0])
            else:
                features.append(0)  # Default to category 0

        # 🔹 Encode numeric and binary features
        numeric_keys = [
            "Weight Change", "Smoke/Alcohol", "Medications", "Stress", 
            "Sleep Issues", "Energy Level", "Symptom Worsening", "Consulted Doctor"
        ]
        
        for key in numeric_keys:
            features.append(request.dynamicUserDetails.get(key, 0))  # Default 0
        
        # 🔹 Convert to NumPy array and scale
        features = np.array(features).reshape(1, -1)
        if scaler:
            features = scaler.transform(features)

        print(f"✅ Processed Features: {features.shape}")  # Debugging
        return features

    except Exception as e:
        print(f"⚠️ Error in feature processing: {str(e)}")
        return None


# ✅ Predict route
logging.basicConfig(level=logging.INFO)

@app.post("/survey_question")
def predict_survey_question(request: SurveyAnalysisRequest):
    try:
        logging.info(f"📥 Received Input: {json.dumps(request.dict(), indent=2)}")

        processed_data = process_input_data(request)
        if processed_data is None:
            raise HTTPException(status_code=400, detail="400: Failed to preprocess input data")

        # ✅ Make prediction
        prediction = model.predict(processed_data)
        predicted_index = np.argmax(prediction)
        predicted_health_issue = target_encoder.inverse_transform([predicted_index])[0] if target_encoder else "Unknown"

        # ✅ Generate AI-selected survey question
        question = f"How has your {request.selectedSymptom} changed recently?"
        trigger_severity_analysis = predicted_health_issue in ["Diabetes", "Hypertension", "Asthma"]

        logging.info(f"📤 Predicted: {predicted_health_issue}, Question: {question}")

        return SurveyAnalysisResponse(question=question, triggerSeverityAnalysis=trigger_severity_analysis)

    except Exception as e:
        logging.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
