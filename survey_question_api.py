from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle

# ✅ Load the trained model
MODEL_PATH = "survey_model.h5"
PREPROCESSING_PATH = "preprocessing.pkl"

model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load preprocessing objects (OneHotEncoder, LabelEncoders & Scaler)
with open(PREPROCESSING_PATH, "rb") as f:
    preprocessors = pickle.load(f)

onehot_encoder = preprocessors["onehot_encoder"]
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

# ✅ Function to process input data
def process_input_data(request):
    try:
        # 🔹 Extract numerical features
        numerical_features = [
            request.dynamicUserDetails.get("age", 30),  # Default age if missing
            request.dynamicUserDetails.get("weight", 70),
            request.dynamicUserDetails.get("height", 175),
            request.dynamicUserDetails.get("energy_level", 3)
        ]

        # 🔸 Extract categorical values for one-hot encoding
        categorical_values = [[
            request.dynamicUserDetails.get("diet", "Balanced"),
            request.dynamicUserDetails.get("exercise", "Regular"),
            request.dynamicUserDetails.get("symptom_trend", "Stable")
        ]]

        # 🔸 Apply one-hot encoding
        encoded_categorical = onehot_encoder.transform(categorical_values)

        # 🔸 Combine numerical and categorical features
        final_features = np.concatenate([numerical_features, encoded_categorical[0]])

        # 🔸 Scale the combined features
        final_features = scaler.transform([final_features])

        return final_features

    except Exception as e:
        print("⚠️ Error in feature processing:", str(e))
        return None

# ✅ Predict route
@app.post("/survey_question")
def predict_survey_question(request: SurveyAnalysisRequest):
    try:
        # ✅ Process input data
        processed_data = process_input_data(request)

        if processed_data is None:
            raise HTTPException(status_code=400, detail="Failed to preprocess input data")

        # ✅ Make prediction
        prediction = model.predict(processed_data)
        predicted_index = np.argmax(prediction)
        predicted_health_issue = target_encoder.inverse_transform([predicted_index])[0]

        # ✅ Generate AI-selected survey question
        question = f"How has your {request.selectedSymptom} changed recently?"
        trigger_severity_analysis = predicted_health_issue in ["Diabetes", "Hypertension", "Asthma"]

        return SurveyAnalysisResponse(question=question, triggerSeverityAnalysis=trigger_severity_analysis)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
