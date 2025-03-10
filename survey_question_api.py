import os
from flask import Flask, request, jsonify

import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained ML model
with open("survey_question_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the trained TF-IDF vectorizer for symptom text processing
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Home Page
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Survey Question Prediction API!"})

@app.route("/survey_question", methods=["POST"])
def predict_survey_question():
    try:
        # Parse JSON request
        data = request.get_json()

        # Required fields
        required_fields = ["symptoms", "stress", "sleep", "energy", "symptom_worsening", "diet", "exercise", "medications", "smoking"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Mapping categorical values to numerical
        diet_mapping = {"balanced": 0, "high_protein": 1, "vegetarian": 2, "fast_food": 3}
        exercise_mapping = {"none": 0, "light": 1, "moderate": 2, "intense": 3}
        symptom_worsening_mapping = {"no_change": 0, "worsening": 1, "improving": 2}

        # Convert symptoms to numerical using TF-IDF
        symptoms_text = [data["symptoms"]]
        symptoms_features = tfidf_vectorizer.transform(symptoms_text).toarray()

        # Convert other inputs to numerical format
        other_features = np.array([
            int(data["stress"]),
            int(data["sleep"]),
            int(data["energy"]),
            symptom_worsening_mapping.get(data["symptom_worsening"].lower(), 0),
            diet_mapping.get(data["diet"].lower(), 0),
            exercise_mapping.get(data["exercise"].lower(), 0),
            int(data["medications"]),
            int(data["smoking"])
        ]).reshape(1, -1)

        # Combine all features (symptoms + other health factors)
        input_features = np.hstack((symptoms_features, other_features))

        # Predict the next survey question
        predicted_question = model.predict(input_features)[0]

        return jsonify({"next_survey_question": predicted_question})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use dynamic port
    app.run(host="0.0.0.0", port=port)
