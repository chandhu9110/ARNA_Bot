import pickle
import joblib
import json
import random
# import os
from pathlib import Path

# Load Fertilizer Recommendation Model


def recommend_fertilizer(disease):
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_DIR = BASE_DIR / "Models"
    DATASET_DIR = BASE_DIR / "Dataset"

    fertilizer_model_path = MODEL_DIR / "Recomendation_chatbot_model.pkl"
    tokenizer_path = MODEL_DIR / "recomendation_vectorizer.pkl"
    json_path = DATASET_DIR / "Fartillizer_recom.json"
    try:
        # Load model
        fertilizer_model = joblib.load(fertilizer_model_path)

        # Load tokenizer
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        # Load intents
        with open(json_path, "r") as f:
            intents = json.load(f)

    except Exception as e:
        print("Model loading error:", e)
        return "Fertilizer recommendation unavailable"
    # Tokenize the disease index or name (modify based on tokenizer format)
    if disease=="Normal":
      return "Your Crop is Healthy No Need Fertilizers"
    if disease == "invalid":
        return " "
    promt="Recommend the Fertilizer for"+disease
    tokenized_input = tokenizer.transform([promt])
    predicted_intent = fertilizer_model.predict(tokenized_input)[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            if ':' in response:
              return response.split(':')[1].strip()
            break

    # return predicted_intent
    return response