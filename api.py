from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import json

# Load the trained model and feature names
model = joblib.load("model.pkl")
with open("features.json", "r") as f:
    feature_names = json.load(f)

# Initialize the FastAPI app
app = FastAPI()

# Dynamically create the Pydantic model with all features as float
ModelInput = create_model('ModelInput', **{name: (float, ...) for name in feature_names})

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: ModelInput):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Make a prediction
    try:
        prediction = model.predict(input_df)
        # Convert prediction to a standard Python int type
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
