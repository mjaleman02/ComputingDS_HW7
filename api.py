from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model
import joblib
import pandas as pd
import json

# Load the model and the .json with the feature names
model = joblib.load("model.pkl")
with open("features.json", "r") as f:
    feature_names = json.load(f)

# Initialize FastAPI
app = FastAPI()

ModelInput = create_model('ModelInput', **{name: (float, ...) for name in feature_names})

# Define the endpoint 
@app.post("/predict")
async def predict(input_data: ModelInput):
    input_df = pd.DataFrame([input_data.dict()])
    try:
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
