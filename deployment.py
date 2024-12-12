from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI()
MODEL_NAME = "harth_rfc"

def load_production_model ():
    try:
        model_uri = f"models:/harth_rfc/Production"
        print(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Could not load the production model.")


# Load model when API starts up
model = load_production_model()


# Define input schema
class PredictionRequest(BaseModel):
    features: list[float]


# Define output schema
class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    An endpoint that allows us to make predictions using the model.
    """
    try:
        # Perform prediction
        prediction = model.predict([request.features])
        # If model provides probabilities, handle them
        probabilities = model.predict_proba([request.features]) if hasattr(model, "predict_proba") else [None]

        return PredictionResponse(
            prediction=int(prediction[0]),
            probability=probabilities[0][1] if probabilities[0] else None
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
