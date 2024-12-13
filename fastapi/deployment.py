from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from contextlib import asynccontextmanager
import asyncio

MODEL_NAME = "harth_rfc"
POLL_INTERVAL = 20  # seconds
model = None  # For storing our ML Model once it's available
mlflow.set_tracking_uri("http://mlflow:5000")


async def wait_for_production_model():
    """Wait asynchronously until a production model is available."""
    global model
    while True:
        try:
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            for version in model_versions:
                if "production" in version.aliases:  # Check for alias 'production' 
                    print(f"Found production model: Version {version.version}")

                    # Load the production model
                    model_uri = f"models:/{MODEL_NAME}/{version.version}"
                    model = mlflow.pyfunc.load_model(model_uri)
                    print(f"Loaded production model: {model_uri}")
                    return
        except Exception as e:
            print(f"Error checking model registry: {e}")
        
        print(f"No production model found. Retrying in {POLL_INTERVAL} seconds...")
        await asyncio.sleep(POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management for FastAPI app."""
    task = asyncio.create_task(wait_for_production_model())  # Start polling for the model in the background
    try:
        yield
    finally:
        task.cancel()
        print("Shutting down FastAPI app.")


# Define input schema
class PredictionRequest(BaseModel):
    features: list[float]


# Define output schema
class PredictionResponse(BaseModel):
    prediction: int


# Initialize FastAPI app with a lifespan context
app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    """Make predictions using the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        # Prepare features
        features = [data.features]

        # Get prediction
        prediction = model.predict(features)  # Predicted class
        predicted_class = prediction[0]

        return {"prediction": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        return {"status": "waiting for model"}
    return {"status": "ready"}