import logging
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (once at startup, not per-request)
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"
TARGET_NAMES = {0: "class_0", 1: "class_1", 2: "class_2"}

ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the app starts; clean up on shutdown."""
    global ml_model
    logger.info("Loading model from %s", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run train.py first."
        )
    ml_model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Wine Classifier API",
    description="Serves a Decision-Tree classifier trained on the Wine dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class WineData(BaseModel):
    """Input features — 13 physicochemical properties of the wine."""
    alcohol:                    float = Field(..., gt=0, description="Alcohol content")
    malic_acid:                 float = Field(..., gt=0, description="Malic acid")
    ash:                        float = Field(..., gt=0, description="Ash")
    alcalinity_of_ash:          float = Field(..., gt=0, description="Alcalinity of ash")
    magnesium:                  float = Field(..., gt=0, description="Magnesium")
    total_phenols:              float = Field(..., gt=0, description="Total phenols")
    flavanoids:                 float = Field(..., ge=0, description="Flavanoids")
    nonflavanoid_phenols:       float = Field(..., ge=0, description="Non-flavanoid phenols")
    proanthocyanins:            float = Field(..., ge=0, description="Proanthocyanins")
    color_intensity:            float = Field(..., gt=0, description="Color intensity")
    hue:                        float = Field(..., gt=0, description="Hue")
    od280_od315_of_diluted_wines: float = Field(..., gt=0, description="OD280/OD315 of diluted wines")
    proline:                    float = Field(..., gt=0, description="Proline")


class WineResponse(BaseModel):
    prediction: int
    wine_class: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.get("/info", status_code=status.HTTP_200_OK)
async def model_info():
    """Return basic metadata about the loaded model."""
    return {
        "model_type": type(ml_model).__name__,
        "features": [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
            "total_phenols", "flavanoids", "nonflavanoid_phenols",
            "proanthocyanins", "color_intensity", "hue",
            "od280_od315_of_diluted_wines", "proline",
        ],
        "target_names": TARGET_NAMES,
    }


@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[
            wine_features.alcohol,
            wine_features.malic_acid,
            wine_features.ash,
            wine_features.alcalinity_of_ash,
            wine_features.magnesium,
            wine_features.total_phenols,
            wine_features.flavanoids,
            wine_features.nonflavanoid_phenols,
            wine_features.proanthocyanins,
            wine_features.color_intensity,
            wine_features.hue,
            wine_features.od280_od315_of_diluted_wines,
            wine_features.proline,
        ]]
        prediction = ml_model.predict(features)
        label = int(prediction[0])
        return WineResponse(
            prediction=label,
            wine_class=TARGET_NAMES.get(label, "unknown"),
        )
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))