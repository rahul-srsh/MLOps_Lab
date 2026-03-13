"""
Standalone prediction utility for the Wine model.
The FastAPI app loads the model at startup, so this is only for quick CLI testing.

Usage:
    python predict.py 13.2 1.78 2.14 11.2 100 2.65 2.76 0.26 1.28 4.38 1.05 3.40 1050
"""
import sys
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "wine_model.pkl"
TARGET_NAMES = {0: "class_0", 1: "class_1", 2: "class_2"}


def predict_data(X):
    model = joblib.load(MODEL_PATH)
    return model.predict(X)


if __name__ == "__main__":
    if len(sys.argv) != 14:
        print("Usage: python predict.py <alcohol> <malic_acid> <ash> "
              "<alcalinity_of_ash> <magnesium> <total_phenols> <flavanoids> "
              "<nonflavanoid_phenols> <proanthocyanins> <color_intensity> "
              "<hue> <od280_od315> <proline>")
        sys.exit(1)

    features = [[float(v) for v in sys.argv[1:]]]
    pred = predict_data(features)
    label = int(pred[0])
    print(f"Prediction: {label} ({TARGET_NAMES.get(label, 'unknown')})")