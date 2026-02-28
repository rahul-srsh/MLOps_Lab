import logging
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # 1. Load Dataset
    logger.info("Loading Wine dataset...")
    wine = load_wine()
    X, y = wine.data, wine.target
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Train Model
    logger.info("Training GradientBoostingClassifier...")
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training complete!")

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    # 5. Save Model
    model_path = "wine_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to '{model_path}'")

    # 6. Reload & Predict
    logger.info("Reloading model and running a sample prediction...")
    loaded_model = joblib.load(model_path)
    sample = X_test[0].reshape(1, -1)
    prediction = loaded_model.predict(sample)
    predicted_class = wine.target_names[prediction[0]]
    actual_class = wine.target_names[y_test[0]]
    print(f"\n── Sample Prediction ──────────────────────────────────")
    print(f"  Predicted : {predicted_class}")
    print(f"  Actual    : {actual_class}")
    print(f"  Match     : {'✅ Correct' if predicted_class == actual_class else '❌ Wrong'}")