from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data import load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"


def fit_model(X_train, y_train):
    """Train a Decision Tree Classifier and return it."""
    dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_classifier.fit(X_train, y_train)
    return dt_classifier


def save_model(model, path: Path):
    """Persist the trained model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = fit_model(X_train, y_train)

    # --- evaluate before saving ---
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print("\nClassification Report (test set):")
    print(classification_report(y_test, model.predict(X_test),
                                target_names=["class_0", "class_1", "class_2"]))

    save_model(model, MODEL_PATH)