import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data along with the scaler.
    """
    print("Starting data preprocessing...")
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    # Drop missing values
    df = df.dropna()
    print(f"After dropping NAs: {df.shape[0]} rows")
    
    # Select clustering features
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    
    # Apply MinMaxScaler
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    print("Data normalized using MinMaxScaler")
    
    # Return both scaled data and the scaler (needed for test data)
    result = {
        'scaled_data': clustering_data_minmax,
        'scaler': min_max_scaler
    }
    
    clustering_serialized_data = pickle.dumps(result)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data, finds optimal k,
    saves the optimal model, and returns SSE list and optimal k.
    """
    print("Building K-Means models...")
    data_bytes = base64.b64decode(data_b64)
    result = pickle.loads(data_bytes)
    df = result['scaled_data']
    scaler = result['scaler']
    
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    
    # Test k from 1 to 50
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    
    # Find optimal k using elbow method
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    print(f"Optimal number of clusters (elbow method): {optimal_k}")
    
    # Train final model with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, **kmeans_kwargs)
    final_kmeans.fit(df)
    print(f"Final model trained with k={optimal_k}")
    
    # Save the optimal model and scaler
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    model_package = {
        'model': final_kmeans,
        'scaler': scaler,
        'optimal_k': optimal_k
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_package, f)
    
    print(f"Model saved to {output_path}")
    
    # Return SSE and optimal k
    return {
        'sse': sse,
        'optimal_k': optimal_k
    }


def load_model_elbow(filename: str, model_info: dict):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # Load the saved model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    
    with open(output_path, "rb") as f:
        model_package = pickle.load(f)
    
    loaded_model = model_package['model']
    scaler = model_package['scaler']
    optimal_k = model_package['optimal_k']
    
    # Extract SSE from the dict (THIS IS THE KEY FIX!)
    sse = model_info['sse']
    
    # Elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")
    
    # Load and preprocess test data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    df = df.dropna()
    
    # Select features and scale (IMPORTANT!)
    test_features = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    test_scaled = scaler.transform(test_features)
    
    # Predict on scaled test data
    pred = loaded_model.predict(test_scaled)[0]
    
    # Ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred
