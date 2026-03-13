---
- Original Lab Reference: [FastAPI lab video](https://www.youtube.com/watch?v=KReburHqRIQ&list=PLcS4TrUUc53LeKBIyXAaERFKBJ3dvc9GZ&index=4)
- Original Blog: [FastAPI Lab-1](https://www.mlwithramin.com/blog/fastapi-lab1)

---

## Overview

This lab exposes an ML model as a REST API using
[FastAPI](https://fastapi.tiangolo.com/) and
[uvicorn](https://www.uvicorn.org/).

**What's different from the original lab:**

| Original Lab | This Version |
|---|---|
| Iris dataset (4 features, 3 classes) | **Wine dataset (13 features, 3 classes)** |
| No input validation | **Pydantic `Field` constraints on every feature** |
| Model reloaded on every request | **Model loaded once at startup via `lifespan`** |
| Fragile relative paths | **`pathlib`-based paths relative to `__file__`** |
| Response was a raw integer | **Response includes class name** |
| No model evaluation | **Prints accuracy + classification report** |
| No CORS | **CORS middleware enabled** |
| No `/info` endpoint | **`/info` returns model metadata** |
| No logging | **Logging on startup and errors** |

### Workflow

1. Train a Decision Tree Classifier on the **Wine** dataset.
2. Serve the trained model through a FastAPI + uvicorn API.

---

## Setting Up the Lab

```bash
# 1. Create & activate a virtual environment
python -m venv fastapi_lab1_env
source fastapi_lab1_env/bin/activate   # macOS / Linux
# fastapi_lab1_env\Scripts\activate    # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** `fastapi[all]` in `requirements.txt` pulls in optional extras
> including **uvicorn**.

### Project Structure

```
mlops_labs
└── fastapi_lab1
    ├── assets/
    ├── fastapi_lab1_env/
    ├── model/
    │   └── wine_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── main.py
    │   ├── predict.py      ← standalone CLI utility
    │   └── train.py
    ├── README.md
    └── requirements.txt
```

---

## Running the Lab

All commands assume you are inside **`src/`**:

```bash
cd src
```

### 1. Train the Model

```bash
python train.py
```

Expected output:

```
Train accuracy: 1.0000
Test  accuracy: 0.8889

Classification Report (test set):
              precision    recall  f1-score   support
     class_0       0.95      0.95      0.95        20
     class_1       0.86      0.86      0.86        21
     class_2       0.85      0.85      0.85        13
    accuracy                           0.89        54

Model saved to …/model/wine_model.pkl
```

### 2. Start the API Server

```bash
uvicorn main:app --reload
```

> `main` = the Python file, `app` = the FastAPI instance.
> `--reload` restarts on code changes (development only).

### 3. Test the Endpoints

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

| Method | Endpoint   | Description                             |
|--------|------------|-----------------------------------------|
| GET    | `/`        | Health check                            |
| GET    | `/info`    | Model metadata (type, features, classes)|
| POST   | `/predict` | Predict wine class from 13 features     |

#### Example `/predict` request body

```json
{
  "alcohol": 13.2,
  "malic_acid": 1.78,
  "ash": 2.14,
  "alcalinity_of_ash": 11.2,
  "magnesium": 100,
  "total_phenols": 2.65,
  "flavanoids": 2.76,
  "nonflavanoid_phenols": 0.26,
  "proanthocyanins": 1.28,
  "color_intensity": 4.38,
  "hue": 1.05,
  "od280_od315_of_diluted_wines": 3.40,
  "proline": 1050
}
```

#### Example response

```json
{
  "prediction": 0,
  "wine_class": "class_0"
}
```

#### curl example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14,
    "alcalinity_of_ash": 11.2, "magnesium": 100,
    "total_phenols": 2.65, "flavanoids": 2.76,
    "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28,
    "color_intensity": 4.38, "hue": 1.05,
    "od280_od315_of_diluted_wines": 3.40, "proline": 1050
  }'
```

#### Optional CLI prediction

```bash
python predict.py 13.2 1.78 2.14 11.2 100 2.65 2.76 0.26 1.28 4.38 1.05 3.40 1050
# Prediction: 0 (class_0)
```

---

## About the Wine Dataset

The [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
contains 178 samples of wine from three different cultivars grown in
the same region of Italy. Each sample has 13 physicochemical features
(alcohol content, malic acid, ash, etc.) and belongs to one of three
classes (`class_0`, `class_1`, `class_2`).

---

## FastAPI Quick Reference

### App instance & server

```python
app = FastAPI()
```

```bash
uvicorn main:app --reload
```

### Route decorators

```python
@app.get("/endpoint")
async def read_endpoint(): ...

@app.post("/endpoint")
async def create_endpoint(body: SomeModel): ...
```

### Error handling

```python
from fastapi import HTTPException
raise HTTPException(status_code=404, detail="Not found")
```

See the [FastAPI error-handling docs](https://fastapi.tiangolo.com/tutorial/handling-errors/)
for more patterns.