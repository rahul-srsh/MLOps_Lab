# Heart Disease Prediction Lab Documentation

This documentation provides a step-by-step guide to a data science lab focused on predicting heart disease using Python. The lab covers data preprocessing, model training, model registration, and batch inference using MLflow, XGBoost, and scikit-learn.

## Prerequisites

Before starting the lab, ensure that you have the following:

- Python environment set up with required libraries installed.
- Dataset: You will need `heart.csv` (UCI Cleveland Heart Disease dataset) in the `data/` directory.

## Step 1: Importing Data

In this step, we load the heart disease dataset using the Pandas library.

```python
import pandas as pd

data = pd.read_csv("data/heart.csv")
```

## Step 2: Exploring Data

In this step, we'll explore the dataset by examining its shape and the first few rows.

```python
print(data.shape)
data.head()
```

## Step 3: Data Preprocessing

In this step, we'll perform data preprocessing tasks to prepare the dataset for model training.

```python
data.rename(columns=lambda x: x.strip().replace(' ', '_'), inplace=True)
print(data.dtypes)
print(data.describe())
```

### Explanation:
We clean column names by stripping whitespace and replacing spaces with underscores. The dataset contains 13 clinical features and a binary `target` column (1 = heart disease present, 0 = absent).

## Step 4: Data Visualization

In this step, we'll visualize the distribution of the target variable.

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='target', data=data)
plt.title('Heart Disease Distribution (0 = No, 1 = Yes)')
plt.show()
```

## Step 5: Correlation Heatmap

In this step, we'll create a correlation heatmap to understand feature relationships.

```python
plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()
```

## Step 6: Exploratory Data Analysis (EDA)

In this step, we'll perform EDA by creating box plots for continuous features against the target variable.

```python
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, len(continuous_cols), figsize=(25, 5))

for i, col in enumerate(continuous_cols):
    sns.boxplot(x='target', y=col, data=data, ax=axes[i])
    axes[i].set_title(f'{col} vs Target')

plt.tight_layout()
plt.show()
```

## Step 7: Checking for Missing Data

In this step, we'll check for missing data within the dataset.

```python
print("Missing values per column:")
print(data.isna().sum())
```

## Step 8: Data Splitting

In this step, we'll split the dataset into training, validation, and test sets (60/20/20).

```python
from sklearn.model_selection import train_test_split

X = data.drop(["target"], axis=1)
y = data["target"]

X_train, X_rem, y_train, y_rem = train_test_split(
    X, y, train_size=0.6, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rem, y_rem, test_size=0.5, random_state=42
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
```

## Step 9: Building a Baseline Model with XGBoost

In this step, we'll create a baseline model using XGBoost and log its performance using MLflow.

```python
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import sklearn
import xgboost
import time


class XGBModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


with mlflow.start_run(run_name='untuned_xgboost'):
    params = {
        'n_estimators': 100,
        'max_depth': 4,
        'learning_rate': 0.1,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds_test = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, preds_test)

    mlflow.log_params(params)
    mlflow.log_metric('auc', auc_score)
    print(f"Test AUC: {auc_score:.4f}")

    preds_binary = model.predict(X_test)
    print(classification_report(y_test, preds_binary))

    wrapped = XGBModelWrapper(model)
    signature = infer_signature(X_train, wrapped.predict(None, X_train))

    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            f"cloudpickle=={cloudpickle.__version__}",
            f"scikit-learn=={sklearn.__version__}",
            f"xgboost=={xgboost.__version__}",
        ],
        additional_conda_channels=None,
    )

    mlflow.pyfunc.log_model(
        "xgboost_model",
        python_model=wrapped,
        conda_env=conda_env,
        signature=signature,
    )
```

### Explanation:
We create an XGBoost classifier with 100 estimators, max depth of 4, and a learning rate of 0.1. The model is trained on the training data with early stopping evaluation on the validation set. We log parameters, the AUC metric, and the model itself using MLflow. A wrapper class `XGBModelWrapper` returns the probability of the positive class (heart disease present). We also generate a classification report with precision, recall, and F1 scores alongside the AUC.

## Step 10: Feature Importance Analysis

In this step, we analyze feature importance to identify which features have the most impact on predicting heart disease.

```python
feat_imp = pd.DataFrame(
    model.feature_importances_,
    index=X_train.columns.tolist(),
    columns=['importance']
).sort_values('importance', ascending=False)

print(feat_imp)

feat_imp.plot(kind='barh', figsize=(10, 6), legend=False)
plt.xlabel('Importance')
plt.title('XGBoost Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### Explanation:
We extract feature importances from the trained XGBoost model and display them as both a sorted table and a horizontal bar chart for easy visualization.

## Step 11: Model Registration in MLflow Model Registry

In this step, we'll register the trained model in the MLflow Model Registry for version tracking and management.

```python
run_id = mlflow.search_runs(
    filter_string='tags.mlflow.runName = "untuned_xgboost"'
).iloc[0].run_id

model_name = "heart_disease_prediction"
model_version = mlflow.register_model(
    f"runs:/{run_id}/xgboost_model", model_name
)

time.sleep(15)
```

### Explanation:
We retrieve the run ID of the MLflow run where the model was trained. We register the model under the name `heart_disease_prediction` in the Model Registry. A delay is added to ensure the registration process completes.

## Step 12: Transitioning Model Version to Production

In this step, we'll transition the newly registered model version to the "Production" stage.

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
)
```

### Explanation:
We use the MlflowClient to transition the model version to "Production." You can now refer to the model using the path `models:/heart_disease_prediction/production`.

## Step 13: Model Inference and Evaluation

In this step, we'll load the production model and verify its performance.

```python
loaded_model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, loaded_model.predict(X_test)):.4f}')
```

### Explanation:
We load the production model from the registry and compute the AUC on the test set as a sanity check. This should match the AUC logged during training.

## Step 14: Serving the Model for Real-Time Inference

Serve the model from the terminal:

```bash
mlflow models serve -m models:/heart_disease_prediction/production -h 0.0.0.0 -p 5001
```

## Step 15: Performing Real-Time Inference

Send requests to the deployed model's API endpoint:

```python
import requests
import json

url = 'http://localhost:5001/invocations'
data_dict = {"dataframe_split": X_test.to_dict(orient='split')}
response = requests.post(url, json=data_dict)
predictions = response.json()
print(predictions)
```

## Conclusion

In this lab, we covered the full ML lifecycle for a heart disease prediction task:

- **Data preparation**: Loading the UCI Cleveland dataset, exploring features, checking for missing values, and splitting into train/val/test sets.
- **Model training**: Building an XGBoost classifier with MLflow experiment tracking, logging parameters, metrics (AUC, precision, recall, F1), and the model artifact.
- **Model deployment**: Registering the model in MLflow Model Registry, transitioning to production, and serving for real-time inference.
- **Inference**: Both batch inference via `model.predict()` and real-time inference via the REST API endpoint.