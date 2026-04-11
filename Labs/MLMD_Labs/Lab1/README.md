# Lab 1 — ML Metadata: Breast Cancer Classification Pipeline

This lab demonstrates how to use [ML Metadata (MLMD)](https://www.tensorflow.org/tfx/guide/mlmd) to track every stage of a real multi-step ML pipeline, from raw data ingestion through model evaluation and cross-experiment comparison.

---

## What This Lab Does

The notebook builds a complete four-step classification pipeline on the **Breast Cancer Wisconsin** dataset and uses MLMD to record every artifact, execution, event, and context produced along the way. Two model variants (LogisticRegression and RandomForestClassifier) are trained and compared entirely through metadata store queries — no manual file inspection needed.

### Pipeline Architecture

```
[train DataSet] ──┐
                  ├──► [Data Validation] ──► [Schema]
[eval  DataSet] ──┘              │
                                 │
[train DataSet + eval DataSet + Schema]
                  │
                  ▼
         [Feature Engineering] ──► [PreprocessedData (train)]
                                 └──► [PreprocessedData (eval)]
                                              │
              ┌───────────────────────────────┤
              │                               │
              ▼                               │
  [Model Training v1 (LogReg)] ──► [TrainedModel v1]
              │                               │
              ▼                               │
  [Model Evaluation v1] ──► [EvaluationMetrics v1]
                                              │
              ┌───────────────────────────────┘
              │
              ▼
  [Model Training v2 (RandomForest)] ──► [TrainedModel v2]
              │
              ▼
  [Model Evaluation v2] ──► [EvaluationMetrics v2]
```

A **failed training attempt** (bad `max_iter` hyperparameter) is also recorded before the successful v1 run, demonstrating MLMD's ability to audit failures alongside successes.

---

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**
- Source: `sklearn.datasets.load_breast_cancer()` — no internet download required
- 569 total samples, 30 real-valued features
- Binary classification: 0 = malignant, 1 = benign
- Features: measurements of digitised images of fine needle aspirate (FNA) of breast masses (radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, fractal dimension — each with mean, standard error, and worst values)

| Split | File | Rows | Notes |
|-------|------|------|-------|
| Train | `data/train/data.csv` | ~398 | Includes `target` column |
| Eval | `data/eval/data.csv` | ~114 | Includes `target` column |
| Serving | `data/serving/data.csv` | ~57 | No `target` column |

Splits are stratified (70 / 20 / 10) with `random_state=42` for reproducibility.

---

## File Structure

```
Lab1/
├── C2_W3_Lab_1_MLMetadata.ipynb   Main notebook
├── README.md                       This file
├── schema.pbtxt                    Legacy schema from original lab
├── data/
│   ├── train/data.csv              Breast cancer training split
│   ├── eval/data.csv               Breast cancer eval split
│   └── serving/data.csv            Breast cancer serving split (no target)
├── img/
│   └── mlmd_overview.png           MLMD architecture diagram
└── artifacts/                      Created when the notebook runs
    ├── schema.pbtxt                TFDV-inferred schema
    ├── preprocessed/
    │   ├── train.csv               StandardScaler-transformed training data
    │   ├── eval.csv                StandardScaler-transformed eval data
    │   └── scaler.joblib           Fitted StandardScaler object
    ├── models/
    │   ├── logreg_v1.joblib        Trained LogisticRegression model
    │   └── rf_v2.joblib            Trained RandomForestClassifier model
    └── metrics/
        ├── metrics_v1.json         Eval metrics for LogReg
        └── metrics_v2.json         Eval metrics for RandomForest
```

> `mlmd.sqlite` is also created in the lab root when the notebook runs — this is the persistent metadata store.

---

## MLMD Components Registered

### Artifact Types (5)

| Type | Properties | Represents |
|------|-----------|------------|
| `DataSet` | name, split, version, num_rows | Raw CSV input splits |
| `Schema` | name, version | TFDV-inferred `.pbtxt` schema file |
| `PreprocessedData` | name, split, scaler_path | StandardScaler-transformed CSV |
| `TrainedModel` | name, model_type, version | Serialised sklearn model (`.joblib`) |
| `EvaluationMetrics` | name, model_version, accuracy, f1_score | JSON metrics file (accuracy + F1 stored as queryable `DOUBLE` properties) |

### Execution Types (4)

| Type | Extra Properties | Runs |
|------|-----------------|------|
| `Data Validation` | state | TFDV statistics + schema inference + eval anomaly check |
| `Feature Engineering` | state, scaler | StandardScaler fit on train, transform on train + eval |
| `Model Training` | state, model_type, hyperparameters | sklearn `model.fit()` |
| `Model Evaluation` | state | sklearn `model.predict()` + classification metrics |

### Context Types (2)

| Type | Properties | Groups |
|------|-----------|--------|
| `Pipeline` | pipeline_name, run_id | All 7 executions and 9 artifacts from the full run |
| `Experiment` | model_type, note | Training + evaluation steps and outputs for one model variant |

### Executions Recorded (7)

| ID | Type | State | Notes |
|----|------|-------|-------|
| 1 | Data Validation | COMPLETED | Train + eval stats, schema inferred |
| 2 | Feature Engineering | COMPLETED | StandardScaler fit + transform |
| 3 | Model Training | **FAILED** | `max_iter=1` triggers ConvergenceWarning |
| 4 | Model Training | COMPLETED | LogisticRegression v1 |
| 5 | Model Evaluation | COMPLETED | Evaluates LogReg on eval split |
| 6 | Model Training | COMPLETED | RandomForestClassifier v2 |
| 7 | Model Evaluation | COMPLETED | Evaluates RandomForest on eval split |

---

## Lineage Queries

Three lineage queries are demonstrated at the end of the notebook:

1. **Reverse lineage** — Starting from the v2 (RandomForest) model artifact, walk backwards through the event graph to identify the original raw CSV files used to produce it.

2. **Forward lineage** — Starting from the raw training CSV, perform a BFS traversal forward through all executions to discover every artifact derived from it (preprocessed data, both models, both metrics files).

3. **Cross-experiment comparison** — Query all `EvaluationMetrics` artifacts directly from the metadata store and print a side-by-side comparison table using the `accuracy` and `f1_score` scalar properties stored on the artifacts — no file I/O required.

---

## Changes from the Original Lab

The original lab (`C2_W3_Lab_1_MLMetadata.ipynb`) was a minimal walkthrough that covered only schema generation with one dataset, one execution, and one context. The following changes were made:

### 1. Dataset Replaced
- **Before:** Chicago Taxi dataset downloaded from a GCP public bucket via `urllib`
- **After:** Breast Cancer Wisconsin dataset loaded from `sklearn.datasets.load_breast_cancer()`. No internet connection required. Data is split into stratified train / eval / serving CSVs within the notebook.

### 2. Storage Backend Upgraded to Persistent SQLite
- **Before:** `connection_config.fake_database.SetInParent()` — in-memory only, lost on kernel restart
- **After:** `connection_config.sqlite.filename_uri = './mlmd.sqlite'` — writes a real SQLite database file that persists between runs and can be inspected with any SQLite browser

### 3. Artifact Types Expanded (3 → 5)
- **Before:** `DataSet`, `Schema`, `statistics`
- **After:** `DataSet`, `Schema`, `PreprocessedData`, `TrainedModel`, `EvaluationMetrics`
- `EvaluationMetrics` stores `accuracy` and `f1_score` as `DOUBLE` properties directly on the artifact so they can be compared across experiments without loading the JSON file
- `DataSet` gains a `num_rows` property for data volume tracking
- `PreprocessedData` records the `scaler_path` so the fitted scaler can be located from metadata alone

### 4. Execution Types Expanded (1 → 4)
- **Before:** `Data Validation` only
- **After:** `Data Validation`, `Feature Engineering`, `Model Training`, `Model Evaluation`
- `Model Training` records `model_type` and `hyperparameters` (as a JSON string) so every training run's configuration is searchable in the store

### 5. Full 4-Step Pipeline Built
- **Before:** One execution (TFDV schema generation from training split only)
- **After:** Seven executions covering the full ML lifecycle:
  - Data Validation runs TFDV on both train and eval splits and compares eval statistics against the inferred schema to detect anomalies
  - Feature Engineering fits a `StandardScaler` on the training split, transforms both splits, and saves the fitted scaler alongside the preprocessed CSVs
  - Model Training × 2 (LogisticRegression v1, RandomForestClassifier v2)
  - Model Evaluation × 2 (one per model)

### 6. Failed Execution Recorded
- **Before:** Not demonstrated
- **After:** A training run with `max_iter=1` deliberately triggers a `ConvergenceWarning` (treated as an error). The execution is recorded in the metadata store with `state = FAILED`. This demonstrates the real-world pattern of auditing failed runs alongside successful ones.

### 7. Two Experiment Variants with Cross-Experiment Comparison
- **Before:** No model training or comparison
- **After:** LogisticRegression (v1) and RandomForestClassifier (v2) are both trained on the same preprocessed data. Each experiment gets its own `Experiment` context grouping its training execution, model artifact, and metrics artifact. A lineage query at the end compares both experiments from the metadata store without opening any files.

### 8. Context Types Expanded (1 → 2)
- **Before:** One `Experiment` context named `"Demo"` with a single `note` property
- **After:**
  - `Pipeline` context groups all 7 executions and 9 artifacts from the entire run, with `pipeline_name` and `run_id` (UUID) properties
  - `Experiment` context (one per model variant) groups only the training + evaluation steps and their outputs, enabling per-experiment artifact queries

### 9. Lineage Queries Expanded (1 → 3)
- **Before:** One reverse-lineage query: schema → execution → input dataset
- **After:**
  - Reverse lineage from model artifact back to raw CSV (two hops: training → feature engineering → raw data)
  - Forward lineage BFS from raw training CSV to all downstream artifacts across the entire graph
  - Cross-experiment comparison querying `EvaluationMetrics` artifact properties directly from the store

### 10. Real Artifacts Produced on Disk
- **Before:** Only a `schema.pbtxt` file was written
- **After:** The notebook produces an `artifacts/` directory containing the inferred schema, preprocessed CSVs, the fitted scaler, two trained model files, and two metrics JSON files — all with their paths registered in the metadata store

---

## Dependencies

The notebook requires the same environment as the original lab:

```
tensorflow >= 2.15
tensorflow-data-validation >= 1.15
ml-metadata
scikit-learn
pandas
numpy
joblib
```

---

## How to Run

1. Open `C2_W3_Lab_1_MLMetadata.ipynb` in Jupyter
2. Run all cells top to bottom (`Kernel → Restart & Run All`)
3. The `artifacts/` directory and `mlmd.sqlite` file will be created automatically
4. Re-running the notebook on an existing `mlmd.sqlite` will append new entries — delete `mlmd.sqlite` first for a clean run
