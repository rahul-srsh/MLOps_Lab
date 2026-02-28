# Docker Lab 1 — Wine Classification with GradientBoosting

## Overview

This lab demonstrates how to containerize a machine learning model using Docker.
The application trains a **GradientBoostingClassifier** on the **Wine dataset**, evaluates it, saves the model, and runs a sample prediction — all inside a Docker container.

---

## Modifications from Original Lab

### `src/main.py`
| | Original | Modified |
|---|---|---|
| Dataset | Iris (150 samples, 4 features) | Wine (178 samples, 13 features) |
| Model | RandomForestClassifier | GradientBoostingClassifier |
| Logging | `print()` only | `logging` module with timestamps |
| Evaluation | None | Accuracy + Classification Report |
| Post-save test | None | Reloads model and runs sample prediction |

### `src/requirements.txt`
| | Original | Modified |
|---|---|---|
| Versions | Not pinned | Pinned for reproducibility |
| numpy | Not listed | Explicitly added |

### `Dockerfile`
| | Original | Modified |
|---|---|---|
| Base image | `python:3.10` (~900MB) | `python:3.10-slim` (~130MB) |
| Labels | None | Added maintainer + description |
| Layer caching | Not optimized | `requirements.txt` copied first |
| pip | No flags | `--no-cache-dir` for smaller image |

---

## Project Structure

```
Lab1/
├── Dockerfile
└── src/
    ├── main.py
    └── requirements.txt
```

---

## What the App Does

1. **Loads** the Wine dataset (178 samples, 13 features, 3 classes)
2. **Splits** data into 80% train / 20% test
3. **Trains** a GradientBoostingClassifier
4. **Evaluates** the model — prints accuracy and classification report
5. **Saves** the model as `wine_model.pkl`
6. **Reloads** the model and runs a sample prediction to validate it

---

## Docker Concepts

![Docker Flow](taxonomy-of-docker-terms-and-concepts.webp)

### Dockerfile → Image → Container

```
Dockerfile  --build-->  Docker Image  --run-->  Docker Container
```

- **Dockerfile** — instructions to build the image
- **Image** — static, read-only blueprint (~130MB with slim base)
- **Container** — running instance of the image

---

## Getting Started

### Prerequisites
- [Docker Desktop](https://www.docker.com/get-started) installed and running

### Step 1 — Build the Docker Image

```bash
# Navigate to Lab1 folder
cd path/to/Lab1

# Build the image
docker build -t wine-classifier:v1 .
```

### Step 2 — Run the Container

```bash
docker run --name wine-lab1 wine-classifier:v1
```

### Step 3 — Expected Output

```
2026-02-28 — INFO — Loading Wine dataset...
2026-02-28 — INFO — Dataset loaded: 178 samples, 13 features, 3 classes
2026-02-28 — INFO — Train size: 142 | Test size: 36
2026-02-28 — INFO — Training GradientBoostingClassifier...
2026-02-28 — INFO — Model training complete!
2026-02-28 — INFO — Model Accuracy: 97.22%

── Classification Report ──────────────────────────────
              precision    recall  f1-score   support
     class_0       1.00      1.00      1.00        14
     class_1       1.00      0.93      0.96        14
     class_2       0.89      1.00      0.94         8

── Sample Prediction ──────────────────────────────────
  Predicted : class_0
  Actual    : class_0
  Match     : ✅ Correct
```

### Step 4 — Verify

```bash
# Should show Exited (0) — clean exit
docker ps -a

# View logs again anytime
docker logs wine-lab1
```

---

## Docker Cheatsheet

### Images
```bash
docker images                        # list all images
docker build -t <name>:<tag> .       # build image
docker rmi <image_id>                # remove image
```

### Containers
```bash
docker ps                            # list running containers
docker ps -a                         # list all containers
docker run --name <name> <image>     # run a container
docker logs <container_id>           # view logs
docker stop <container_id>           # stop container
docker rm <container_id>             # remove container
```

### Cleanup
```bash
docker rm wine-lab1                  # remove this lab's container
docker rmi wine-classifier:v1        # remove this lab's image
docker system prune -f               # remove all unused resources
```

---

## Why These Modifications?

| Modification | Reason |
|---|---|
| **Wine dataset** | More features (13 vs 4) — better demonstrates real ML complexity |
| **GradientBoosting** | Boosting algorithms are widely used in production MLOps pipelines |
| **Pinned dependencies** | Reproducibility — same versions every time the container is built |
| **Logging** | Standard practice in production; timestamps help with debugging |
| **Evaluation metrics** | Model accuracy must always be measured, not just assumed |
| **Slim base image** | Smaller images = faster pulls, less storage, better for CI/CD |
| **Layer caching** | Separating `COPY requirements.txt` from `COPY src/` speeds up rebuilds |
