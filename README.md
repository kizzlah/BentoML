# BentoML Iris Classifier Example

This repository demonstrates how to build and deploy a machine learning model using BentoML.

## Project Structure

- `iris_classifier.py`: Trains a scikit-learn model on the Iris dataset and saves it with BentoML
- `service.py`: Defines a BentoML service that serves the trained model
- `bentofile.yaml`: Configuration for packaging the service as a Bento
- `requirements.txt`: Python dependencies

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python iris_classifier.py
```

This will train a Random Forest classifier on the Iris dataset and save it to the BentoML model store.

### 3. Serve the model locally

```bash
bentoml serve service:svc
```

This will start a local API server at http://localhost:3000

### 4. Build a Bento for deployment

```bash
bentoml build
```

This will package the service and its dependencies into a Bento, which can be deployed to various platforms.

### 5. Deploy the Bento

```bash
bentoml deploy
```

## API Usage

### Predict endpoint

```bash
curl -X POST \
  http://localhost:3000/predict \
  -H 'Content-Type: application/json' \
  -d '[[5.1, 3.5, 1.4, 0.2]]'
```

### Predict probabilities endpoint

```bash
curl -X POST \
  http://localhost:3000/predict_proba \
  -H 'Content-Type: application/json' \
  -d '[[5.1, 3.5, 1.4, 0.2]]'
```

## Learn More

- [BentoML Documentation](https://docs.bentoml.org/)
- [BentoML GitHub Repository](https://github.com/bentoml/BentoML)
