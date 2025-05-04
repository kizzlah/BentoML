"""
BentoML Service for Iris Classification

This file defines a BentoML service that serves the Iris classifier model.
It can be used to deploy the model as a REST API.
"""

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, JSON


# Get the latest iris_classifier model from the BentoML model store
iris_classifier_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

# Create a BentoML service
svc = bentoml.Service("iris_classifier_service", runners=[iris_classifier_runner])


@svc.api(input=NumpyNdarray(), output=JSON())
async def predict(input_data: np.ndarray) -> dict:
    """
    Predict the Iris flower class from input features.
    
    Args:
        input_data: 2-D array with shape (batch_size, 4) containing features:
                   sepal length, sepal width, petal length, petal width
    
    Returns:
        Dictionary with prediction results
    """
    result = await iris_classifier_runner.predict.async_run(input_data)
    iris_classes = ['setosa', 'versicolor', 'virginica']
    predictions = [iris_classes[i] for i in result]
    
    return {
        "predictions": predictions,
        "input_shape": input_data.shape,
    }


@svc.api(input=NumpyNdarray(), output=JSON())
async def predict_proba(input_data: np.ndarray) -> dict:
    """
    Get probability estimates for each Iris class.
    
    Args:
        input_data: 2-D array with shape (batch_size, 4) containing features
    
    Returns:
        Dictionary with class probabilities
    """
    result = await iris_classifier_runner.predict_proba.async_run(input_data)
    iris_classes = ['setosa', 'versicolor', 'virginica']
    
    probabilities = []
    for probs in result:
        probabilities.append({
            class_name: float(prob)
            for class_name, prob in zip(iris_classes, probs)
        })
    
    return {
        "probabilities": probabilities,
        "input_shape": input_data.shape,
    }
