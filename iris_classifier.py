"""
Iris Classifier Example with BentoML

This example demonstrates how to:
1. Train a simple scikit-learn model on the Iris dataset
2. Save the model with BentoML
3. Create a BentoML service for serving predictions
4. Package the service as a Bento for deployment
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import bentoml
from bentoml.io import NumpyNdarray, JSON


# 1. Load and prepare the Iris dataset
def train_model():
    print("Loading Iris dataset and training model...")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Save the model with BentoML
    saved_model = bentoml.sklearn.save_model(
        "iris_classifier",
        model,
        signatures={
            "predict": {"batchable": True},
            "predict_proba": {"batchable": True},
        },
        metadata={
            "accuracy": float(accuracy),
            "dataset": "iris",
            "framework": "scikit-learn",
        }
    )
    print(f"Model saved: {saved_model.tag}")
    return saved_model.tag


# 2. Define a BentoML service for serving the model
class IrisClassifierService:
    def __init__(self, model_tag):
        # Load the model from the BentoML model store
        self.model = bentoml.sklearn.load_model(model_tag)
        self.iris_classes = ['setosa', 'versicolor', 'virginica']
    
    @bentoml.api(input=NumpyNdarray(), output=JSON())
    def predict(self, input_data):
        """
        Predict the Iris class from input features
        
        Args:
            input_data: 2-D array with shape (batch_size, 4) containing features:
                        sepal length, sepal width, petal length, petal width
        
        Returns:
            List of predicted class names
        """
        predictions = self.model.predict(input_data)
        return [self.iris_classes[prediction] for prediction in predictions]
    
    @bentoml.api(input=NumpyNdarray(), output=JSON())
    def predict_proba(self, input_data):
        """
        Get probability estimates for each Iris class
        
        Args:
            input_data: 2-D array with shape (batch_size, 4) containing features
        
        Returns:
            List of dictionaries mapping class names to probabilities
        """
        probabilities = self.model.predict_proba(input_data)
        result = []
        for probs in probabilities:
            result.append({
                class_name: float(prob)
                for class_name, prob in zip(self.iris_classes, probs)
            })
        return result


# 3. Create a BentoML service
def create_service(model_tag):
    # Create a new BentoML service
    svc = bentoml.Service("iris_classifier_service", runners=[bentoml.sklearn.get(model_tag)])
    
    # Define API endpoints
    @svc.api(input=NumpyNdarray(), output=JSON())
    def predict(input_array):
        result = svc.runners.sklearn_runner.predict.run(input_array)
        return {"prediction": result.tolist()}
    
    @svc.api(input=NumpyNdarray(), output=JSON())
    def predict_proba(input_array):
        result = svc.runners.sklearn_runner.predict_proba.run(input_array)
        iris_classes = ['setosa', 'versicolor', 'virginica']
        return {
            "probabilities": [
                {class_name: float(prob) for class_name, prob in zip(iris_classes, probs)}
                for probs in result
            ]
        }
    
    return svc


if __name__ == "__main__":
    # Train and save the model
    model_tag = train_model()
    print(f"Model trained and saved with tag: {model_tag}")
    print("To serve the model, run: bentoml serve iris_classifier:latest")
    
    # Example of how to use the model for predictions
    print("\nExample prediction:")
    import bentoml
    model = bentoml.sklearn.load_model(model_tag)
    # Sample data: [sepal length, sepal width, petal length, petal width]
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example of Iris setosa
    prediction = model.predict(sample)
    iris_classes = ['setosa', 'versicolor', 'virginica']
    print(f"Predicted class: {iris_classes[prediction[0]]}")
