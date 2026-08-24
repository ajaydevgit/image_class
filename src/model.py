"""
Machine Learning Classifier Module.

Implements Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN) classification models,
along with model training, prediction, and evaluation metrics interfaces.
"""

import numpy as np
from typing import Dict, Any, Union
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


class ImageClassifier:
    """Unified Machine Learning Image Classifier for SVM and k-NN algorithms."""

    def __init__(self, model_type: str = "svm", **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type (str): Model algorithm ('svm' or 'knn').
            **kwargs: Hyperparameters passed to scikit-learn estimator.
        """
        self.model_type = model_type.lower()
        self.kwargs = kwargs
        self.model = self._build_model()
        self.is_trained = False

    def _build_model(self) -> Union[SVC, KNeighborsClassifier]:
        """Instantiate underlying scikit-learn estimator."""
        if self.model_type == "svm":
            # Default SVM parameters (RBF kernel, C=10.0, gamma='scale')
            C = self.kwargs.get("C", 10.0)
            kernel = self.kwargs.get("kernel", "rbf")
            gamma = self.kwargs.get("gamma", "scale")
            return SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        elif self.model_type == "knn":
            # Default k-NN parameters (n_neighbors=5, weights='distance')
            n_neighbors = self.kwargs.get("n_neighbors", 5)
            weights = self.kwargs.get("weights", "distance")
            return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        else:
            raise ValueError(f"Unsupported model_type: '{self.model_type}'. Choose 'svm' or 'knn'.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "ImageClassifier":
        """
        Train the model on feature vectors.
        
        Args:
            X_train (np.ndarray): Training feature matrix (N, D).
            y_train (np.ndarray): Target label array (N,).
            
        Returns:
            ImageClassifier: Fitted instance.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input test samples.
        
        Args:
            X_test (np.ndarray): Test feature matrix (M, D).
            
        Returns:
            np.ndarray: Predicted labels array (M,).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained (.fit) before making predictions.")
        return self.model.predict(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test set.
        
        Args:
            X_test (np.ndarray): Test feature matrix.
            y_test (np.ndarray): True target labels.
            
        Returns:
            Dict[str, Any]: Metrics dictionary including accuracy, precision, recall, f1, and confusion matrix.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "model_type": self.model_type.upper(),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "y_true": y_test,
            "y_pred": y_pred,
            "classification_report": report
        }
