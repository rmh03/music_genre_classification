from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from src.model.models import (
    create_knn_model, create_svm_model, create_logreg_model,
    create_decision_tree_model, create_random_forest_model,
    create_cnn_model, create_lstm_model
)
from src.utils.logger import Logger
import tensorflow as tf
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, logger):
    """Evaluate model with comprehensive metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.log("Classification Report:")
    logger.log(classification_report(y_test, y_pred))
    logger.log("\nConfusion Matrix:")
    logger.log(str(confusion_matrix(y_test, y_pred)))
    logger.log(f"\nAccuracy: {accuracy:.4f}")
    return accuracy

def train_sklearn_models(X_train, y_train):
    """
    Train traditional ML models without evaluation.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.

    Returns:
        dict: Dictionary of trained sklearn models.
    """
    logger = Logger()
    models = {
        'KNN': create_knn_model(),
        'SVM': create_svm_model(),
        'Logistic Regression': create_logreg_model(),
        'Decision Tree': create_decision_tree_model(),
        'Random Forest': create_random_forest_model()
    }
    
    trained_models = {}
    for name, model in models.items():
        logger.log(f"\n=== Training {name} ===")
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def train_sklearn_models_with_tuning(X_train, y_train):
    """
    Train sklearn models with hyperparameter tuning.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.

    Returns:
        dict: Dictionary of trained sklearn models with best hyperparameters.
    """
    logger = Logger()
    models = {
        'SVM': {
            'model': create_svm_model(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'Random Forest': {
            'model': create_random_forest_model(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Logistic Regression': {
            'model': create_logreg_model(),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'KNN': {
            'model': create_knn_model(),
            'params': {
                'n_neighbors': range(1, 21),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'Decision Tree': {
            'model': create_decision_tree_model(),
            'params': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        }
    }

    trained_models = {}
    for name, config in models.items():
        logger.log(f"\n=== Training {name} with Hyperparameter Tuning ===")
        grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        trained_models[name] = grid_search.best_estimator_
        logger.log(f"Best Parameters for {name}: {grid_search.best_params_}")
    return trained_models

def train_dl_models(X_train, y_train, num_classes):
    """
    Train deep learning models without evaluation.

    Parameters:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        num_classes (int): Number of output classes.

    Returns:
        dict: Dictionary of trained deep learning models.
    """
    logger = Logger()
    # Reshape for CNN/LSTM
    X_train_3d = X_train[..., np.newaxis]
    
    # CNN Model
    cnn_model = create_cnn_model(input_shape=(X_train_3d.shape[1], 1), num_classes=num_classes)
    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.log("\n=== Training CNN ===")
    cnn_model.fit(X_train_3d, y_train, epochs=20, batch_size=32, verbose=1)
    
    # LSTM Model
    lstm_model = create_lstm_model(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.log("\n=== Training LSTM ===")
    lstm_model.fit(X_train_3d, y_train, epochs=20, batch_size=32, verbose=1)
    
    return {'CNN': cnn_model, 'LSTM': lstm_model}