from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Input
from tensorflow.keras.optimizers import Adam
from src.model.models import (
    create_knn_model, create_svm_model, create_logreg_model,
    create_random_forest_model, create_cnn_model, create_lstm_model
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
    """
    logger = Logger()
    models = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': range(1, 21),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'SVM': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=5000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['lbfgs', 'liblinear']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200],
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
        logger.log(f"Best Accuracy for {name}: {grid_search.best_score_:.4f}")
    return trained_models

def train_dl_models(X_train, y_train, num_classes):
    """
    Train deep learning models (CNN and LSTM).
    """
    logger = Logger()
    models = {}

    # CNN Model
    logger.log("\n=== Training CNN ===")
    cnn_model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    cnn_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train[..., None], y_train, epochs=20, batch_size=32, verbose=1)
    models['CNN'] = cnn_model

    # LSTM Model
    logger.log("\n=== Training LSTM ===")
    lstm_model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    lstm_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_train[..., None], y_train, epochs=20, batch_size=32, verbose=1)
    models['LSTM'] = lstm_model

    return models