import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from src.utils.config import Config
from src.utils.logger import Logger
import joblib

def save_confusion_matrix(y_true, y_pred, labels, model_name, output_dir):
    """
    Save the confusion matrix plot to the specified directory.

    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        labels (list): List of class labels.
        model_name (str): Name of the model.
        output_dir (str): Directory to save the plot.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plot_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")

def save_test_results(results, output_dir):
    """
    Save the test results to a text file.

    Parameters:
        results (str): Test results as a string.
        output_dir (str): Directory to save the results.
    """
    results_path = os.path.join(output_dir, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(results)
    print(f"Test results saved to {results_path}")

def test_models():
    """
    Evaluate models and save test results and confusion matrix plots.
    """
    # Initialize Config and Logger
    config = Config()
    logger = Logger(log_file="test_results.log")
    output_dir = os.path.join(config.docs_dir, "test_results")
    os.makedirs(output_dir, exist_ok=True)

    # Load features from audio_features.csv
    logger.log("Loading features from audio_features.csv...")
    features_csv_path = os.path.join(config.data_dir, "processed", "audio_features.csv")
    df = pd.read_csv(features_csv_path)

    # Separate features and labels
    X = df.drop(columns=["filename", "label"])  # Drop filename and label columns
    y = df["label"]

    # Drop extra columns to ensure consistency with training
    extra_columns = ["harmony_mean", "harmony_var", "length", "perceptr_mean", "perceptr_var"]
    for col in extra_columns:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Encode labels
    y_encoded = pd.factorize(y)[0]
    labels = pd.factorize(y)[1]  # Get the class labels

    # Load scaler
    scaler_path = os.path.join(config.models_dir, "scaler.pkl")
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)  # Scale the features

    # Test traditional ML models
    logger.log("\nTesting Traditional ML Models...")
    sklearn_model_paths = {
        "KNN": os.path.join(config.models_dir, "knn_tuned.pkl"),
        "SVM": os.path.join(config.models_dir, "svm_tuned.pkl"),
        "Random Forest": os.path.join(config.models_dir, "random_forest_tuned.pkl")
    }

    results = ""
    for model_name, model_path in sklearn_model_paths.items():
        logger.log(f"Loading {model_name} model from {model_path}...")
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average='weighted')
        recall = recall_score(y_encoded, y_pred, average='weighted')
        f1 = f1_score(y_encoded, y_pred, average='weighted')

        # Log results
        model_results = (
            f"\n=== {model_name} Evaluation ===\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
        )
        logger.log(model_results)
        results += model_results

        # Save confusion matrix
        save_confusion_matrix(y_encoded, y_pred, labels, model_name, output_dir)

    # Test deep learning models
    logger.log("\nTesting Deep Learning Models...")
    dl_model_paths = {
        "CNN": os.path.join(config.models_dir, "cnn.h5")
    }

    for model_name, model_path in dl_model_paths.items():
        logger.log(f"Loading {model_name} model from {model_path}...")
        model = load_model(model_path)

        # Reshape X for CNN (add channel dimension)
        X_3d = X[..., np.newaxis]

        # Predict using the CNN model
        y_pred_probs = model.predict(X_3d)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculate metrics
        accuracy = accuracy_score(y_encoded, y_pred)
        precision = precision_score(y_encoded, y_pred, average='weighted')
        recall = recall_score(y_encoded, y_pred, average='weighted')
        f1 = f1_score(y_encoded, y_pred, average='weighted')

        # Log results
        model_results = (
            f"\n=== {model_name} Evaluation ===\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1 Score: {f1:.4f}\n"
        )
        logger.log(model_results)
        results += model_results

        # Save confusion matrix
        save_confusion_matrix(y_encoded, y_pred, labels, model_name, output_dir)

    # Save all test results to a text file
    save_test_results(results, output_dir)

if __name__ == "__main__":
    test_models()