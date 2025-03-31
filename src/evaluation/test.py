from src.evaluation.metrics import evaluate_sklearn_model, evaluate_dl_model
from src.utils.config import Config
from src.utils.logger import Logger
from sklearn.externals import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

def test_models():
    """
    Test both traditional ML and deep learning models on the test dataset.
    """
    # Initialize Config and Logger
    config = Config()
    logger = Logger(log_file="test_results.log")

    # Load features from audio_features.csv
    logger.log("Loading features from audio_features.csv...")
    features_csv_path = config.audio_features_csv
    df = pd.read_csv(features_csv_path)

    # Separate features and labels
    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    # Encode labels
    y_encoded = pd.factorize(y)[0]

    # Load scaler
    scaler_path = config.models_dir + "/scaler.pkl"
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)

    # Test traditional ML models
    logger.log("\nTesting Traditional ML Models...")
    sklearn_model_paths = {
        "KNN": config.models_dir + "/knn_tuned.pkl",
        "SVM": config.models_dir + "/svm_tuned.pkl",
        "Logistic Regression": config.models_dir + "/logistic_regression_tuned.pkl",
        "Random Forest": config.models_dir + "/random_forest_tuned.pkl"
    }
    for name, path in sklearn_model_paths.items():
        model = joblib.load(path)
        logger.log(f"Loaded {name} model from {path}")
        evaluate_sklearn_model(model, X, y_encoded, name, logger)

    # Test deep learning models
    logger.log("\nTesting Deep Learning Models...")
    dl_model_paths = {
        "CNN": config.models_dir + "/cnn.h5",
        "LSTM": config.models_dir + "/lstm.h5"
    }
    for name, path in dl_model_paths.items():
        model = load_model(path)
        logger.log(f"Loaded {name} model from {path}")
        X_3d = X[..., np.newaxis]  # Reshape for deep learning models
        evaluate_dl_model(model, X_3d, y_encoded, name, logger)

if __name__ == "__main__":
    test_models()