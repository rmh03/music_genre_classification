from src.model.train import train_sklearn_models, train_dl_models
from src.evaluation.metrics import evaluate_sklearn_model, evaluate_dl_model
from src.data_processing.load_data import load_audio_files, encode_labels
from src.utils.config import Config
from src.utils.logger import Logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_models():
    """
    Test both traditional ML and deep learning models on the test dataset.
    """
    # Initialize Config and Logger
    config = Config()
    logger = Logger(log_file="test_results.log")
    dataset_path = config.audio_dir

    # Load and preprocess data
    logger.log("Loading data for testing...")
    X, y = load_audio_files(dataset_path, max_files_per_genre=100, augment=False)
    y_encoded, label_encoder = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and test traditional ML models
    logger.log("\nTesting Traditional ML Models...")
    sklearn_models = train_sklearn_models(X_train, y_train)
    for name, model in sklearn_models.items():
        evaluate_sklearn_model(model, X_test, y_test, name, logger)

    # Train and test deep learning models
    logger.log("\nTesting Deep Learning Models...")
    dl_models = train_dl_models(X_train, y_train, num_classes=len(label_encoder.classes_))
    for name, model in dl_models.items():
        evaluate_dl_model(model, X_test[..., np.newaxis], y_test, name, logger)

if __name__ == "__main__":
    test_models()