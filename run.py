from src.data_processing.load_data import load_audio_files, encode_labels
from src.data_processing.feature_engineering import extract_and_plot_features
from src.model.train import train_sklearn_models, train_dl_models
from src.utils.config import Config
from src.utils.logger import Logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    # Initialize Config and Logger
    config = Config()
    logger = Logger()

    # Load and preprocess data
    logger.log("Loading and preprocessing data...")
    X, y = load_audio_files(config.audio_dir, max_files_per_genre=100, augment=False)
    y_encoded, label_encoder = encode_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for future use
    scaler_path = os.path.join(config.base_dir, "models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.log(f"Scaler saved to {scaler_path}")

    # Train traditional ML models
    logger.log("\nTraining Traditional ML Models...")
    sklearn_models = train_sklearn_models(X_train, y_train)
    for name, model in sklearn_models.items():
        model_path = os.path.join(config.base_dir, "models", f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        logger.log(f"{name} model saved to {model_path}")

    # Train deep learning models
    logger.log("\nTraining Deep Learning Models...")
    dl_models = train_dl_models(X_train, y_train, num_classes=len(label_encoder.classes_))
    for name, model in dl_models.items():
        model_path = os.path.join(config.base_dir, "models", f"{name.lower()}.h5")
        model.save(model_path)
        logger.log(f"{name} model saved to {model_path}")

    # Extract and plot features for a sample audio file
    audio_dir = config.audio_dir
    sample_audio = os.path.join(audio_dir, "classical", "classical.00000.wav")  # Example file path
    logger.log(f"Processing {sample_audio}...")
    extract_and_plot_features(sample_audio)

if __name__ == "__main__":
    main()