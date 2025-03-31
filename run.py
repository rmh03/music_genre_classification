from src.model.train import train_sklearn_models_with_tuning, train_dl_models
from src.utils.config import Config
from src.utils.logger import Logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os

def main():
    # Initialize Config and Logger
    config = Config()
    logger = Logger()

    # Load features from audio_features.csv
    logger.log("Loading features from audio_features.csv...")
    features_csv_path = config.audio_features_csv
    df = pd.read_csv(features_csv_path)

    # Separate features and labels
    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    # Encode labels
    y_encoded = pd.factorize(y)[0]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for future use
    scaler_path = os.path.join(config.models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.log(f"Scaler saved to {scaler_path}")

    # Train traditional ML models with hyperparameter tuning
    logger.log("\nTraining Traditional ML Models with Hyperparameter Tuning...")
    sklearn_models = train_sklearn_models_with_tuning(X_train, y_train)
    for name, model in sklearn_models.items():
        model_path = os.path.join(config.models_dir, f"{name.lower().replace(' ', '_')}_tuned.pkl")
        joblib.dump(model, model_path)
        logger.log(f"{name} model saved to {model_path}")

    # Train deep learning models
    logger.log("\nTraining Deep Learning Models...")
    dl_models = train_dl_models(X_train, y_train, num_classes=len(set(y_encoded)))
    for name, model in dl_models.items():
        model_path = os.path.join(config.models_dir, f"{name.lower()}.h5")
        model.save(model_path)
        logger.log(f"{name} model saved to {model_path}")

if __name__ == "__main__":
    main()