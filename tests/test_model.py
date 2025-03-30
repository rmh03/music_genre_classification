import unittest
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model.models import (
    create_knn_model, create_svm_model, create_logreg_model,
    create_decision_tree_model, create_random_forest_model,
    create_cnn_model, create_lstm_model
)
from src.data_processing.load_data import load_audio_files, encode_labels
from src.utils.config import Config
from src.utils.logger import Logger
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestModelInference(unittest.TestCase):
    def setUp(self):
        """Set up data and models for testing."""
        self.logger = Logger(log_file="test_model.log")
        self.config = Config()
        self.dataset_path = self.config.audio_dir

        # Load and preprocess data
        self.logger.log("Loading and preprocessing data for testing...")
        X, y = load_audio_files(self.dataset_path, max_files_per_genre=10, augment=False)
        y_encoded, self.label_encoder = encode_labels(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test

    def test_sklearn_models(self):
        """Test inference for traditional ML models."""
        models = {
            "KNN": create_knn_model(),
            "SVM": create_svm_model(),
            "Logistic Regression": create_logreg_model(),
            "Decision Tree": create_decision_tree_model(),
            "Random Forest": create_random_forest_model()
        }

        for name, model in models.items():
            with self.subTest(model=name):
                self.logger.log(f"\n=== Testing {name} ===")
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                accuracy = np.mean(y_pred == self.y_test)
                self.logger.log(f"{name} Accuracy: {accuracy:.4f}")
                self.assertGreater(accuracy, 0.3, f"{name} accuracy should be greater than 0.3.")

    def test_cnn_model(self):
        """Test inference for CNN model."""
        self.logger.log("\n=== Testing CNN Model ===")
        cnn_model = create_cnn_model(input_shape=(self.X_train.shape[1], 1), num_classes=len(self.label_encoder.classes_))
        cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Reshape data for CNN
        X_train_3d = self.X_train[..., np.newaxis]
        X_test_3d = self.X_test[..., np.newaxis]

        cnn_model.fit(X_train_3d, self.y_train, epochs=5, batch_size=16, verbose=0)
        loss, accuracy = cnn_model.evaluate(X_test_3d, self.y_test, verbose=0)
        self.logger.log(f"CNN Accuracy: {accuracy:.4f}")
        self.assertGreater(accuracy, 0.3, "CNN accuracy should be greater than 0.3.")

    def test_lstm_model(self):
        """Test inference for LSTM model."""
        self.logger.log("\n=== Testing LSTM Model ===")
        lstm_model = create_lstm_model(input_shape=(self.X_train.shape[1], 1), num_classes=len(self.label_encoder.classes_))
        lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Reshape data for LSTM
        X_train_3d = self.X_train[..., np.newaxis]  # Add a new axis for timesteps
        X_test_3d = self.X_test[..., np.newaxis]

        # Train the LSTM model with more epochs and a smaller batch size
        lstm_model.fit(X_train_3d, self.y_train, epochs=50, batch_size=16, verbose=0)
        loss, accuracy = lstm_model.evaluate(X_test_3d, self.y_test, verbose=0)
        self.logger.log(f"LSTM Accuracy: {accuracy:.4f}")
        self.assertGreater(accuracy, 0.3, "LSTM accuracy should be greater than 0.3.")

class TestModelCreation(unittest.TestCase):
    def test_knn_model(self):
        """Test KNN model creation."""
        model = create_knn_model()
        self.assertIsNotNone(model, "KNN model should not be None.")

    def test_svm_model(self):
        """Test SVM model creation."""
        model = create_svm_model()
        self.assertIsNotNone(model, "SVM model should not be None.")

    def test_logreg_model(self):
        """Test Logistic Regression model creation."""
        model = create_logreg_model()
        self.assertIsNotNone(model, "Logistic Regression model should not be None.")

    def test_decision_tree_model(self):
        """Test Decision Tree model creation."""
        model = create_decision_tree_model()
        self.assertIsNotNone(model, "Decision Tree model should not be None.")

    def test_random_forest_model(self):
        """Test Random Forest model creation."""
        model = create_random_forest_model()
        self.assertIsNotNone(model, "Random Forest model should not be None.")

    def test_cnn_model(self):
        """Test CNN model creation."""
        input_shape = (128, 1)
        num_classes = 10
        model = create_cnn_model(input_shape, num_classes)
        self.assertIsNotNone(model, "CNN model should not be None.")
        self.assertEqual(model.input_shape[1:], input_shape, "CNN input shape mismatch.")

    def test_lstm_model(self):
        """Test LSTM model creation."""
        input_shape = (128, 1)
        num_classes = 10
        model = create_lstm_model(input_shape, num_classes)
        self.assertIsNotNone(model, "LSTM model should not be None.")
        self.assertEqual(model.input_shape[1:], input_shape, "LSTM input shape mismatch.")

if __name__ == "__main__":
    unittest.main()