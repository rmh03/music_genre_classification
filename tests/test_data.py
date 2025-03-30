import unittest
from src.data_processing.load_data import load_audio_files, encode_labels
from src.utils.config import Config
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        """Set up configuration and paths."""
        self.config = Config()
        self.dataset_path = self.config.audio_dir

    def test_load_audio_files(self):
        """Test if audio files are loaded and features are extracted correctly."""
        # Ensure the dataset path exists
        self.assertTrue(os.path.exists(self.dataset_path), "Dataset path does not exist.")
        self.assertTrue(len(os.listdir(self.dataset_path)) > 0, "Dataset directory is empty.")

        # Load audio files
        X, y = load_audio_files(self.dataset_path, max_files_per_genre=5, augment=False)

        # Validate features and labels
        self.assertIsInstance(X, np.ndarray, "Features should be a numpy array.")
        self.assertIsInstance(y, np.ndarray, "Labels should be a numpy array.")
        self.assertGreater(len(X), 0, "Features should not be empty.")
        self.assertGreater(len(y), 0, "Labels should not be empty.")
        self.assertEqual(len(X), len(y), "Features and labels should have the same length.")

    def test_encode_labels(self):
        """Test if labels are encoded correctly."""
        labels = ["classical", "jazz", "rock", "classical"]
        encoded_labels, label_encoder = encode_labels(labels)

        # Validate encoded labels
        self.assertEqual(len(encoded_labels), len(labels), "Encoded labels should match input labels.")
        self.assertEqual(len(label_encoder.classes_), len(set(labels)), "Label encoder should have unique classes.")
        self.assertTrue(all(isinstance(label, int) for label in encoded_labels), "Encoded labels should be integers.")
        for i, label in enumerate(labels):
            self.assertEqual(label_encoder.inverse_transform([encoded_labels[i]])[0], label, "Encoded label mapping is incorrect.")

if __name__ == "__main__":
    unittest.main()