import os

class Config:
    def __init__(self):
        # Base directory
        self.base_dir = "H:/ML project/Project"

        # Data directories
        self.data_dir = os.path.join(self.base_dir, "data")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")

        # CSV files
        self.audio_features_csv = os.path.join(self.processed_data_dir, "audio_features.csv")

        # Model directory
        self.models_dir = os.path.join(self.base_dir, "models")

        # Logs directory
        self.logs_dir = os.path.join(self.base_dir, "logs")

        # Ensure directories exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)