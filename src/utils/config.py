import os
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = self.base_dir / "data"
        
        # Paths
        self.raw_data = self.data_dir / "raw"
        self.csv_path = self.raw_data / "features_3_sec.csv"
        self.audio_dir = self.raw_data / "genres_original"
        self.figures_dir = self.base_dir / "src" / "docs" / "figures"
        
        # Model params
        self.model_path = self.base_dir / "models" / "genre_classifier.h5"
        self.epochs = 50
        self.batch_size = 32
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.raw_data, exist_ok=True)
        os.makedirs(self.base_dir / "models", exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)