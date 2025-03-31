import os

class Config:
    def __init__(self):
        # Base directory of the project
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

        # Directory paths
        self.data_dir = os.path.join(self.base_dir, "data")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.logs_dir = os.path.join(self.base_dir, "logs")
        self.docs_dir = os.path.join(self.base_dir, "docs")  # Add docs directory
        self.figures_dir = os.path.join(self.docs_dir, "figures")  # Add figures directory inside docs

        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)