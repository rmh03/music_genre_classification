import logging
from pathlib import Path

class Logger:
    def __init__(self, log_file="training.log"):
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / log_file

        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger()

    def log(self, message):
        print(message)
        self.logger.info(message)

    def log_metrics(self, model_name, metrics):
        self.log(f"Model: {model_name}")
        for metric, value in metrics.items():
            self.log(f"{metric}: {value:.4f}")