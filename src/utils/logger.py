import logging
from pathlib import Path

class Logger:
    def __init__(self, log_file="training.log"):
        """
        Initialize the logger.

        Parameters:
            log_file (str): Name of the log file.
        """
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
        """
        Log a message to the console and the log file.

        Parameters:
            message (str): The message to log.
        """
        print(message)
        self.logger.info(message)

    def log_metrics(self, model_name, metrics):
        """
        Log evaluation metrics for a model.

        Parameters:
            model_name (str): Name of the model.
            metrics (dict): Dictionary of evaluation metrics.
        """
        self.log(f"\n=== Evaluation Results for {model_name} ===")
        for metric, value in metrics.items():
            if isinstance(value, (float, int)):
                self.log(f"{metric}: {value:.4f}")
            else:
                self.log(f"{metric}:\n{value}")

    def log_multiline(self, title, content):
        """
        Log multi-line content (e.g., confusion matrix or classification report).

        Parameters:
            title (str): Title of the content.
            content (str): Multi-line string content to log.
        """
        self.log(f"\n{title}")
        for line in content.splitlines():
            self.log(line)