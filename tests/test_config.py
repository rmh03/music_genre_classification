from src.utils.config import Config

def test_config():
    config = Config()
    print(f"Docs directory: {config.docs_dir}")
    print(f"Figures directory: {config.figures_dir}")

if __name__ == "__main__":
    test_config()