import os
import pathlib
import torch

class Config:
    """
    Configuration settings for the Operator Assistant project.
    """

    # 1. Path Configurations
    BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    TRAINED_MODEL_PATH = MODELS_DIR / "operator_assistant_model.pt"
    TOKENIZER_PATH = MODELS_DIR / "tokenizer.json"
    LOG_FILE = LOGS_DIR / "app.log"

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Model Parameters
    MODEL_NAME = "bert-base-uncased" # Example model
    EMBEDDING_DIM = 768
    MAX_SEQ_LENGTH = 128
    NUM_CLASSES = 5 # Example: Number of intent classes
    DROPOUT_RATE = 0.1

    # 3. Training Parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1.0
    SEED = 42
    EARLY_STOPPING_PATIENCE = 3
    EVALUATION_STRATEGY = "epoch" # Or "steps"
    SAVE_STRATEGY = "epoch" # Or "steps"
    SAVE_TOTAL_LIMIT = 2 # Keep only the best 2 checkpoints

    # 4. Environment Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1
    LOGGING_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    TENSORBOARD_LOG_DIR = LOGS_DIR / "tensorboard"

    # Ensure TensorBoard log directory exists
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate the config object for easy import
config = Config()

if __name__ == "__main__":
    # Example of accessing configuration values
    print(f"Base Directory: {config.BASE_DIR}")
    print(f"Data Directory: {config.DATA_DIR}")
    print(f"Model Name: {config.MODEL_NAME}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Device: {config.DEVICE}")
    print(f"Log File Path: {config.LOG_FILE}")
    print(f"Trained Model Path: {config.TRAINED_MODEL_PATH}")