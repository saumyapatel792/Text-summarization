import os

# MLflow Configurations
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Text_Summarization_System")
MODEL_REGISTRY_NAME = os.getenv("MODEL_REGISTRY_NAME", "TextSummarizerBest")

# Dataset Configurations
DATASET_NAME = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
# Use a small subset for benchmarking and hyperparameter tuning in local/CI runs
BENCHMARK_SAMPLE_SIZE = int(os.getenv("BENCHMARK_SAMPLE_SIZE", "10"))
TUNE_SAMPLE_SIZE = int(os.getenv("TUNE_SAMPLE_SIZE", "10"))
TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", "5"))

# Candidate Models for Benchmarking
CANDIDATE_MODELS = [
    "sshleifer/distilbart-cnn-6-6",
    "t5-small",
    "google/flan-t5-small"
]

# Default best model to fall back on if registry is unavailable or for local default initialization
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "sshleifer/distilbart-cnn-6-6")

# Generation Parameters Configuration
DEFAULT_MIN_LENGTH = 30
DEFAULT_MAX_LENGTH = 142
