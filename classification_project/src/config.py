import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "data/raw/train.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data/processed/cleaned.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models/news_classifier.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "results/metrics.txt")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
