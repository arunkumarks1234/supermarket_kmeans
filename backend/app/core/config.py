import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "Mall_Customers.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "kmeans_model.joblib"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"

# Constants
RANDOM_STATE = 42
