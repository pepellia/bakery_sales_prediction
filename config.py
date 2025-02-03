import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Input data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "0_DataPreparation")
COMPETITION_DATA_DIR = os.path.join(DATA_DIR, "input", "competition_data")

# Competition data
TRAIN_PATH = os.path.join(COMPETITION_DATA_DIR, "train.csv")
TEST_PATH = os.path.join(COMPETITION_DATA_DIR, "test.csv")
WEATHER_PATH = os.path.join(COMPETITION_DATA_DIR, "wetter.csv")
KIWO_PATH = os.path.join(COMPETITION_DATA_DIR, "kiwo.csv")

COMPILED_DATA_DIR = os.path.join(DATA_DIR, "input", "compiled_data")
WINDJAMMER_PATH = os.path.join(COMPILED_DATA_DIR, "windjammer.csv")
HOLIDAYS_PATH = os.path.join(COMPILED_DATA_DIR, "Feiertage-SH.csv")
SCHOOL_HOLIDAYS_PATH = os.path.join(COMPILED_DATA_DIR, "Schulferientage-SH.csv")

# Output directories
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "3_Model", "output")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "3_Model", "visualizations") 