import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data paths
INPUT_DIR = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'input')
COMPETITION_DATA_DIR = os.path.join(INPUT_DIR, 'competition_data')
COMPILED_DATA_DIR = os.path.join(INPUT_DIR, 'compiled_data')

# Competition data files
TRAIN_PATH = os.path.join(COMPETITION_DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(COMPETITION_DATA_DIR, 'test.csv')
SAMPLE_SUBMISSION_PATH = os.path.join(COMPETITION_DATA_DIR, 'sample_submission.csv')
WEATHER_PATH = os.path.join(COMPETITION_DATA_DIR, 'wetter.csv')
KIWO_PATH = os.path.join(COMPETITION_DATA_DIR, 'kiwo.csv')

# Compiled data files
HOLIDAYS_PATH = os.path.join(COMPILED_DATA_DIR, 'Feiertage-SH.csv')
SCHOOL_HOLIDAYS_PATH = os.path.join(COMPILED_DATA_DIR, 'Schulferientage-SH.csv')
WINDJAMMER_PATH = os.path.join(COMPILED_DATA_DIR, 'windjammer.csv')

# Output directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'output')
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')

# Visualization directories
VIZ_DIR = os.path.join(PROJECT_ROOT, '0_DataPreparation', 'visualizations')
CORRELATION_VIZ_DIR = os.path.join(VIZ_DIR, 'correlations')
FEATURE_VIZ_DIR = os.path.join(VIZ_DIR, 'features')

# Create directories if they don't exist
for directory in [COMPETITION_DATA_DIR, COMPILED_DATA_DIR, 
                 FEATURES_DIR, CORRELATION_VIZ_DIR, FEATURE_VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

# Product group mapping
WARENGRUPPEN = {
    1: "Brot",
    2: "Brötchen",
    3: "Croissant",
    4: "Feingebäck",
    5: "Kuchen",
    6: "Saisonales Brot"
}

def get_warengruppe_name(warengruppe_id):
    """Get the name of a product group by its ID"""
    return WARENGRUPPEN.get(warengruppe_id, "Unknown") 