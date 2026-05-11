from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "toy_data.csv"
MODEL_DIR = PROJECT_ROOT / "outputs"
BEST_MODEL_PATH = MODEL_DIR / "best_choquet_model.pt"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.json"

RANDOM_SEED = 42
NUM_CLASSES = 2
AGENT_NAMES = [
    "Semantic",
    "Emotion",
    "Intention",
    "Lexical",
    "Consistency",
]

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 16
EPOCHS = 35
LEARNING_RATE = 0.035
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"
