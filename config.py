import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_DATA_PATH = PROJECT_ROOT / "data" / "toy_data.csv"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


DATA_PATH = _resolve_path(os.getenv("DATA_PATH", str(PROJECT_ROOT / "data" / "raw_data" / "clickbait" / "zongxiang.csv")))
DATA_AUTOGENERATE_DEMO = os.getenv("DATA_AUTOGENERATE_DEMO", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
MODEL_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = MODEL_DIR / "runs"
BEST_MODEL_PATH = MODEL_DIR / "best_choquet_model.pt"
MODEL_SUMMARY_PATH = MODEL_DIR / "model_summary.json"
LLM_CACHE_PATH = MODEL_DIR / "llm_cache_zongxiang.json"
CHOQUET_MODE = os.getenv("CHOQUET_MODE", "inspired").strip().lower()

RANDOM_SEED = 42
NUM_CLASSES = 2
AGENT_NAMES = [
    "Semantic",
    "Emotion",
    "Intention",
    "Lexical",
    "Consistency",
]

# Default to rule agents so the project runs without any API key. LLM settings
# point at an OpenAI-compatible gateway and can be overridden by env vars.
AGENT_BACKEND = os.getenv("AGENT_BACKEND", "rule").strip().lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
LLM_MODEL_CANDIDATES = [
    item.strip()
    for item in os.getenv(
        "LLM_MODEL_CANDIDATES",
        "gemini-3.1-flash-lite,gemini-3.1-flash,gemini-2.5-flash-lite,gemini-2.0-flash-lite,gemini-2.0-flash",
    ).split(",")
    if item.strip()
]
LLM_API_KEY_ENV = os.getenv("LLM_API_KEY_ENV", "XIAOHU_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://xiaohumini.site/v1").strip()
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
)

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "35"))
LEARNING_RATE = 0.035
WEIGHT_DECAY = 1e-4
DEVICE = os.getenv("DEVICE", "auto").strip().lower()
RUN_SAMPLE_LIMIT = int(os.getenv("RUN_SAMPLE_LIMIT", "0"))
