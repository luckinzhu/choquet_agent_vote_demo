import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    AGENT_BACKEND,
    DATA_AUTOGENERATE_DEMO,
    DATA_PATH,
    LLM_API_KEY_ENV,
    LLM_CACHE_PATH,
    LLM_MODEL,
    LLM_PROVIDER,
    RANDOM_SEED,
    RUN_SAMPLE_LIMIT,
    has_llm_api_key,
)
from src.dataset import load_dataset  # noqa: E402
from src.model import MultiAgentChoquetModel  # noqa: E402
from src.utils import set_seed  # noqa: E402


def main() -> int:
    set_seed(RANDOM_SEED)
    backend = AGENT_BACKEND if AGENT_BACKEND in {"llm", "hybrid"} else "hybrid"
    if LLM_PROVIDER != "local" and not has_llm_api_key():
        print(f"Missing API key. Set LLM_API_KEY in .env or environment before precomputing LLM outputs.")
        return 2

    df = load_dataset(DATA_PATH, allow_generate_demo=DATA_AUTOGENERATE_DEMO)
    if RUN_SAMPLE_LIMIT > 0 and RUN_SAMPLE_LIMIT < len(df):
        df = df.sample(n=RUN_SAMPLE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"RUN_SAMPLE_LIMIT active: precomputing {len(df)} rows.")
    else:
        print(f"Precomputing all {len(df)} rows.")

    print(f"backend: {backend}")
    print(f"model: {LLM_MODEL}")
    print(f"cache path: {LLM_CACHE_PATH}")
    print(f"planned calls: {len(df) * 5} for {len(df)} samples x 5 agents")

    model = MultiAgentChoquetModel(num_classes=2, device="cpu", agent_backend=backend)
    model.warm_agent_cache(df)

    count = 0
    if LLM_CACHE_PATH.exists():
        try:
            count = len(json.loads(LLM_CACHE_PATH.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            count = -1
    print(f"cache entries after precompute: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
