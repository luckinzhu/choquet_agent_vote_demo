from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ===================== Editable CONFIG =====================
CONFIG = {
    # Choquet mode:
    # "inspired" = original Choquet-inspired pairwise aggregation
    # "discrete_2additive" = discrete 2-additive Choquet approximation
    "CHOQUET_MODE": "inspired",

    # Agent backend:
    # "rule" = rule agents, no API key needed
    # "hybrid" = prefer LLM, fallback to rule on failure/cache miss
    # "llm" = LLM only, fail fast on errors/cache miss
    "AGENT_BACKEND": "rule",

    # Fast test settings:
    # "8" means use 8 samples; "" or None means full dataset.
    "RUN_SAMPLE_LIMIT": "8",

    # "2" means quick test; "" or None uses config.py defaults.
    "EPOCHS": "2",
    "BATCH_SIZE": "2",

    # LLM gateway settings. Do not put the real API key here.
    "LLM_PROVIDER": "openai_compatible",
    "LLM_MODEL": "gemini-3.1-flash-lite",
    "LLM_API_KEY_ENV": "XIAOHU_API_KEY",
    "LLM_BASE_URL": "https://xiaohumini.site/v1",

    # Optional workflow switches:
    "RUN_TEST_GATEWAY": False,
    "RUN_PRECOMPUTE_LLM": False,
    "RUN_MAIN": True,

    # Stop remaining steps if one subprocess fails.
    "STOP_ON_ERROR": True,
}

# Preset examples:
# 1) Quick inspired test:
#    CHOQUET_MODE="inspired", AGENT_BACKEND="rule", RUN_SAMPLE_LIMIT="8", EPOCHS="2", BATCH_SIZE="2"
# 2) Quick discrete_2additive test:
#    CHOQUET_MODE="discrete_2additive", AGENT_BACKEND="rule", RUN_SAMPLE_LIMIT="8", EPOCHS="2", BATCH_SIZE="2"
# 3) Full rule training:
#    CHOQUET_MODE="inspired", AGENT_BACKEND="rule", RUN_SAMPLE_LIMIT="", EPOCHS="", BATCH_SIZE=""
# 4) LLM hybrid small-sample validation:
#    CHOQUET_MODE="inspired", AGENT_BACKEND="hybrid", RUN_SAMPLE_LIMIT="8", EPOCHS="2", BATCH_SIZE="2",
#    RUN_TEST_GATEWAY=True, RUN_PRECOMPUTE_LLM=True, RUN_MAIN=True
# ===========================================================

ENV_KEYS = [
    "CHOQUET_MODE",
    "AGENT_BACKEND",
    "RUN_SAMPLE_LIMIT",
    "EPOCHS",
    "BATCH_SIZE",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY_ENV",
    "LLM_BASE_URL",
]


def set_or_clear_env(key: str, value: object) -> None:
    if value is None or str(value).strip() == "":
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)


def apply_config() -> None:
    for key in ENV_KEYS:
        set_or_clear_env(key, CONFIG.get(key))


def api_key_present() -> bool:
    key_env = os.environ.get("LLM_API_KEY_ENV", "XIAOHU_API_KEY")
    return bool(os.environ.get(key_env))


def print_config(project_root: Path) -> None:
    print("=== auto_run.py configuration ===")
    print(f"Project root     : {project_root}")
    for key in [
        "AGENT_BACKEND",
        "CHOQUET_MODE",
        "RUN_SAMPLE_LIMIT",
        "EPOCHS",
        "BATCH_SIZE",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY_ENV",
    ]:
        print(f"{key:16s}: {os.environ.get(key, '')}")
    print(f"API key detected : {api_key_present()}")
    print("=================================")


def run_step(title: str, args: Iterable[str], project_root: Path) -> int:
    print(f"\n=== {title} ===")
    command = [sys.executable, *args]
    print("Command:", " ".join(command))
    result = subprocess.run(command, cwd=project_root, env=os.environ.copy())
    print(f"{title} exit code: {result.returncode}")
    return result.returncode


def main() -> int:
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    apply_config()
    print_config(project_root)

    steps: list[tuple[str, list[str]]] = []
    if CONFIG.get("RUN_TEST_GATEWAY"):
        steps.append(("Gateway smoke test", ["scripts/test_llm_gateway.py"]))
    if CONFIG.get("RUN_PRECOMPUTE_LLM"):
        steps.append(("Precompute LLM outputs", ["scripts/precompute_llm_outputs.py"]))
    if CONFIG.get("RUN_MAIN"):
        steps.append(("Run main.py", ["main.py"]))

    if not steps:
        print("No steps selected. Enable RUN_TEST_GATEWAY, RUN_PRECOMPUTE_LLM, or RUN_MAIN in CONFIG.")
        return 0

    success = True
    for title, args in steps:
        code = run_step(title, args, project_root)
        if code != 0:
            success = False
            print(f"Step failed: {title}, exit code={code}")
            if CONFIG.get("STOP_ON_ERROR", True):
                break

    print("\n=== auto_run.py summary ===")
    print(f"Success          : {success}")
    print(f"Run outputs dir  : {project_root / 'outputs' / 'runs'}")
    print(f"Latest model     : {project_root / 'outputs' / 'best_choquet_model.pt'}")
    print(f"Latest summary   : {project_root / 'outputs' / 'model_summary.json'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
