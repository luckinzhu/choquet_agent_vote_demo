from __future__ import annotations

import os
import re
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
    "CHOQUET_MODE": "discrete_2additive",

    # Agent backend:
    # "rule" = rule agents, no API key needed
    # "hybrid" = prefer LLM, fallback to rule on failure/cache miss
    # "llm" = LLM only, fail fast on errors/cache miss
    "AGENT_BACKEND": "llm",

    # Fast test settings:
    # "8" means use 8 samples; "" or None means full dataset.
    "RUN_SAMPLE_LIMIT": "8",

    # "2" means quick test; "" or None uses config.py defaults.
    "EPOCHS": "2",
    "BATCH_SIZE": "2",

    # LLM gateway settings.
    "LLM_PROVIDER": "openai_compatible",
    "LLM_MODEL": "gemini-3.1-flash-lite",
    "LLM_API_KEY_ENV": "XIAOHU_API_KEY",

    # Optional local private key. Leave as "" or None to read the system env var.
    # Do not print this value. Do not commit auto_run.py with a real key filled in.
    "LLM_API_KEY_VALUE": "sk-LebPbwmUbesvY2o115OPhG0C3phFZEcM4rSWxuo8n8Lx8Bn9",

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
#    LLM_API_KEY_ENV="XIAOHU_API_KEY", LLM_API_KEY_VALUE="" or your local key,
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


def looks_like_inline_api_key(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return bool(re.match(r"^(sk-|AIza|eyJ)[A-Za-z0-9_.-]{12,}$", text))


def apply_config() -> tuple[bool, str]:
    for key in ENV_KEYS:
        set_or_clear_env(key, CONFIG.get(key))

    api_key_env = str(CONFIG.get("LLM_API_KEY_ENV") or "XIAOHU_API_KEY").strip()
    api_key_value = CONFIG.get("LLM_API_KEY_VALUE")

    if looks_like_inline_api_key(api_key_env) and not api_key_value:
        print("ERROR: LLM_API_KEY_ENV appears to contain a real API key.")
        print('Set LLM_API_KEY_ENV to an env var name like "XIAOHU_API_KEY" and put the key in LLM_API_KEY_VALUE or the system environment.')
        return False, "invalid_key_env"

    if api_key_value is not None and str(api_key_value).strip() != "":
        os.environ[api_key_env] = str(api_key_value).strip()
        return True, "CONFIG"
    if os.environ.get(api_key_env):
        return True, "environment"
    return True, "missing"


def api_key_status() -> tuple[bool, str]:
    api_key_env = os.environ.get("LLM_API_KEY_ENV", "XIAOHU_API_KEY")
    config_value = CONFIG.get("LLM_API_KEY_VALUE")
    if config_value is not None and str(config_value).strip() != "":
        return True, "CONFIG"
    if os.environ.get(api_key_env):
        return True, "environment"
    return False, "missing"


def print_config(project_root: Path) -> None:
    detected, source = api_key_status()
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
    print(f"API key detected : {detected}")
    print(f"API key source   : {source}")
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
    config_ok, _ = apply_config()
    if not config_ok:
        return 2
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


