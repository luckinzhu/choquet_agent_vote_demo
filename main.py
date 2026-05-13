import json
import os
import re
import shutil
from datetime import datetime

import torch

from config import (
    AGENT_BACKEND,
    BEST_MODEL_PATH,
    BATCH_SIZE,
    CHOQUET_MODE,
    DATA_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    LLM_API_KEY_ENV,
    LLM_BASE_URL,
    LLM_CACHE_ENABLED,
    LLM_CACHE_PATH,
    LLM_MODEL,
    LLM_MODEL_CANDIDATES,
    LLM_PROVIDER,
    MODEL_DIR,
    MODEL_SUMMARY_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    RUN_SAMPLE_LIMIT,
    RUNS_DIR,
    TRAIN_RATIO,
    VALID_RATIO,
    WEIGHT_DECAY,
)
from src.dataset import ensure_toy_data, load_and_split, split_dataframe
from src.evaluate import (
    compare_methods,
    export_readable_model_summary,
    print_interpretability_summary,
    print_sample_decisions,
)
from src.model import MultiAgentChoquetModel
from src.train import evaluate_model, train_choquet_model
from src.utils import format_metrics_table, set_seed


VALID_BACKENDS = {"rule", "llm", "hybrid"}
VALID_CHOQUET_MODES = {"inspired", "discrete_2additive"}


def _has_api_key() -> bool:
    return bool(os.getenv(LLM_API_KEY_ENV))


def _resolve_device() -> str:
    requested = (DEVICE or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested DEVICE={DEVICE!r}, but CUDA is not available. Falling back to CPU.")
        return "cpu"
    return requested


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def _create_run_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in RUNS_DIR.iterdir():
        if path.is_dir():
            prefix = path.name.split("_", 1)[0]
            if prefix.isdigit():
                existing.append(int(prefix))
    run_index = max(existing, default=0) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (
        f"{run_index:04d}_{timestamp}_{_safe_slug(AGENT_BACKEND)}_"
        f"{_safe_slug(CHOQUET_MODE)}_{_safe_slug(LLM_MODEL)}"
    )
    run_dir = RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_run_json(run_dir, name: str, payload: dict) -> None:
    (run_dir / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _runtime_snapshot(effective_device: str, extra: dict | None = None) -> dict:
    data = {
        "agent_backend_requested": AGENT_BACKEND,
        "choquet_mode": CHOQUET_MODE,
        "llm_provider": LLM_PROVIDER,
        "llm_model": LLM_MODEL,
        "llm_model_candidates": LLM_MODEL_CANDIDATES,
        "llm_base_url": LLM_BASE_URL,
        "llm_api_key_env": LLM_API_KEY_ENV,
        "api_key_detected": _has_api_key(),
        "llm_cache_enabled": LLM_CACHE_ENABLED,
        "llm_cache_path": str(LLM_CACHE_PATH),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device_requested": DEVICE,
        "device_effective": effective_device,
        "cuda_available": torch.cuda.is_available(),
        "run_sample_limit": RUN_SAMPLE_LIMIT,
    }
    if extra:
        data.update(extra)
    return data


def _print_runtime_config(run_dir, effective_device: str) -> None:
    print("=== Runtime Configuration ===")
    print(f"AGENT_BACKEND: {AGENT_BACKEND}")
    print(f"CHOQUET_MODE: {CHOQUET_MODE}")
    print(f"DEVICE requested/effective: {DEVICE} / {effective_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"LLM_MODEL: {LLM_MODEL}")
    print(f"LLM_MODEL_CANDIDATES: {', '.join(LLM_MODEL_CANDIDATES)}")
    print(f"LLM_BASE_URL: {LLM_BASE_URL}")
    print(f"LLM_API_KEY_ENV: {LLM_API_KEY_ENV}")
    print(f"API key detected: {_has_api_key()}")
    print(f"LLM_CACHE_ENABLED: {LLM_CACHE_ENABLED}")
    print(f"LLM cache path: {LLM_CACHE_PATH}")
    print(f"Run output dir: {run_dir}")


def _runtime_agent_backend() -> str:
    backend = AGENT_BACKEND.strip().lower()
    if backend not in VALID_BACKENDS:
        print(f"Unsupported AGENT_BACKEND={AGENT_BACKEND!r}. Use rule, llm, or hybrid.")
        print("Falling back to AGENT_BACKEND=rule for this run.")
        return "rule"

    needs_key = backend in {"llm", "hybrid"} and LLM_PROVIDER != "local"
    if needs_key and not _has_api_key():
        print(
            f"AGENT_BACKEND={backend} requires environment variable {LLM_API_KEY_ENV}, "
            "but it is not set."
        )
        if backend == "llm":
            print("LLM mode requires a working API key or a complete precomputed cache. Exiting before training.")
            return "missing_llm_key"
        print("Hybrid mode will use rule fallback for this run.")
        return "rule"
    return backend


def _validate_choquet_mode() -> str:
    mode = CHOQUET_MODE.strip().lower()
    if mode not in VALID_CHOQUET_MODES:
        print(f"Unsupported CHOQUET_MODE={CHOQUET_MODE!r}. Falling back to inspired.")
        return "inspired"
    return mode


def _ensure_cache_file() -> None:
    if LLM_CACHE_ENABLED and not LLM_CACHE_PATH.exists():
        LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LLM_CACHE_PATH.write_text("{}", encoding="utf-8")


def _smoke_test_llm_agents(model: MultiAgentChoquetModel) -> tuple[bool, list[str]]:
    print("\n=== LLM Agent Smoke Test ===")
    text = "震惊！这个方法让所有人都忍不住点开，真相终于曝光"
    task_description = "中文点击诱导/标题党检测：class_0=非标题党，class_1=标题党或点击诱导"
    failures = []
    for agent in model.agents:
        output = agent.predict_one(text, task_description)
        probs = output["probs"]
        conf = float(output["confidence"])
        explanation = str(output["explanation"])
        print(
            f"{agent.name:12s} probs={[round(float(x), 4) for x in probs]} "
            f"confidence={conf:.3f} explanation={explanation}"
        )
        if getattr(agent, "last_used_fallback", False):
            error = getattr(agent, "last_error", "unknown LLM failure")
            failures.append(f"{agent.name}: {error}")
    if failures:
        print("Smoke test detected LLM failures:")
        for item in failures:
            print(f"  - {item}")
        return False, failures
    print("Smoke test passed: all LLM agents returned parseable JSON.")
    return True, []


def _copy_latest_artifacts(run_model_path, run_summary_path) -> None:
    if run_model_path.exists():
        shutil.copy2(run_model_path, BEST_MODEL_PATH)
    if run_summary_path.exists():
        shutil.copy2(run_summary_path, MODEL_SUMMARY_PATH)


def main():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_cache_file()
    effective_device = _resolve_device()
    choquet_mode = _validate_choquet_mode()
    run_dir = _create_run_dir()
    run_model_path = run_dir / "best_choquet_model.pt"
    run_summary_path = run_dir / "model_summary.json"
    _print_runtime_config(run_dir, effective_device)
    _write_run_json(run_dir, "run_config.json", _runtime_snapshot(effective_device, {"choquet_mode_effective": choquet_mode}))

    backend = _runtime_agent_backend()
    if backend == "missing_llm_key":
        _write_run_json(run_dir, "run_result.json", _runtime_snapshot(effective_device, {"status": "missing_llm_key"}))
        return

    df = ensure_toy_data(DATA_PATH)
    if RUN_SAMPLE_LIMIT > 0 and RUN_SAMPLE_LIMIT < len(df):
        df = df.sample(n=RUN_SAMPLE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"\nRUN_SAMPLE_LIMIT active: using {len(df)} rows for this run.")
    print(f"\nDataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(df.groupby(["task_name", "label"]).size().to_string())

    if RUN_SAMPLE_LIMIT > 0:
        train_df, valid_df, test_df = split_dataframe(
            df,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            seed=RANDOM_SEED,
        )
    else:
        train_df, valid_df, test_df = load_and_split(
            DATA_PATH,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            seed=RANDOM_SEED,
        )
    print(f"\nSplit: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    model = MultiAgentChoquetModel(
        num_classes=NUM_CLASSES,
        device=effective_device,
        agent_backend=backend,
        choquet_mode=choquet_mode,
    )
    smoke_ok = None
    smoke_failures = []

    if backend in {"llm", "hybrid"}:
        smoke_ok, smoke_failures = _smoke_test_llm_agents(model)
        if not smoke_ok and backend == "llm":
            print("LLM mode smoke test failed. Exiting before training.")
            print("First failure:", smoke_failures[0] if smoke_failures else "unknown")
            _write_run_json(
                run_dir,
                "run_result.json",
                _runtime_snapshot(
                    effective_device,
                    {
                        "status": "llm_smoke_failed",
                        "effective_backend": backend,
                        "smoke_failures": smoke_failures,
                    },
                ),
            )
            return
        if not smoke_ok and backend == "hybrid":
            print("Hybrid smoke test failed. Rebuilding with rule agents for this run.")
            backend = "rule"
            model = MultiAgentChoquetModel(
                num_classes=NUM_CLASSES,
                device=effective_device,
                agent_backend=backend,
                choquet_mode=choquet_mode,
            )

    # Fit TF-IDF relevance on all available text/task descriptions. This is not
    # label leakage; it only builds a vocabulary for relevance estimation.
    model.fit_relevance(df)

    if backend in {"llm", "hybrid"}:
        print("\n=== Precomputing LLM Agent Output Cache ===")
        print("LLM calls happen here. Training epochs switch to cache-only reads afterward.")
        model.warm_agent_cache(df)
        model.set_llm_cache_only(True)
        print(f"LLM cache saved to: {LLM_CACHE_PATH}")

    print(f"\n=== Training Choquet Layer ({choquet_mode}) ===")
    try:
        model, best_valid = train_choquet_model(
            model=model,
            train_df=train_df,
            valid_df=valid_df,
            model_path=run_model_path,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            device=effective_device,
        )
    except Exception as exc:
        if backend == "llm":
            print("Training stopped, likely because an LLM cache entry is missing.")
            print("Run scripts/precompute_llm_outputs.py before llm-mode training.")
        raise exc
    print(
        "\nBest validation: "
        f"loss={best_valid['loss']:.4f}, acc={best_valid['accuracy']:.4f}, "
        f"macro_f1={best_valid['macro_f1']:.4f}"
    )
    print(f"Saved best model to: {run_model_path}")

    print("\n=== Test Metrics for Trained Model ===")
    test_metrics = evaluate_model(model, test_df, BATCH_SIZE, effective_device, use_pairwise=True)
    print(
        f"loss={test_metrics['loss']:.4f}, acc={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, recall={test_metrics['recall']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    print("\n=== Baseline Comparison ===")
    rows = compare_methods(model, test_df)
    print(format_metrics_table(rows))

    print_interpretability_summary(model, test_df)
    print_sample_decisions(model, test_df, n_samples=5, seed=RANDOM_SEED)
    export_readable_model_summary(model, test_df, run_summary_path)
    _copy_latest_artifacts(run_model_path, run_summary_path)
    print(f"\nReadable model summary saved to: {run_summary_path}")
    print(f"Latest model copied to: {BEST_MODEL_PATH}")
    print(f"Latest summary copied to: {MODEL_SUMMARY_PATH}")

    monotonicity = model.layer.monotonicity_diagnostics()
    if choquet_mode == "discrete_2additive":
        print("\n=== Capacity Monotonicity Diagnostic ===")
        print(monotonicity)

    _write_run_json(
        run_dir,
        "run_result.json",
        _runtime_snapshot(
            effective_device,
            {
                "status": "completed",
                "effective_backend": backend,
                "choquet_mode_effective": choquet_mode,
                "smoke_ok": smoke_ok,
                "smoke_failures": smoke_failures,
                "best_validation": best_valid,
                "test_metrics": test_metrics,
                "monotonicity_diagnostic": monotonicity,
                "run_model_path": str(run_model_path),
                "run_summary_path": str(run_summary_path),
            },
        ),
    )
    print(f"Run metadata saved to: {run_dir / 'run_result.json'}")
    print("\nDone. Agents produced fixed outputs; only the selected Choquet aggregation layer was trained.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
