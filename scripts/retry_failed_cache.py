import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from config import LLM_CACHE_PATH, LLM_PROVIDER, has_llm_api_key  # noqa: E402
from integrity_checker import is_success_entry  # noqa: E402
from src.cache import LLMCache  # noqa: E402
from src.model import MultiAgentChoquetModel  # noqa: E402


@dataclass
class RetryItem:
    cache_key: str
    agent_name: str
    text: str
    task_description: str
    model_name: str
    retry_count: int


@dataclass
class ScanResult:
    total_cache_records: int
    success_records: int
    failed_records: int
    skipped_max_retry: int
    retry_items: List[RetryItem]
    retry_count_distribution: Counter


@dataclass
class RetryResult:
    requested: int = 0
    recovered: int = 0
    still_failed: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry FAILED LLM cache records and overwrite them on success.")
    parser.add_argument("--max-retry", type=int, default=100, help="Skip FAILED records at or above this retry count.")
    parser.add_argument("--base-url", type=str, default=None, help="")
    parser.add_argument("--model", type=str, default=None, help="")
    parser.add_argument("--api-key", type=str, default=None, help="")
    parser.add_argument("--api-key-env", type=str, default=None, help="")
    return parser.parse_args()


def classify_error(error: object) -> str:
    text = str(error or "")
    if "sensitive words detected" in text.lower() or "local:sensitive_words" in text.lower():
        return "SensitiveWords"
    if "IncompleteRead" in text:
        return "IncompleteRead"
    if "SSLError" in text or "SSL" in text:
        return "SSLError"
    if "HTTP 429" in text or "HTTP429" in text or "429" in text:
        return "HTTP429"
    if "TimeoutError" in text or "Timeout" in text or "timed out" in text.lower():
        return "TimeoutError"
    if "ConnectionError" in text or "Connection refused" in text or "Network error" in text:
        return "ConnectionError"
    return "Other"


def is_fallback_error(error: object) -> bool:
    text = str(error or "").lower()
    fallback_keywords = [
        "sensitive words detected",
        "local:sensitive_words",
        "winerror 10054",
        "远程主机强迫关闭了一个现有的连接",
        "connection reset",
        "connectionreseterror",
        "incompleteread",
        "sslerror",
        "timeout",
        "timed out",
    ]
    return any(keyword in text for keyword in fallback_keywords)


def build_fallback_success_record(item: RetryItem, error_message: str) -> Dict[str, object]:
    explanation = (
        "LLM request failed due to API gateway/network issue; "
        "fallback neutral probabilities were assigned to keep data complete."
    )
    raw_payload = {
        "class_0_probability": 0.5,
        "class_1_probability": 0.5,
        "confidence": 0.0,
        "explanation": explanation,
    }
    return {
        "status": "SUCCESS",
        "text": item.text,
        "task_description": item.task_description,
        "agent_name": item.agent_name,
        "model_name": item.model_name,
        "raw_text": json.dumps(raw_payload, ensure_ascii=False),
        "output": {
            "probs": [0.5, 0.5],
            "confidence": 0.0,
            "explanation": explanation,
        },
        "fallback": True,
        "fallback_reason": error_message,
        "fallback_note": (
            "This record was marked SUCCESS only to keep the cache complete. "
            "The probabilities are neutral fallback values, not a real LLM prediction."
        ),
        "retry_count": item.retry_count + 1,
        "last_retry": datetime.now(timezone.utc).isoformat(),
    }


def build_model() -> MultiAgentChoquetModel:
    return MultiAgentChoquetModel(num_classes=2, device="cpu", agent_backend="llm")


def scan_failed_cache(cache: LLMCache, model: MultiAgentChoquetModel, max_retry: int) -> ScanResult:
    agents_by_name = {agent.name: agent for agent in model.agents}
    success_records = 0
    failed_records = 0
    skipped_max_retry = 0
    retry_items: List[RetryItem] = []
    retry_count_distribution: Counter = Counter()

    for cache_key, entry in cache._items.items():
        if not isinstance(entry, dict):
            continue
        status = entry.get("status")
        if status == "SUCCESS":
            success_records += 1
            continue
        if status != "FAILED":
            continue

        failed_records += 1
        agent_name = str(entry.get("agent_name", ""))
        retry_count = int(entry.get("retry_count", 0) or 0)
        retry_count_distribution[retry_count] += 1

        agent = agents_by_name.get(agent_name)
        if agent is None:
            print(f"[WARN] unknown agent in FAILED cache; skip. key={cache_key} agent={agent_name}")
            skipped_max_retry += 1
            continue

        cache_model_name = str(entry.get("model_name", ""))
        if not cache_model_name:
            print(f"[WARN] FAILED cache record has no model_name; skip. key={cache_key} agent={agent_name}")
            skipped_max_retry += 1
            continue

        if retry_count >= max_retry:
            skipped_max_retry += 1
            continue

        retry_items.append(
            RetryItem(
                cache_key=cache_key,
                agent_name=agent_name,
                text=str(entry.get("text", "")),
                task_description=str(entry.get("task_description", "")),
                model_name=cache_model_name,
                retry_count=retry_count,
            )
        )

    return ScanResult(
        total_cache_records=len(cache._items),
        success_records=success_records,
        failed_records=failed_records,
        skipped_max_retry=skipped_max_retry,
        retry_items=retry_items,
        retry_count_distribution=retry_count_distribution,
    )


def print_startup_report(scan: ScanResult) -> None:
    print("==================================================")
    print("Cache Retry Runner")
    print("==================")
    print()
    print(f"Cache records: {scan.total_cache_records}")
    print(f"SUCCESS records: {scan.success_records}")
    print(f"FAILED records: {scan.failed_records}")
    print()
    print(f"Need Retry: {len(scan.retry_items)}")
    print(f"Skipped(MaxRetry): {scan.skipped_max_retry}")
    print()
    print("Retry count distribution:")
    if scan.retry_count_distribution:
        for retry_count, count in sorted(scan.retry_count_distribution.items()):
            print(f"retry_count={retry_count:<3d}: {count}")
    else:
        print("none")
    print()
    print("Cache file:")
    print(LLM_CACHE_PATH)
    print()
    print("==================================================")


class PlainProgress:
    def __init__(self, total: int):
        self.total = total
        self.current = 0

    def __enter__(self):
        print(f"Retrying cache: 0/{self.total}")
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, value: int = 1) -> None:
        self.current += value
        print(f"Retrying cache: {self.current}/{self.total}")

    def set_postfix(self, **_: object) -> None:
        return None


def progress_bar(total: int):
    if tqdm is None:
        return PlainProgress(total)
    return tqdm(total=total, desc="Retrying cache", unit="req")


def overwrite_success_record(cache: LLMCache, cache_key: str, success_entry: Dict[str, object], retry_count: int) -> None:
    success_entry["status"] = "SUCCESS"
    success_entry["retry_count"] = retry_count + 1
    cache._items[cache_key] = success_entry
    cache._flush()


def overwrite_fallback_record(cache: LLMCache, cache_key: str, fallback_entry: Dict[str, object]) -> None:
    cache._items[cache_key] = fallback_entry
    cache._flush()


def set_agent_model(agent, model_name: str) -> None:
    llm_client = getattr(agent, "llm_client", None)
    if llm_client is None:
        return
    llm_client.model = model_name


def execute_retries(
    cache: LLMCache,
    retry_items: Iterable[RetryItem],
    model: MultiAgentChoquetModel,
) -> Tuple[RetryResult, Dict[str, int]]:
    agents_by_name = {agent.name: agent for agent in model.agents}
    result = RetryResult()
    error_counter: Dict[str, int] = defaultdict(int)
    retry_items = list(retry_items)
    if not retry_items:
        return result, error_counter

    with progress_bar(len(retry_items)) as progress:
        for item in retry_items:
            result.requested += 1
            agent = agents_by_name[item.agent_name]
            set_agent_model(agent, item.model_name)
            try:
                agent._predict_one_prepared(item.text, item.text, item.task_description)
                success_entry = agent.cache._items.get(item.cache_key)
                if is_success_entry(success_entry):
                    overwrite_success_record(cache, item.cache_key, success_entry, item.retry_count)
                    result.recovered += 1
                else:
                    result.still_failed += 1
            except Exception as exc:
                error_type = classify_error(exc)
                error_counter[error_type] += 1
                if is_fallback_error(exc):
                    fallback_entry = build_fallback_success_record(item, str(exc))
                    overwrite_fallback_record(cache, item.cache_key, fallback_entry)
                    result.recovered += 1
                    error_counter["Fallback"] += 1
                    print()
                    print("[FALLBACK]")
                    print("Assigned neutral probabilities: [0.5, 0.5]")
                    print(f"key={item.cache_key}")
                    print(f"agent={item.agent_name}")
                    print(f"error={error_type}")
                    print(f"detail={str(exc)}")
                    continue

                result.still_failed += 1
                print()
                print("[FAILED]")
                print(f"key={item.cache_key}")
                print(f"agent={item.agent_name}")
                print(f"error={error_type}")
                print(f"detail={str(exc)}")
            finally:
                progress.update(1)
                progress.set_postfix(recovered=result.recovered, failed=result.still_failed)

    return result, error_counter


def print_error_report(error_counter: Dict[str, int]) -> None:
    print("===================================")
    print("ERROR REPORT")
    print("============")
    print()
    for label in ["SensitiveWords", "Fallback", "IncompleteRead", "SSLError", "HTTP429", "TimeoutError", "ConnectionError", "Other"]:
        print(f"{label:<15}: {int(error_counter.get(label, 0))}")
    print()


def print_retry_summary(scan: ScanResult, result: RetryResult) -> None:
    skipped = scan.skipped_max_retry
    recovery_rate = (result.recovered / result.requested * 100.0) if result.requested else 0.0
    print("==================================================")
    print("Retry Summary")
    print("=============")
    print()
    print(f"FAILED records  : {scan.failed_records}")
    print(f"Requested       : {result.requested}")
    print(f"Recovered       : {result.recovered}")
    print(f"Still Failed    : {result.still_failed}")
    print(f"Skipped         : {skipped}")
    print(f"Recovery Rate   : {recovery_rate:.2f}%")
    print()
    print("==================================================")


def count_failed_records(cache: LLMCache) -> int:
    cache._load()
    return sum(1 for entry in cache._items.values() if isinstance(entry, dict) and entry.get("status") == "FAILED")


def main() -> int:
    args = parse_args()
    
    # Apply command-line overrides before loading config
    if args.base_url:
        import os
        os.environ["LLM_BASE_URL"] = args.base_url
        print(f"[CONFIG] Overriding LLM_BASE_URL: {args.base_url}")
    
    if args.model:
        import os
        os.environ["LLM_MODEL"] = args.model
        print(f"[CONFIG] Overriding LLM_MODEL: {args.model}")
    
    if args.api_key:
        import os
        # Determine which env var name to use for the API key
        api_key_env = args.api_key_env or "LLM_API_KEY"
        os.environ[api_key_env] = args.api_key
        # Also set LLM_API_KEY_ENV so config.py knows which env var to read
        os.environ["LLM_API_KEY_ENV"] = api_key_env
        print(f"[CONFIG] Setting API key in env var '{api_key_env}' (value hidden)")
    elif args.api_key_env:
        # Only override the env var name, not the value
        import os
        os.environ["LLM_API_KEY_ENV"] = args.api_key_env
        print(f"[CONFIG] Using API key from env var: {args.api_key_env}")
    
    cache = LLMCache(LLM_CACHE_PATH)
    model = build_model()
    scan = scan_failed_cache(cache, model, args.max_retry)
    print_startup_report(scan)

    if scan.retry_items and LLM_PROVIDER != "local" and not has_llm_api_key():
        print("Missing API key. Set LLM_API_KEY in .env or environment.")
        return 2

    result, error_counter = execute_retries(cache, scan.retry_items, model)
    print_error_report(error_counter)
    print_retry_summary(scan, result)

    remaining_failed = count_failed_records(cache)
    print(f"Remaining FAILED records: {remaining_failed}")
    return 0 if remaining_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
