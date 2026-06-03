import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from config import (  # noqa: E402
    AGENT_NAMES,
    DATA_AUTOGENERATE_DEMO,
    DATA_PATH,
    LLM_CACHE_PATH,
    RANDOM_SEED,
    RUN_SAMPLE_LIMIT,
)
from integrity_checker import (  # noqa: E402
    check_cache_integrity,
    check_dataset_integrity,
    failed_error_distribution,
    missing_agent_distribution,
)
from src.cache import LLMCache  # noqa: E402
from src.dataset import load_dataset  # noqa: E402
from src.model import MultiAgentChoquetModel  # noqa: E402


REQUIRED_AGENTS = tuple(AGENT_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report cache or dataset LLM-output integrity.")
    parser.add_argument(
        "--dataset",
        action="store_true",
        help="Check the full configured dataset. Default checks only samples already present in cache.",
    )
    return parser.parse_args()


def print_counter(counter: Counter, empty_label: str = "None") -> None:
    if not counter:
        print(empty_label)
        return
    width = max(len(str(key)) for key in counter)
    for key, value in counter.most_common():
        print(f"{key:<{width}} : {value}")


def retry_count_distribution() -> Counter:
    counter = Counter()
    for entry in LLMCache(LLM_CACHE_PATH)._items.values():
        if isinstance(entry, dict) and entry.get("status") == "SUCCESS":
            counter[int(entry.get("retry_count", 0) or 0)] += 1
    return counter


def print_success_retry_distribution() -> None:
    print("------------------------------------------------")
    print("SUCCESS Retry Distribution")
    print("------------------------------------------------")
    print()
    counter = retry_count_distribution()
    if not counter:
        print("None")
        return
    for retry_count, count in sorted(counter.items()):
        if retry_count == 0:
            print(f"Recovered without retry : {count}")
        elif retry_count == 1:
            print(f"Recovered after 1 retry : {count}")
        else:
            print(f"Recovered after {retry_count} retries : {count}")


def agent_coverage(details: dict) -> Counter:
    counter = Counter()
    for item in details.values():
        present = set(REQUIRED_AGENTS) - set(item.get("missing_agents", []))
        for agent in present:
            counter[agent] += 1
    return counter


def agent_count_distribution(details: dict) -> Counter:
    counter = Counter()
    for item in details.values():
        counter[int(item.get("success_agents", 0) or 0)] += 1
    return counter


def print_agent_coverage(details: dict, total_samples: int) -> None:
    print("------------------------------------------------")
    print("Agent Coverage")
    print("------------------------------------------------")
    print()
    coverage = agent_coverage(details)
    width = max(len(agent) for agent in REQUIRED_AGENTS)
    for agent in REQUIRED_AGENTS:
        print(f"{agent:<{width}} {coverage.get(agent, 0)}/{total_samples}")


def print_agent_count_distribution(details: dict) -> None:
    print("------------------------------------------------")
    print("Incomplete Sample Distribution")
    print("------------------------------------------------")
    print()
    distribution = agent_count_distribution(details)
    for count in range(len(REQUIRED_AGENTS), 0, -1):
        print(f"拥有{count}个Agent：{distribution.get(count, 0)}")


def print_missing_examples(details: dict, limit: int = 10) -> None:
    print("------------------------------------------------")
    print("First Missing Agent Samples")
    print("------------------------------------------------")
    print()
    examples = [item for item in details.values() if item.get("missing_agents")]
    if not examples:
        print("None")
        return
    for idx, item in enumerate(examples[:limit], 1):
        missing = ", ".join(item.get("missing_agents", []))
        print(f"Sample #{idx}")
        print(f"Missing: {missing}")
        print(f"Model: {item.get('model_name', '')}")
        print()


def report_cache_mode() -> int:
    result = check_cache_integrity(LLM_CACHE_PATH)
    total = int(result["cache_samples"])
    complete = int(result["complete_samples"])
    incomplete = int(result["incomplete_samples"])
    completeness = (complete / total * 100.0) if total else 0.0

    print("================================================")
    print("CACHE REPORT")
    print("================================================")
    print()
    print(f"Total Samples: {total}")
    print()
    print(f"Complete Samples: {complete}")
    print(f"Incomplete Samples: {incomplete}")
    print()
    print("Completeness:")
    print(f"{completeness:.1f}%")
    print()
    print("SUCCESS records:", result["success_records"])
    print("FAILED records:", result["failed_records"])
    print()
    print("------------------------------------------------")
    print("Missing Agent Distribution")
    print("------------------------------------------------")
    print()
    print_counter(missing_agent_distribution(result["details"]))
    print()
    print_agent_coverage(result["details"], total)
    print()
    print_agent_count_distribution(result["details"])
    print()
    print_missing_examples(result["details"], limit=10)
    print()
    print("------------------------------------------------")
    print("FAILED Error Distribution")
    print("------------------------------------------------")
    print()
    print_counter(failed_error_distribution(LLM_CACHE_PATH))
    print()
    print_success_retry_distribution()
    print()
    print("CACHE PATH:", LLM_CACHE_PATH)
    print()
    return 0


def report_dataset_mode() -> int:
    df = load_dataset(DATA_PATH, allow_generate_demo=DATA_AUTOGENERATE_DEMO)
    if RUN_SAMPLE_LIMIT > 0 and RUN_SAMPLE_LIMIT < len(df):
        df = df.sample(n=RUN_SAMPLE_LIMIT, random_state=RANDOM_SEED).reset_index(drop=True)

    model = MultiAgentChoquetModel(num_classes=2, device="cpu", agent_backend="llm")
    result = check_dataset_integrity(df, model.agents, LLM_CACHE_PATH)
    expected = int(result["expected_records"])
    success = int(result["success_records"])
    missing = int(result["missing_records"])
    completeness = (success / expected * 100.0) if expected else 0.0

    print("================================================")
    print("DATASET INTEGRITY REPORT")
    print("================================================")
    print()
    print(f"Dataset Samples: {result['dataset_samples']}")
    print(f"Expected Records: {expected}")
    print(f"SUCCESS Records: {success}")
    print(f"Missing Records: {missing}")
    print()
    print("Completeness:")
    print(f"{completeness:.2f}%")
    print()
    print(f"Complete Samples: {result['complete_samples']}")
    print(f"Incomplete Samples: {result['incomplete_samples']}")
    print()
    print("------------------------------------------------")
    print("Missing Agent Distribution")
    print("------------------------------------------------")
    print()
    print_counter(missing_agent_distribution(result["details"]))
    print()
    print("CACHE PATH:", LLM_CACHE_PATH)
    print()
    return 0 if missing == 0 else 1


def main() -> int:
    args = parse_args()
    if args.dataset:
        return report_dataset_mode()
    return report_cache_mode()


if __name__ == "__main__":
    raise SystemExit(main())
