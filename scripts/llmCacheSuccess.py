import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LLM_CACHE_PATH  # noqa: E402
from integrity_checker import failed_error_distribution  # noqa: E402
from src.cache import LLMCache  # noqa: E402


def main() -> int:
    cache = LLMCache(LLM_CACHE_PATH)
    stats = defaultdict(lambda: {"total": 0, "success": 0, "failed": 0, "other": 0})

    for item in cache._items.values():
        agent = str(item.get("agent_name", "UNKNOWN"))
        status = str(item.get("status", "UNKNOWN"))
        stats[agent]["total"] += 1
        if status == "SUCCESS":
            stats[agent]["success"] += 1
        elif status == "FAILED":
            stats[agent]["failed"] += 1
        else:
            stats[agent]["other"] += 1

    print("\n" + "=" * 80)
    print("LLM CACHE STATUS ANALYSIS")
    print("=" * 80)

    grand_total = 0
    grand_success = 0
    grand_failed = 0
    grand_other = 0

    for agent, item in sorted(stats.items()):
        total = item["total"]
        success = item["success"]
        failed = item["failed"]
        other = item["other"]
        success_rate = success / total * 100 if total else 0.0
        failed_rate = failed / total * 100 if total else 0.0

        grand_total += total
        grand_success += success
        grand_failed += failed
        grand_other += other

        print(
            f"{agent:12s} | total={total:5d} | "
            f"success={success:5d} ({success_rate:6.2f}%) | "
            f"failed={failed:5d} ({failed_rate:6.2f}%) | "
            f"other={other:5d}"
        )

    print("\n" + "-" * 80)
    overall_success = grand_success / grand_total * 100 if grand_total else 0.0
    overall_failed = grand_failed / grand_total * 100 if grand_total else 0.0
    print(
        f"OVERALL      | total={grand_total} | "
        f"success={grand_success} ({overall_success:.2f}%) | "
        f"failed={grand_failed} ({overall_failed:.2f}%) | "
        f"other={grand_other}"
    )

    print("\n" + "=" * 80)
    print("ERROR TYPE DISTRIBUTION")
    print("=" * 80)
    for error_type, count in failed_error_distribution(LLM_CACHE_PATH).most_common():
        print(f"{error_type:20s}: {count}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
