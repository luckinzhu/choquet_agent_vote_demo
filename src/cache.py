import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_error(error_message: object) -> str:
    text = str(error_message or "").strip()
    if not text:
        return "Unknown"
    if "IncompleteRead" in text:
        return "IncompleteRead"
    if "SSL" in text:
        return "SSL_ERROR"
    if "HTTP 429" in text or "HTTP429" in text:
        return "HTTP429"
    if "ConnectionError" in text or "Network error" in text:
        return "ConnectionError"
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)(?:\(|:|$)", text)
    return match.group(1) if match else "LLMError"


class LLMCache:
    """Persistent JSON cache for LLM agent outputs.

    Key fields intentionally include the model name so changing providers/models
    does not silently reuse old judgments.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: Dict[str, Dict[str, object]] = {}
        if self.path.exists():
            self._load()

    @staticmethod
    def make_key(text: str, task_description: str, agent_name: str, model_name: str) -> str:
        raw = json.dumps(
            {
                "text": text,
                "task_description": task_description,
                "agent_name": agent_name,
                "model_name": model_name,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        try:
            self._items = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._items = {}
        changed = self.migrate_legacy_items(flush=False)
        if changed:
            self._flush()

    def _flush(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(self.path)

    def get(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
    ) -> Optional[Dict[str, object]]:
        key = self.make_key(text, task_description, agent_name, model_name)
        item = self._items.get(key)
        if not item:
            return None
        if item.get("status") != "SUCCESS":
            return None
        output = item.get("output", {})
        try:
            return {
                "probs": np.array(output["probs"], dtype=np.float32),
                "confidence": float(output["confidence"]),
                "explanation": str(output["explanation"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def set(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        output: Dict[str, object],
        raw_text: str,
    ) -> None:
        key = self.make_key(text, task_description, agent_name, model_name)
        probs = output["probs"]
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        self._items[key] = {
            "status": "SUCCESS",
            "text": text,
            "task_description": task_description,
            "agent_name": agent_name,
            "model_name": model_name,
            "raw_text": raw_text,
            "output": {
                "probs": probs,
                "confidence": float(output["confidence"]),
                "explanation": str(output["explanation"]),
            },
        }
        self._flush()

    def set_failed(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
        error_message: object,
    ) -> None:
        key = self.make_key(text, task_description, agent_name, model_name)
        existing = self._items.get(key, {})
        retry_count = int(existing.get("retry_count", 0) or 0) + 1
        self._items[key] = {
            "status": "FAILED",
            "text": text,
            "task_description": task_description,
            "agent_name": agent_name,
            "model_name": model_name,
            "error_type": classify_error(error_message),
            "error_message": str(error_message),
            "retry_count": retry_count,
            "last_retry": utc_now_iso(),
        }
        self._flush()

    def get_entry(
        self,
        text: str,
        task_description: str,
        agent_name: str,
        model_name: str,
    ) -> Optional[Dict[str, object]]:
        key = self.make_key(text, task_description, agent_name, model_name)
        return self._items.get(key)

    def migrate_legacy_items(self, flush: bool = True) -> bool:
        changed = False
        for item in self._items.values():
            if not isinstance(item, dict) or item.get("status"):
                continue
            raw_text = str(item.get("raw_text", ""))
            output = item.get("output", {})
            probs = output.get("probs") if isinstance(output, dict) else None
            is_neutral_placeholder = (
                isinstance(probs, list)
                and len(probs) == 2
                and abs(float(probs[0]) - 0.5) < 1e-6
                and abs(float(probs[1]) - 0.5) < 1e-6
                and float(output.get("confidence", 1.0)) <= 0.31
            )
            if raw_text.startswith("FALLBACK_NEUTRAL_AFTER_LLM_ERROR:") or is_neutral_placeholder:
                error_message = raw_text.split(":", 1)[1].strip() if ":" in raw_text else raw_text
                item.pop("output", None)
                item["status"] = "FAILED"
                item["error_type"] = classify_error(error_message)
                item["error_message"] = error_message
                item["retry_count"] = int(item.get("retry_count", 0) or 0)
                item["last_retry"] = item.get("last_retry") or utc_now_iso()
            else:
                item["status"] = "SUCCESS"
            changed = True
        if changed and flush:
            self._flush()
        return changed

    def warm(self, agents, df) -> None:
        """Generate all agent outputs once before training begins."""
        texts = df["text"].tolist()
        task_descriptions = df["task_description"].tolist()
        records = df.to_dict("records")
        for agent in agents:
            agent.predict_batch(texts, task_descriptions, records=records)
