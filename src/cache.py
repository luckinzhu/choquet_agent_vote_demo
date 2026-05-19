import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np


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

    def warm(self, agents, df) -> None:
        """Generate all agent outputs once before training begins."""
        texts = df["text"].tolist()
        task_descriptions = df["task_description"].tolist()
        records = df.to_dict("records")
        for agent in agents:
            agent.predict_batch(texts, task_descriptions, records=records)
