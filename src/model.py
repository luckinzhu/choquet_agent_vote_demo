from typing import Dict

import numpy as np
import torch

from .agents import build_agents, run_agents
from .choquet_layer import ChoquetInspiredVotingLayer
from .embeddings import AGENT_DESCRIPTIONS, TfidfRelevanceEstimator
from .utils import to_tensor


class MultiAgentChoquetModel:
    """Thin orchestration wrapper around fixed agents + trainable aggregator.

    Agents can be rule-based or LLM-backed. They are inference-only; the only
    trainable module remains the selected Choquet aggregation layer mode.
    """

    def __init__(
        self,
        num_classes: int = 2,
        device: str = "cpu",
        agent_backend: str = None,
        choquet_mode: str = "inspired",
    ):
        self.device = device
        self.choquet_mode = choquet_mode
        self.agents = build_agents(agent_backend)
        self.agent_names = [a.name for a in self.agents]
        self.agent_descriptions = AGENT_DESCRIPTIONS
        self.relevance = TfidfRelevanceEstimator(self.agent_descriptions)
        self.layer = ChoquetInspiredVotingLayer(
            len(self.agents),
            num_classes,
            mode=choquet_mode,
        ).to(device)

    def fit_relevance(self, df):
        self.relevance.fit(df["task_description"].tolist(), df["text"].tolist())
        return self

    def set_llm_cache_only(self, cache_only: bool = True) -> None:
        for agent in self.agents:
            setter = getattr(agent, "set_cache_only", None)
            if callable(setter):
                setter(cache_only)

    def warm_agent_cache(self, df) -> None:
        texts = df["text"].tolist()
        task_descriptions = df["task_description"].tolist()
        run_agents(self.agents, texts, task_descriptions)

    def make_inputs(self, df_or_texts, task_descriptions=None) -> Dict[str, object]:
        if hasattr(df_or_texts, "__getitem__") and task_descriptions is None:
            texts = df_or_texts["text"].tolist()
            task_descriptions = df_or_texts["task_description"].tolist()
        else:
            texts = list(df_or_texts)
            task_descriptions = list(task_descriptions)

        agent_probs, agent_conf, explanations = run_agents(self.agents, texts, task_descriptions)
        task_rel = self.relevance.task_relevance(task_descriptions)
        sample_rel = self.relevance.sample_relevance(texts)
        return {
            "texts": texts,
            "task_descriptions": task_descriptions,
            "agent_probs_np": agent_probs,
            "agent_conf_np": agent_conf,
            "task_rel_np": task_rel,
            "sample_rel_np": sample_rel,
            "explanations": explanations,
            "agent_probs": to_tensor(agent_probs, self.device),
            "agent_confidences": to_tensor(agent_conf, self.device),
            "task_relevance": to_tensor(task_rel, self.device),
            "sample_relevance": to_tensor(sample_rel, self.device),
        }

    def logits_from_inputs(self, inputs: Dict[str, object], use_pairwise: bool = True, details: bool = False):
        return self.layer(
            inputs["agent_probs"],
            inputs["agent_confidences"],
            inputs["task_relevance"],
            inputs["sample_relevance"],
            use_pairwise=use_pairwise,
            return_details=details,
        )

    @torch.no_grad()
    def predict(self, df, use_pairwise: bool = True, return_details: bool = False):
        self.layer.eval()
        inputs = self.make_inputs(df)
        output = self.logits_from_inputs(inputs, use_pairwise=use_pairwise, details=return_details)
        if return_details:
            logits, details = output
        else:
            logits, details = output, None
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        if return_details:
            return preds, probs.cpu().numpy(), details, inputs
        return preds, probs.cpu().numpy()

    def state_dict(self):
        return self.layer.state_dict()

    def load_state_dict(self, state_dict):
        self.layer.load_state_dict(state_dict)
