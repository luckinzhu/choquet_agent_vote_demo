from itertools import combinations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoquetInspiredVotingLayer(nn.Module):
    """Trainable 2-additive Choquet-inspired voting layer.

    A full Choquet capacity over K agents needs 2^K set capacities. This demo
    uses a 2-additive approximation: singleton capacities plus pairwise
    capacity-like interactions. It keeps the non-additive "agents can reinforce
    or overlap with each other" idea while reducing complexity to O(K^2).
    """

    def __init__(self, num_agents: int, num_classes: int, pair_scale: float = 0.45):
        super().__init__()
        self.num_agents = num_agents
        self.num_classes = num_classes
        self.pair_indices: List[Tuple[int, int]] = list(combinations(range(num_agents), 2))
        self.num_pairs = len(self.pair_indices)
        self.pair_scale = pair_scale

        # single_weight_i = softmax(a_i task_i + b_i sample_i + c_i conf_i + bias_i)
        self.a = nn.Parameter(torch.ones(num_agents) * 0.7)
        self.b = nn.Parameter(torch.ones(num_agents) * 0.7)
        self.c = nn.Parameter(torch.ones(num_agents) * 0.4)
        self.single_bias = nn.Parameter(torch.zeros(num_agents))

        # pair_weight_ij = sigmoid(u_ij task_i task_j + v_ij sample_i sample_j
        #                          + r_ij agreement_ij + m_ij)
        self.u = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
        self.v = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
        self.v = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
        self.r = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
        self.pair_bias = nn.Parameter(torch.zeros(self.num_pairs))

        # A small global affine calibration helps CrossEntropyLoss optimize the
        # positive score vector without changing the aggregation mechanism.
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))

    def _pair_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        left = torch.stack([x[:, i] for i, _ in self.pair_indices], dim=1)
        right = torch.stack([x[:, j] for _, j in self.pair_indices], dim=1)
        return left, right

    def forward(
        self,
        agent_probs: torch.Tensor,
        agent_confidences: torch.Tensor,
        task_relevance: torch.Tensor,
        sample_relevance: torch.Tensor,
        use_pairwise: bool = True,
        return_details: bool = False,
    ):
        single_logits = (
            self.a * task_relevance
            + self.b * sample_relevance
            + self.c * agent_confidences
            + self.single_bias
        )
        single_weights = F.softmax(single_logits, dim=1)
        single_contribution = torch.sum(single_weights.unsqueeze(-1) * agent_probs, dim=1)

        pi, pj = self._pair_tensors(agent_probs)
        task_i, task_j = self._pair_tensors(task_relevance)
        sample_i, sample_j = self._pair_tensors(sample_relevance)
        pair_task = task_i * task_j
        pair_sample = sample_i * sample_j

        # Agreement can also be implemented with cosine similarity. With
        # probability vectors, 1 - mean L1 distance is stable and easy to read.
        agreement = 1.0 - torch.mean(torch.abs(pi - pj), dim=-1)
        pair_logits = self.u * pair_task + self.v * pair_sample + self.r * agreement + self.pair_bias
        pair_weights = torch.sigmoid(pair_logits)

        # Choquet-inspired interaction. Replace pi * pj with torch.minimum(pi, pj)
        # to try a min-style fuzzy interaction.
        interaction = pi * pj
        if use_pairwise:
            pair_contribution = self.pair_scale * torch.sum(
                pair_weights.unsqueeze(-1) * interaction, dim=1
            ) / max(1, self.num_pairs)
        else:
            pair_contribution = torch.zeros_like(single_contribution)

        scores = single_contribution + pair_contribution
        logits = scores * torch.clamp(self.logit_scale, min=0.1, max=20.0) + self.logit_bias

        if not return_details:
            return logits
        details: Dict[str, torch.Tensor] = {
            "single_weights": single_weights,
            "pair_weights": pair_weights,
            "single_contribution": single_contribution,
            "pair_contribution": pair_contribution,
            "agreement": agreement,
        }
        return logits, details
