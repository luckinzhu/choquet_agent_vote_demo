import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .utils import classification_metrics


def majority_voting(agent_probs: np.ndarray) -> np.ndarray:
    agent_preds = np.argmax(agent_probs, axis=-1)
    preds = []
    for row, probs in zip(agent_preds, agent_probs):
        counts = np.bincount(row, minlength=probs.shape[-1])
        if len(set(counts.tolist())) == 1:
            preds.append(int(np.argmax(probs.mean(axis=0))))
        else:
            preds.append(int(np.argmax(counts)))
    return np.array(preds)


def average_probability_voting(agent_probs: np.ndarray) -> np.ndarray:
    return np.argmax(agent_probs.mean(axis=1), axis=-1)


@torch.no_grad()
def learned_dynamic_voting(model, df, use_pairwise: bool) -> np.ndarray:
    preds, _ = model.predict(df, use_pairwise=use_pairwise, return_details=False)
    return preds


def compare_methods(model, test_df) -> List[Dict[str, float]]:
    inputs = model.make_inputs(test_df)
    y_true = test_df["label"].values
    methods = [
        ("Majority Voting", majority_voting(inputs["agent_probs_np"])),
        ("Average Probability Voting", average_probability_voting(inputs["agent_probs_np"])),
        ("Dynamic Single-Agent Weighting", learned_dynamic_voting(model, test_df, use_pairwise=False)),
        ("Choquet-inspired Pairwise Voting", learned_dynamic_voting(model, test_df, use_pairwise=True)),
    ]
    rows = []
    for name, preds in methods:
        metrics = classification_metrics(y_true, preds)
        rows.append({"method": name, **metrics})
    return rows


@torch.no_grad()
def aggregate_details_by_task(model, df) -> Dict[str, Dict[str, np.ndarray]]:
    result = {}
    for task_name, task_df in df.groupby("task_name"):
        preds, probs, details, _ = model.predict(task_df, use_pairwise=True, return_details=True)
        result[task_name] = {
            "single_weights": details["single_weights"].cpu().numpy().mean(axis=0),
            "pair_weights": details["pair_weights"].cpu().numpy().mean(axis=0),
            "preds": preds,
            "probs": probs,
        }
    return result


def print_interpretability_summary(model, df, top_pairs: int = 5) -> None:
    summary = aggregate_details_by_task(model, df)
    pair_names = [
        f"{model.agent_names[i]} + {model.agent_names[j]}" for i, j in model.layer.pair_indices
    ]

    print("\n=== Task-level Average Single Weights ===")
    for task_name, values in summary.items():
        print(f"\nTask: {task_name}")
        for name, weight in zip(model.agent_names, values["single_weights"]):
            print(f"  {name:12s}: {weight:.4f}")

    print("\n=== Task-level Top Pairwise Weights ===")
    for task_name, values in summary.items():
        order = np.argsort(values["pair_weights"])[::-1][:top_pairs]
        print(f"\nTask: {task_name}")
        for idx in order:
            print(f"  {pair_names[idx]:28s}: {values['pair_weights'][idx]:.4f}")


def print_sample_decisions(model, df, n_samples: int = 4, seed: int = 42) -> None:
    show_df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    preds, probs, details, inputs = model.predict(show_df, use_pairwise=True, return_details=True)
    single = details["single_weights"].cpu().numpy()
    pair = details["pair_weights"].cpu().numpy()
    pair_names = [
        f"{model.agent_names[i]} + {model.agent_names[j]}" for i, j in model.layer.pair_indices
    ]

    print("\n=== Sample-level Decision Traces ===")
    for row_idx, row in show_df.iterrows():
        print("\n--- Sample ---")
        print(f"Task: {row['task_name']}")
        print(f"Text: {row['text']}")
        print(f"Gold label: {row['label']} | Final prediction: {preds[row_idx]} | Final probs: {probs[row_idx].round(4)}")
        print("Agent predictions:")
        for agent_idx, name in enumerate(model.agent_names):
            agent_prob = inputs["agent_probs_np"][row_idx, agent_idx]
            agent_pred = int(np.argmax(agent_prob))
            conf = inputs["agent_conf_np"][row_idx, agent_idx]
            explanation = inputs["explanations"][agent_idx][row_idx]
            print(
                f"  {name:12s} pred={agent_pred} conf={conf:.3f} "
                f"prob={agent_prob.round(3)} | {explanation}"
            )
        print("Single weights:")
        for name, weight in zip(model.agent_names, single[row_idx]):
            print(f"  {name:12s}: {weight:.4f}")
        top = np.argsort(pair[row_idx])[::-1][:3]
        print("Top pairwise weights:")
        for idx in top:
            print(f"  {pair_names[idx]:28s}: {pair[row_idx, idx]:.4f}")


def export_readable_model_summary(model, df, output_path: Path) -> None:
    """Export a UTF-8 JSON companion file for the binary PyTorch checkpoint.

    The .pt file is intentionally binary and will look garbled in text editors.
    This JSON is the human-readable artifact to inspect in PyCharm.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_summary = aggregate_details_by_task(model, df)
    pair_names = [
        f"{model.agent_names[i]} + {model.agent_names[j]}" for i, j in model.layer.pair_indices
    ]

    state = {
        key: value.detach().cpu().numpy().round(6).tolist()
        for key, value in model.layer.state_dict().items()
    }
    readable = {
        "note": (
            "best_choquet_model.pt is a binary PyTorch checkpoint. "
            "Open this JSON file for a readable UTF-8 summary."
        ),
        "agent_names": model.agent_names,
        "pair_names": pair_names,
        "trainable_parameters": state,
        "task_level_average_weights": {},
    }

    for task_name, values in task_summary.items():
        single_map = {
            agent: round(float(weight), 6)
            for agent, weight in zip(model.agent_names, values["single_weights"])
        }
        pair_map = {
            pair: round(float(weight), 6)
            for pair, weight in zip(pair_names, values["pair_weights"])
        }
        top_pairs = dict(
            sorted(pair_map.items(), key=lambda item: item[1], reverse=True)[:5]
        )
        readable["task_level_average_weights"][task_name] = {
            "single_weights": single_map,
            "pair_weights": pair_map,
            "top_pair_weights": top_pairs,
        }

    output_path.write_text(
        json.dumps(readable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
