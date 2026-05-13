from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.choquet_layer import ChoquetInspiredVotingLayer  # noqa: E402


def manual_demo() -> None:
    scores = {"A1": 0.8, "A2": 0.6, "A3": 0.9, "A4": 0.4}
    order = sorted(scores, key=scores.get)
    sorted_values = [scores[name] for name in order]
    diffs = [sorted_values[0]] + [sorted_values[i] - sorted_values[i - 1] for i in range(1, len(sorted_values))]
    upper_sets = [order[i:] for i in range(len(order))]

    capacities = {
        frozenset(["A4", "A2", "A1", "A3"]): 1.00,
        frozenset(["A2", "A1", "A3"]): 0.85,
        frozenset(["A1", "A3"]): 0.70,
        frozenset(["A3"]): 0.45,
    }
    choquet = sum(diff * capacities[frozenset(upper)] for diff, upper in zip(diffs, upper_sets))

    print("Manual finite/discrete Choquet formula demo")
    print("Scores: A4=0.4, A2=0.6, A1=0.8, A3=0.9")
    print("Sorted order: " + " < ".join(order))
    print("Diffs: " + ", ".join(f"{value:.1f}" for value in diffs))
    print("Upper sets:")
    for upper in upper_sets:
        print("  {" + ",".join(upper) + "}")
    print("C(x) = 0.4 mu({A4,A2,A1,A3}) + 0.2 mu({A2,A1,A3}) + 0.2 mu({A1,A3}) + 0.1 mu({A3})")
    print(f"C(x) = {choquet:.4f}")


def layer_demo() -> None:
    # Agent index mapping: A1=0, A2=1, A3=2, A4=3.
    agent_probs = torch.tensor([[[0.8], [0.6], [0.9], [0.4]]], dtype=torch.float32)
    layer = ChoquetInspiredVotingLayer(4, 1, mode="discrete_2additive")
    with torch.no_grad():
        # Make the 2-additive approximation purely additive for a transparent check:
        # m(A1)=0.2, m(A2)=0.3, m(A3)=0.4, m(A4)=0.1, all pair terms = 0.
        layer.raw_single_m.copy_(torch.log(torch.tensor([0.2, 0.3, 0.4, 0.1])))
        layer.raw_pair_m.zero_()
        layer.logit_scale.fill_(1.0)
        layer.logit_bias.zero_()
    dummy = torch.ones(1, 4)
    logits, details = layer(agent_probs, dummy, dummy, dummy, return_details=True)

    # Expected additive capacity values for sorted upper sets:
    # {A4,A2,A1,A3}=1.0, {A2,A1,A3}=0.9, {A1,A3}=0.6, {A3}=0.4
    expected = 0.4 * 1.0 + 0.2 * 0.9 + 0.2 * 0.6 + 0.1 * 0.4
    print("\nLayer discrete_2additive demo with additive Mobius terms")
    print("Mobius single weights: A1=0.2, A2=0.3, A3=0.4, A4=0.1; pair terms=0")
    print("Expected score = 0.4*1.0 + 0.2*0.9 + 0.2*0.6 + 0.1*0.4")
    print(f"Expected score = {expected:.4f}")
    print(f"Layer score    = {float(logits[0, 0].detach()):.4f}")
    print("Sorted capacities from layer:", details["sorted_capacities"][0, :, 0].detach().numpy().round(4).tolist())
    print("Monotonicity diagnostic:", layer.monotonicity_diagnostics())


if __name__ == "__main__":
    manual_demo()
    layer_demo()

