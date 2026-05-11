import torch

from config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    DATA_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    MODEL_SUMMARY_PATH,
    NUM_CLASSES,
    RANDOM_SEED,
    TRAIN_RATIO,
    VALID_RATIO,
    WEIGHT_DECAY,
)
from src.dataset import ensure_toy_data, load_and_split
from src.evaluate import (
    compare_methods,
    export_readable_model_summary,
    print_interpretability_summary,
    print_sample_decisions,
)
from src.model import MultiAgentChoquetModel
from src.train import evaluate_model, train_choquet_model
from src.utils import format_metrics_table, set_seed


def main():
    set_seed(RANDOM_SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = ensure_toy_data(DATA_PATH)
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: {len(df)}")
    print(df.groupby(["task_name", "label"]).size().to_string())

    train_df, valid_df, test_df = load_and_split(
        DATA_PATH,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        seed=RANDOM_SEED,
    )
    print(f"\nSplit: train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    model = MultiAgentChoquetModel(num_classes=NUM_CLASSES, device=DEVICE)
    # Fit TF-IDF relevance on all available text/task descriptions. This is not
    # label leakage; it only builds a vocabulary for relevance estimation.
    model.fit_relevance(df)

    print("\n=== Training Choquet-inspired Pairwise Voting Layer ===")
    model, best_valid = train_choquet_model(
        model=model,
        train_df=train_df,
        valid_df=valid_df,
        model_path=BEST_MODEL_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    )
    print(
        "\nBest validation: "
        f"loss={best_valid['loss']:.4f}, acc={best_valid['accuracy']:.4f}, "
        f"macro_f1={best_valid['macro_f1']:.4f}"
    )
    print(f"Saved best model to: {BEST_MODEL_PATH}")

    print("\n=== Test Metrics for Trained Model ===")
    test_metrics = evaluate_model(model, test_df, BATCH_SIZE, DEVICE, use_pairwise=True)
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
    export_readable_model_summary(model, test_df, MODEL_SUMMARY_PATH)
    print(f"\nReadable model summary saved to: {MODEL_SUMMARY_PATH}")

    print("\nDone. The demo validated: fixed agents + task relevance + sample relevance + trainable pairwise aggregation.")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
