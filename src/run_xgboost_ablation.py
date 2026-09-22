"""Run the XGBoost ablation experiment and summarize the results."""

import json
import subprocess
import sys

from src.config import get_xgboost_metrics_file


VARIANTS = ["full", "no_synthetic", "no_ip_risk"]


def run_command(module_name: str, variant: str) -> None:
    """Run one project module for a specific ablation variant.

    This method executes the requested module with the given feature variant
    and raises an error immediately if the command fails.

    Args:
        module_name (str): Module to run with `python -m`.
        variant (str): Ablation variant to pass to the module.

    Returns:
        None
    """
    # Use the normal module entrypoints so ablation runs exercise the same code paths as manual runs.
    command = [sys.executable, "-m", module_name, "--variant", variant]
    subprocess.run(command, check=True)


def load_metrics(variant: str) -> dict[str, object]:
    """Load the saved XGBoost metrics for one ablation variant.

    Args:
        variant (str): Ablation variant name.

    Returns:
        dict[str, object]: Parsed metrics payload for the variant.
    """
    metrics_file = get_xgboost_metrics_file(variant)
    with metrics_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_confusion_values(metrics_payload: dict[str, object], split_name: str) -> tuple[int, int]:
    """Extract false positives and false negatives from a confusion matrix.

    Args:
        metrics_payload (dict[str, object]): Metrics payload for one variant.
        split_name (str): Metrics split name, such as `validation_metrics`.

    Returns:
        tuple[int, int]: False positives and false negatives.
    """
    confusion = metrics_payload[split_name]["confusion_matrix"]
    false_positives = int(confusion[0][1])
    false_negatives = int(confusion[1][0])
    return false_positives, false_negatives


def print_split_table(metrics_by_variant: dict[str, dict[str, object]], split_name: str, label: str) -> None:
    """Print a compact comparison table for one evaluation split.

    Args:
        metrics_by_variant (dict[str, dict[str, object]]): Metrics keyed by variant.
        split_name (str): Metrics split name, such as `validation_metrics`.
        label (str): Human-readable split label.

    Returns:
        None
    """
    print(f"\n{label}")
    header = (
        f"{'variant':<15}"
        f"{'precision':>11}"
        f"{'recall':>11}"
        f"{'f1':>11}"
        f"{'pr_auc':>11}"
        f"{'roc_auc':>11}"
        f"{'fp':>8}"
        f"{'fn':>8}"
    )
    print(header)
    print("-" * len(header))

    for variant in VARIANTS:
        split_metrics = metrics_by_variant[variant][split_name]
        false_positives, false_negatives = get_confusion_values(metrics_by_variant[variant], split_name)
        print(
            f"{variant:<15}"
            f"{split_metrics['precision']:>11.4f}"
            f"{split_metrics['recall']:>11.4f}"
            f"{split_metrics['f1']:>11.4f}"
            f"{split_metrics['pr_auc']:>11.4f}"
            f"{split_metrics['roc_auc']:>11.4f}"
            f"{false_positives:>8}"
            f"{false_negatives:>8}"
        )


def interpret_results(metrics_by_variant: dict[str, dict[str, object]]) -> None:
    """Print a short interpretation of the ablation results.

    Args:
        metrics_by_variant (dict[str, dict[str, object]]): Metrics keyed by variant.

    Returns:
        None
    """
    FRAUD_COST = 280.0
    FALSE_BLOCK_COST = 12.50

    full_pr_auc = metrics_by_variant["full"]["test_metrics"]["pr_auc"]
    no_synthetic_pr_auc = metrics_by_variant["no_synthetic"]["test_metrics"]["pr_auc"]
    no_ip_risk_pr_auc = metrics_by_variant["no_ip_risk"]["test_metrics"]["pr_auc"]

    full_fp, _ = get_confusion_values(metrics_by_variant["full"], "test_metrics")
    no_synthetic_fp, _ = get_confusion_values(metrics_by_variant["no_synthetic"], "test_metrics")
    no_ip_risk_fp, _ = get_confusion_values(metrics_by_variant["no_ip_risk"], "test_metrics")

    no_synthetic_drop = full_pr_auc - no_synthetic_pr_auc
    no_ip_risk_drop = full_pr_auc - no_ip_risk_pr_auc

    if no_synthetic_drop >= 0.05:
        synthetic_message = "Removing all synthetic features caused a major drop in test PR-AUC."
    else:
        synthetic_message = "Removing all synthetic features did not cause a major test PR-AUC drop."

    if no_ip_risk_drop >= 0.03:
        ip_risk_message = "Removing only synthetic_ip_risk_score caused a noticeable drop in test PR-AUC."
    else:
        ip_risk_message = "Removing only synthetic_ip_risk_score caused little test PR-AUC change."

    balance_scores = {}
    for variant in VARIANTS:
        false_positives, false_negatives = get_confusion_values(metrics_by_variant[variant], "test_metrics")
        balance_scores[variant] = -(false_positives * FALSE_BLOCK_COST + false_negatives * FRAUD_COST)

    best_balance_variant = max(balance_scores, key=balance_scores.get)

    print("\nInterpretation")
    print(synthetic_message)
    print(ip_risk_message)
    print(
        "Lowest estimated test-time cost based on false positives and false negatives: "
        f"{best_balance_variant}."
    )
    print(
        f"Test PR-AUC comparison: full={full_pr_auc:.4f}, "
        f"no_synthetic={no_synthetic_pr_auc:.4f}, "
        f"no_ip_risk={no_ip_risk_pr_auc:.4f}."
    )
    print(
        f"Test false positives: full={full_fp}, "
        f"no_synthetic={no_synthetic_fp}, "
        f"no_ip_risk={no_ip_risk_fp}."
    )


def main() -> None:
    """Run the full XGBoost ablation workflow and compare results.

    This method builds features and trains XGBoost for each ablation variant,
    loads the saved metrics, and prints a compact comparison summary.

    Args:
        None

    Returns:
        None
    """
    for variant in VARIANTS:
        print(f"\nBuilding features for variant: {variant}")
        run_command("src.build_features", variant)

    for variant in VARIANTS:
        print(f"\nTraining XGBoost for variant: {variant}")
        run_command("src.train_xgboost", variant)

    metrics_by_variant = {variant: load_metrics(variant) for variant in VARIANTS}

    print("\nXGBoost Ablation Summary")
    print_split_table(metrics_by_variant, "validation_metrics", "Validation Metrics")
    print_split_table(metrics_by_variant, "test_metrics", "Test Metrics")
    interpret_results(metrics_by_variant)


if __name__ == "__main__":
    main()
