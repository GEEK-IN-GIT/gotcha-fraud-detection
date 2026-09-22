"""Generate project figures from saved metrics artifacts."""

import json

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR, METRICS_DIR, get_threshold_sweep_file, get_xgboost_feature_importance_file


VARIANT = "full"
SELECTED_THRESHOLD = 0.80


def load_json(path):
    """Load a JSON artifact from disk."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_threshold_sweep_figure() -> None:
    """Create the validation threshold sweep chart for the full variant."""
    threshold_sweep = load_json(get_threshold_sweep_file(VARIANT))
    focused_results = [
        result for result in threshold_sweep
        if 0.50 <= result["threshold"] <= 0.90
    ]

    thresholds = [result["threshold"] for result in focused_results]
    precision_values = [result["precision"] for result in focused_results]
    recall_values = [result["recall"] for result in focused_results]
    f1_values = [result["f1"] for result in focused_results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision_values, color="blue", linestyle="-", label="Precision")
    plt.plot(thresholds, recall_values, color="orange", linestyle="-", label="Recall")
    plt.plot(thresholds, f1_values, color="green", linestyle="-", label="F1")
    plt.axvline(SELECTED_THRESHOLD, color="red", linestyle="--", label="Selected threshold")
    plt.title("Validation Threshold Sweep — Full Variant")
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()

    output_file = FIGURES_DIR / "threshold_sweep_full.png"
    plt.savefig(output_file, format="png")
    plt.close()
    print(f"Saved figure: {output_file}")


def save_feature_importance_figure() -> None:
    """Create the top-feature-importance chart for the full XGBoost model."""
    feature_importance = load_json(get_xgboost_feature_importance_file(VARIANT))
    top_features = feature_importance[:15]

    feature_names = [
        record["feature"].removeprefix("numeric__").removeprefix("categorical__")
        for record in top_features
    ]
    importance_values = [record["importance"] for record in top_features]

    plt.figure(figsize=(10, 8))
    plt.barh(feature_names, importance_values, color="steelblue")
    plt.title("Top 15 Feature Importances — XGBoost Full Variant")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.gca().invert_yaxis()
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    output_file = FIGURES_DIR / "feature_importance_full.png"
    plt.savefig(output_file, format="png")
    plt.close()
    print(f"Saved figure: {output_file}")


def save_model_comparison_figure() -> None:
    """Create the grouped comparison chart for baseline and XGBoost test metrics."""
    baseline_metrics = load_json(METRICS_DIR / "baseline_logistic_regression_metrics_full.json")
    xgboost_metrics = load_json(METRICS_DIR / "xgboost_tuned_test_metrics_full.json")

    baseline_scores = baseline_metrics["test_metrics"]
    xgboost_scores = xgboost_metrics

    print(f"Loaded baseline test_metrics: {baseline_scores}")

    metric_keys = ["precision", "recall", "f1", "pr_auc"]
    metric_labels = ["Precision", "Recall", "F1", "PR-AUC"]
    baseline_values = [baseline_scores[key] for key in metric_keys]
    xgboost_values = [xgboost_scores[key] for key in metric_keys]

    positions = range(len(metric_labels))
    bar_width = 0.35
    baseline_positions = [position - (bar_width / 2) for position in positions]
    xgboost_positions = [position + (bar_width / 2) for position in positions]

    plt.figure(figsize=(10, 6))
    baseline_bars = plt.bar(
        baseline_positions,
        baseline_values,
        width=bar_width,
        color="steelblue",
        label="Logistic Regression",
    )
    xgboost_bars = plt.bar(
        xgboost_positions,
        xgboost_values,
        width=bar_width,
        color="darkorange",
        label="XGBoost",
    )

    plt.title("Logistic Regression vs XGBoost — Test Set Performance")
    plt.text(
        0.5,
        0.97,
        "Evaluated on held-out future transactions (chronological split)",
        transform=plt.gca().transAxes,
        ha="center",
        va="top",
    )
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.xticks(list(positions), metric_labels)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="upper left")

    for bars in (baseline_bars, xgboost_bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    output_file = FIGURES_DIR / "model_comparison.png"
    plt.savefig(output_file, format="png")
    plt.close()
    print(f"Saved figure: {output_file}")


def main() -> None:
    """Generate the configured project figures from saved metrics artifacts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Touch METRICS_DIR here so the script clearly depends on the saved metrics location from config.
    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Metrics directory not found: {METRICS_DIR}")

    save_threshold_sweep_figure()
    save_feature_importance_figure()
    save_model_comparison_figure()


if __name__ == "__main__":
    main()
