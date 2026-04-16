"""CLI for a simple bag-of-words baseline over JSON datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a TF-IDF + PCA + logistic-regression baseline on a JSON dataset."
    )
    parser.add_argument("data_path", help="Path to the JSON file to analyze.")
    parser.add_argument(
        "--records-key",
        default="",
        help=(
            "Optional dotted path to the list of records inside the JSON file. "
            "Examples: 'records', 'data.items'. Leave empty when the file is already a list."
        ),
    )
    parser.add_argument(
        "--text-key",
        default="lyrics",
        help="Dotted path to the text field in each record.",
    )
    parser.add_argument(
        "--label-key",
        default="metadata.artist",
        help="Dotted path to the class label field in each record.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=64,
        help="Requested PCA dimensionality; clipped automatically when needed.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of stratified CV folds; clipped automatically when needed.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum iterations for logistic regression.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of permutations for the null distribution.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for scikit-learn cross-validation. Use -1 for all cores.",
    )
    parser.add_argument(
        "--plot-out",
        default="bow_baseline_plot.png",
        help="Where to save the permutation histogram.",
    )
    parser.add_argument(
        "--confusion-matrix-out",
        default="bow_baseline_confusion_matrix.png",
        help="Where to save the artist confusion matrix heatmap.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving both the permutation histogram and confusion matrix.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Base random seed for PCA, CV, and permutations.",
    )
    return parser


def _get_nested_value(item: dict[str, Any], dotted_key: str) -> Any:
    current: Any = item
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def _load_records(data_path: str, records_key: str) -> list[dict[str, Any]]:
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if records_key:
        raw = _get_nested_value(raw, records_key)

    if not isinstance(raw, list):
        raise ValueError(
            "Expected a JSON list of records. Use --records-key if the records "
            "live under a nested object."
        )

    records = [record for record in raw if isinstance(record, dict)]
    if len(records) != len(raw):
        raise ValueError("Every record in the dataset must be a JSON object.")
    return records


def _extract_dataset(
    records: list[dict[str, Any]], text_key: str, label_key: str
) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    for index, record in enumerate(records):
        try:
            text_value = _get_nested_value(record, text_key)
            label_value = _get_nested_value(record, label_key)
        except KeyError as exc:
            raise KeyError(f"Record {index} is missing required key '{exc.args[0]}'.") from exc

        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError(f"Record {index} has an empty or non-string text field at '{text_key}'.")
        if not isinstance(label_value, str) or not label_value.strip():
            raise ValueError(
                f"Record {index} has an empty or non-string label field at '{label_key}'."
            )

        texts.append(text_value)
        labels.append(label_value)

    return texts, labels


def _resolve_n_splits(labels: list[str], requested_folds: int) -> int:
    label_counts = Counter(labels)
    min_class_count = min(label_counts.values())
    n_splits = min(requested_folds, min_class_count)
    if n_splits < 2:
        raise ValueError(
            "Need at least 2 examples in every class for stratified cross-validation."
        )
    return n_splits


def main() -> None:
    args = _build_parser().parse_args()

    records = _load_records(args.data_path, args.records_key)
    texts, labels = _extract_dataset(records, args.text_key, args.label_key)

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    if n_classes < 2:
        raise ValueError("Need at least 2 distinct labels to run the baseline.")

    chance_level = 1.0 / n_classes
    n_splits = _resolve_n_splits(labels, args.folds)
    print(f"Loaded {len(texts)} records from {n_classes} classes.")
    print(f"Using text key '{args.text_key}' and label key '{args.label_key}'.")

    print("Extracting TF-IDF BoW features...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=args.max_features)
    x_sparse = vectorizer.fit_transform(texts)
    n_samples, n_features = x_sparse.shape
    if n_features == 0:
        raise ValueError("TF-IDF produced zero features. Check your text field contents.")

    max_pca_components = min(args.pca_components, n_samples, n_features)
    print(f"Running PCA down to {max_pca_components} components...")
    pca = PCA(n_components=max_pca_components, random_state=args.random_seed)
    x_pca = pca.fit_transform(x_sparse.toarray())

    classifier = LogisticRegression(
        solver="lbfgs",
        max_iter=args.max_iter,
        random_state=args.random_seed,
    )
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.random_seed,
    )

    print(f"Training linear probe (stratified {n_splits}-fold CV)...")
    scores = cross_val_score(
        classifier,
        x_pca,
        labels,
        cv=cv,
        scoring="accuracy",
        n_jobs=args.jobs,
    )
    mean_acc = scores.mean()
    std_acc = scores.std()

    print("Collecting out-of-fold predictions for confusion matrix...")
    predicted_labels = cross_val_predict(
        classifier,
        x_pca,
        labels,
        cv=cv,
        n_jobs=args.jobs,
    )
    confusion = confusion_matrix(
        labels,
        predicted_labels,
        labels=unique_labels,
        normalize="true",
    )

    print(f"BoW Probe Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
    print(f"Chance Level: {chance_level:.3f}")

    print(f"Running permutation test with {args.permutations} permutations...")
    null_scores = []
    y_arr = np.array(labels)
    for i in range(args.permutations):
        rng = np.random.RandomState(args.random_seed + i)
        y_shuffled = rng.permutation(y_arr)
        perm_scores = cross_val_score(
            classifier,
            x_pca,
            y_shuffled,
            cv=cv,
            scoring="accuracy",
            n_jobs=args.jobs,
        )
        null_scores.append(perm_scores.mean())
        if (i + 1) % 100 == 0 or i + 1 == args.permutations:
            print(f"  Completed {i + 1}/{args.permutations} permutations...")

    null_scores_arr = np.array(null_scores)
    p_value = (np.sum(null_scores_arr >= mean_acc) + 1.0) / (args.permutations + 1)

    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Null mean: {null_scores_arr.mean():.3f} +/- {null_scores_arr.std():.3f}")

    if args.no_plot:
        print("Skipping plot generation (--no-plot).")
        return

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 5))
    sns.histplot(
        null_scores_arr,
        bins=30,
        color="silver",
        stat="density",
        label=f"Null Distribution (n={args.permutations})",
        alpha=0.8,
    )
    plt.axvline(
        mean_acc,
        color=sns.color_palette("muted")[0],
        linewidth=2.5,
        linestyle="-",
        label=f"Observed Accuracy ({mean_acc:.3f})",
    )
    plt.axvline(
        chance_level,
        color="lightcoral",
        linestyle="--",
        linewidth=2,
        label=f"Chance ({chance_level:.3f})",
    )
    plt.title(
        "Permutation Null Distribution - BoW Baseline\n"
        f"p-value: {p_value:.4f} | PCA: {max_pca_components} | Folds: {n_splits}"
    )
    plt.xlabel("Probe Accuracy")
    plt.ylabel("Density")
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()

    plot_path = Path(args.plot_out)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")

    matrix_path = Path(args.confusion_matrix_out)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    side = max(8, min(22, 0.7 * len(unique_labels) + 4))
    plt.figure(figsize=(side, side))
    sns.heatmap(
        confusion,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar_kws={"label": "Row-normalized accuracy"},
        square=True,
    )
    plt.title("Confusion Matrix (bow_baseline)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {matrix_path}")


if __name__ == "__main__":
    main()
