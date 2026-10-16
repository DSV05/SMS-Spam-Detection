"""Train and evaluate the SMS spam classification pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "raw" / "SMSSpamCollection"
DEFAULT_MODEL_PATH = ROOT_DIR / "model.pkl"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "training_report.json"
DEFAULT_HUMAN_REPORT_PATH = ROOT_DIR / "reports" / "training_report.md"
DEFAULT_ACADEMIC_REPORT_PATH = ROOT_DIR / "docs" / "model-evaluation.md"
LABEL_ORDER = ["ham", "spam"]
RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGGER = logging.getLogger(__name__)


def load_dataset(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the original headerless UTF-8 TSV without altering message text."""
    if not path.is_file():
        raise FileNotFoundError(
            f"SMS dataset not found at {path}. Extract sms+spam+collection.zip "
            "into data/raw/ before training."
        )

    data = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="error",
    )
    if data.empty:
        raise ValueError(f"SMS dataset at {path} contains no records.")

    if data[["label", "message"]].eq("").any().any():
        raise ValueError("The SMS dataset contains an empty label or message.")

    observed_labels = set(data["label"].unique())
    unexpected_labels = observed_labels.difference(LABEL_ORDER)
    if unexpected_labels:
        raise ValueError(f"Unexpected SMS labels found: {sorted(unexpected_labels)}")

    if observed_labels != set(LABEL_ORDER):
        raise ValueError("The SMS dataset must contain both ham and spam labels.")
    return data


def _dataset_statistics(data: pd.DataFrame, source: Path) -> dict[str, Any]:
    return {
        "source_file": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "encoding": "UTF-8",
        "delimiter": "tab",
        "header": False,
        "columns": ["label", "message"],
        "records": int(len(data)),
        "unique_messages": int(data["message"].nunique()),
        "class_distribution": {
            label: int(count)
            for label, count in data["label"].value_counts().sort_index().items()
        },
        "missing_cells": int(data[["label", "message"]].eq("").sum().sum()),
        "unknown_labels": sorted(set(data["label"]).difference(LABEL_ORDER)),
        "messages_with_conflicting_labels": int(
            data.groupby("message")["label"].nunique().gt(1).sum()
        ),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "duplicate_handling": "Retained unchanged; no rows were removed.",
        "label_handling": "Original ham/spam labels retained without remapping.",
    }


def split_dataset(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split unique messages first so repeated texts cannot cross the boundary."""
    labels_per_message = data.groupby("message")["label"].nunique()
    if labels_per_message.gt(1).any():
        raise ValueError(
            "A message occurs with conflicting labels; a leakage-safe "
            "stratified message split cannot be made."
        )

    unique_examples = data.drop_duplicates(subset="message", keep="first")
    train_groups, test_groups, _, _ = train_test_split(
        unique_examples["message"],
        unique_examples["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=unique_examples["label"],
    )
    train_messages = set(train_groups)
    training_mask = data["message"].isin(train_messages)
    return (
        data.loc[training_mask, "message"],
        data.loc[~training_mask, "message"],
        data.loc[training_mask, "label"],
        data.loc[~training_mask, "label"],
    )


def train_and_evaluate(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    human_report_path: Path = DEFAULT_HUMAN_REPORT_PATH,
    academic_report_path: Path = DEFAULT_ACADEMIC_REPORT_PATH,
) -> dict[str, Any]:
    """Train a leakage-safe TF-IDF/MultinomialNB pipeline and save its report."""
    data = load_dataset(data_path)
    x_train, x_test, y_train, y_test = split_dataset(data)
    message_overlap = len(set(x_train).intersection(x_test))

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", MultinomialNB()),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    spam_precision, spam_recall, spam_f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=["spam"],
        average=None,
        zero_division=0,
    )
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "generated_at_utc": generated_at,
        "dataset": _dataset_statistics(data, data_path),
        "split": {
            "target_test_size": TEST_SIZE,
            "actual_test_fraction": float(len(x_test) / len(data)),
            "random_state": RANDOM_STATE,
            "stratified": True,
            "grouped_by_message": True,
            "unique_message_overlap": message_overlap,
            "training_unique_messages": int(x_train.nunique()),
            "testing_unique_messages": int(x_test.nunique()),
            "training_records": int(len(x_train)),
            "testing_records": int(len(x_test)),
            "training_class_distribution": {
                label: int(count)
                for label, count in y_train.value_counts().sort_index().items()
            },
            "testing_class_distribution": {
                label: int(count)
                for label, count in y_test.value_counts().sort_index().items()
            },
        },
        "model": {
            "pipeline": ["TfidfVectorizer(ngram_range=(1, 2))", "MultinomialNB"],
            "random_state": RANDOM_STATE,
            "scikit_learn_version": sklearn.__version__,
            "vectorizer_fit_only_on_training_data": True,
        },
        "metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "spam_precision": float(spam_precision[0]),
            "spam_recall": float(spam_recall[0]),
            "spam_f1_score": float(spam_f1[0]),
            "confusion_matrix_labels": LABEL_ORDER,
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=LABEL_ORDER
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=LABEL_ORDER,
                target_names=LABEL_ORDER,
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    human_report_path.parent.mkdir(parents=True, exist_ok=True)
    academic_report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    report["model"]["artifact_sha256"] = hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    human_report = render_human_report(report)
    human_report_path.write_text(human_report, encoding="utf-8")
    academic_report_path.write_text(human_report, encoding="utf-8")

    LOGGER.info("Saved fitted pipeline to %s", model_path)
    LOGGER.info("Saved evaluation report to %s", report_path)
    LOGGER.info("Saved human-readable report to %s", human_report_path)
    LOGGER.info("Saved academic evaluation report to %s", academic_report_path)
    print(f"Records: {len(data)} (train={len(x_train)}, test={len(x_test)})")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(
        "Spam precision: "
        f"{report['metrics']['spam_precision']:.4f}; "
        f"spam recall: {report['metrics']['spam_recall']:.4f}; "
        f"spam F1: {report['metrics']['spam_f1_score']:.4f}"
    )
    print("Confusion matrix (rows=true, columns=predicted; ham, spam):")
    for row in report["metrics"]["confusion_matrix"]:
        print(row)
    print(f"Model: {model_path}")
    print(f"Report: {report_path}")
    return report


def render_human_report(report: dict[str, Any]) -> str:
    """Render generated metrics as a readable report without fixed values."""
    dataset = report["dataset"]
    split = report["split"]
    metrics = report["metrics"]
    model = report["model"]
    confusion = metrics["confusion_matrix"]
    classification = metrics["classification_report"]

    lines = [
        "# SMS Spam Detection — Model Evaluation",
        "",
        f"Generated (UTC): {report['generated_at_utc']}",
        "",
        "## Dataset and split",
        "",
        f"- Source file: `{dataset['source_file']}` (SHA-256 `{dataset['source_sha256']}`)",
        f"- Records: {dataset['records']}; unique message texts: {dataset['unique_messages']}",
        f"- Class counts: {dataset['class_distribution']}",
        f"- Exact duplicate rows retained: {dataset['exact_duplicate_rows']}",
        f"- Missing cells: {dataset['missing_cells']}; conflicting-label messages: "
        f"{dataset['messages_with_conflicting_labels']}",
        f"- Split: {split['training_records']} train / {split['testing_records']} test "
        f"(target test fraction {split['target_test_size']:.0%}; actual "
        f"{split['actual_test_fraction']:.2%})",
        f"- Message-text overlap: {split['unique_message_overlap']}",
        f"- Random seed: {split['random_state']}; stratified by unique message",
        "",
        "## Model and evaluation",
        "",
        f"- Pipeline: {' → '.join(model['pipeline'])}",
        f"- scikit-learn: {model['scikit_learn_version']}",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro precision / recall / F1: "
        f"{classification['macro avg']['precision']:.4f} / "
        f"{classification['macro avg']['recall']:.4f} / "
        f"{classification['macro avg']['f1-score']:.4f}",
        f"- HAM precision / recall / F1: "
        f"{classification['ham']['precision']:.4f} / "
        f"{classification['ham']['recall']:.4f} / "
        f"{classification['ham']['f1-score']:.4f}",
        f"- SPAM precision / recall / F1: "
        f"{classification['spam']['precision']:.4f} / "
        f"{classification['spam']['recall']:.4f} / "
        f"{classification['spam']['f1-score']:.4f}",
        "",
        "Confusion matrix (rows = actual, columns = predicted; ham, spam):",
        "",
        "```text",
        f"{confusion[0]}",
        f"{confusion[1]}",
        "```",
        "",
        "## Classification report",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("ham", "spam"):
        item = classification[label]
        lines.append(
            f"| {label.upper()} | {item['precision']:.4f} | {item['recall']:.4f} "
            f"| {item['f1-score']:.4f} | {int(item['support'])} |"
        )
    for label in ("macro avg", "weighted avg"):
        item = classification[label]
        lines.append(
            f"| {label} | {item['precision']:.4f} | {item['recall']:.4f} "
            f"| {item['f1-score']:.4f} | {int(item['support'])} |"
        )
    lines.extend(
        [
            "",
            "Accuracy alone can be misleading because ham examples outnumber spam "
            "examples. Spam recall and the confusion matrix show how many spam "
            "messages were missed; precision shows the false-alarm trade-off. "
            "These metrics describe one held-out split, not guaranteed future performance.",
            "",
            "Model probabilities are not calibrated and should not be treated as certainty.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--human-report", type=Path, default=DEFAULT_HUMAN_REPORT_PATH
    )
    parser.add_argument(
        "--academic-report", type=Path, default=DEFAULT_ACADEMIC_REPORT_PATH
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train_and_evaluate(
        args.data,
        args.model,
        args.report,
        args.human_report,
        args.academic_report,
    )


if __name__ == "__main__":
    main()
