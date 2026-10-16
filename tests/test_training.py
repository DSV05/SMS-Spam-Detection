from pathlib import Path

import joblib
import json
from sklearn.pipeline import Pipeline

from train_model import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    load_dataset,
    split_dataset,
)


def test_actual_dataset_is_loaded_without_dropping_duplicates() -> None:
    data = load_dataset(DEFAULT_DATA_PATH)
    assert len(data) == 5574
    assert data["label"].value_counts().to_dict() == {"ham": 4827, "spam": 747}
    assert data.duplicated().sum() == 403
    assert not data[["label", "message"]].eq("").any().any()


def test_split_keeps_duplicate_messages_together_without_dropping_rows() -> None:
    data = load_dataset(DEFAULT_DATA_PATH)
    x_train, x_test, y_train, y_test = split_dataset(data)
    assert len(x_train) + len(x_test) == len(data)
    assert len(y_train) + len(y_test) == len(data)
    assert not set(x_train).intersection(x_test)
    assert x_train.duplicated().sum() + x_test.duplicated().sum() == 403


def test_trained_pipeline_is_serialized_and_predicts_actual_messages() -> None:
    model_path = Path(DEFAULT_MODEL_PATH)
    assert model_path.is_file(), "Run python train_model.py before this test."
    pipeline = joblib.load(model_path)
    assert isinstance(pipeline, Pipeline)
    assert set(pipeline.classes_) == {"ham", "spam"}
    result = pipeline.predict(["Call now to claim your prize"])
    assert len(result) == 1
    assert result[0] in {"ham", "spam"}


def test_machine_and_human_reports_match_saved_model() -> None:
    report_path = DEFAULT_DATA_PATH.parents[2] / "reports" / "training_report.json"
    human_report_path = (
        DEFAULT_DATA_PATH.parents[2] / "docs" / "model-evaluation.md"
    )
    assert report_path.is_file()
    assert human_report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"]["artifact_sha256"]
    assert report["generated_at_utc"]
    assert "Confusion matrix" in human_report_path.read_text(encoding="utf-8")
