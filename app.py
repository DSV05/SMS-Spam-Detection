"""Flask web application and JSON API for SMS spam classification."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import joblib
import sklearn
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "model.pkl"
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "training_report.json"
MAX_MESSAGE_LENGTH = 10_000
MODEL_LABELS = ("ham", "spam")


def _load_model(path: Path) -> tuple[Any | None, str | None]:
    try:
        model = joblib.load(path)
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Model artifact has no prediction method.")
        model_classes = {str(label) for label in model.classes_}
        if model_classes != set(MODEL_LABELS):
            raise ValueError("Model artifact has unsupported classes.")
        return model, None
    except (
        OSError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
        ImportError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
    ) as exc:
        logging.getLogger(__name__).error(
            "Unable to load model artifact (%s).", type(exc).__name__
        )
        return None, type(exc).__name__


def _load_evaluation_report(
    path: Path, model_path: Path
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Evaluation report must contain a JSON object.")
        if report.get("report_schema_version") != 1:
            raise ValueError("Unsupported evaluation report schema.")
        if "generated_at_utc" not in report:
            raise ValueError("Evaluation report has no generation timestamp.")
        model_metadata = report["model"]
        dataset = report["dataset"]
        split = report["split"]
        metrics = report["metrics"]
        classification_report = metrics["classification_report"]
        if model_metadata["scikit_learn_version"] != sklearn.__version__:
            raise ValueError(
                "Evaluation report scikit-learn version does not match runtime."
            )
        artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if model_metadata["artifact_sha256"] != artifact_hash:
            raise ValueError("Evaluation report does not match the current model artifact.")
        required_metadata = (
            "source_sha256",
            "records",
            "unique_messages",
            "class_distribution",
            "exact_duplicate_rows",
            "missing_cells",
        )
        if any(name not in dataset for name in required_metadata):
            raise ValueError("Evaluation report is missing dataset metadata.")
        if any(name not in split for name in ("testing_records", "unique_message_overlap")):
            raise ValueError("Evaluation report is missing split metadata.")
        if any(
            name not in metrics
            for name in (
                "accuracy",
                "spam_precision",
                "spam_recall",
                "spam_f1_score",
                "confusion_matrix",
                "confusion_matrix_labels",
            )
        ):
            raise ValueError("Evaluation report is missing required metrics.")
        for label in MODEL_LABELS:
            if label not in classification_report:
                raise ValueError("Evaluation report is missing a class report.")
            for metric_name in ("precision", "recall", "f1-score", "support"):
                if metric_name not in classification_report[label]:
                    raise ValueError("Evaluation report contains incomplete class metrics.")
        return report, None
    except (
        OSError,
        json.JSONDecodeError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logging.getLogger(__name__).error(
            "Evaluation report is unavailable or invalid (%s).", type(exc).__name__
        )
        return None, type(exc).__name__


def _make_prediction(model: Any, message: str) -> dict[str, Any]:
    started = time.perf_counter()
    label = str(model.predict([message])[0])
    if label not in MODEL_LABELS:
        raise RuntimeError("The model returned a label outside the supported classes.")

    response: dict[str, Any] = {
        "label": label,
        "confidence": None,
        "probabilities": None,
        "confidence_note": (
            "Model probabilities are estimates, not guarantees, and have not been "
            "calibrated."
        ),
    }
    if hasattr(model, "predict_proba"):
        class_names = [str(item) for item in model.classes_]
        values = model.predict_proba([message])[0]
        probabilities = {
            name: float(value) for name, value in zip(class_names, values)
        }
        if set(probabilities) != set(MODEL_LABELS):
            raise RuntimeError("The model probability classes are not supported.")
        response["probabilities"] = probabilities
        response["confidence"] = probabilities.get(label)
    response["prediction_time_ms"] = round(
        (time.perf_counter() - started) * 1000, 3
    )
    return response


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        MODEL_PATH=str(DEFAULT_MODEL_PATH),
        REPORT_PATH=str(DEFAULT_REPORT_PATH),
        MODEL=None,
        MAX_CONTENT_LENGTH=32 * 1024,
    )
    if test_config:
        app.config.update(test_config)

    configured_model = app.config.get("MODEL")
    if configured_model is not None:
        app.extensions["sms_model"] = configured_model
        app.extensions["sms_model_error"] = None
        app.extensions["sms_evaluation_report"] = None
        app.extensions["sms_report_error"] = "not_associated_with_injected_model"
    else:
        model_path = Path(app.config["MODEL_PATH"])
        model, error = _load_model(model_path)
        app.extensions["sms_model"] = model
        app.extensions["sms_model_error"] = error
        if model is not None:
            report, report_error = _load_evaluation_report(
                Path(app.config["REPORT_PATH"]), model_path
            )
        else:
            report, report_error = None, "model_unavailable"
        app.extensions["sms_evaluation_report"] = report
        app.extensions["sms_report_error"] = report_error

    def model_unavailable() -> bool:
        return app.extensions["sms_model"] is None

    def report_unavailable() -> tuple[Any, int]:
        return jsonify(error="Verified evaluation metadata is unavailable."), 503

    def validate_message(value: Any) -> tuple[str | None, str | None]:
        if not isinstance(value, str):
            return None, "The message field must be a string."
        message = value.strip()
        if not message:
            return None, "The message must not be empty."
        if len(message) > MAX_MESSAGE_LENGTH:
            return None, (
                f"The message must be at most {MAX_MESSAGE_LENGTH:,} characters."
            )
        return message, None

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_error: RequestEntityTooLarge):
        return jsonify(error="Request body is too large."), 413

    @app.get("/")
    def home():
        return render_template(
            "index.html",
            message="",
            prediction=None,
            error=None,
            model_available=not model_unavailable(),
            evaluation_report=app.extensions["sms_evaluation_report"],
        )

    @app.post("/")
    def predict_form():
        raw_message = request.form.get("message")
        message, error = validate_message(raw_message)
        if error:
            return (
                render_template(
                    "index.html",
                    message=raw_message or "",
                    prediction=None,
                    error=error,
                    model_available=not model_unavailable(),
                    evaluation_report=app.extensions["sms_evaluation_report"],
                ),
                400,
            )
        if model_unavailable():
            return (
                render_template(
                    "index.html",
                    message=message,
                    prediction=None,
                    error="The model is unavailable. Train the model and try again.",
                    model_available=False,
                    evaluation_report=app.extensions["sms_evaluation_report"],
                ),
                503,
            )
        try:
            prediction = _make_prediction(app.extensions["sms_model"], message)
        except (RuntimeError, TypeError, ValueError):
            app.logger.exception("Model prediction failed for form submission.")
            return (
                render_template(
                    "index.html",
                    message=message,
                    prediction=None,
                    error="The message could not be classified. Please try again.",
                    model_available=True,
                    evaluation_report=app.extensions["sms_evaluation_report"],
                ),
                500,
            )
        return render_template(
            "index.html",
            message=message,
            prediction=prediction,
            error=None,
            model_available=True,
            evaluation_report=app.extensions["sms_evaluation_report"],
        )

    @app.get("/health")
    def health():
        available = not model_unavailable()
        return (
            jsonify(
                status="ok" if available else "unavailable",
                model_loaded=available,
                evaluation_report_available=(
                    app.extensions["sms_evaluation_report"] is not None
                ),
            ),
            200 if available else 503,
        )

    @app.get("/api/model-info")
    def model_info():
        model = app.extensions["sms_model"]
        report = app.extensions["sms_evaluation_report"]
        if model is None:
            return jsonify(error="The model is unavailable."), 503
        if report is None:
            return report_unavailable()
        return jsonify(
            model_type="TF-IDF + Multinomial Naive Bayes",
            classes=[str(label) for label in model.classes_],
            generated_at_utc=report["generated_at_utc"],
            scikit_learn_version=report["model"]["scikit_learn_version"],
            dataset=report["dataset"],
            split=report["split"],
            evaluation=report["metrics"],
        )

    @app.get("/api/metrics")
    def metrics():
        report = app.extensions["sms_evaluation_report"]
        if report is None:
            return report_unavailable()
        return jsonify(report["metrics"])

    @app.post("/prediction")
    def prediction():
        if not request.is_json:
            return jsonify(error="Content-Type must be application/json."), 415
        try:
            payload = request.get_json()
        except BadRequest:
            return jsonify(error="Malformed JSON request body."), 400
        if not isinstance(payload, dict):
            return jsonify(error="JSON request body must be an object."), 400
        if "message" not in payload:
            return jsonify(error="Missing required field: message."), 400

        message, error = validate_message(payload["message"])
        if error:
            return jsonify(error=error), 400
        if model_unavailable():
            return jsonify(error="The model is unavailable. Train the model first."), 503
        try:
            result = _make_prediction(app.extensions["sms_model"], message)
        except (RuntimeError, TypeError, ValueError):
            app.logger.exception("Model prediction failed for API request.")
            return jsonify(error="The message could not be classified."), 500
        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5000, debug=False)
