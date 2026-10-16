from __future__ import annotations

from typing import Any

import pytest
import joblib

from app import DEFAULT_MODEL_PATH, DEFAULT_REPORT_PATH, create_app


class DeterministicModel:
    classes_ = ["ham", "spam"]

    def predict(self, messages: list[str]) -> list[str]:
        return [
            "ham" if "ham-like" in message else "spam"
            for message in messages
        ]

    def predict_proba(self, messages: list[str]) -> list[list[float]]:
        return [
            [0.85, 0.15] if "ham-like" in message else [0.15, 0.85]
            for message in messages
        ]


class LabelOnlyModel:
    classes_ = ["ham", "spam"]

    def predict(self, messages: list[str]) -> list[str]:
        return ["ham" for _ in messages]


@pytest.fixture
def client():
    app = create_app({"TESTING": True, "MODEL": DeterministicModel()})
    return app.test_client()


def test_home_page_returns_form(client: Any) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert b"SMS Spam Detection System" in response.data
    assert b'name="message"' in response.data
    assert b'id="sample-select"' in response.data
    assert b"Not calibrated" in response.data


def test_health_endpoint_reports_loaded_model(client: Any) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {
        "model_loaded": True,
        "status": "ok",
        "evaluation_report_available": False,
    }


def test_prediction_returns_label_and_estimated_confidence(client: Any) -> None:
    response = client.post(
        "/prediction",
        json={"message": "A message to classify"},
    )
    assert response.status_code == 200
    result = response.get_json()
    assert result["label"] == "spam"
    assert result["confidence"] == pytest.approx(0.85)
    assert result["probabilities"] == {"ham": pytest.approx(0.15), "spam": 0.85}
    assert "not guarantees" in result["confidence_note"]
    assert result["prediction_time_ms"] >= 0


@pytest.mark.parametrize(
    ("message", "expected_label"),
    [("A ham-like message", "ham"), ("A spam-like message", "spam")],
)
def test_valid_class_predictions(client: Any, message: str, expected_label: str) -> None:
    response = client.post("/prediction", json={"message": message})
    assert response.status_code == 200
    assert response.get_json()["label"] == expected_label


def test_prediction_rejects_missing_message(client: Any) -> None:
    response = client.post("/prediction", json={})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Missing required field: message."}


@pytest.mark.parametrize("message", ["", "  \t"])
def test_prediction_rejects_empty_message(client: Any, message: str) -> None:
    response = client.post("/prediction", json={"message": message})
    assert response.status_code == 400
    assert "must not be empty" in response.get_json()["error"]


def test_prediction_rejects_malformed_json(client: Any) -> None:
    response = client.post(
        "/prediction",
        data='{"message":',
        content_type="application/json",
    )
    assert response.status_code == 400
    assert response.get_json() == {"error": "Malformed JSON request body."}


@pytest.mark.parametrize("message", [None, 42, ["message"], {"text": "message"}])
def test_prediction_rejects_non_string_message(client: Any, message: Any) -> None:
    response = client.post("/prediction", json={"message": message})
    assert response.status_code == 400
    assert response.get_json() == {"error": "The message field must be a string."}


def test_prediction_requires_json_content_type(client: Any) -> None:
    response = client.post("/prediction", data="message=hello")
    assert response.status_code == 415
    assert response.get_json() == {
        "error": "Content-Type must be application/json."
    }


def test_prediction_works_without_probability_support() -> None:
    app = create_app({"TESTING": True, "MODEL": LabelOnlyModel()})
    result = app.test_client().post(
        "/prediction", json={"message": "An ordinary message"}
    )
    assert result.status_code == 200
    assert result.get_json()["label"] == "ham"
    assert result.get_json()["confidence"] is None
    assert result.get_json()["probabilities"] is None


def test_prediction_rejects_non_object_json(client: Any) -> None:
    response = client.post("/prediction", json=["not", "an", "object"])
    assert response.status_code == 400
    assert response.get_json() == {"error": "JSON request body must be an object."}


def test_prediction_rejects_message_over_limit(client: Any) -> None:
    response = client.post("/prediction", json={"message": "x" * 10_001})
    assert response.status_code == 400
    assert "10,000 characters" in response.get_json()["error"]


def test_prediction_rejects_oversized_request_body(client: Any) -> None:
    response = client.post(
        "/prediction",
        data="x" * 40_000,
        content_type="application/json",
    )
    assert response.status_code == 413
    assert response.get_json() == {"error": "Request body is too large."}


def test_unavailable_model_returns_service_unavailable(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(tmp_path / "missing-model.pkl"),
        }
    )
    client = app.test_client()
    health = client.get("/health")
    prediction = client.post("/prediction", json={"message": "Hello"})
    assert health.status_code == 503
    assert health.get_json() == {
        "model_loaded": False,
        "status": "unavailable",
        "evaluation_report_available": False,
    }
    assert prediction.status_code == 503
    assert prediction.get_json() == {
        "error": "The model is unavailable. Train the model first."
    }


def test_unavailable_model_does_not_expose_internal_path(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(tmp_path / "private" / "missing-model.pkl"),
        }
    )
    client = app.test_client()
    response = client.get("/api/model-info")
    assert response.status_code == 503
    assert str(tmp_path) not in response.get_data(as_text=True)


def test_invalid_model_artifact_is_not_reported_healthy(tmp_path) -> None:
    model_path = tmp_path / "invalid-model.pkl"
    joblib.dump({"not": "a classifier"}, model_path)
    app = create_app({"TESTING": True, "MODEL_PATH": str(model_path)})
    response = app.test_client().get("/health")
    assert response.status_code == 503
    assert response.get_json()["model_loaded"] is False


def test_model_info_and_metrics_are_verified_against_current_artifact() -> None:
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(DEFAULT_MODEL_PATH),
            "REPORT_PATH": str(DEFAULT_REPORT_PATH),
        }
    )
    client = app.test_client()
    info_response = client.get("/api/model-info")
    metrics_response = client.get("/api/metrics")
    assert info_response.status_code == 200
    assert metrics_response.status_code == 200
    info = info_response.get_json()
    metrics = metrics_response.get_json()
    assert info["model_type"] == "TF-IDF + Multinomial Naive Bayes"
    assert info["classes"] == ["ham", "spam"]
    assert info["evaluation"]["accuracy"] == metrics["accuracy"]


def test_stale_report_is_unavailable(tmp_path) -> None:
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(DEFAULT_MODEL_PATH),
            "REPORT_PATH": str(tmp_path / "missing-report.json"),
        }
    )
    client = app.test_client()
    assert client.get("/health").get_json()["model_loaded"] is True
    assert client.get("/api/metrics").status_code == 503


def test_invalid_evaluation_report_is_not_exposed(tmp_path) -> None:
    report_path = tmp_path / "invalid-report.json"
    report_path.write_text("[]", encoding="utf-8")
    app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(DEFAULT_MODEL_PATH),
            "REPORT_PATH": str(report_path),
        }
    )
    client = app.test_client()
    assert client.get("/health").get_json()["model_loaded"] is True
    assert client.get("/health").get_json()["evaluation_report_available"] is False
    assert client.get("/api/model-info").status_code == 503
    assert client.get("/api/metrics").status_code == 503


def test_form_shows_prediction(client: Any) -> None:
    response = client.post("/", data={"message": "A test message"})
    assert response.status_code == 200
    assert b"SPAM" in response.data
