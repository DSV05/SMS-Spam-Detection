# Existing Project Audit

**Audit date:** 2026-10-15  
**Scope:** Existing `CV-Project` workspace, supplied practical DOCX, original
dataset ZIP, trained model, application, tests, and Docker configuration.

## Source materials inspected

- `2CEAI702_PRACTICAL_FILE__FORMAT.docx` is present in the workspace. It defines
  four practicals: convert a notebook/model into a Flask interface and API;
  Dockerize and exercise the image/container lifecycle; deploy on AWS EC2; and
  deploy the same image separately on GCP.
- `sms+spam+collection.zip` contains `SMSSpamCollection` and `readme`.
- `SMSSpamCollection` is the original headerless UTF-8 tab-separated dataset.
  Literal quotation marks in messages require quote handling to be disabled
  during TSV parsing. It has 5,574 records: 4,827 ham and 747 spam, no empty
  fields, 403 exact duplicate rows, 5,171 unique message texts, and no messages
  with conflicting labels.
- The extracted dataset files remain unchanged in `data/raw/`.

## Baseline project and verification

Before this upgrade the workspace already contained a TF-IDF /
Multinomial Naive Bayes training pipeline, a serialized model, basic Flask
routes, a simple form, tests, Docker configuration, practical guidance, and a
README. The baseline test suite was executed with the project virtual
environment: **18 passed**.

The baseline Docker image `sms-spam-app:1.0` was built and exercised. Home,
health, and prediction endpoints worked. Container logs exposed a
scikit-learn persistence-version mismatch, which was addressed by pinning
scikit-learn to 1.7.2, regenerating the model, and rebuilding the image. The
Docker container's lifecycle was exercised only on the explicitly named
project container; no cloud resource was created.

## Gaps identified and addressed

| Area | Audit finding | Upgrade |
| --- | --- | --- |
| UI | The initial page was a basic form without navigation, samples, live health, probability visualization, or project methodology. | Replaced with responsive, accessible dashboard layout, asynchronous form, sample selector, measured prediction time, class probabilities, and responsible-use warning. |
| Model metadata | The fitted artifact and metrics report had no shared artifact fingerprint or generation time. | Report now records UTC generation timestamp, scikit-learn version, artifact SHA-256, split details, and dataset quality. |
| Evaluation | There was no generated human-readable report, and message duplicates could occur across a row-wise split. | Generates JSON and Markdown reports; stratifies unique messages and keeps every duplicate row in the same split, preventing message-text overlap. |
| API | No endpoints exposed verified model information or metrics. Oversized inputs and report freshness lacked focused tests. | Adds `/api/model-info` and `/api/metrics`; verifies evaluation report hash/version; tests invalid, oversized, missing, stale, and valid requests. |
| Documentation | No concise audit, architecture diagram, or model evaluation document existed. | Added this audit, system architecture, generated evaluation document, and expanded the academic README and evidence checklist. |
| Cloud practicals | Instructions existed, but no AWS or GCP deployment had been performed. | Retained and expanded step-by-step instructions with security and cost cautions; deployment evidence remains pending. |

## Current status and remaining practical work

- Local training, Flask endpoints, tests, and Docker build/runtime checks have
  been executed successfully after the changes.
- Final automated suite: **29 passed**. The final Docker image's health check
  reached `healthy`; its home, readiness, model-info, metrics, and prediction
  endpoints were verified, and its named test container was removed after
  lifecycle checks.
- Evaluation values and UI summary are generated from the current, hash-matched
  report; the server does not expose stale report metrics as current.
- AWS EC2 and GCP deployment have not been executed. They require account
  access, billing approval, and manual resource provisioning. The practical
  evidence checklist marks their screenshots pending.
- Practical screenshot captures still need to be taken by the student from the
  actual environment; none are fabricated here.
