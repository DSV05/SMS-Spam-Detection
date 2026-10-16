# System Architecture

## End-to-end flow

```text
Original sms+spam+collection.zip
              |
              v
data/raw/SMSSpamCollection
  UTF-8 · tab-separated · no header
              |
              v
Validation and data-quality report
  labels · empty fields · duplicates · conflicting message labels
              |
              v
Stratified unique-message train/test split
  repeated rows retained and assigned as one group
              |
              v
Training-only TF-IDF vectorization
              |
              v
Multinomial Naive Bayes
              |
              v
Fitted scikit-learn Pipeline (model.pkl)
       |                 |
       v                 v
training_report.json   human-readable evaluation
  hash/version/metrics  reports and docs
       |
       v
Flask application
  Web UI · /health · /prediction · /api/model-info · /api/metrics
       |
       v
Docker image: sms-spam-app:1.0
  Python 3.11 · pinned model runtime · Gunicorn · port 5000
       |
       +-------------------------------+
       |                               |
       v                               v
AWS EC2 Ubuntu VM                 GCP Compute Engine VM
manual student deployment        separate manual deployment
```

## Data and training

The dataset loader reads the supplied tab-separated file with explicit UTF-8
encoding and literal-text quote handling. It validates the two expected fields,
known `ham`/`spam` labels, and non-empty content. Dataset-level duplicates are
not deleted. The split is stratified over unique message strings and uses a
fixed seed; all occurrences of a message are assigned to one side. This
prevents identical text from appearing in both train and test while preserving
all rows. TF-IDF vocabulary and weights are learned inside the scikit-learn
pipeline during `.fit()` on training messages only.

The serialized pipeline includes both the fitted `TfidfVectorizer` and
`MultinomialNB`, so inference uses the same transformations as training.
`training_report.json` records the source-data hash, pipeline/runtime version,
trained artifact hash, split counts, and computed test metrics. A generated
human-readable copy is written to `reports/training_report.md` and
`docs/model-evaluation.md`.

## Serving and readiness

`app.py` loads the fitted artifact once when the Flask application factory
creates the app; the application does not retrain at startup. `/health` returns
HTTP 200 only when the model loaded successfully, and 503 otherwise.
`/prediction` validates a JSON message, runs the fitted pipeline, returns the
label and supported class probabilities, and measures model prediction
duration. Probability outputs are estimates, not calibrated guarantees.

`/api/model-info` and `/api/metrics` expose report values only when the report's
scikit-learn version and model SHA-256 match the current runtime and artifact.
Missing, malformed, or stale evaluation metadata is reported as unavailable
rather than silently presented as current.

## Docker and deployment boundaries

The Docker image contains the Flask serving code, HTML/CSS/JavaScript, and
trained model. It runs Gunicorn bound to `0.0.0.0:5000` as a non-root user.
Training data and evaluation files are excluded from the runtime image.

AWS and GCP are distinct deployment exercises using the same image. Cloud
credentials and private keys remain outside the repository. Security groups
and firewall rules should restrict SSH to the user's address and application
access to intended clients. Cloud provisioning is manual and may incur charges;
no successful cloud deployment is claimed without actual verification.
