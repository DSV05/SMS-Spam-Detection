# SMS Spam Detection using Multinomial Naive Bayes

A small, reproducible machine-learning application that classifies an SMS as
`spam` or `ham`. It uses the user's original SMS Spam Collection archive, a
scikit-learn TF-IDF/Multinomial Naive Bayes pipeline, Flask, and Joblib.

## Project status

- The supplied data has been extracted and validated locally.
- The fitted model and computed evaluation report have been generated.
- The Flask interface and JSON API are implemented and covered by automated
  tests.
- The Docker image built successfully. The container's home, health, and
  prediction endpoints, logs, inspection, and start/stop/remove lifecycle were
  exercised locally; the test container was removed afterward.
- No AWS or GCP resources have been created. Cloud deployment instructions below
  are guides, not evidence of a deployment.

## Abstract and problem statement

Unwanted SMS messages can contain fraudulent offers, requests, and links.
This project demonstrates a supervised text classifier that learns from
labelled SMS examples and returns a class label for a submitted message. The
classifier is intended for coursework and experimentation, not as a guarantee
that a message is safe.

## Introduction

Short-message classification is a practical natural-language-processing task:
the system must turn free-form text into numerical features and infer a class
from previously labelled examples. This project demonstrates that workflow as
an end-to-end MLOps practical, from dataset validation and reproducible
training to a testable API and container image.

## Problem statement

The application receives an SMS message and predicts whether it resembles
`ham` (legitimate, non-spam) or `spam` (unwanted or spam-like) messages in the
provided corpus. A predicted HAM label must not be interpreted as proof that a
message or its sender is safe.

## Objectives

- Preserve and use the provided dataset rather than substitute generated data.
- Train a complete text preprocessing/classification pipeline without test-set
  leakage.
- Evaluate the held-out test set and save its computed metrics.
- Serve the same fitted pipeline through a responsive web form and JSON API.
- Exercise the app with pytest and package it in Docker.
- Document repeatable AWS EC2 and GCP VM deployment procedures.

## Dataset and data quality

The supplied `sms+spam+collection.zip` contains `SMSSpamCollection` and a
dataset `readme`. The data file is headerless, UTF-8, tab-separated, with
`label` and `message` fields. Quotes in message text are literal; the loader
therefore disables CSV quote interpretation.

Validation of the supplied file found:

| Finding | Value |
| --- | ---: |
| Records | 5,574 |
| `ham` labels | 4,827 |
| `spam` labels | 747 |
| Empty labels/messages | 0 |
| Exact duplicate rows | 403 |
| Unknown labels | 0 |

The raw extracted file remains unchanged under `data/raw/`. Duplicate rows are
retained, and the original `ham` and `spam` labels are preserved. The dataset
SHA-256, data-quality results, split counts, and evaluation metrics are recorded
in `reports/training_report.json`.

## Technologies and model

- Python 3.10 or 3.11
- pandas, NumPy, scikit-learn, Joblib
- Flask and Gunicorn
- pytest
- Docker

## Software and hardware requirements

- Windows PowerShell, Linux, or macOS.
- Python 3.10 or 3.11 for local training and the Flask development server.
- A modest computer with at least 2 GB available memory is sufficient for this
  dataset and model. Training is CPU-only.
- Docker Desktop/Engine is optional for Practical 2 and required for building
  and exercising the container image.
- AWS or GCP account access, billing authorization, and permission to create
  virtual machines are needed only if completing the cloud practicals.

The saved scikit-learn `Pipeline` contains `TfidfVectorizer(ngram_range=(1, 2))`
followed by `MultinomialNB`. It is fitted on the training split only, then
serialized as `model.pkl`. The split is reproducible (`random_state=42`,
target 80% training / 20% testing, stratified over unique messages). Repeated
messages are assigned to the same split to prevent text overlap, but all
original rows remain in training or testing. No label mapping changes the
source labels. scikit-learn is pinned to 1.7.2, which supports Python 3.10 and
3.11 and matches the version used to serialize the model; preserve that match
when moving `model.pkl` to another environment.

## Actual evaluation

Metrics below are computed by `train_model.py` from the held-out 1,112-message
test set, with duplicate message groups kept out of both splits; they are not
illustrative values.

| Metric | Result |
| --- | ---: |
| Accuracy | 0.9442 |
| Spam precision | 1.0000 |
| Spam recall | 0.5894 |
| Spam F1 | 0.7417 |

Confusion matrix, rows = actual class and columns = predicted class, ordered
`ham`, `spam`:

```text
[[961, 0],
 [ 62, 89]]
```

The complete classification report and split distributions are in
`reports/training_report.json`.

Accuracy alone is insufficient for this imbalanced dataset: predicting the
majority class can mask poor spam detection. Therefore the report also includes
class-specific precision, recall, F1, support, and a confusion matrix. The
spam-class recall reveals the false negatives; the held-out split and corpus
still do not establish performance on future messages.

## System architecture

```text
Original SMS archive → validation → stratified grouped split
→ training-only TF-IDF → Multinomial Naive Bayes
→ serialized pipeline + evaluation reports
→ Flask web/API → Docker image → separate AWS EC2 / GCP VM exercises
```

See [system-architecture.md](./docs/system-architecture.md) for the detailed
data and serving flow.

## Preprocessing methodology

The raw data is headerless UTF-8 TSV. A tab separates the source label from the
message, and quote characters inside message text are preserved literally.
The loader rejects empty fields and unexpected labels; it does not strip,
rewrite, or relabel original rows. Exact duplicate rows are retained.
Duplicate message groups are assigned to one side of the stratified
train/test split to prevent text overlap. The vectorizer is part of the
scikit-learn pipeline, so its vocabulary and inverse-document-frequency
weights are fitted only on training data and reused unchanged at inference.

## Algorithm overview

TF-IDF (term frequency–inverse document frequency) represents a message as a
sparse vector of token and token-pair weights. The vectorizer uses word
unigrams and bigrams. Multinomial Naive Bayes estimates class likelihoods for
the non-negative feature counts/weights and predicts the more likely class.
This is a compact, interpretable baseline for text classification. Its
`predict_proba` outputs are not calibrated by this project.

## Training and evaluation methodology

The training script validates the actual corpus, records data-quality
statistics, stratifies unique messages with `random_state=42`, fits the
pipeline on the training subset, and computes metrics on the held-out subset.
It writes:

- `model.pkl` — complete fitted inference pipeline.
- `reports/training_report.json` — machine-readable metrics and verified model
  metadata.
- `reports/training_report.md` — human-readable generated report.
- `docs/model-evaluation.md` — generated academic evaluation report.

The Flask app verifies the evaluation report's model hash and scikit-learn
version before making evaluation metrics available.

Only load the project-generated `model.pkl` or another artifact from a trusted
source. Joblib uses Python object serialization, which is not safe for
untrusted artifacts.

## Project layout

```text
.
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── data/
│   └── raw/
│       ├── SMSSpamCollection
│       └── readme
├── reports/
│   ├── training_report.json
│   └── training_report.md
├── templates/
│   └── index.html
├── static/
│   ├── app.js
│   └── style.css
├── tests/
│   ├── test_app.py
│   └── test_training.py
└── docs/
    ├── model-evaluation.md
    ├── project-audit.md
    ├── system-architecture.md
    └── practical-evidence.md
```

The original archive `sms+spam+collection.zip` and practical instruction
document are also kept in the project root. Do not put cloud credentials or
private SSH keys in this folder.

## Environment setup

Use Python 3.10 or 3.11. The implementation and tests in this workspace were
run with Python 3.11.9.

### Windows PowerShell

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, use the per-process policy only for the
current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On Windows, run the Flask development server as described below. Gunicorn is
used in the Linux Docker image and cloud VMs; it is not the Windows server.

### Linux / macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train and evaluate

The archive has already been extracted into `data/raw/`. To repeat training:

```powershell
python train_model.py
```

```bash
python train_model.py
```

The script validates the dataset, makes the stratified split, fits the vectorizer
and classifier on training messages only, evaluates the test messages, then
writes `model.pkl` and `reports/training_report.json`. To use non-default
locations, pass `--data`, `--model`, and/or `--report`. The command fails
explicitly if the dataset is absent or invalid.

## Run the Flask application

Make sure `model.pkl` exists, then from the project root:

```powershell
python app.py
```

Open <http://127.0.0.1:5000>. For Linux production/container use, the Docker
image starts Gunicorn bound to `0.0.0.0:5000`; the development server is not a
production server.

Routes:

| Route | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Responsive message-entry form |
| `/` | POST | Classify form message |
| `/health` | GET | Model readiness and evaluation-report status; 200 when model loaded, 503 otherwise |
| `/prediction` | POST | JSON message classification |
| `/api/model-info` | GET | Verified model, dataset, split, and evaluation metadata |
| `/api/metrics` | GET | Computed metrics from the matching evaluation report |

The model-information response includes the UTC report/training timestamp,
class labels, dataset hash and counts, split details, and computed evaluation
metrics. Both metadata endpoints return 503 if the report is missing, invalid,
stale, or does not match the model artifact/runtime version.

### Prediction API

Request:

```json
{
  "message": "Congratulations! You have won a free prize"
}
```

Example PowerShell request:

```powershell
$body = @{ message = "Congratulations! You have won a free prize" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/prediction `
  -ContentType "application/json" -Body $body
```

Example `curl` request:

```bash
curl -X POST http://127.0.0.1:5000/prediction \
  -H "Content-Type: application/json" \
  -d '{"message":"Congratulations! You have won a free prize"}'
```

Normal-message example (the actual class depends on the trained model):

```bash
curl -X POST http://127.0.0.1:5000/prediction \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi, are we still meeting for lunch?"}'
```

A successful response has this shape. The values below are placeholders; the
label and numbers are generated by the fitted model at runtime:

```text
{
  "label": "<ham or spam>",
  "confidence": <model probability estimate>,
  "probabilities": {
    "ham": <model probability estimate>,
    "spam": <model probability estimate>
  },
  "prediction_time_ms": <measured model prediction duration>,
  "confidence_note": "Model probabilities are estimates, not guarantees, and have not been calibrated."
}
```

`confidence` is the model's class probability estimate; it has not been
calibrated and is not a guarantee.

Invalid request examples:

```bash
# Missing field: HTTP 400
curl -X POST http://127.0.0.1:5000/prediction \
  -H "Content-Type: application/json" -d '{}'

# Empty message: HTTP 400
curl -X POST http://127.0.0.1:5000/prediction \
  -H "Content-Type: application/json" -d '{"message":"  "}'

# Wrong value type: HTTP 400
curl -X POST http://127.0.0.1:5000/prediction \
  -H "Content-Type: application/json" -d '{"message":42}'

# Malformed JSON: HTTP 400
curl -X POST http://127.0.0.1:5000/prediction \
  -H "Content-Type: application/json" -d '{"message":'
```

The API requires `Content-Type: application/json`, rejects non-string/empty
messages, and caps a message at 10,000 characters. A missing model produces a
clear 503 response rather than silently training during application startup.
The JSON API also rejects request bodies larger than 32 KiB with HTTP 413.
For Postman, create a POST request to `/prediction`, select **Body → raw → JSON**,
and send the request object shown above.

## Automated tests

Run the test suite from the project root:

```powershell
python -m pytest -q
```

```bash
python -m pytest -q
```

`tests/test_app.py` uses a deterministic fixture model for API behavior.
`tests/test_training.py` checks the actual dataset, verifies duplicate messages
stay within one split, and loads the trained pipeline to predict a message. The
current result in this workspace is **29 passed**.

## Docker

Docker Desktop/Engine must be running before building. In this workspace Docker
29.6.1 was used, the `sms-spam-app:1.0` image built, and the project container
was started and checked. The container was stopped, restarted, and removed after
verification.

From the project root:

```powershell
docker --version
docker info
docker build -t sms-spam-app:1.0 .
docker images
docker run -d --name sms-spam-container -p 5000:5000 sms-spam-app:1.0
docker ps
docker logs sms-spam-container
docker inspect sms-spam-container
```

Verify the endpoints:

```powershell
Invoke-WebRequest http://127.0.0.1:5000/health
$body = @{ message = "Hi, are we still meeting for lunch?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/prediction `
  -ContentType "application/json" -Body $body
```

When finished, operate only on this project's named container:

```powershell
docker stop sms-spam-container
docker start sms-spam-container
docker rm -f sms-spam-container
```

The image uses Python 3.11 slim, installs the declared requirements, runs as a
non-root user, and starts Gunicorn with two workers. The pinned scikit-learn
version matches the version used to serialize `model.pkl`. The original dataset
and Markdown reports are excluded from the image; the compact JSON report is
included so `/api/model-info`, `/api/metrics`, and the UI can show verified
evaluation results. Docker's health check calls `/health` and reports unhealthy
if the model is not ready.

## AWS EC2 deployment (Practical 3)

These instructions are not a deployment claim. You need an AWS account, billing
authorization, EC2 permissions, and an SSH key. Review current prices and stop
or terminate billable resources when finished.

1. In the AWS Console, select a region (for example Mumbai, `ap-south-1`) and
   open EC2.
2. Launch an Ubuntu LTS instance. Choose an instance with at least 2 GiB memory
   where available (for example a suitable `t3.small` equivalent); check your
   quota, regional availability, and price before creating it.
3. Create/download an SSH key pair and store the private key outside the
   repository with restrictive permissions.
4. In the instance security group allow TCP 22 only from **My IP** (`/32`).
   Allow TCP 5000 only from your trusted client address range when practical.
   `0.0.0.0/0` exposes the app publicly and should be used only when required.
5. Wait for the instance checks, then connect from PowerShell or a terminal.
   Replace the placeholders:

   ```bash
   ssh -i "YOUR_KEY_PATH.pem" ubuntu@YOUR_EC2_PUBLIC_IP
   ```

6. In the SSH session install and verify Docker:

   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo systemctl enable --now docker
   sudo docker --version
   sudo docker info
   ```

7. On your local machine (with Docker running), save the already-built image
   and copy it to the instance:

   ```powershell
   docker save -o .\sms-spam-app-1.0.tar sms-spam-app:1.0
   scp -i "YOUR_KEY_PATH.pem" .\sms-spam-app-1.0.tar ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/
   ```

8. Back in the instance SSH session, load and start it:

   ```bash
   sudo docker load -i /home/ubuntu/sms-spam-app-1.0.tar
   sudo docker images
   sudo docker run -d --name sms-spam-container --restart unless-stopped \
     -p 5000:5000 sms-spam-app:1.0
   sudo docker ps
   sudo docker logs sms-spam-container
   ```

9. From an allowed client, open `http://YOUR_EC2_PUBLIC_IP:5000` and
   `http://YOUR_EC2_PUBLIC_IP:5000/health`. Send a POST to
   `http://YOUR_EC2_PUBLIC_IP:5000/prediction` using one of the API examples above.
10. To stop/remove only this app container: `sudo docker stop
    sms-spam-container` and `sudo docker rm sms-spam-container`. Stop or
    terminate the EC2 instance in the console when finished to avoid ongoing
    charges; deleting an instance may not delete separately billed storage or
    addresses.

If access fails, check instance status checks and public IP, security-group
source/port rules, `sudo docker ps`, `sudo docker logs sms-spam-container`, and
that port 5000 is published.

## Google Cloud Platform deployment (Practical 4)

This is a separate deployment exercise using the same image. No GCP project or
VM has been created from this workspace. Confirm billing, permissions, regional
pricing, and cleanup steps before provisioning resources.

1. In the Google Cloud Console create/select a project, link billing if
   required, and enable the Compute Engine API.
2. Create an Ubuntu LTS VM in a suitable region/zone. Choose at least 2 GiB
   memory where available and review machine/disk costs.
3. Create a VPC firewall ingress rule for TCP 5000 targeted only to this VM
   (use a network tag or service account). Restrict the source CIDR to trusted
   clients where possible; `0.0.0.0/0` permits any IPv4 source.
4. Connect with the browser SSH button. Install Docker:

   ```bash
   sudo apt-get update
   sudo apt-get install -y docker.io
   sudo systemctl enable --now docker
   sudo docker --version
   sudo docker info
   ```

5. On the local machine, save the Docker image:

   ```powershell
   docker save -o .\sms-spam-app-1.0.tar sms-spam-app:1.0
   ```

   Install the Google Cloud CLI if needed, authenticate in a local terminal,
   and select your project. Replace the placeholders:

   ```powershell
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   gcloud compute scp .\sms-spam-app-1.0.tar YOUR_VM_NAME:~/ `
     --zone YOUR_VM_ZONE --project YOUR_PROJECT_ID
   ```

6. In the VM's browser SSH terminal, import and run the image:

   ```bash
   sudo docker load -i ~/sms-spam-app-1.0.tar
   sudo docker images
   sudo docker run -d --name sms-spam-container --restart unless-stopped \
     -p 5000:5000 sms-spam-app:1.0
   sudo docker ps
   sudo docker logs sms-spam-container
   ```

7. Find the VM external IPv4 address. Verify
   `http://YOUR_GCP_EXTERNAL_IP:5000`, `/health`, and a POST to `/prediction` from
   an address allowed by the firewall.
8. When finished, remove only this app container with
   `sudo docker rm -f sms-spam-container`, then stop/delete the VM and any
   separately billed disks, static IPs, or firewall resources no longer needed.

If access fails, verify the VM external IP, firewall target tag/source/port,
container status and logs, and that Docker published `5000:5000`.

## Limitations and future enhancements

The corpus is imbalanced and English-language, and it reflects its original
collection sources and period. Grouped splitting prevents exact message text
from appearing in both subsets, but one split does not establish general
performance. The spam recall of the current experiment indicates that some
spam examples are missed. The model does not perform probability calibration,
sender verification, URL reputation checking, or real-time threat analysis.

Potential future work includes repeated stratified group evaluation, tuning
model hyperparameters using training-only cross-validation, probability
calibration evaluated on a separate validation set, broader contemporary
datasets with clear provenance, and explicit error analysis. These extensions
should be measured and documented rather than presented as implemented
capabilities.

## Conclusion

This practical implements a reproducible text-classification lifecycle using
the supplied dataset: validation, grouped stratified evaluation, training-only
TF-IDF, Multinomial Naive Bayes, model serialization, a Flask interface/API,
automated tests, and Docker packaging. The computed results and remaining
limitations are documented in the generated evaluation report. AWS EC2 and GCP
remain separate manual exercises; their setup instructions are not deployment
evidence.

- AWS deployment URL : http://16.4.38.19:5000/

