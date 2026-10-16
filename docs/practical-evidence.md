# Practical Evidence Checklist

Evidence status: all screenshots below are **pending** until captured from the
actual environment. Do not use fabricated output, URLs, or deployment evidence.

### Practical 1

- [ ] Environment setup — create/activate `.venv` in PowerShell and run
  `python --version`; capture Python 3.10 or 3.11 and the activated environment.
- [ ] Dependency installation — run `python -m pip install -r requirements.txt`;
  capture successful completion.
- [ ] Dataset inspection — inspect the ZIP entries and show
  `data/raw/SMSSpamCollection`; capture the archive contents and extracted file.
- [ ] Training output — run `python train_model.py`; capture dataset and
  train/test record counts.
- [ ] Evaluation metrics — capture the printed accuracy, spam precision/recall,
  confusion matrix, and the corresponding `reports/training_report.json` and
  `docs/model-evaluation.md`.
- [ ] Saved model — show `model.pkl` in the project root and its file details
  after training; show the model artifact SHA-256 in the generated report.
- [ ] Project structure — show the project root with the app, data, reports,
  templates, static, tests, and docs folders.
- [ ] Flask interface — run `python app.py` and open
  `http://127.0.0.1:5000`; capture the form.
- [ ] Prediction results — submit an SMS through the form and capture the
  actual class, probability bars, measured prediction duration, and
  responsible-use warning returned by this trained model.
- [ ] Responsive controls — select a sample message, show the character count,
  and capture the mobile layout if responsive evidence is required.
- [ ] Health check — request `http://127.0.0.1:5000/health`; capture its actual
  JSON response.
- [ ] Model metadata — request `/api/model-info` and `/api/metrics`; capture
  the verified dataset, split, and computed report values.
- [ ] Pytest results — run `python -m pytest -q`; capture the actual summary.

### Practical 2

- [ ] Docker version — run `docker --version`; capture the installed version.
- [ ] Docker image build — run
  `docker build -t sms-spam-app:1.0 .`; capture successful completion.
- [ ] Docker images — run `docker images`; capture `sms-spam-app:1.0`.
- [ ] Running container — run
  `docker run -d --name sms-spam-container -p 5000:5000 sms-spam-app:1.0`
  and `docker ps`; capture the running container and port mapping. Capture
  `docker inspect sms-spam-container` showing its health status.
- [ ] Application in browser — open `http://127.0.0.1:5000`; capture the
  container-served interface.
- [ ] Successful API response — POST a message to
  `http://127.0.0.1:5000/prediction`; capture the actual JSON response. Also
  capture `/api/model-info` and `/api/metrics` from the container.
- [ ] Logs — run `docker logs sms-spam-container`; capture the container logs.
- [ ] Container lifecycle commands — capture `docker inspect
  sms-spam-container`, `docker stop sms-spam-container`,
  `docker start sms-spam-container`, and `docker rm -f sms-spam-container`
  executed against this project container only.

### Practical 3

- [ ] EC2 configuration — capture the actual Ubuntu instance setup, region,
  instance type, storage, and key-pair selection (never expose the private key).
- [ ] Security group — capture inbound SSH restricted to your IP and the
  application port restricted to trusted clients when practical.
- [ ] SSH session — capture a successful terminal connection with the public IP
  obscured if required by your submission policy.
- [ ] Docker installation — capture `sudo docker --version` and successful
  `sudo docker info`.
- [ ] Image transfer/build — capture the actual Docker image transfer and
  `sudo docker load` output.
- [ ] Running container — capture `sudo docker ps` and the app port mapping.
- [ ] Browser access — capture the actual EC2 public URL serving the application.
- [ ] Successful prediction — capture the actual prediction API response from
  the deployed EC2 system.

### Practical 4

- [ ] GCP project — capture the selected project and Compute Engine API status.
- [ ] VM configuration — capture the actual Ubuntu VM name, zone, machine type,
  and boot disk.
- [ ] Firewall rule — capture the actual TCP 5000 rule, target, and allowed
  source CIDR.
- [ ] SSH session — capture a connected browser-based SSH terminal.
- [ ] Docker installation — capture `sudo docker --version` and successful
  `sudo docker info`.
- [ ] Image transfer/build — capture the image transfer and `sudo docker load`
  output on the VM.
- [ ] Running container — capture `sudo docker ps` and the app port mapping.
- [ ] Browser access — capture the actual GCP external URL serving the app.
- [ ] Successful prediction — capture the actual prediction API response from
  the deployed GCP system.
