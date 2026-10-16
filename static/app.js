(() => {
  const form = document.querySelector("#prediction-form");
  const messageField = document.querySelector("#message");
  const sampleSelect = document.querySelector("#sample-select");
  const counter = document.querySelector("#character-count");
  const errorBox = document.querySelector("#form-error");
  const classifyButton = document.querySelector("#classify-button");
  const clearButton = document.querySelector("#clear-button");
  const buttonText = classifyButton.querySelector(".button-text");
  const resultCard = document.querySelector("#result-card");
  const resultPlaceholder = document.querySelector("#result-placeholder");
  const resultContent = document.querySelector("#result-content");
  const healthStatus = document.querySelector("#health-status");
  const healthLabel = document.querySelector("#health-label");

  const updateCounter = () => {
    counter.textContent = `${messageField.value.length.toLocaleString()} / 10,000`;
  };

  const showError = (message) => {
    errorBox.textContent = message;
    errorBox.hidden = false;
    messageField.setAttribute("aria-invalid", "true");
  };

  const clearError = () => {
    errorBox.textContent = "";
    errorBox.hidden = true;
    messageField.removeAttribute("aria-invalid");
  };

  const setBar = (name, value) => {
    const percent = Math.max(0, Math.min(100, value * 100));
    document.querySelector(`#${name}-value`).textContent = `${percent.toFixed(1)}%`;
    const bar = document.querySelector(`#${name}-bar`);
    bar.setAttribute("aria-valuenow", String(Math.round(percent)));
    bar.querySelector("span").style.width = `${percent}%`;
  };

  const displayResult = (result) => {
    const isSpam = result.label === "spam";
    resultCard.dataset.class = isSpam ? "spam" : "ham";
    document.querySelector("#result-label").textContent = isSpam ? "SPAM" : "HAM";
    document.querySelector("#result-meaning").textContent = isSpam
      ? "The message is classified as unwanted or spam-like."
      : "The message is classified as legitimate, non-spam correspondence.";
    document.querySelector("#prediction-time").textContent =
      `Model time: ${Number(result.prediction_time_ms).toFixed(3)} ms`;
    document.querySelector("#confidence-note").textContent =
      result.confidence_note ||
      "Model probability estimates are not guarantees and may be uncalibrated.";

    const probabilities = result.probabilities;
    const probabilityPanel = document.querySelector("#probability-panel");
    if (
      probabilities &&
      Number.isFinite(probabilities.ham) &&
      Number.isFinite(probabilities.spam)
    ) {
      setBar("ham", probabilities.ham);
      setBar("spam", probabilities.spam);
      probabilityPanel.hidden = false;
    } else {
      probabilityPanel.hidden = true;
    }
    resultPlaceholder.hidden = true;
    resultContent.hidden = false;
    resultCard.focus({ preventScroll: true });
  };

  messageField.addEventListener("input", () => {
    updateCounter();
    clearError();
  });

  sampleSelect.addEventListener("change", () => {
    if (sampleSelect.value) {
      messageField.value = sampleSelect.value;
      updateCounter();
      clearError();
      messageField.focus();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();
    const message = messageField.value.trim();
    if (!message) {
      showError("Enter a message before classification.");
      messageField.focus();
      return;
    }
    if (message.length > 10_000) {
      showError("The message must be at most 10,000 characters.");
      messageField.focus();
      return;
    }

    classifyButton.disabled = true;
    classifyButton.classList.add("is-loading");
    classifyButton.setAttribute("aria-busy", "true");
    buttonText.textContent = "Analyzing…";
    try {
      const response = await fetch("/prediction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body.error || "The message could not be classified.");
      }
      displayResult(body);
    } catch (error) {
      showError(
        error instanceof Error
          ? error.message
          : "Unable to contact the classifier. Please try again.",
      );
    } finally {
      classifyButton.disabled = false;
      classifyButton.classList.remove("is-loading");
      classifyButton.removeAttribute("aria-busy");
      buttonText.textContent = "Analyze message";
    }
  });

  clearButton.addEventListener("click", () => {
    window.setTimeout(() => {
      updateCounter();
      clearError();
      sampleSelect.value = "";
      resultCard.removeAttribute("data-class");
      resultContent.hidden = true;
      resultPlaceholder.hidden = false;
    });
  });

  const checkHealth = async () => {
    try {
      const response = await fetch("/health", { cache: "no-store" });
      const health = await response.json();
      const ready = response.ok && health.model_loaded === true;
      healthStatus.dataset.state = ready ? "ready" : "unavailable";
      healthLabel.textContent = ready ? "Model ready" : "Model unavailable";
    } catch {
      healthStatus.dataset.state = "unavailable";
      healthLabel.textContent = "Status unavailable";
    }
  };

  updateCounter();
  checkHealth();
})();
