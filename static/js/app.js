const form = document.getElementById("predict-form");
const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const statusLine = document.getElementById("status-line");
const predictButton = document.getElementById("predict-button");
const previewWrap = document.getElementById("preview-wrap");
const previewImage = document.getElementById("preview-image");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");
const predictedEmotion = document.getElementById("predicted-emotion");
const predictedConfidence = document.getElementById("predicted-confidence");
const confidenceBars = document.getElementById("confidence-bars");

const emotionOrder = ["Calm", "Stressed", "Angry", "Focused"];

function fileFromEvent(event) {
  if (event.dataTransfer?.files?.length) {
    return event.dataTransfer.files[0];
  }
  if (event.target?.files?.length) {
    return event.target.files[0];
  }
  return null;
}

function setStatus(message, isError = false) {
  statusLine.textContent = message;
  statusLine.style.color = isError ? "#a1262a" : "#42564a";
}

function setPreview(file) {
  if (!file) {
    previewWrap.classList.add("hidden");
    previewImage.removeAttribute("src");
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  previewImage.src = objectUrl;
  previewWrap.classList.remove("hidden");
}

function renderBars(probabilities) {
  confidenceBars.innerHTML = "";

  emotionOrder.forEach((emotion) => {
    const score = Number(probabilities[emotion] || 0);
    const percent = Math.round(score * 100);

    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.textContent = emotion;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${percent}%`;

    const value = document.createElement("span");
    value.textContent = `${percent}%`;

    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    confidenceBars.appendChild(row);
  });
}

async function submitPrediction(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/predict", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    const message = payload.message || "Prediction failed.";
    throw new Error(message);
  }

  return payload;
}

function applyResult(result) {
  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");

  predictedEmotion.textContent = result.predicted_emotion;
  predictedConfidence.textContent = `Confidence: ${Math.round(result.confidence * 100)}%`;
  renderBars(result.probabilities || {});
}

fileInput.addEventListener("change", (event) => {
  const file = fileFromEvent(event);
  setPreview(file);
  setStatus(file ? `Selected: ${file.name}` : "", false);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove("drag-active");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = fileFromEvent(event);
  if (!file) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setPreview(file);
  setStatus(`Selected: ${file.name}`);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = fileInput.files[0];

  if (!file) {
    setStatus("Please choose an image first.", true);
    return;
  }

  try {
    predictButton.disabled = true;
    setStatus("Running CNN inference...");
    const result = await submitPrediction(file);
    applyResult(result);
    setStatus(`Prediction complete: ${result.predicted_emotion}`);
  } catch (error) {
    setStatus(error.message || "Prediction failed.", true);
  } finally {
    predictButton.disabled = false;
  }
});
