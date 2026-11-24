const CLASS_CONFIG = [
  { name: "Apple", color: "#fb7185" },
  { name: "Lemon", color: "#facc15" },
  { name: "Potato", color: "#c084fc" },
  { name: "Tomato", color: "#f97316" },
  { name: "Carrot", color: "#38bdf8" },
];

const modelSelect = document.getElementById("modelSelect");
const modelPill = document.getElementById("modelPill");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileMeta = document.getElementById("fileMeta");
const controlDeck = document.querySelector(".control-deck");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const confPill = document.getElementById("confPill");
const predictBtn = document.getElementById("predictBtn");
const btnSpinner = document.getElementById("btnSpinner");
const btnText = document.getElementById("btnText");
const alertBanner = document.getElementById("alertBanner");
const stageBox = document.getElementById("stageBox");
const emptyState = document.getElementById("emptyState");
const stageImg = document.getElementById("stageImg");
const viewSwitch = document.getElementById("viewSwitch");
const switchButtons = Array.from(document.querySelectorAll(".switch-btn"));
const speedPill = document.getElementById("speedPill");
const totalCountEl = document.getElementById("totalCount");
const heroSubEl = document.getElementById("heroSub");
const classCardsEl = document.getElementById("classCards");
const detStreamEl = document.getElementById("detStream");
const detBadgeEl = document.getElementById("detBadge");
const resetBtn = document.getElementById("resetBtn");
const cameraStartBtn = document.getElementById("cameraStartBtn");
const cameraStopBtn = document.getElementById("cameraStopBtn");
const cameraVideo = document.getElementById("cameraVideo");
const cameraCanvas = document.getElementById("cameraCanvas");

let selectedFile = null;
let originalDataUrl = null;
let annotatedDataUrl = null;
let activeViewMode = "annotated";
let cameraStream = null;
let cameraTimer = null;

function showError(msg) {
  alertBanner.textContent = msg;
  alertBanner.hidden = false;
}

function clearError() {
  alertBanner.textContent = "";
  alertBanner.hidden = true;
}

function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  const kb = bytes / 1024;
  return kb < 1024 ? `${kb.toFixed(1)} KB` : `${(kb / 1024).toFixed(2)} MB`;
}

function buildClassCards(counts = {}, total = 0) {
  const orderedNames = [
    ...CLASS_CONFIG.map((c) => c.name),
    ...Object.keys(counts).filter((k) => !CLASS_CONFIG.some((c) => c.name === k)),
  ];

  classCardsEl.innerHTML = orderedNames
    .map((name) => {
      const cfg = CLASS_CONFIG.find((c) => c.name === name) || { color: "#38bdf8" };
      const count = Number(counts[name] ?? 0);
      const pct = total > 0 ? Math.min(100, Math.round((count / total) * 100)) : 0;
      const activeCls = count > 0 ? "cls-card active" : "cls-card";

      return `
        <div class="${activeCls}" style="--cls-color: ${cfg.color};">
          <div class="cls-card-top">
            <div class="cls-left">
              <span class="cls-swatch"></span>
              <span class="cls-title">${name}</span>
            </div>
            <span class="cls-count">${count} adet</span>
          </div>
          <div class="cls-bar-track">
            <div class="cls-bar-fill" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderDetections(detections = []) {
  detBadgeEl.textContent = `${detections.length} kutu`;

  if (!detections.length) {
    detStreamEl.innerHTML = `<div class="det-empty">Seçili güven eşiğinde nesne bulunamadı. Eşiği düşürmeyi deneyebilirsin.</div>`;
    return;
  }

  detStreamEl.innerHTML = detections
    .map(
      (item) => `
      <div class="det-item">
        <span class="det-name">#${item.id} • ${item.class_name}</span>
        <span class="det-conf">%${Number(item.confidence).toFixed(1)}</span>
      </div>
    `
    )
    .join("");
}

function setViewMode(mode) {
  activeViewMode = mode;
  switchButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });

  if (mode === "original" && originalDataUrl) {
    stageImg.src = originalDataUrl;
  } else if (annotatedDataUrl) {
    stageImg.src = annotatedDataUrl;
  } else if (originalDataUrl) {
    stageImg.src = originalDataUrl;
  }
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading || !selectedFile;
  modelSelect.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnText.textContent = isLoading ? "GPU Analiz Yapıyor..." : "Analizi Başlat";
  stageBox.classList.toggle("scanning", isLoading);
}


async function startCamera() {
  clearError();
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Bu tarayıcı kamera erişimini desteklemiyor.");
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.hidden = false;
    cameraStartBtn.disabled = true;
    cameraStopBtn.disabled = false;
    heroSubEl.textContent = "Canlı kamera açık";
    cameraTimer = window.setInterval(captureCameraFrame, 1800);
  } catch (err) {
    showError(`Kamera açılamadı: ${err.message}`);
  }
}

function stopCamera() {
  if (cameraTimer) window.clearInterval(cameraTimer);
  cameraTimer = null;
  if (cameraStream) cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraVideo.srcObject = null;
  cameraVideo.hidden = true;
  cameraStartBtn.disabled = false;
  cameraStopBtn.disabled = true;
}

function captureCameraFrame() {
  if (!cameraStream || !cameraVideo.videoWidth) return;
  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  const ctx = cameraCanvas.getContext("2d");
  ctx.drawImage(cameraVideo, 0, 0);
  cameraCanvas.toBlob((blob) => {
    if (!blob) return;
    selectedFile = new File([blob], "camera_frame.jpg", { type: "image/jpeg" });
    runPrediction();
  }, "image/jpeg", 0.86);
}

async function runPrediction() {
  if (!selectedFile) return;

  clearError();
  setLoading(true);

  const chosenModel = modelSelect.value;
  modelPill.textContent = `Model: ${chosenModel}`;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("conf", confSlider.value);
  formData.append("model_name", chosenModel);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      showError(data.error || "Analiz sırasında bir hata oluştu.");
      return;
    }

    annotatedDataUrl = data.image_base64;
    viewSwitch.hidden = false;
    setViewMode("annotated");

    if (data.model_used) {
      modelSelect.value = data.model_used;
      modelPill.textContent = `Model: ${data.model_used}`;
    }

    speedPill.textContent = `GPU Süre: ${data.elapsed_ms} ms`;
    totalCountEl.textContent = data.total_items;
    heroSubEl.textContent = `${data.model_used} • Eşik ${Number(data.conf_used).toFixed(2)}`;

    buildClassCards(data.counts, data.total_items);
    renderDetections(data.detections);
  } catch (err) {
    showError(`Sunucu bağlantı hatası: ${err.message}`);
  } finally {
    setLoading(false);
  }
}

function handleFileSelection(file) {
  clearError();
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    showError("Lütfen geçerli bir görsel dosyası seçin.");
    return;
  }

  selectedFile = file;
  annotatedDataUrl = null;
  viewSwitch.hidden = true;
  predictBtn.disabled = false;
  fileMeta.textContent = `${file.name} (${formatBytes(file.size)})`;

  const reader = new FileReader();
  reader.onload = (ev) => {
    originalDataUrl = ev.target.result;
    emptyState.hidden = true;
    stageImg.hidden = false;
    setViewMode("original");
    heroSubEl.textContent = "Analiz için hazır";
  };
  reader.readAsDataURL(file);
}

modelSelect.addEventListener("change", () => {
  modelPill.textContent = `Model: ${modelSelect.value}`;
  if (selectedFile) {
    runPrediction();
  }
});

confSlider.addEventListener("input", () => {
  const formatted = parseFloat(confSlider.value).toFixed(2);
  confValue.textContent = formatted;
  confPill.textContent = `Eşik: ${formatted}`;
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", (e) => {
  handleFileSelection(e.target.files?.[0]);
});

["dragenter", "dragover"].forEach((evtName) => {
  dropzone.addEventListener(evtName, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((evtName) => {
  dropzone.addEventListener(evtName, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  });
});

dropzone.addEventListener("drop", (e) => {
  const droppedFile = e.dataTransfer?.files?.[0];
  handleFileSelection(droppedFile);
});

switchButtons.forEach((btn) => {
  btn.addEventListener("click", () => setViewMode(btn.dataset.mode));
});

predictBtn.addEventListener("click", runPrediction);
cameraStartBtn?.addEventListener("click", startCamera);
cameraStopBtn?.addEventListener("click", stopCamera);

resetBtn.addEventListener("click", () => {
  stopCamera();
  clearError();
  selectedFile = null;
  originalDataUrl = null;
  annotatedDataUrl = null;
  fileInput.value = "";
  fileMeta.textContent = "JPG, PNG, WEBP desteklenir • Henüz görsel seçilmedi";
  predictBtn.disabled = true;
  viewSwitch.hidden = true;
  stageImg.hidden = true;
  stageImg.removeAttribute("src");
  emptyState.hidden = false;
  speedPill.textContent = "GPU Süre: —";
  totalCountEl.textContent = "0";
  heroSubEl.textContent = "Görsel bekleniyor";

  const defaultConf = parseFloat(controlDeck.dataset.defaultConf || "0.35").toFixed(2);
  confSlider.value = defaultConf;
  confValue.textContent = defaultConf;
  confPill.textContent = `Eşik: ${defaultConf}`;

  buildClassCards({}, 0);
  renderDetections([]);
});

buildClassCards({}, 0);

