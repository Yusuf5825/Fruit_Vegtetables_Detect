import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODELS_DIR = Path(os.environ.get("MODEL_DIR", PROJECT_DIR / "models")).expanduser().resolve()
DEFAULT_CONF = 0.35
IMGSZ = 640
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_LABEL = (
    f"{torch.cuda.get_device_name(0)} (CUDA)"
    if torch.cuda.is_available()
    else "CPU"
)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def discover_models(models_dir: Path) -> dict[str, Path]:
    """Find model weights stored as models/<run_name>/best.pt."""
    found: dict[str, Path] = {}
    if not models_dir.exists():
        return found
    for pt_path in sorted(models_dir.rglob("best.pt")):
        if pt_path.is_file():
            rel = pt_path.relative_to(models_dir)
            label = str(rel.parent) if str(rel.parent) != "." else pt_path.stem
            found[label] = pt_path.resolve()
    return found


AVAILABLE_MODELS = discover_models(MODELS_DIR)
LOADED_MODELS: dict[str, YOLO] = {}
DEFAULT_MODEL_NAME = (
    "fruit_vegetables_veri2_v1"
    if "fruit_vegetables_veri2_v1" in AVAILABLE_MODELS
    else next(iter(AVAILABLE_MODELS), "")
)


def get_model(model_key: str | None) -> tuple[str, YOLO]:
    """Load and cache a selected model on the best available device."""
    current_registry = discover_models(MODELS_DIR)
    if not current_registry:
        raise FileNotFoundError(
            f"Model ağırlığı bulunamadı. best.pt dosyasını '{MODELS_DIR}' altına ekleyin."
        )

    selected_key = model_key if model_key in current_registry else DEFAULT_MODEL_NAME
    if selected_key not in current_registry:
        selected_key = next(iter(current_registry))

    if selected_key not in LOADED_MODELS:
        model = YOLO(str(current_registry[selected_key]))
        model.to(DEVICE)
        LOADED_MODELS[selected_key] = model
    return selected_key, LOADED_MODELS[selected_key]


@app.route("/")
def index():
    model_names = list(discover_models(MODELS_DIR))
    active_name = DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in model_names else ""
    return render_template(
        "index.html",
        gpu_label=DEVICE_LABEL,
        default_conf=DEFAULT_CONF,
        model_names=model_names,
        active_model=active_name,
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Lütfen bir fotoğraf dosyası seçin."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Dosya adı boş olamaz."}), 400
    if Path(file.filename).suffix.lower() not in ALLOWED_EXTS:
        return jsonify({"error": "Desteklenmeyen format. JPG, PNG veya WEBP yükleyin."}), 400

    try:
        conf = float(request.form.get("conf", DEFAULT_CONF))
    except (TypeError, ValueError):
        conf = DEFAULT_CONF
    conf = max(0.05, min(0.95, conf))
    requested_model = request.form.get("model_name", "").strip() or None

    try:
        used_model_name, model = get_model(requested_model)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    raw_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"error": "Görsel okunamadı veya bozuk."}), 400

    start = time.perf_counter()
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False,
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 1)

    result = results[0]
    annotated_bgr = result.plot()
    ok_enc, buffer = cv2.imencode(
        ".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    )
    if not ok_enc:
        return jsonify({"error": "Çıktı görseli encode edilemedi."}), 500

    image_b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    class_names = model.names
    counts = {name: 0 for name in class_names.values()}
    detections = []
    if result.boxes is not None:
        for index, box in enumerate(result.boxes, start=1):
            class_id = int(box.cls[0].item())
            class_name = class_names.get(class_id, str(class_id))
            confidence = round(float(box.conf[0].item()) * 100.0, 1)
            counts[class_name] = counts.get(class_name, 0) + 1
            detections.append(
                {"id": index, "class_name": class_name, "confidence": confidence}
            )

    return jsonify(
        {
            "image_base64": f"data:image/jpeg;base64,{image_b64}",
            "counts": counts,
            "total_items": sum(counts.values()),
            "detections": detections,
            "elapsed_ms": elapsed_ms,
            "conf_used": conf,
            "model_used": used_model_name,
        }
    )


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
