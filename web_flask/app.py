import base64
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

# ==============================================================================
# AYARLAR
# ==============================================================================
ROOT_DIR = Path(r"C:\Projeler\Fruits_detect")
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_CONF = 0.35
IMGSZ = 640
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def discover_models(models_dir: Path) -> dict[str, Path]:
    """models klasörü altındaki tüm best.pt ağırlıklarını bulur."""
    found: dict[str, Path] = {}
    if not models_dir.exists():
        return found

    for pt_path in sorted(models_dir.rglob("best.pt")):
        if not pt_path.is_file():
            continue
        rel = pt_path.relative_to(models_dir)
        # Örn: fruit_vegetables_veri2_v1/best.pt -> fruit_vegetables_veri2_v1
        label = str(rel.parent) if str(rel.parent) != "." else pt_path.stem
        found[label] = pt_path.resolve()

    return found


def init_gpu() -> tuple[int, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı! GPU aktif bir ortam gerekli.")
    torch.backends.cudnn.benchmark = True
    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    return device_id, f"{gpu_name} (cuda:0)"


DEVICE_ID, GPU_LABEL = init_gpu()
AVAILABLE_MODELS: dict[str, Path] = discover_models(MODELS_DIR)
LOADED_MODELS: dict[str, YOLO] = {}


def get_model(model_key: str | None) -> tuple[str, YOLO]:
    """İstenen modeli belleğe alır (önbellekli) ve döndürür."""
    current_registry = discover_models(MODELS_DIR)
    if not current_registry:
        raise FileNotFoundError(f"'{MODELS_DIR}' altında hiç best.pt modeli bulunamadı.")

    selected_key = model_key if model_key in current_registry else None
    if selected_key is None:
        preferred = "fruit_vegetables_veri2_v1"
        selected_key = preferred if preferred in current_registry else next(iter(current_registry.keys()))

    pt_path = current_registry[selected_key]

    if selected_key not in LOADED_MODELS:
        model = YOLO(str(pt_path))
        model.to(f"cuda:{DEVICE_ID}")
        dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
        model.predict(source=dummy, imgsz=IMGSZ, device=DEVICE_ID, verbose=False)
        LOADED_MODELS[selected_key] = model

    return selected_key, LOADED_MODELS[selected_key]


# Başlangıçta varsayılan modeli ısıt
DEFAULT_MODEL_NAME, _ = get_model("fruit_vegetables_veri2_v1")


@app.route("/")
def index():
    model_names = list(discover_models(MODELS_DIR).keys())
    active_name = DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in model_names else (model_names[0] if model_names else "")
    return render_template(
        "index.html",
        gpu_label=GPU_LABEL,
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

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": "Desteklenmeyen format. JPG, PNG veya WEBP yükleyin."}), 400

    try:
        conf = float(request.form.get("conf", DEFAULT_CONF))
    except ValueError:
        conf = DEFAULT_CONF
    conf = max(0.05, min(0.95, conf))

    requested_model = request.form.get("model_name", "").strip() or None

    try:
        used_model_name, model = get_model(requested_model)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404

    raw_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return jsonify({"error": "Görsel okunamadı veya bozuk."}), 400

    t0 = time.perf_counter()
    results = model.predict(
        source=frame_bgr,
        conf=conf,
        imgsz=IMGSZ,
        device=DEVICE_ID,
        verbose=False,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

    res = results[0]
    annotated_bgr = res.plot()

    ok_enc, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok_enc:
        return jsonify({"error": "Çıktı görseli encode edilemedi."}), 500

    img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    counts: dict[str, int] = {name: 0 for name in model.names.values()}
    detections: list[dict] = []

    if res.boxes is not None and len(res.boxes) > 0:
        for idx, box in enumerate(res.boxes, start=1):
            cls_id = int(box.cls[0].item())
            cls_name = model.names.get(cls_id, str(cls_id))
            score = round(float(box.conf[0].item()) * 100.0, 1)
            counts[cls_name] = counts.get(cls_name, 0) + 1
            detections.append({
                "id": idx,
                "class_name": cls_name,
                "confidence": score,
            })

    total_items = sum(counts.values())

    return jsonify({
        "image_base64": f"data:image/jpeg;base64,{img_b64}",
        "counts": counts,
        "total_items": total_items,
        "detections": detections,
        "elapsed_ms": elapsed_ms,
        "conf_used": conf,
        "model_used": used_model_name,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
