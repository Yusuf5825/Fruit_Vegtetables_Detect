import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

# ==============================================================================
# AYARLAR
# ==============================================================================
MODEL_PATH = Path(r"C:/Projeler/Fruits_detect/models/fruit_vegetables_veri4_v1/best.pt")
CAM_INDEX = 0
CONF_THRES = 0.50
IMGSZ = 800


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model bulunamadı: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı!")

    torch.backends.cudnn.benchmark = True
    gpu_name = torch.cuda.get_device_name(0)

    model = YOLO(str(MODEL_PATH))
    model.to("cuda:0")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı!")

    print(f"GPU Aktif: {gpu_name} | Çıkmak için 'q' tuşuna bas.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            t0 = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=CONF_THRES,
                imgsz=IMGSZ,
                device=0,
                verbose=False,
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / dt_ms if dt_ms > 0 else 0.0

            res = results[0]
            annotated = res.plot()

            found: list[str] = []
            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = res.names.get(cls_id, str(cls_id))
                    conf_val = float(box.conf[0].item()) * 100.0
                    found.append(f"{cls_name} ({conf_val:.0f}%)")

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            det_text = ", ".join(found) if found else "Tespit yok"
            print(f"[{ts}] {dt_ms:.1f} ms ({fps:.1f} FPS) -> {det_text}", flush=True)

            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | conf: {CONF_THRES}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Kamera Tespit (GPU)", annotated)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()