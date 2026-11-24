import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou
from ultralytics import YOLO
import yaml

# ==============================================================================
# AYARLAR
# ==============================================================================
ROOT = Path(r"C:\Projeler\Fruits_detect")
TEST_DIR = ROOT / "kendi_test_verim"
MODELS_DIR = ROOT / "models"
OUT_JSON_DIR = ROOT / "test_sonuclari_json"

CLASSES = {0: "Apple", 1: "Lemon", 2: "Potato", 3: "Tomato", 4: "Carrot"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.50


def prepare_test_yaml(test_dir: Path, out_dir: Path) -> Path:
    """kendi_test_verim klasörüne dokunmadan geçici bir test_eval.yaml üretir."""
    imgs_dir = test_dir / "images"
    if not imgs_dir.exists():
        raise FileNotFoundError(f"Görsel klasörü bulunamadı: {imgs_dir}")

    # images altında train/val/test alt klasörü varsa onu kullan, yoksa doğrudan images klasörünü ver
    has_split_subdirs = any((imgs_dir / s).is_dir() for s in ("train", "val", "test"))
    test_entry = "images/test" if (imgs_dir / "test").is_dir() else ("images" if not has_split_subdirs else "images/val")

    yaml_data = {
        "path": str(test_dir.resolve()),
        "train": test_entry,
        "val": test_entry,
        "test": test_entry,
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = out_dir / "_kendi_test_eval.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    return yaml_path


def find_image_files(test_dir: Path) -> list[Path]:
    imgs_root = test_dir / "images"
    if (imgs_root / "test").is_dir():
        search_root = imgs_root / "test"
    else:
        search_root = imgs_root
    return [
        p for p in sorted(search_root.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def load_gt_boxes_xyxy(lbl_path: Path, img_w: int, img_h: int) -> tuple[torch.Tensor, np.ndarray]:
    if not lbl_path.is_file():
        return torch.empty((0, 4), dtype=torch.float32), np.empty((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    cls_ids: list[int] = []
    for raw in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
        cls_ids.append(cid)

    if not boxes:
        return torch.empty((0, 4), dtype=torch.float32), np.empty((0,), dtype=np.int64)
    return torch.tensor(boxes, dtype=torch.float32), np.array(cls_ids, dtype=np.int64)


def compute_ious(
    model: YOLO,
    img_files: list[Path],
    test_dir: Path,
    device_id: int,
) -> tuple[float | None, dict[int, float | None]]:
    """Her görsel için aynı sınıftaki gerçek-tahmin kutularının en iyi IoU eşleşmelerini toplar."""
    all_matched_ious: list[float] = []
    per_class_ious: dict[int, list[float]] = {cid: [] for cid in CLASSES}

    for img_path in img_files:
        # Etiket yolunu bul (images/.../x.jpg -> labels/.../x.txt)
        try:
            rel = img_path.relative_to(test_dir / "images")
            lbl_path = (test_dir / "labels" / rel).with_suffix(".txt")
        except ValueError:
            lbl_path = test_dir / "labels" / f"{img_path.stem}.txt"

        preds = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMGSZ,
            device=device_id,
            verbose=False,
        )
        res = preds[0]
        img_h, img_w = res.orig_shape

        gt_boxes, gt_cls = load_gt_boxes_xyxy(lbl_path, img_w, img_h)
        if len(gt_boxes) == 0:
            continue

        if res.boxes is None or len(res.boxes) == 0:
            continue

        pred_boxes = res.boxes.xyxy.detach().cpu().float()
        pred_cls = res.boxes.cls.detach().cpu().numpy().astype(np.int64)

        iou_matrix = box_iou(gt_boxes, pred_boxes).numpy()  # [num_gt, num_pred]

        for g_idx in range(len(gt_boxes)):
            g_c = int(gt_cls[g_idx])
            same_cls_pred_indices = np.where(pred_cls == g_c)[0]
            if len(same_cls_pred_indices) == 0:
                continue
            best_iou = float(iou_matrix[g_idx, same_cls_pred_indices].max())
            all_matched_ious.append(best_iou)
            if g_c in per_class_ious:
                per_class_ious[g_c].append(best_iou)

    overall_mean_iou = round(float(np.mean(all_matched_ious)), 4) if all_matched_ious else None
    class_mean_iou = {
        cid: (round(float(np.mean(vals)), 4) if vals else None)
        for cid, vals in per_class_ious.items()
    }
    return overall_mean_iou, class_mean_iou


def safe_float(val, digits: int = 4) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    return round(float(val), digits)


def evaluate_model(pt_path: Path, yaml_path: Path, img_files: list[Path], device_id: int) -> dict:
    t0 = time.perf_counter()
    model = YOLO(str(pt_path))
    model.to(f"cuda:{device_id}")

    metrics = model.val(
        data=str(yaml_path),
        split="test",
        imgsz=IMGSZ,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=device_id,
        plots=False,
        save=False,
        verbose=False,
    )

    overall_iou, class_ious = compute_ious(model, img_files, TEST_DIR, device_id)
    elapsed_sec = round(time.perf_counter() - t0, 2)

    box = metrics.box
    prec_macro = safe_float(box.mp)
    rec_macro = safe_float(box.mr)
    f1_macro = safe_float(np.mean(box.f1) if hasattr(box.f1, "__len__") else box.f1)
    map50_macro = safe_float(box.map50)
    map50_95_macro = safe_float(box.map)

    per_class_report: dict[str, dict] = {}
    raw_ap_cls = getattr(box, "ap_class_index", None)
    if raw_ap_cls is None:
        ap_class_indices: list[int] = []
    elif hasattr(raw_ap_cls, "tolist"):
        ap_class_indices = [int(x) for x in raw_ap_cls.tolist()]
    else:
        ap_class_indices = [int(x) for x in raw_ap_cls]

    class_metrics_map: dict[int, tuple[float, float, float, float, float]] = {}
    for idx, cid_int in enumerate(ap_class_indices):
        p_val = safe_float(box.p[idx]) if idx < len(box.p) else 0.0
        r_val = safe_float(box.r[idx]) if idx < len(box.r) else 0.0
        f1_val = safe_float(box.f1[idx]) if idx < len(box.f1) else 0.0
        ap50_val = safe_float(box.ap50[idx]) if idx < len(box.ap50) else 0.0
        ap50_95_val = safe_float(box.ap[idx]) if idx < len(box.ap) else 0.0
        class_metrics_map[cid_int] = (p_val, r_val, f1_val, ap50_val, ap50_95_val)

    for cid, cname in CLASSES.items():
        p_v, r_v, f1_v, ap50_v, ap50_95_v = class_metrics_map.get(cid, (0.0, 0.0, 0.0, 0.0, 0.0))
        per_class_report[cname] = {
            "class_id": cid,
            "mAP50": ap50_v,
            "mAP50_95": ap50_95_v,
            "precision": p_v,
            "recall": r_v,
            "f1": f1_v,
            "mean_matched_iou": class_ious.get(cid),
        }

    return {
        "model_name": pt_path.parent.name,
        "weights_path": str(pt_path.resolve()),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_sec": elapsed_sec,
        "settings": {
            "device": f"cuda:{device_id}",
            "imgsz": IMGSZ,
            "conf": CONF_THRES,
            "iou": IOU_THRES,
            "test_dataset": str(TEST_DIR.resolve()),
        },
        "overall_metrics": {
            "mAP50": map50_macro,
            "mAP50_95": map50_95_macro,
            "precision": prec_macro,
            "recall": rec_macro,
            "f1": f1_macro,
            "mean_matched_iou": overall_iou,
        },
        "per_class_metrics": per_class_report,
    }



def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı! Değerlendirme GPU üzerinde çalışacak şekilde ayarlandı.")
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models klasörü bulunamadı: {MODELS_DIR}")

    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = prepare_test_yaml(TEST_DIR, OUT_JSON_DIR)
    img_files = find_image_files(TEST_DIR)

    pt_files = sorted(MODELS_DIR.rglob("best.pt"))
    if not pt_files:
        raise FileNotFoundError(f"{MODELS_DIR} altında hiç best.pt bulunamadı.")

    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    print(f"GPU: {gpu_name} | Bulunan model sayısı: {len(pt_files)} | Test görseli: {len(img_files)}")

    ozet_liste: list[dict] = []

    for pt_path in pt_files:
        print(f"Test ediliyor -> {pt_path.parent.name} ({pt_path})")
        report = evaluate_model(pt_path, yaml_path, img_files, device_id)

        model_json_path = OUT_JSON_DIR / f"{pt_path.parent.name}_test_sonuc.json"
        with model_json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        om = report["overall_metrics"]
        print(
            f"  mAP@50: {om['mAP50']:.4f} | mAP@50-95: {om['mAP50_95']:.4f} | "
            f"P: {om['precision']:.4f} | R: {om['recall']:.4f} | F1: {om['f1']:.4f} | "
            f"Mean IoU: {om['mean_matched_iou']}"
        )

        ozet_liste.append({
            "model_name": report["model_name"],
            "json_file": model_json_path.name,
            "overall_metrics": om,
        })

    # Tüm modelleri tek bakışta karşılaştırmak için özet JSON da kaydedilir
    ozet_path = OUT_JSON_DIR / "_tum_modeller_ozet.json"
    with ozet_path.open("w", encoding="utf-8") as f:
        json.dump(ozet_liste, f, indent=2, ensure_ascii=False)

    print(f"\nTamamlandı. Tüm JSON dosyaları şurada: {OUT_JSON_DIR}")


if __name__ == "__main__":
    main()
