from pathlib import Path

ROOT = Path(r"C:\Projeler\Fruits_detect")
for name in ("yolo_fruits_export", "coco_out", "tomato_dataset", "merged_dataset"):
    d = ROOT / name
    if not d.exists():
        print(f"{name:20s} -> klasör yok")
        continue
    imgs = sum(1 for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    txts = sum(1 for p in d.rglob("*.txt") if p.name != "classes.txt")
    print(f"{name:20s} -> görsel: {imgs}, etiket (.txt): {txts}")