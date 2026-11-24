import random
import shutil
from pathlib import Path
import yaml

# Sabit seed sayesinde bölünme tekrarlanabilir olur
RANDOM_SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

BASE_DIR = Path(r"C:/Projeler/Fruits_detect/yolo_fruits_export")
IMAGES_ROOT = BASE_DIR / "images"
LABELS_ROOT = BASE_DIR / "labels"
YAML_PATH = BASE_DIR / "dataset.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_pairs() -> list[tuple[Path, Path | None]]:
    """Mevcut image ve varsa karşılık gelen label dosyalarını toplar."""
    image_files = [
        p for p in IMAGES_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    pairs: list[tuple[Path, Path | None]] = []
    for img_path in image_files:
        rel_from_images = img_path.relative_to(IMAGES_ROOT)
        candidate_label = (LABELS_ROOT / rel_from_images).with_suffix(".txt")

        if candidate_label.exists():
            lbl_path = candidate_label
        else:
            # Farklı alt klasörde (ör. labels/val/xxx.txt) olma ihtimaline karşı stem ile ara
            matches = list(LABELS_ROOT.rglob(f"{img_path.stem}.txt"))
            lbl_path = matches[0] if matches else None

        pairs.append((img_path, lbl_path))

    return pairs


def prepare_split_dirs() -> None:
    for split in ("train", "val", "test"):
        (IMAGES_ROOT / split).mkdir(parents=True, exist_ok=True)
        (LABELS_ROOT / split).mkdir(parents=True, exist_ok=True)


def move_pairs(pairs: list[tuple[Path, Path | None]], split_name: str) -> None:
    target_img_dir = IMAGES_ROOT / split_name
    target_lbl_dir = LABELS_ROOT / split_name

    for img_path, lbl_path in pairs:
        dest_img = target_img_dir / img_path.name
        shutil.move(str(img_path), str(dest_img))

        dest_lbl = target_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path is not None and lbl_path.exists():
            shutil.move(str(lbl_path), str(dest_lbl))
        else:
            # Boş etiket dosyası (negatif/etiketsiz kare uyumluluğu için)
            dest_lbl.write_text("", encoding="utf-8")


def cleanup_empty_dirs() -> None:
    for root in (IMAGES_ROOT, LABELS_ROOT):
        for sub in sorted(root.rglob("*"), reverse=True):
            if sub.is_dir() and sub.name not in {"train", "val", "test"}:
                try:
                    sub.rmdir()
                except OSError:
                    pass


def update_dataset_yaml(total_train: int, total_val: int, total_test: int) -> None:
    data = {}
    if YAML_PATH.exists():
        with YAML_PATH.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data["path"] = str(BASE_DIR.resolve())
    data["train"] = "images/train"
    data["val"] = "images/val"
    data["test"] = "images/test"

    if "names" in data and isinstance(data["names"], (list, dict)):
        data["nc"] = len(data["names"])

    with YAML_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"dataset.yaml güncellendi -> train: {total_train}, val: {total_val}, test: {total_test}")


def main() -> None:
    pairs = collect_pairs()
    if not pairs:
        print("Ayrılacak görüntü bulunamadı. Zaten ayrılmış olabilir.")
        return

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    prepare_split_dirs()
    move_pairs(train_pairs, "train")
    move_pairs(val_pairs, "val")
    move_pairs(test_pairs, "test")
    cleanup_empty_dirs()
    update_dataset_yaml(len(train_pairs), len(val_pairs), len(test_pairs))


if __name__ == "__main__":
    main()