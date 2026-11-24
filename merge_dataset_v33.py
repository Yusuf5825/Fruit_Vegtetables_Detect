import hashlib
import random
import shutil
from pathlib import Path
import yaml

# ==============================================================================
# HEDEF SINIF SIRASI: 0: Apple, 1: Lemon, 2: Potato, 3: Tomato, 4: Carrot
# ==============================================================================
CLASSES = ["Apple", "Lemon", "Potato", "Tomato", "Carrot"]

ROOT = Path(r"C:\Projeler\Fruits_detect")
BASE_MERGED = ROOT / "merged_dataset_v2"
OUT_MERGED = ROOT / "merged_dataset_v3"

# apple.v4i.yolov8 -> tüm etiketli kutular 0 (Apple)
# carrot.v1i.yolov8 -> tek etiket / tüm kutular 4 (Carrot)
NEW_SOURCES = [
    {"name": "apple_v4i", "dir": ROOT / "apple.v4i.yolov8", "target_id": 0},
    {"name": "carrot_v1i", "dir": ROOT / "carrot.v1i.yolov8", "target_id": 4},
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = (0.75, 0.15, 0.10)  # train, val, test
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_label_for_image(img_path: Path, src_dir: Path) -> Path | None:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        cand = Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
        if cand.is_file():
            return cand

    same_txt = img_path.with_suffix(".txt")
    if same_txt.is_file():
        return same_txt

    for m in src_dir.rglob(f"{img_path.stem}.txt"):
        if m.is_file() and m.name != "classes.txt":
            return m
    return None


def map_all_boxes_to_id(lbl_path: Path, target_id: int) -> list[str]:
    """Dosyadaki her geçerli kutu satırının sınıf ID'sini doğrudan target_id yapar."""
    out: list[str] = []
    for raw in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        coords = " ".join(parts[1:5])
        out.append(f"{target_id} {coords}")
    return out


def prepare_out_dirs() -> None:
    if OUT_MERGED.exists():
        shutil.rmtree(OUT_MERGED)
    for sp in SPLITS:
        (OUT_MERGED / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT_MERGED / "labels" / sp).mkdir(parents=True, exist_ok=True)


def copy_existing_merged_v2(seen_hashes: set[str]) -> dict[str, int]:
    """merged_dataset_v2 içindeki verileri kendi split'leriyle v3'e kopyalar."""
    copied = {"train": 0, "val": 0, "test": 0}
    if not BASE_MERGED.exists():
        raise FileNotFoundError(f"merged_dataset_v2 bulunamadı: {BASE_MERGED}")

    for sp in SPLITS:
        img_dir = BASE_MERGED / "images" / sp
        lbl_dir = BASE_MERGED / "labels" / sp
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.rglob("*")):
            if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS):
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.is_file():
                continue

            digest = sha256_file(img_path)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            ext = img_path.suffix.lower()
            shutil.copy2(img_path, OUT_MERGED / "images" / sp / f"{img_path.stem}{ext}")
            shutil.copy2(lbl_path, OUT_MERGED / "labels" / sp / f"{img_path.stem}.txt")
            copied[sp] += 1

    return copied


def collect_new_unique_items(seen_hashes: set[str]) -> tuple[list[tuple[str, str, Path, str]], int, int]:
    new_items: list[tuple[str, str, Path, str]] = []
    skipped_dup = 0
    skipped_missing = 0

    for src in NEW_SOURCES:
        src_dir: Path = src["dir"]
        if not src_dir.exists():
            print(f"[Uyarı] Klasör bulunamadı, atlandı: {src_dir}")
            continue

        target_id: int = src["target_id"]
        src_name: str = src["name"]

        for img_path in sorted(src_dir.rglob("*")):
            if not (img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS):
                continue

            lbl_path = find_label_for_image(img_path, src_dir)
            if lbl_path is None:
                skipped_missing += 1
                continue

            lines = map_all_boxes_to_id(lbl_path, target_id)
            if not lines:
                skipped_missing += 1
                continue

            digest = sha256_file(img_path)
            if digest in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(digest)

            new_items.append((src_name, digest[:10], img_path, "\n".join(lines) + "\n"))

    return new_items, skipped_dup, skipped_missing


def write_yaml() -> None:
    data = {
        "path": str(OUT_MERGED.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with (OUT_MERGED / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    prepare_out_dirs()
    seen_hashes: set[str] = set()

    base_counts = copy_existing_merged_v2(seen_hashes)
    new_items, skipped_dup, skipped_missing = collect_new_unique_items(seen_hashes)

    rng = random.Random(SEED)
    rng.shuffle(new_items)

    n = len(new_items)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])

    assigned = (
        [("train", it) for it in new_items[:n_train]]
        + [("val", it) for it in new_items[n_train : n_train + n_val]]
        + [("test", it) for it in new_items[n_train + n_val :]]
    )

    added_counts = {"train": 0, "val": 0, "test": 0}
    for idx, (sp, (src_name, short_hash, img_path, lbl_text)) in enumerate(assigned):
        ext = img_path.suffix.lower()
        stem = f"{src_name}_{short_hash}_{idx:05d}"
        shutil.copy2(img_path, OUT_MERGED / "images" / sp / f"{stem}{ext}")
        (OUT_MERGED / "labels" / sp / f"{stem}.txt").write_text(lbl_text, encoding="utf-8")
        added_counts[sp] += 1

    write_yaml()

    total_train = base_counts["train"] + added_counts["train"]
    total_val = base_counts["val"] + added_counts["val"]
    total_test = base_counts["test"] + added_counts["test"]

    print(f"Oluşturuldu -> {OUT_MERGED}")
    print(f"merged_dataset_v2'den korunan (train/val/test) : {base_counts['train']} / {base_counts['val']} / {base_counts['test']}")
    print(f"Yeni eklenen tekil (train/val/test)          : {added_counts['train']} / {added_counts['val']} / {added_counts['test']}")
    print(f"Toplam v3 (train/val/test)                   : {total_train} / {total_val} / {total_test}")
    print(f"Atlanan yinelenen                            : {skipped_dup} | Etiketsiz/boş: {skipped_missing}")


if __name__ == "__main__":
    main()