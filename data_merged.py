import hashlib
import random
import shutil
from pathlib import Path
import yaml

# ==============================================================================
# 1. HEDEF SINIF YAPISI (0: Apple, 1: Lemon, 2: Potato, 3: Tomato, 4: Carrot)
# ==============================================================================
CLASSES = ["Apple", "Lemon", "Potato", "Tomato", "Carrot"]

ROOT = Path(r"C:/Projeler/Fruits_detect")
OUT_DIR = ROOT / "merged_dataset"

# Kaynak klasörler ve içlerindeki mevcut sınıf ID'sinin hedef ID'ye dönüşümü
SOURCES = [
    {
        "name": "yolo_fruits_export",
        "dir": ROOT / "yolo_fruits_export",
        "id_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},  # Zaten aynı sıra
    },
    {
        "name": "coco_out",
        "dir": ROOT / "coco_out",
        "id_map": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},  # Zaten aynı sıra
    },
    {
        "name": "carrot_v2i",
        "dir": ROOT / "carrot.v2i.yolov8",
        "id_map": {0: 4},  # Tek sınıf (0) -> 4: Carrot
    },
    {
        "name": "lemon_v2i",
        "dir": ROOT / "lemon.v2i.yolov8",
        "id_map": {0: 1},  # Tek sınıf (0) -> 1: Lemon
    },
    {
        "name": "tomato_v7i",
        "dir": ROOT / "tomato.v7i.yolov8",
        "id_map": {0: 3},  # Tek sınıf (0) -> 3: Tomato
    },
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_label_for_image(img_path: Path, src_dir: Path) -> Path | None:
    """Roboflow (split/images/xxx.jpg + split/labels/xxx.txt) ve standart YOLO yapısını destekler."""
    # 1) .../images/foo.jpg -> .../labels/foo.txt
    parts = list(img_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        alt_parts = parts[:idx] + ["labels"] + parts[idx + 1:]
        cand = Path(*alt_parts).with_suffix(".txt")
        if cand.exists():
            return cand

    # 2) Aynı klasörde foo.txt
    same_txt = img_path.with_suffix(".txt")
    if same_txt.exists():
        return same_txt

    # 3) Kaynak klasör içinde aynı stem ile .txt ara
    for m in src_dir.rglob(f"{img_path.stem}.txt"):
        if m.is_file():
            return m
    return None


def remap_label_text(txt_path: Path, id_map: dict[int, int]) -> str:
    out_lines: list[str] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_id = int(float(parts[0]))
        except ValueError:
            continue
        if old_id not in id_map:
            continue
        parts[0] = str(id_map[old_id])
        out_lines.append(" ".join(parts))
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def detect_split_hint(img_path: Path, src_dir: Path) -> str | None:
    rel_parts = [p.lower() for p in img_path.relative_to(src_dir).parts]
    if "train" in rel_parts:
        return "train"
    if "val" in rel_parts or "valid" in rel_parts or "validation" in rel_parts:
        return "val"
    if "test" in rel_parts:
        return "test"
    return None


def collect_unique_items() -> list[dict]:
    seen_hashes: set[str] = set()
    items: list[dict] = []
    duplicates = 0

    for src in SOURCES:
        src_dir: Path = src["dir"]
        if not src_dir.exists():
            print(f"[UYARI] Klasör bulunamadı, atlandı: {src_dir}")
            continue

        id_map: dict[int, int] = src["id_map"]
        img_files = [
            p for p in src_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

        for img_path in img_files:
            lbl_path = find_label_for_image(img_path, src_dir)
            if lbl_path is None:
                continue

            mapped_txt = remap_label_text(lbl_path, id_map)
            if not mapped_txt.strip():
                continue

            # İçerik bazlı tekilleştirme: aynı görsel farklı klasör/split'te varsa tek kayda düşer
            digest = file_sha256(img_path)
            if digest in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(digest)

            items.append({
                "source": src["name"],
                "hash": digest[:12],
                "img_path": img_path,
                "label_text": mapped_txt,
                "split_hint": detect_split_hint(img_path, src_dir),
            })

    print(f"Tekil geçerli görsel sayısı: {len(items)} (Atlanan yinelenen görsel: {duplicates})")
    return items


def assign_splits_no_leakage(items: list[dict]) -> dict[str, list[dict]]:
    """
    Her tekil hash'i yalnızca tek bir split'e (train, val veya test) atar.
    Mevcut split ipucu varsa korur, yoksa %70 / %15 / %15 oranında böler.
    """
    rng = random.Random(SEED)
    by_hint: dict[str | None, list[dict]] = {"train": [], "val": [], "test": [], None: []}
    for it in items:
        by_hint[it["split_hint"]].append(it)

    splits = {"train": [], "val": [], "test": []}

    for hint in ("train", "val", "test"):
        rng.shuffle(by_hint[hint])
        splits[hint].extend(by_hint[hint])

    unknown = by_hint[None]
    rng.shuffle(unknown)
    n = len(unknown)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    splits["train"].extend(unknown[:n_train])
    splits["val"].extend(unknown[n_train:n_train + n_val])
    splits["test"].extend(unknown[n_train + n_val:])

    # Son güvenlik kontrolü: split'ler arası hash kesişimi sıfır olmalı
    h_train = {x["hash"] for x in splits["train"]}
    h_val = {x["hash"] for x in splits["val"]}
    h_test = {x["hash"] for x in splits["test"]}
    assert not (h_train & h_val), "Veri sızıntısı: train ve val kesişiyor!"
    assert not (h_train & h_test), "Veri sızıntısı: train ve test kesişiyor!"
    assert not (h_val & h_test), "Veri sızıntısı: val ve test kesişiyor!"

    return splits


def write_dataset(splits: dict[str, list[dict]]) -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split_name, group in splits.items():
        img_out = OUT_DIR / "images" / split_name
        lbl_out = OUT_DIR / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for idx, it in enumerate(group):
            ext = it["img_path"].suffix.lower()
            base_name = f"{it['source']}_{it['hash']}_{idx:05d}"
            shutil.copy2(it["img_path"], img_out / f"{base_name}{ext}")
            (lbl_out / f"{base_name}.txt").write_text(it["label_text"], encoding="utf-8")

    yaml_data = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with (OUT_DIR / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(
        f"Tamamlandı -> {OUT_DIR} | "
        f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}"
    )


def main() -> None:
    items = collect_unique_items()
    splits = assign_splits_no_leakage(items)
    write_dataset(splits)


if __name__ == "__main__":
    main()