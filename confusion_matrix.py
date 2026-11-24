"""Generate a confusion matrix from per-detection class pairs.

Input JSON format:
{"pairs": [{"true_class": "Apple", "predicted_class": "Apple"}, ...]}
"""
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["Apple", "Lemon", "Potato", "Tomato", "Carrot"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON with a pairs list")
    ap.add_argument("--output", default="confusion_matrix.png")
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for pair in data.get("pairs", []):
        if pair.get("true_class") in CLASSES and pair.get("predicted_class") in CLASSES:
            matrix[CLASSES.index(pair["true_class"]), CLASSES.index(pair["predicted_class"])] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Adet")
    ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf", title="Confusion matrix")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180)
    print(f"Kaydedildi: {args.output}")

if __name__ == "__main__":
    main()
