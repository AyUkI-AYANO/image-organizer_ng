#!/usr/bin/env python3
"""Organize images into people/scenery/objects buckets based on content."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

PEOPLE_KEYWORDS = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "bride",
    "groom",
    "scuba diver",
    "baseball player",
}

SCENERY_KEYWORDS = {
    "mountain",
    "valley",
    "volcano",
    "cliff",
    "seashore",
    "lakeside",
    "sandbar",
    "promontory",
    "geyser",
    "alp",
    "coral reef",
    "beach",
    "forest",
}


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    category: str


class ImageOrganizer:
    def __init__(self, topk: int = 5) -> None:
        try:
            from torchvision import models
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'torchvision'. Please run: pip install -r requirements.txt"
            ) from exc

        self.topk = topk
        self.weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=self.weights)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.labels = self.weights.meta["categories"]

    def classify(self, image_path: Path) -> ClassificationResult:
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies. Please run: pip install -r requirements.txt"
            ) from exc

        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)

        conf, idx = torch.topk(probs, self.topk)
        labels = [self.labels[i] for i in idx[0].tolist()]
        confidences = conf[0].tolist()

        top_label = labels[0]
        top_conf = float(confidences[0])
        category = self._map_category(labels)

        return ClassificationResult(label=top_label, confidence=top_conf, category=category)

    def _map_category(self, labels: Iterable[str]) -> str:
        normalized = {label.lower().strip() for label in labels}

        if normalized & PEOPLE_KEYWORDS:
            return "people"
        if normalized & SCENERY_KEYWORDS:
            return "scenery"

        # If neither matches, default to objects.
        return "objects"


def iter_images(source_dir: Path) -> Iterable[Path]:
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def unique_path(dest_path: Path) -> Path:
    if not dest_path.exists():
        return dest_path

    stem, suffix = dest_path.stem, dest_path.suffix
    counter = 1
    while True:
        candidate = dest_path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def organize(
    source_dir: Path,
    output_dir: Path,
    move: bool,
    min_confidence: float,
) -> None:
    organizer = ImageOrganizer()
    action = shutil.move if move else shutil.copy2

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for image_path in iter_images(source_dir):
        total += 1
        result = organizer.classify(image_path)
        category = result.category if result.confidence >= min_confidence else "unknown"

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = unique_path(category_dir / image_path.name)
        action(str(image_path), str(destination))

        print(
            f"[{category.upper()}] {image_path} -> {destination} "
            f"(label={result.label}, conf={result.confidence:.2%})"
        )

    print(f"\nDone. Processed {total} image(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify and organize images into people/scenery/objects folders."
    )
    parser.add_argument("source", type=Path, help="Source directory containing images")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("organized_images"),
        help="Output directory for organized images (default: organized_images)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.20,
        help="Minimum confidence for category assignment, else goes to unknown (default: 0.20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("--min-confidence must be between 0 and 1")

    organize(
        source_dir=args.source,
        output_dir=args.output,
        move=args.move,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
