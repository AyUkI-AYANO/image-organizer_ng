#!/usr/bin/env python3
"""Organize images into semantic buckets with optimized scoring and modern Web UI."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

CATEGORY_KEYWORDS: dict[str, set[str]] = {
    "people": {
        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "bride",
        "groom",
        "scuba diver",
        "baseball player",
        "swimmer",
        "skier",
        "soldier",
        "academic gown",
        "face",
        "portrait",
        "crowd",
    },
    "scenery": {
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
        "desert",
        "ocean",
        "rainforest",
        "waterfall",
        "sunset",
        "sunrise",
        "sky",
        "landscape",
    },
    "animals": {
        "goldfish",
        "tabby",
        "persian cat",
        "siamese cat",
        "egyptian cat",
        "lion",
        "tiger",
        "cheetah",
        "bear",
        "zebra",
        "hippopotamus",
        "ox",
        "ram",
        "llama",
        "camel",
        "elephant",
        "giant panda",
        "koala",
        "otter",
        "chimpanzee",
        "gorilla",
        "dog",
        "retriever",
        "shepherd",
        "poodle",
        "wolf",
        "fox",
        "hare",
        "rabbit",
        "deer",
        "squirrel",
        "bird",
        "eagle",
        "parrot",
        "owl",
        "duck",
        "penguin",
        "flamingo",
        "shark",
        "ray",
        "snake",
        "lizard",
        "turtle",
        "insect",
        "butterfly",
        "bee",
        "spider",
        "fish",
        "mammal",
    },
    "vehicles": {
        "car wheel",
        "sports car",
        "convertible",
        "jeep",
        "limousine",
        "cab",
        "minivan",
        "truck",
        "pickup",
        "trailer truck",
        "fire engine",
        "ambulance",
        "bus",
        "school bus",
        "trolleybus",
        "motor scooter",
        "moped",
        "mountain bike",
        "bicycle",
        "airliner",
        "warplane",
        "airship",
        "space shuttle",
        "bullet train",
        "steam locomotive",
        "submarine",
        "boat",
        "catamaran",
        "sailboat",
        "speedboat",
        "tractor",
        "forklift",
        "locomotive",
        "motorcycle",
        "train",
        "aircraft",
    },
    "food": {
        "pizza",
        "cheeseburger",
        "hotdog",
        "french loaf",
        "bagel",
        "pretzel",
        "ice cream",
        "trifle",
        "potpie",
        "carbonara",
        "guacamole",
        "consomme",
        "red wine",
        "espresso",
        "banana",
        "pineapple",
        "orange",
        "lemon",
        "pomegranate",
        "strawberry",
        "fig",
        "granny smith",
        "mushroom",
        "broccoli",
        "cauliflower",
        "artichoke",
        "soup bowl",
        "plate",
        "dining table",
        "dessert",
        "meal",
        "fruit",
    },
    "buildings": {
        "church",
        "mosque",
        "palace",
        "monastery",
        "dome",
        "library",
        "planetarium",
        "greenhouse",
        "movie theater",
        "restaurant",
        "lighthouse",
        "castle",
        "barn",
        "boathouse",
        "schoolhouse",
        "bell cote",
        "bridge",
        "suspension bridge",
        "viaduct",
        "pier",
        "fountain",
        "gazebo",
        "patio",
        "skyscraper",
        "tower",
        "arch",
        "building",
        "house",
        "apartment",
    },
}

SYNONYM_MAP: dict[str, str] = {
    "automobile": "car",
    "kitty": "cat",
    "puppy": "dog",
    "aeroplane": "airplane",
    "cycle": "bicycle",
    "motorbike": "motorcycle",
    "portraiture": "portrait",
}

TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    category: str
    category_score: float


@dataclass
class ProcessedItem:
    source: str
    destination: str
    action: str
    category: str
    label: str
    confidence: float
    category_score: float


class ImageOrganizer:
    def __init__(self, topk: int = 10) -> None:
        try:
            import torch
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.keyword_meta = self._build_keyword_index()

    def _build_keyword_index(self) -> dict[str, dict[str, set[str]]]:
        prepared: dict[str, dict[str, set[str]]] = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            normalized = {self._normalize_text(kw) for kw in keywords}
            prepared[category] = {
                "phrases": {kw for kw in normalized if kw},
                "tokens": {token for kw in normalized for token in self._tokens(kw)},
            }
        return prepared

    def classify(self, image_path: Path) -> ClassificationResult:
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing dependencies. Please run: pip install -r requirements.txt"
            ) from exc

        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)

        conf, idx = torch.topk(probs, self.topk)
        labels = [self.labels[i] for i in idx[0].tolist()]
        confidences = [float(v) for v in conf[0].tolist()]

        top_label = labels[0]
        top_conf = confidences[0]
        category, category_score = self._map_category(labels, confidences)

        return ClassificationResult(
            label=top_label,
            confidence=top_conf,
            category=category,
            category_score=category_score,
        )

    def _map_category(self, labels: Iterable[str], confidences: Iterable[float]) -> tuple[str, float]:
        scores = {category: 0.0 for category in CATEGORY_KEYWORDS}

        for rank, (label, confidence) in enumerate(zip(labels, confidences), start=1):
            normalized_label = self._normalize_text(label)
            normalized_label = self._apply_synonyms(normalized_label)
            if not normalized_label:
                continue

            label_tokens = self._tokens(normalized_label)
            rank_weight = 1.0 / (1.0 + 0.35 * (rank - 1))
            confidence_weight = math.sqrt(max(confidence, 0.0))
            base_weight = confidence_weight * rank_weight

            for category, meta in self.keyword_meta.items():
                # exact phrase match
                if normalized_label in meta["phrases"]:
                    scores[category] += base_weight * 1.45
                    continue

                token_overlap = len(label_tokens & meta["tokens"])
                if token_overlap:
                    ratio = token_overlap / max(len(label_tokens), 1)
                    scores[category] += base_weight * (0.68 + 0.52 * ratio)

                # weak substring fallback
                if any(p in normalized_label or normalized_label in p for p in meta["phrases"]):
                    scores[category] += base_weight * 0.25

        best_category = max(scores, key=scores.get, default="objects")
        best_score = scores.get(best_category, 0.0)

        if best_score <= 0:
            return "objects", 0.0

        return best_category, best_score

    @staticmethod
    def _apply_synonyms(text: str) -> str:
        return " ".join(SYNONYM_MAP.get(token, token) for token in text.split())

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(TOKEN_SPLIT_RE.sub(" ", text.lower()).split())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(text.split())


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
    min_category_score: float,
    topk: int,
    dry_run: bool = False,
    report_path: Path | None = None,
) -> tuple[int, dict[str, int]]:
    organizer = ImageOrganizer(topk=topk)
    action = shutil.move if move else shutil.copy2
    action_name = "move" if move else "copy"

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    stats: dict[str, int] = {}
    processed_items: list[ProcessedItem] = []
    for image_path in iter_images(source_dir):
        total += 1
        result = organizer.classify(image_path)
        if result.confidence >= min_confidence and result.category_score >= min_category_score:
            category = result.category
        else:
            category = "unknown"

        stats[category] = stats.get(category, 0) + 1
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = unique_path(category_dir / image_path.name)
        if not dry_run:
            action(str(image_path), str(destination))

        processed_items.append(
            ProcessedItem(
                source=str(image_path),
                destination=str(destination),
                action="dry-run" if dry_run else action_name,
                category=category,
                label=result.label,
                confidence=result.confidence,
                category_score=result.category_score,
            )
        )

        print(
            f"[{category.upper()}] {image_path} -> {destination} "
            f"(label={result.label}, conf={result.confidence:.2%}, category_score={result.category_score:.3f}, action={'dry-run' if dry_run else action_name})"
        )

    if report_path is not None:
        report_payload = {
            "version": "5.1.0",
            "source_dir": str(source_dir.resolve()),
            "output_dir": str(output_dir.resolve()),
            "dry_run": dry_run,
            "action": action_name,
            "min_confidence": min_confidence,
            "min_category_score": min_category_score,
            "topk": topk,
            "total": total,
            "stats": stats,
            "items": [item.__dict__ for item in processed_items],
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Report saved to: {report_path.resolve()}")

    print(f"\nDone. Processed {total} image(s).")
    return total, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify and organize images into people/scenery/animals/"
            "vehicles/food/buildings/objects folders."
        )
    )
    parser.add_argument("source", type=Path, nargs="?", help="Source directory containing images")
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
        default=0.22,
        help="Minimum model confidence, else goes to unknown (default: 0.22)",
    )
    parser.add_argument(
        "--min-category-score",
        type=float,
        default=0.16,
        help="Minimum category match score, else goes to unknown (default: 0.16)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K predictions used for semantic scoring (default: 10)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch web GUI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview classification and destinations without moving/copying files",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report output path",
    )
    return parser.parse_args()


def launch_gui() -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit("Gradio is not available. Please run: pip install -r requirements.txt") from exc

    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["Inter", "ui-sans-serif", "system-ui"],
    )

    def run_web_organize(
        source_dir: str,
        output_dir: str,
        move_files: bool,
        min_conf: float,
        min_cat_score: float,
        topk: int,
        dry_run: bool,
        report_path: str,
    ) -> str:
        source_path = Path(source_dir.strip())
        output_path = Path(output_dir.strip() or "organized_images")

        if not source_path.exists() or not source_path.is_dir():
            return "❌ Source directory does not exist."

        if not (0.0 <= min_conf <= 1.0 and 0.0 <= min_cat_score <= 1.0):
            return "❌ Threshold must be in [0, 1]."

        total, stats = organize(
            source_dir=source_path,
            output_dir=output_path,
            move=move_files,
            min_confidence=min_conf,
            min_category_score=min_cat_score,
            topk=max(3, int(topk)),
            dry_run=dry_run,
            report_path=Path(report_path.strip()) if report_path.strip() else None,
        )

        lines = [f"✅ Done. Processed {total} image(s).", f"📁 Output: {output_path.resolve()}"]
        if dry_run:
            lines.append("🧪 Dry-run enabled: no files were moved/copied.")
        if report_path.strip():
            lines.append(f"🧾 Report: {Path(report_path.strip()).resolve()}")
        lines.append("\nCategory distribution:")
        for key in sorted(stats):
            lines.append(f"- {key}: {stats[key]}")
        return "\n".join(lines)

    with gr.Blocks(theme=theme, title="Image Organizer NG 5.1.0") as app:
        gr.Markdown(
            """
            # ✨ Image Organizer NG 5.1.0
            **更优雅的界面 + 更稳健的分类算法 + 可追溯报告**  
            融合 Top-K 结果、词义归一化、分层打分策略，并新增 Dry-Run 与 JSON 报告导出。
            """
        )

        with gr.Row():
            source_input = gr.Textbox(label="Source directory", placeholder="/path/to/images")
            output_input = gr.Textbox(label="Output directory", value=str(Path("organized_images").resolve()))

        with gr.Row():
            min_conf_input = gr.Slider(0, 1, value=0.22, step=0.01, label="Min confidence")
            min_score_input = gr.Slider(0, 1, value=0.16, step=0.01, label="Min category score")
            topk_input = gr.Slider(3, 20, value=10, step=1, label="Top-K predictions")

        with gr.Row():
            move_input = gr.Checkbox(label="Move files instead of copy", value=False)
            dry_run_input = gr.Checkbox(label="Dry-run (no file write)", value=False)

        report_input = gr.Textbox(
            label="Report path (optional)",
            placeholder="./reports/organize_report.json",
        )
        run_button = gr.Button("🚀 Start Organizing", variant="primary")
        output = gr.Textbox(label="Run summary", lines=12)

        run_button.click(
            fn=run_web_organize,
            inputs=[
                source_input,
                output_input,
                move_input,
                min_conf_input,
                min_score_input,
                topk_input,
                dry_run_input,
                report_input,
            ],
            outputs=output,
        )

    app.launch(server_name="0.0.0.0", server_port=7860)


def main() -> None:
    args = parse_args()

    if args.gui:
        launch_gui()
        return

    if args.source is None:
        raise SystemExit("Source directory is required when not using --gui")

    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("--min-confidence must be between 0 and 1")
    if not (0.0 <= args.min_category_score <= 1.0):
        raise SystemExit("--min-category-score must be between 0 and 1")

    organize(
        source_dir=args.source,
        output_dir=args.output,
        move=args.move,
        min_confidence=args.min_confidence,
        min_category_score=args.min_category_score,
        topk=max(3, args.topk),
        dry_run=args.dry_run,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
