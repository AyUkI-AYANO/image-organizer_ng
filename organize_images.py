#!/usr/bin/env python3
"""Image organizer v2: simple CLI + GUI for people/scenery/objects sorting."""

from __future__ import annotations

import argparse
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

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
    """Model wrapper used by both CLI and GUI."""

    def __init__(self, topk: int = 5) -> None:
        try:
            from torchvision import models
        except ImportError as exc:
            raise SystemExit(
                "Missing dependency 'torchvision'. Run: pip install -r requirements.txt"
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
                "Missing dependencies. Run: pip install -r requirements.txt"
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

        return "objects"


def iter_images(source_dir: Path) -> list[Path]:
    return [
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


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
    progress: Callable[[str], None] | None = None,
) -> int:
    organizer = ImageOrganizer()
    action = shutil.move if move else shutil.copy2
    images = iter_images(source_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(images, start=1):
        result = organizer.classify(image_path)
        category = result.category if result.confidence >= min_confidence else "unknown"

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = unique_path(category_dir / image_path.name)
        action(str(image_path), str(destination))

        if progress:
            progress(
                f"[{idx}/{len(images)}] {image_path.name} -> {category} "
                f"(label={result.label}, conf={result.confidence:.2%})"
            )

    if progress:
        progress(f"完成：共处理 {len(images)} 张图片。")

    return len(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image organizer v2: CLI or GUI mode for image classification."
    )
    parser.add_argument("source", nargs="?", type=Path, help="Source directory")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("organized_images"),
        help="Output directory (default: organized_images)",
    )
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.20,
        help="Confidence threshold in [0,1] (default: 0.20)",
    )
    parser.add_argument("--gui", action="store_true", help="Launch desktop GUI")
    return parser.parse_args()


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    class OrganizerApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("图片整理器 v2.0")
            self.geometry("760x520")

            self.source_var = tk.StringVar()
            self.output_var = tk.StringVar(value=str(Path("organized_images").resolve()))
            self.mode_var = tk.StringVar(value="copy")
            self.conf_var = tk.StringVar(value="0.20")

            self._build_ui(ttk)

        def _build_ui(self, ttk_module: object) -> None:
            frame = ttk_module.Frame(self, padding=16)
            frame.pack(fill="both", expand=True)

            ttk_module.Label(frame, text="步骤 1：选择图片目录").grid(row=0, column=0, sticky="w")
            ttk_module.Entry(frame, textvariable=self.source_var, width=72).grid(row=1, column=0, sticky="we")
            ttk_module.Button(frame, text="浏览", command=self.pick_source).grid(row=1, column=1, padx=8)

            ttk_module.Label(frame, text="步骤 2：选择输出目录").grid(row=2, column=0, pady=(14, 0), sticky="w")
            ttk_module.Entry(frame, textvariable=self.output_var, width=72).grid(row=3, column=0, sticky="we")
            ttk_module.Button(frame, text="浏览", command=self.pick_output).grid(row=3, column=1, padx=8)

            options = ttk_module.Frame(frame)
            options.grid(row=4, column=0, columnspan=2, sticky="we", pady=(14, 0))
            ttk_module.Label(options, text="步骤 3：运行参数").grid(row=0, column=0, sticky="w")
            ttk_module.Radiobutton(options, text="复制", variable=self.mode_var, value="copy").grid(
                row=1, column=0, sticky="w"
            )
            ttk_module.Radiobutton(options, text="移动", variable=self.mode_var, value="move").grid(
                row=1, column=1, sticky="w", padx=(10, 0)
            )
            ttk_module.Label(options, text="最低置信度:").grid(row=1, column=2, padx=(16, 4))
            ttk_module.Entry(options, textvariable=self.conf_var, width=8).grid(row=1, column=3)

            self.run_button = ttk_module.Button(frame, text="开始整理", command=self.start)
            self.run_button.grid(row=5, column=0, sticky="w", pady=(14, 0))

            self.log = tk.Text(frame, height=16, wrap="word")
            self.log.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(12, 0))
            frame.grid_rowconfigure(6, weight=1)
            frame.grid_columnconfigure(0, weight=1)

        def pick_source(self) -> None:
            path = filedialog.askdirectory(title="选择图片目录")
            if path:
                self.source_var.set(path)

        def pick_output(self) -> None:
            path = filedialog.askdirectory(title="选择输出目录")
            if path:
                self.output_var.set(path)

        def append_log(self, text: str) -> None:
            self.log.insert("end", f"{text}\n")
            self.log.see("end")

        def start(self) -> None:
            source = Path(self.source_var.get().strip())
            output = Path(self.output_var.get().strip())

            try:
                min_conf = float(self.conf_var.get().strip())
            except ValueError:
                messagebox.showerror("参数错误", "最低置信度必须是数字（例如 0.2）。")
                return

            if not source.exists() or not source.is_dir():
                messagebox.showerror("路径错误", "请先选择有效的图片目录。")
                return
            if not (0.0 <= min_conf <= 1.0):
                messagebox.showerror("参数错误", "最低置信度必须在 0 到 1 之间。")
                return

            self.run_button.config(state="disabled")
            self.append_log("开始整理，请稍候...（首次运行可能下载模型）")

            def worker() -> None:
                try:
                    total = organize(
                        source_dir=source,
                        output_dir=output,
                        move=self.mode_var.get() == "move",
                        min_confidence=min_conf,
                        progress=lambda msg: self.after(0, self.append_log, msg),
                    )
                    self.after(
                        0,
                        lambda: messagebox.showinfo("完成", f"整理完成，共处理 {total} 张图片。"),
                    )
                except Exception as exc:  # noqa: BLE001
                    self.after(0, lambda: messagebox.showerror("运行失败", str(exc)))
                finally:
                    self.after(0, lambda: self.run_button.config(state="normal"))

            threading.Thread(target=worker, daemon=True).start()

    app = OrganizerApp()
    app.mainloop()


def main() -> None:
    args = parse_args()

    if args.gui:
        run_gui()
        return

    if args.source is None:
        raise SystemExit("CLI模式需要提供 source 目录，或使用 --gui 启动界面。")
    if not args.source.exists() or not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise SystemExit("--min-confidence must be between 0 and 1")

    def print_progress(msg: str) -> None:
        print(msg)

    total = organize(
        source_dir=args.source,
        output_dir=args.output,
        move=args.move,
        min_confidence=args.min_confidence,
        progress=print_progress,
    )
    print(f"\nDone. Processed {total} image(s).")


if __name__ == "__main__":
    main()
