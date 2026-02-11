#!/usr/bin/env python3
"""One-command launcher for Image Organizer NG 5 Web UI."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED_PACKAGES = ["torch", "torchvision", "PIL", "gradio"]


def ensure_dependencies() -> None:
    missing: list[str] = []
    for module_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)

    if not missing:
        return

    print(f"Missing dependencies: {', '.join(missing)}")
    print("Installing from requirements.txt ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def main() -> None:
    ensure_dependencies()
    script = Path(__file__).with_name("organize_images.py")
    subprocess.check_call([sys.executable, str(script), "--gui"])


if __name__ == "__main__":
    main()
