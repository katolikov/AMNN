#!/usr/bin/env python3
"""Generate Android mipmap icons from a source PNG image.

Usage:
    python generate_icons.py icon_source.png

Generates ic_launcher.png in all required mipmap-* directories.
Requires Pillow: pip install Pillow
"""
import sys
from pathlib import Path

MIPMAP_SIZES = {
    "mipmap-mdpi": 48,
    "mipmap-hdpi": 72,
    "mipmap-xhdpi": 96,
    "mipmap-xxhdpi": 144,
    "mipmap-xxxhdpi": 192,
}

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <source_icon.png>")
        sys.exit(1)

    source = Path(sys.argv[1])
    if not source.exists():
        print(f"Error: {source} not found")
        sys.exit(1)

    try:
        from PIL import Image
    except ImportError:
        # Fallback to macOS sips
        import subprocess
        res_dir = Path(__file__).parent / "app" / "src" / "main" / "res"
        for folder, size in MIPMAP_SIZES.items():
            out_dir = res_dir / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "ic_launcher.png"
            subprocess.run([
                "sips", "-z", str(size), str(size),
                str(source), "--out", str(out_file)
            ], check=True, capture_output=True)
            print(f"  {folder}: {size}x{size}")
        print("Done (using sips)")
        return

    img = Image.open(source).convert("RGBA")
    res_dir = Path(__file__).parent / "app" / "src" / "main" / "res"

    for folder, size in MIPMAP_SIZES.items():
        out_dir = res_dir / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        resized = img.resize((size, size), Image.LANCZOS)
        out_file = out_dir / "ic_launcher.png"
        resized.save(out_file, "PNG")
        print(f"  {folder}: {size}x{size}")

    print("Done")


if __name__ == "__main__":
    main()
