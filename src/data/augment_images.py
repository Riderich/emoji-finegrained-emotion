"""
Image augmentation utilities for emoji dataset.

This script applies rotation (±15°), brightness (±20%), and contrast
adjustments (±15%) to images. Outputs are saved in a destination
directory, preserving filename prefixes.
"""

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageEnhance


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for source and destination paths."""
    # 中文说明：命令行参数包括源目录 --src 与目标目录 --dst
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True, help="Source image directory")
    p.add_argument("--dst", type=str, required=True, help="Destination directory")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    """Create directory if not exists."""
    # 中文说明：若目标目录不存在则创建
    path.mkdir(parents=True, exist_ok=True)


def rotate_image(img: Image.Image, degrees: float) -> Image.Image:
    """Rotate image by given degrees, expanding canvas if necessary."""
    # 中文说明：旋转图像并使用透明背景填充超出区域
    return img.rotate(degrees, resample=Image.BICUBIC, expand=False)


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Adjust brightness by factor (1.0 means no change)."""
    # 中文说明：亮度因子>1增加亮度，<1降低亮度
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Adjust contrast by factor (1.0 means no change)."""
    # 中文说明：对比度因子>1增强对比度，<1降低对比度
    return ImageEnhance.Contrast(img).enhance(factor)


def augment_file(src_path: Path, dst_dir: Path) -> Tuple[int, int]:
    """Apply augmentations to a single image and save outputs.

    Parameters
    ----------
    src_path : Path
        Path to the source image file.
    dst_dir : Path
        Directory where augmented images will be saved.

    Returns
    -------
    Tuple[int, int]
        A tuple of (saved_count, failed_count).
    """
    # 中文说明：对单张图片应用旋转、亮度、对比度增强，总计生成多张样本
    base = src_path.stem
    try:
        img = Image.open(src_path).convert("RGBA")
    except Exception:
        return (0, 1)

    saved, failed = 0, 0
    variants = []
    # 旋转±15°
    variants.append((rotate_image(img, +15), f"{base}_rot_p15.png"))
    variants.append((rotate_image(img, -15), f"{base}_rot_m15.png"))
    # 亮度±20%
    variants.append((adjust_brightness(img, 1.2), f"{base}_bri_p20.png"))
    variants.append((adjust_brightness(img, 0.8), f"{base}_bri_m20.png"))
    # 对比度+15%
    variants.append((adjust_contrast(img, 1.15), f"{base}_con_p15.png"))

    for var_img, name in variants:
        try:
            out_path = dst_dir / name
            var_img.save(out_path)
            saved += 1
        except Exception:
            failed += 1

    return (saved, failed)


def main() -> None:
    """Entry point: augment images in the source directory."""
    # 中文说明：读取源目录内所有 PNG，逐一增强并保存到目标目录
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    ensure_dir(dst)

    files = list(src.glob("*.png"))
    print(f"[Augment] Source: {src} | Files: {len(files)} | Output: {dst}")
    saved_total, failed_total = 0, 0
    for f in files:
        s, fa = augment_file(f, dst)
        saved_total += s
        failed_total += fa
    print(f"[Augment] Saved: {saved_total} | Failed: {failed_total}")


if __name__ == "__main__":
    main()