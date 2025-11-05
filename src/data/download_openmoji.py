"""
Download OpenMoji face emojis and normalize images for dataset preparation.

This script fetches OpenMoji metadata, filters smileys & emotion category,
and downloads PNG assets. Images are resized to 72x72 to align with the
project's standard.
"""

import os
import json
import io
from typing import List, Dict
from pathlib import Path

import requests
from PIL import Image


OPENMOJI_JSON_URL = (
    "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/data/openmoji.json"
)
OPENMOJI_PNG_BASE = (
    "https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/72x72/"
)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    # 中文说明：确保目录存在，若不存在则创建
    path.mkdir(parents=True, exist_ok=True)


def fetch_metadata() -> List[Dict]:
    """Fetch OpenMoji metadata as a list of emoji entries."""
    # 中文说明：请求 OpenMoji 的 JSON 元数据，包含每个 emoji 的分类与代码信息
    resp = requests.get(OPENMOJI_JSON_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def filter_face_emojis(meta: List[Dict]) -> List[Dict]:
    """Filter entries to smileys & emotion category (face-related groups)."""
    # 中文说明：筛选「Smileys & Emotion」类别且子分组为 face-* 的条目，近似对齐面部表情
    allowed_groups = {
        "face-smiling",
        "face-affection",
        "face-tongue",
        "face-hand",
        "face-neutral-skeptical",
        "face-sleepy",
        "face-unwell",
        "face-hat",
        "face-glasses",
        "face-concerned",
        "face-negative",
        "face-costume",
    }
    result = []
    for e in meta:
        if e.get("group") == "Smileys & Emotion" and e.get("subgroups"):
            # 中文说明：OpenMoji 的新版字段可能为 subgroups 或 subgroup，兼容处理
            subgroups = e.get("subgroups") or [e.get("subgroup")]
            if any(s in allowed_groups for s in subgroups if s):
                result.append(e)
        elif e.get("group") == "Smileys & Emotion" and e.get("subgroup") in allowed_groups:
            result.append(e)
    return result


def download_and_resize(hexcode: str, out_dir: Path) -> bool:
    """Download a single PNG by hexcode and resize to 72x72.

    Parameters
    ----------
    hexcode : str
        The hex code of the emoji (e.g., '1F60A').
    out_dir : Path
        Destination directory to save the normalized PNG.

    Returns
    -------
    bool
        True if success, False otherwise.
    """
    # 中文说明：下载 PNG 文件并用 Pillow 统一尺寸到 72x72；若下载失败返回 False
    url = f"{OPENMOJI_PNG_BASE}{hexcode}.png"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return False
        img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        img = img.resize((72, 72), Image.LANCZOS)
        ensure_dir(out_dir)
        out_path = out_dir / f"{hexcode}.png"
        img.save(out_path)
        return True
    except Exception:
        return False


def main() -> None:
    """Entry point: download face emojis from OpenMoji and normalize images."""
    # 中文说明：主流程包括：拉取元数据、筛选面部 emoji、批量下载与保存
    dest = Path("data/emoji_images/openmoji")
    ensure_dir(dest)

    meta = fetch_metadata()
    faces = filter_face_emojis(meta)

    print(f"[OpenMoji] Total entries: {len(meta)} | Faces filtered: {len(faces)}")
    ok, fail = 0, 0
    for e in faces:
        hexcode = e.get("hexcode") or e.get("hexcodeAlt")
        if not hexcode:
            fail += 1
            continue
        if download_and_resize(hexcode, dest):
            ok += 1
        else:
            fail += 1

    print(f"[OpenMoji] Saved: {ok} | Failed: {fail} | Output: {dest}")


if __name__ == "__main__":
    main()