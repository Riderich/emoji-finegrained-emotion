"""Rewrite image URLs to local file paths in mapping files.

English
-------
This utility scans mapping files (JSON/CSV) and adds a new field `image_path`
that points to the locally saved emoji image under a platform-specific folder.
It keeps original remote URL fields (e.g., `img_url`/`image`) intact, but
augments entries with the local path for downstream usage.

中文说明
--------
该脚本为 JSON/CSV 映射文件新增 `image_path` 字段，指向本地已保存的表情图片路径。
不会删除原有的远程链接（如 `img_url`/`image`），仅补充本地路径，方便后续使用。
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rewriting image paths.

    Returns
    -------
    argparse.Namespace
        Arguments including input/output paths and image base directory.
    """
    # 中文说明：支持处理 JSON 或 CSV；支持指定平台与图片根目录，默认覆盖原文件
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", type=str, default=None, help="Input JSON path")
    p.add_argument("--out_json", type=str, default=None, help="Output JSON path (defaults to overwrite input)")
    p.add_argument("--in_csv", type=str, default=None, help="Input CSV path")
    p.add_argument("--out_csv", type=str, default=None, help="Output CSV path (defaults to overwrite input)")
    p.add_argument("--platform", type=str, default=None, help="Platform name (e.g., baidu, weibo, douyin, bilibili)")
    p.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Absolute base directory where images are stored, with subfolder per platform",
    )
    return p.parse_args()


def infer_platform_from_filename(path: Path) -> Optional[str]:
    """Infer platform name from file name like 'weibo_emojiall_map.json'.

    中文说明：从文件名推测平台（如 weibo_emojiall_map.json -> weibo）。
    """
    name = path.name.lower()
    for p in ("baidu", "weibo", "douyin", "bilibili"):
        if p in name:
            return p
    return None


def basename_from_url(url: str) -> Optional[str]:
    """Extract the last path segment from a URL.

    中文说明：从 URL 中提取文件名（末段）。
    """
    if not url:
        return None
    # 简单分割 URL，获取最后一段作为文件名
    seg = url.split("?")[0].split("#")[0].rstrip("/").split("/")[-1]
    return seg or None


def compose_local_path(images_root: Path, platform: str, filename: str) -> Path:
    """Compose the local image path given root, platform, and filename.

    中文说明：根据图片根目录、平台与文件名拼接出本地图片完整路径。
    """
    return images_root / platform / filename


def pick_image_url_entry(entry: dict) -> Optional[str]:
    """Pick a plausible remote image URL field from an entry.

    English
    -------
    Tries common keys: 'img_url', 'image', 'image_url', 'img', 'image_src'.

    中文说明：尝试常见字段名，取出远程图片链接。
    """
    for k in ("img_url", "image", "image_url", "img", "image_src"):
        v = entry.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def process_json(in_path: Path, out_path: Path, platform: Optional[str], images_root: Path) -> None:
    """Process a JSON mapping file, augmenting entries with `image_path`.

    中文说明：为 JSON 映射文件每条记录补充 `image_path` 本地路径；若无法定位文件则跳过。
    """
    data = json.loads(in_path.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of entries")
    plat = platform or infer_platform_from_filename(in_path)
    if not plat:
        raise ValueError("Cannot infer platform; please provide --platform")

    def resolve_local_image_path(entry: dict) -> Optional[str]:
        """Resolve a local image path for a JSON entry.

        English
        -------
        Build candidate filenames from URL and codepoints, handle '-new' variants,
        and search both platform and '<platform>_unmapped' directories.

        中文说明
        -------
        从远程 URL 与 codepoints 生成候选文件名，处理 '-new' 变体，在平台与未映射目录中查找，返回匹配的本地路径。
        """
        url = pick_image_url_entry(entry)
        fname = basename_from_url(url) if url else None

        # 中文：构造候选文件名集合
        candidate_names = []
        if fname:
            candidate_names.append(fname)  # 原始文件名，如 1f642.png
            base, ext = os.path.splitext(fname)
            if ext.lower() == ".png":
                # 中文：若是 '-new.png'，增加去掉 '-new' 的常规变体
                if base.endswith("-new"):
                    candidate_names.append(f"{base[:-4]}.png")  # 去掉 -new

        # 中文：根据首个码点添加模糊匹配关键词（不带扩展名）
        cps = entry.get("codepoints") or []
        if isinstance(cps, list) and cps:
            first = str(cps[0]).upper().replace("U+", "")
            if first:
                candidate_names.append(first.lower())

        plat_dir = images_root / plat
        plat_unmapped = images_root / f"{plat}_unmapped"

        def try_find(cand: str) -> Optional[Path]:
            # 中文：若包含扩展名，优先直接命中完整文件名
            if "." in cand:
                p = plat_dir / cand
                if p.exists():
                    return p
                p2 = plat_unmapped / cand
                if p2.exists():
                    return p2
            # 中文：否则进行模糊查找（包含子串）
            for dir_ in (plat_dir, plat_unmapped):
                if dir_.exists():
                    for fp in dir_.iterdir():
                        if fp.is_file() and cand in fp.name.lower():
                            return fp
            return None

        for cand in candidate_names:
            found = try_find(str(cand).lower())
            if found:
                return str(found)
        return None

    for e in data:
        local_path_str = resolve_local_image_path(e)
        if local_path_str:
            e["image_path"] = local_path_str

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def process_csv(in_path: Path, out_path: Path, platform: Optional[str], images_root: Path) -> None:
    """Process a CSV mapping file, adding `image_path` column.

    中文说明：为 CSV 文件新增 `image_path` 列；若能匹配到本地图片则填入路径。
    """
    with in_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", "utf-8")
        return
    header = rows[0]
    # 中文说明：确定远程链接所在列以及 image_path 列索引
    url_idx = None
    for k in ("img_url", "image", "image_url", "img", "image_src"):
        if k in header:
            url_idx = header.index(k)
            break
    if url_idx is None:
        raise ValueError("CSV must contain an image URL column (img_url/image/image_url/img/image_src)")
    if "image_path" in header:
        path_idx = header.index("image_path")
    else:
        header.append("image_path")
        path_idx = len(header) - 1
    out_rows = [header]
    plat = platform or infer_platform_from_filename(in_path)
    if not plat:
        raise ValueError("Cannot infer platform; please provide --platform")

    def resolve_from_row(row: list) -> Optional[str]:
        """Resolve local path for a CSV row.

        English
        -------
        Build candidates like JSON path resolution, including '-new' fix and
        codepoint fuzzy search.

        中文说明
        -------
        构造候选名并在平台目录与未映射目录中查找，与 JSON 保持一致的解析逻辑。
        """
        url = row[url_idx] if url_idx < len(row) else ""
        fname = basename_from_url(url) if url else None
        candidate_names = []
        if fname:
            candidate_names.append(fname)
            base, ext = os.path.splitext(fname)
            if ext.lower() == ".png" and base.endswith("-new"):
                candidate_names.append(f"{base[:-4]}.png")

        # 中文：尝试从 codepoints 列获取模糊关键词（若存在该列）
        cp_idx = header.index("codepoints") if "codepoints" in header else None
        if cp_idx is not None and cp_idx < len(row):
            cp_cell = row[cp_idx]
            # 中文：统一解析为首个 codepoint（支持 JSON 风格或单值）
            first = None
            if cp_cell:
                if cp_cell.startswith("[") and cp_cell.endswith("]"):
                    try:
                        arr = json.loads(cp_cell)
                        if isinstance(arr, list) and arr:
                            first = str(arr[0])
                    except Exception:
                        first = None
                else:
                    first = cp_cell
            if first:
                first = first.upper().replace("U+", "").lower()
                candidate_names.append(first)

        plat_dir = images_root / plat
        plat_unmapped = images_root / f"{plat}_unmapped"

        def try_find(cand: str) -> Optional[Path]:
            if "." in cand:
                p = plat_dir / cand
                if p.exists():
                    return p
                p2 = plat_unmapped / cand
                if p2.exists():
                    return p2
            for dir_ in (plat_dir, plat_unmapped):
                if dir_.exists():
                    for fp in dir_.iterdir():
                        if fp.is_file() and cand in fp.name.lower():
                            return fp
            return None

        for cand in candidate_names:
            found = try_find(str(cand).lower())
            if found:
                return str(found)
        return None

    for r in rows[1:]:
        path_val = resolve_from_row(r) or ""
        # 确保行长度覆盖到 image_path 列
        if len(r) <= path_idx:
            r.extend([""] * (path_idx - len(r) + 1))
        r[path_idx] = path_val
        out_rows.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        # 中文说明：强制为所有字段加引号，满足地址字段必须带引号的需求
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerows(out_rows)


def main() -> None:
    """CLI entry: rewrite image URLs to local paths for JSON/CSV.

    中文说明：根据命令行参数处理 JSON/CSV 文件，`--images_root` 指向图片根目录；
    若未显式传入平台，将从文件名推断平台。
    """
    args = parse_args()
    if not args.images_root:
        raise ValueError("--images_root is required (absolute base directory)")
    images_root = Path(args.images_root)
    if not images_root.is_absolute():
        # 中文说明：必须使用绝对路径，避免相对路径混淆
        images_root = images_root.resolve()

    if args.in_json:
        in_json = Path(args.in_json)
        out_json = Path(args.out_json) if args.out_json else in_json
        process_json(in_json, out_json, args.platform, images_root)
        print(f"[Rewrite] JSON written: {out_json}")

    if args.in_csv:
        in_csv = Path(args.in_csv)
        out_csv = Path(args.out_csv) if args.out_csv else in_csv
        process_csv(in_csv, out_csv, args.platform, images_root)
        print(f"[Rewrite] CSV written: {out_csv}")


if __name__ == "__main__":
    main()