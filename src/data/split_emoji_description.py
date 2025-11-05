"""Split emoji and description in mapping files.

English
-------
Some mapping files have mixed content in the `emoji` field, e.g., the emoji
character(s) together with a textual description. This utility extracts the
actual emoji characters into `emoji` and moves residual text into a new
`description` field. Works for both JSON and CSV files.

中文说明
--------
部分文件的 `emoji` 字段同时包含表情字符与文字描述。本工具将真正的表情字符提取到
`emoji` 字段，并把剩余的文字描述放到新字段 `description` 中。支持处理 JSON 与 CSV。
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Tuple


# 中文说明：覆盖常见 emoji 码段，便于识别出文本中的表情字符
EMOJI_RANGES = (
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A
    "\u2600-\u26FF"          # Miscellaneous Symbols
    "\u2700-\u27BF"          # Dingbats
    "\U0001F1E6-\U0001F1FF"  # Regional Indicators
    "\U0001F3FB-\U0001F3FF"  # Emoji Modifiers (skin tones)
)

# 英文说明：pattern to find emoji characters; does not combine ZWJ clusters
# 中文说明：匹配单个 emoji 字符（不组合 ZWJ 序列），用于抽取与移除
EMOJI_CHAR_PATTERN = re.compile(fr"[{EMOJI_RANGES}]")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for splitting emoji and description.

    Returns
    -------
    argparse.Namespace
        Arguments including input/output paths for JSON/CSV.
    """
    # 中文说明：支持分别处理 JSON/CSV；可选输出路径，否则原地覆盖
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", type=str, default=None, help="Input JSON path")
    p.add_argument("--out_json", type=str, default=None, help="Output JSON path (defaults to overwrite input)")
    p.add_argument("--in_csv", type=str, default=None, help="Input CSV path")
    p.add_argument("--out_csv", type=str, default=None, help="Output CSV path (defaults to overwrite input)")
    return p.parse_args()


def extract_emoji_and_description(text: str) -> Tuple[str, str]:
    """Extract emoji characters and residual description from a mixed string.

    English
    -------
    - Finds emoji characters by pattern.
    - Joins them in order to form `emoji`.
    - Removes them from original text; trims leftover separators to form `description`.

    中文说明：识别字符串中的表情字符，按出现顺序拼接为 `emoji`；
    去除后剩余文本清理空白与分隔符，得到 `description`。
    """
    if not text:
        return "", ""
    # 中文说明：提取所有 emoji 字符（包括国旗为双区域指示符）
    emoji_chars = EMOJI_CHAR_PATTERN.findall(text)
    emoji_str = "".join(emoji_chars)
    # 中文说明：去除 emoji 字符，保留描述文本
    desc = EMOJI_CHAR_PATTERN.sub("", text)
    # 中文说明：清理多余空白与常见分隔符
    desc = re.sub(r"\s+", " ", desc)
    desc = desc.strip(" -–—|:，,；;()（）[]【】")
    return emoji_str, desc


def process_json(in_path: Path, out_path: Path) -> None:
    """Process JSON list: split `emoji` and create `description` per entry.

    中文说明：遍历 JSON 列表项，分离 `emoji` 与描述；若已有 `description` 列，则覆盖为新值。
    """
    data = json.loads(in_path.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of entries")
    for e in data:
        mixed = e.get("emoji", "")
        emoji_str, desc = extract_emoji_and_description(str(mixed))
        e["emoji"] = emoji_str
        # 中文说明：若已有 description 字段，用新的描述覆盖
        e["description"] = desc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def process_csv(in_path: Path, out_path: Path) -> None:
    """Process CSV: split `emoji` and add/update `description` column.

    中文说明：读取 CSV，将 `emoji` 中的描述拆分到 `description` 列；若列不存在则新增。
    """
    with in_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", "utf-8")
        return
    header = rows[0]
    # 中文说明：找到/新增必要列
    try:
        emoji_idx = header.index("emoji")
    except ValueError:
        raise ValueError("CSV must contain 'emoji' column")
    if "description" in header:
        desc_idx = header.index("description")
    else:
        header.append("description")
        desc_idx = len(header) - 1
    out_rows = [header]
    # 中文说明：逐行处理，拆分 emoji 与描述
    for r in rows[1:]:
        mixed = r[emoji_idx] if emoji_idx < len(r) else ""
        emoji_str, desc = extract_emoji_and_description(mixed)
        r[emoji_idx] = emoji_str
        # 确保行长度覆盖到 description 列
        if len(r) <= desc_idx:
            r.extend([""] * (desc_idx - len(r) + 1))
        r[desc_idx] = desc
        out_rows.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(out_rows)


def main() -> None:
    """CLI entry for splitting emoji and description in JSON/CSV files.

    中文说明：根据命令行参数处理指定的 JSON/CSV 文件，并写回输出路径（默认覆盖原文件）。
    """
    args = parse_args()
    if args.in_json:
        in_json = Path(args.in_json)
        out_json = Path(args.out_json) if args.out_json else in_json
        process_json(in_json, out_json)
        print(f"[Split] JSON written: {out_json}")
    if args.in_csv:
        in_csv = Path(args.in_csv)
        out_csv = Path(args.out_csv) if args.out_csv else in_csv
        process_csv(in_csv, out_csv)
        print(f"[Split] CSV written: {out_csv}")


if __name__ == "__main__":
    main()