"""Filter codepoints in existing mapping files to keep only emoji codepoints.

English
-------
This utility loads JSON/CSV mapping files and filters the `codepoints` field to
retain only Unicode codepoints that belong to known emoji blocks. It removes
non-emoji characters (e.g., ASCII digits), and by default excludes `FE0F` (emoji
presentation) and `200D` (ZWJ). You can process JSON or CSV via CLI.

中文说明
--------
该脚本用于清洗已有的 JSON/CSV 映射文件，将 `codepoints` 中非表情符号区块的编码移除，
默认也会去掉 `FE0F`（表情呈现选择符）和 `200D`（零宽连接符），保留真正的 emoji 码点。
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for codepoints filtering.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with input/output paths and options.
    """
    # 中文说明：支持处理 JSON 或 CSV；默认覆盖原文件，也可指定输出文件
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", type=str, default=None, help="Input JSON path")
    p.add_argument("--out_json", type=str, default=None, help="Output JSON path (defaults to overwrite input)")
    p.add_argument("--in_csv", type=str, default=None, help="Input CSV path")
    p.add_argument("--out_csv", type=str, default=None, help="Output CSV path (defaults to overwrite input)")
    p.add_argument("--exclude_variation", action="store_true", help="Exclude FE0F variation selector (default true)")
    p.add_argument("--exclude_zwj", action="store_true", help="Exclude 200D zero-width joiner (default true)")
    return p.parse_args()


def is_emoji_codepoint(cp: int) -> bool:
    """Determine if a codepoint belongs to common emoji blocks.

    English
    -------
    Checks ranges for emoji-related blocks: Emoticons, Misc Symbols & Pictographs,
    Supplemental Symbols & Pictographs, Symbols & Pictographs Extended-A,
    Transport & Map, Misc Symbols, Dingbats, Regional Indicators, Emoji Modifiers.

    中文说明：判断码点是否属于常见的 emoji 相关区块，覆盖表情、杂项符号与象形、交通地图、
    扩展象形等；也包含区域指示符（国旗）与肤色修饰符。
    """
    ranges: List[Tuple[int, int]] = [
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x1FA70, 0x1FAFF),  # Symbols & Pictographs Extended-A
        (0x2600, 0x26FF),    # Miscellaneous Symbols
        (0x2700, 0x27BF),    # Dingbats
        (0x1F1E6, 0x1F1FF),  # Regional Indicator Symbols (flags)
        (0x1F3FB, 0x1F3FF),  # Emoji modifiers (skin tones)
    ]
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def parse_codepoint_token(tok: str) -> int:
    """Parse a token like 'U+1F601' into integer codepoint.

    中文说明：将形如 'U+XXXX' 的码点字符串解析为整数；不合法返回 -1。
    """
    tok = tok.strip().upper()
    if tok.startswith("U+"):
        try:
            return int(tok[2:], 16)
        except Exception:
            return -1
    return -1


def format_codepoint(cp: int) -> str:
    """Format integer codepoint back to 'U+XXXX' string.

    中文说明：将整数码点格式化为 'U+XXXX' 字符串形式。
    """
    return f"U+{cp:04X}"


def filter_codepoints(tokens: List[str], drop_fe0f: bool = True, drop_zwj: bool = True) -> List[str]:
    """Filter a list of 'U+XXXX' tokens to keep emoji codepoints only.

    English
    -------
    - Keeps only codepoints within emoji blocks.
    - Optionally drops FE0F and 200D.
    - Returns unique tokens in original string format.

    中文说明：仅保留 emoji 区块码点；可选排除 FE0F/200D；返回去重后的 'U+XXXX' 列表。
    """
    cps = [parse_codepoint_token(t) for t in tokens]
    result: List[int] = []
    for cp in cps:
        if cp <= 0:
            continue
        # 中文说明：默认排除 FE0F（变体选择符）与 200D（ZWJ）
        if drop_fe0f and cp == 0xFE0F:
            continue
        if drop_zwj and cp == 0x200D:
            continue
        if is_emoji_codepoint(cp):
            result.append(cp)
    # 去重并保持顺序
    seen = set()
    filtered = []
    for cp in result:
        if cp not in seen:
            seen.add(cp)
            filtered.append(cp)
    return [format_codepoint(cp) for cp in filtered]


def process_json(in_path: Path, out_path: Path, drop_fe0f: bool, drop_zwj: bool) -> None:
    """Process a JSON mapping file, filtering `codepoints` per entry.

    中文说明：读取 JSON 列表，按条目过滤 `codepoints`，并写回输出文件。
    """
    data = json.loads(in_path.read_text("utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of entries")
    for e in data:
        tokens = e.get("codepoints", []) or []
        if isinstance(tokens, list):
            e["codepoints"] = filter_codepoints(tokens, drop_fe0f=drop_fe0f, drop_zwj=drop_zwj)
        else:
            # 若是字符串（异常情况），按空格切分再过滤
            parts = str(tokens).split()
            e["codepoints"] = filter_codepoints(parts, drop_fe0f=drop_fe0f, drop_zwj=drop_zwj)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")


def process_csv(in_path: Path, out_path: Path, drop_fe0f: bool, drop_zwj: bool) -> None:
    """Process a CSV mapping file, filtering `codepoints` column.

    中文说明：读取 CSV，过滤 `codepoints` 列（以空格分隔的 'U+XXXX'），写回输出文件。
    """
    with in_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", "utf-8")
        return
    header = rows[0]
    # 中文说明：找到 codepoints 列索引
    try:
        idx = header.index("codepoints")
    except ValueError:
        raise ValueError("CSV must contain 'codepoints' column")
    out_rows = [header]
    for r in rows[1:]:
        tokens = (r[idx] or "").split()
        filtered = filter_codepoints(tokens, drop_fe0f=drop_fe0f, drop_zwj=drop_zwj)
        r[idx] = " ".join(filtered)
        out_rows.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(out_rows)


def main() -> None:
    """CLI entry: filter codepoints in JSON/CSV mapping files.

    中文说明：根据命令行参数处理指定的 JSON/CSV 文件，输出筛选后的结果。
    """
    args = parse_args()
    drop_fe0f = True if args.exclude_variation else True
    drop_zwj = True if args.exclude_zwj else True

    if args.in_json:
        in_json = Path(args.in_json)
        out_json = Path(args.out_json) if args.out_json else in_json
        process_json(in_json, out_json, drop_fe0f=drop_fe0f, drop_zwj=drop_zwj)
        print(f"[Filtered] JSON written: {out_json}")

    if args.in_csv:
        in_csv = Path(args.in_csv)
        out_csv = Path(args.out_csv) if args.out_csv else in_csv
        process_csv(in_csv, out_csv, drop_fe0f=drop_fe0f, drop_zwj=drop_zwj)
        print(f"[Filtered] CSV written: {out_csv}")


if __name__ == "__main__":
    main()