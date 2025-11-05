"""
Build a consolidated matching table for Emoji-Dis 112 emojis across 4 platforms.

This script reads platform mapping CSV files (baidu, weibo, douyin, bilibili),
derives the canonical Emoji-Dis 112 list from Baidu CSV by default, and outputs
`data/vendor/emoji_dis_match_table.csv` containing, for each emoji:
- emoji character and canonical codepoints
- per-platform display name
- per-platform local image_path (absolute, Windows-style)
- per-platform match status (mapped/unmapped/vendor_only)

Usage:
    python -m src.data.build_match_table \
        --root "E:\\OneDrive - The Chinese University of Hong Kong\\College\\Course Content\\y3\\AIST4010\\project\\emoji-finegrained-emotion" \
        [--canonical "data/vendor/baidu_emojiall_map.csv"]

Notes:
- CSV inputs are expected to have headers: emoji, codepoints, name, img_url,
  detail_url, kind, description, image_path.
- Output CSV uses QUOTE_ALL to ensure paths are quoted.
"""

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# 英文函数注释；中文行间注释在实现中补充
def load_platform_csv(path: Path) -> List[Dict[str, str]]:
    """Load a platform CSV mapping file and return list of row dicts.

    Expects columns: emoji, codepoints, name, kind, image_path, etc.
    Returns empty list if file does not exist.
    """
    # 读取平台CSV映射文件，如果不存在则返回空列表
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "") for k, v in r.items()})
    return rows


def norm_codepoints(cp: str) -> str:
    """Normalize a codepoints string like 'U+1F937 U+2640' to '1f937-2640'.

    This helps create a canonical key for matching across platform CSVs.
    """
    # 规范化codepoints格式：去掉U+，变为小写，空格转为短横连接
    cp = (cp or "").strip()
    if not cp:
        return ""
    parts = []
    for token in cp.split():
        token = token.strip()
        if not token:
            continue
        if token.upper().startswith("U+"):
            token = token[2:]
        parts.append(token.lower())
    return "-".join(parts)


def build_index(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """Build an index dict keyed by normalized codepoints or emoji char.

    Priority: normalized codepoints; fallback: emoji char.
    """
    # 为每个平台的行构建索引：优先使用规范化的codepoints作为键，回退为emoji字符
    index: Dict[str, Dict[str, str]] = {}
    for r in rows:
        key_cp = norm_codepoints(r.get("codepoints", ""))
        key_emoji = (r.get("emoji", "") or "").strip()
        if key_cp:
            index[key_cp] = r
        elif key_emoji:
            # 没有codepoints时用emoji作为键
            index[key_emoji] = r
    return index


def choose_canonical(rows: List[Dict[str, str]]) -> List[Tuple[str, str, str]]:
    """Choose canonical Emoji-Dis 112 list from Baidu rows.

    Returns list of tuples: (emoji, normalized_codepoints, canonical_name).
    Filters out vendor_only and rows without emoji/codepoints.
    """
    # 从Baidu的行中选择作为Emoji-Dis 112的基准列表
    # 过滤掉vendor_only条目，以及缺失emoji或codepoints的行
    result: List[Tuple[str, str, str]] = []
    seen: set = set()
    for r in rows:
        kind = (r.get("kind", "") or "").strip()
        if kind == "vendor_only":
            continue  # 平台专属原创表情不计入Emoji-Dis 112
        emoji = (r.get("emoji", "") or "").strip()
        cp_norm = norm_codepoints(r.get("codepoints", ""))
        if not emoji or not cp_norm:
            continue
        if (emoji, cp_norm) in seen:
            continue
        seen.add((emoji, cp_norm))
        name = (r.get("name", "") or "").strip()
        result.append((emoji, cp_norm, name))
    return result


def match_row(index: Dict[str, Dict[str, str]], emoji: str, cp_norm: str) -> Optional[Dict[str, str]]:
    """Find a matching row in the platform index by codepoints or emoji.

    Returns the row dict or None if not found.
    """
    # 先按规范化codepoints匹配，若失败则按emoji字符匹配
    if cp_norm and cp_norm in index:
        return index[cp_norm]
    if emoji and emoji in index:
        return index[emoji]
    return None


def main():
    # 解析命令行参数：项目根路径和可选的canonical文件
    parser = argparse.ArgumentParser(description="Build Emoji-Dis 112 match table across platforms")
    parser.add_argument("--root", required=True, help="Project root absolute path")
    parser.add_argument("--canonical", default="data/vendor/baidu_emojiall_map.csv", help="Canonical CSV path for Emoji-Dis 112")
    args = parser.parse_args()

    root = Path(args.root)
    # 校验根路径存在
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    # 组装各平台CSV路径（相对项目根）
    vendor_dir = root / "data" / "vendor"
    baidu_csv = vendor_dir / "baidu_emojiall_map.csv"
    weibo_csv = vendor_dir / "weibo_emojiall_map.csv"
    douyin_csv = vendor_dir / "douyin_emojiall_map.csv"
    bilibili_csv = vendor_dir / "bilibili_emojiall_map.csv"

    # 加载各平台CSV
    baidu_rows = load_platform_csv(baidu_csv)
    weibo_rows = load_platform_csv(weibo_csv)
    douyin_rows = load_platform_csv(douyin_csv)
    bilibili_rows = load_platform_csv(bilibili_csv)

    # 构建索引
    idx_baidu = build_index(baidu_rows)
    idx_weibo = build_index(weibo_rows)
    idx_douyin = build_index(douyin_rows)
    idx_bilibili = build_index(bilibili_rows)

    # 基准列表：默认从Baidu CSV抽取Emoji-Dis的112个（过滤vendor_only）
    canonical_rows = load_platform_csv(root / args.canonical) if args.canonical else baidu_rows
    canonical_list = choose_canonical(canonical_rows if canonical_rows else baidu_rows)

    # 输出CSV路径
    out_csv = vendor_dir / "emoji_dis_match_table.csv"

    # 写出匹配盘点表，强制所有字段加引号
    headers = [
        "emoji",
        "codepoints_norm",
        "name_baidu",
        "path_baidu",
        "status_baidu",
        "name_weibo",
        "path_weibo",
        "status_weibo",
        "name_douyin",
        "path_douyin",
        "status_douyin",
        "name_bilibili",
        "path_bilibili",
        "status_bilibili",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        # 遍历112个emoji，逐平台匹配写出记录
        for emoji, cp_norm, canonical_name in canonical_list:
            row_out = {
                "emoji": emoji,
                "codepoints_norm": cp_norm,
                "name_baidu": canonical_name,
                "path_baidu": "",
                "status_baidu": "unmapped",
                "name_weibo": "",
                "path_weibo": "",
                "status_weibo": "unmapped",
                "name_douyin": "",
                "path_douyin": "",
                "status_douyin": "unmapped",
                "name_bilibili": "",
                "path_bilibili": "",
                "status_bilibili": "unmapped",
            }

            # Baidu匹配
            m_baidu = match_row(idx_baidu, emoji, cp_norm)
            if m_baidu:
                row_out["path_baidu"] = (m_baidu.get("image_path", "") or "").strip()
                row_out["status_baidu"] = (m_baidu.get("kind", "") or "").strip() or "mapped"

            # Weibo匹配
            m_weibo = match_row(idx_weibo, emoji, cp_norm)
            if m_weibo:
                row_out["name_weibo"] = (m_weibo.get("name", "") or "").strip()
                row_out["path_weibo"] = (m_weibo.get("image_path", "") or "").strip()
                row_out["status_weibo"] = (m_weibo.get("kind", "") or "").strip() or "mapped"

            # Douyin匹配
            m_douyin = match_row(idx_douyin, emoji, cp_norm)
            if m_douyin:
                row_out["name_douyin"] = (m_douyin.get("name", "") or "").strip()
                row_out["path_douyin"] = (m_douyin.get("image_path", "") or "").strip()
                row_out["status_douyin"] = (m_douyin.get("kind", "") or "").strip() or "mapped"

            # Bilibili匹配
            m_bilibili = match_row(idx_bilibili, emoji, cp_norm)
            if m_bilibili:
                row_out["name_bilibili"] = (m_bilibili.get("name", "") or "").strip()
                row_out["path_bilibili"] = (m_bilibili.get("image_path", "") or "").strip()
                row_out["status_bilibili"] = (m_bilibili.get("kind", "") or "").strip() or "mapped"

            writer.writerow(row_out)

    # 完成后打印输出位置提示
    print(f"Match table written: {out_csv}")


if __name__ == "__main__":
    main()