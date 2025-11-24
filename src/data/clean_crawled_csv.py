"""
中文数据清洗脚本：
1) 仅保留指定列：bvid, emoji_alt, emoji_name, message, sentence, mid, uname, ctime_iso
2) 去除由同一用户（优先 mid，缺失则用 uname）重复发布的完全相同内容（优先 message，缺失则用 sentence）

使用示例（Windows）：
  python -m src.data.clean_crawled_csv --src-dir "emoji-finegrained-emotion\\data\\vendor\\crawl\\by_bvid" \
    --out-dir "emoji-finegrained-emotion\\data\\vendor\\crawl\\cleaned" \
    --combined-out "emoji-finegrained-emotion\\data\\vendor\\crawl\\cleaned_combined.csv"
"""

import os
import io
import csv
import argparse
from typing import List, Dict, Tuple


# 中文说明：统一路径解析函数（跨平台分隔符兼容、相对项目根解析）
def resolve_project_path(p: str) -> str:
    """将输入路径规范化为绝对路径。

    - 统一分隔符到当前系统的 `os.sep`；
    - 绝对路径直接规范化；
    - 相对路径以仓库根目录解析（src/data → 项目根向上两级）。
    - 若误带 'emoji-finegrained-emotion' 前缀，自动剥离一次。
    """
    if not p:
        return ''
    # 中文行间注释：项目根为当前文件的上上级目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # 中文行间注释：统一正反斜杠为当前系统分隔符
    norm = p.replace('\\', os.sep).replace('/', os.sep)
    if os.path.isabs(norm):
        return os.path.normpath(norm)
    prefix = f"emoji-finegrained-emotion{os.sep}"
    if norm.lower().startswith(prefix):
        norm = norm[len(prefix):]
    return os.path.normpath(os.path.join(project_root, norm))


# 目标列：输出文件只保留这些列，顺序固定
TARGET_COLUMNS = [
    'bvid', 'emoji_alt', 'emoji_name', 'message', 'sentence', 'mid', 'uname', 'ctime_iso'
]


def normalize_text(s: str) -> str:
    """中文说明：对文本进行轻量规范化——去除首尾空白并压缩内部空白。
    注意：不进行大小写或全/半角转换，避免过度清洗导致误去重。
    """
    if s is None:
        return ''
    s = s.strip()
    # 将连续空白压缩为单个空格
    return ' '.join(s.split())


def select_and_dedup_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """中文说明：筛选列并按用户+内容去重。
    - 用户ID：优先使用 mid（若为空则用 uname）
    - 内容：优先使用 message（若为空则用 sentence）
    - 去重键：f"{user_key}::{content_key}"，完全相同才算重复
    返回：
      - cleaned_rows：清洗后的行（仅保留目标列）
      - stats：统计信息（总数、保留、重复丢弃、内容缺失丢弃）
    """
    stats = {
        'total': 0,
        'kept': 0,
        'dropped_dupe': 0,
        'dropped_empty': 0,
    }
    seen = set()
    cleaned = []

    for r in rows:
        stats['total'] += 1
        # 读取原字段
        bvid = (r.get('bvid') or '').strip()
        emoji_alt = (r.get('emoji_alt') or '').strip()
        emoji_name = (r.get('emoji_name') or '').strip()
        message = normalize_text(r.get('message') or '')
        sentence = normalize_text(r.get('sentence') or '')
        mid = (r.get('mid') or '').strip()
        uname = (r.get('uname') or '').strip()
        ctime_iso = (r.get('ctime_iso') or '').strip()

        # 内容选择：优先 message，其次 sentence
        content = message if message else sentence
        if not content:
            stats['dropped_empty'] += 1
            continue

        # 用户选择：优先 mid，其次 uname
        user_key = mid if mid else uname
        if not user_key:
            # 若用户也为空，难以定义“同一人”，此处保留但可选丢弃；选择保留以避免过多数据损失
            user_key = 'UNKNOWN_USER'

        # 去重键
        key = f"{user_key}::{content}"
        if key in seen:
            stats['dropped_dupe'] += 1
            continue
        seen.add(key)

        # 只保留目标列
        cleaned.append({
            'bvid': bvid,
            'emoji_alt': emoji_alt,
            'emoji_name': emoji_name,
            'message': message,
            'sentence': sentence,
            'mid': mid,
            'uname': uname,
            'ctime_iso': ctime_iso,
        })
        stats['kept'] += 1

    return cleaned, stats


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    """中文说明：读取CSV并返回字典行。假设首行是表头且包含目标字段。"""
    rows: List[Dict[str, str]] = []
    # 中文行间注释：使用 utf-8-sig 以兼容可能存在的 BOM，避免首列字段名带不可见字符
    with io.open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: str, rows: List[Dict[str, str]]):
    """中文说明：写出CSV，仅包含目标列，使用UTF-8编码。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in TARGET_COLUMNS})


def parse_args():
    parser = argparse.ArgumentParser(description='清洗爬取CSV：列筛选与按用户+内容去重')
    parser.add_argument('--src-dir', default=r'emoji-finegrained-emotion\\data\\vendor\\crawl\\by_bvid',
                        help='源CSV目录（默认：by_bvid）')
    parser.add_argument('--pattern', default='*.csv', help='匹配文件模式（默认：*.csv）')
    parser.add_argument('--out-dir', default=r'emoji-finegrained-emotion\\data\\vendor\\crawl\\cleaned',
                        help='清洗后输出目录')
    parser.add_argument('--combined-out', default=None,
                        help='合并清洗输出文件路径（可选）')
    return parser.parse_args()


def main():
    args = parse_args()
    # 中文说明：参数路径统一规范化为绝对路径，防止跨平台分隔符问题
    src_dir = resolve_project_path(args.src_dir)
    out_dir = resolve_project_path(args.out_dir)
    pattern = args.pattern
    combined_out = resolve_project_path(args.combined_out) if args.combined_out else None

    if not os.path.isdir(src_dir):
        print(f"[error] 源目录不存在：{src_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)

    # 收集待处理CSV文件
    import glob
    files = sorted(glob.glob(os.path.join(src_dir, pattern)))
    if not files:
        print(f"[warn] 未在 {src_dir} 下找到匹配 {pattern} 的CSV文件。")
        return

    combined_rows: List[Dict[str, str]] = []
    total_stats = {'total': 0, 'kept': 0, 'dropped_dupe': 0, 'dropped_empty': 0}

    for fp in files:
        name = os.path.basename(fp)
        rows = read_csv_rows(fp)
        cleaned, stats = select_and_dedup_rows(rows)
        out_fp = os.path.join(out_dir, name)
        write_csv(out_fp, cleaned)

        # 汇总统计与打印
        total_stats['total'] += stats['total']
        total_stats['kept'] += stats['kept']
        total_stats['dropped_dupe'] += stats['dropped_dupe']
        total_stats['dropped_empty'] += stats['dropped_empty']
        print(f"[info] {name}: 总={stats['total']} 保留={stats['kept']} 重复丢弃={stats['dropped_dupe']} 空内容丢弃={stats['dropped_empty']} → 输出={out_fp}")

        # 合并输出收集
        if combined_out:
            combined_rows.extend(cleaned)

    # 写出合并文件（如指定）
    if combined_out:
        write_csv(combined_out, combined_rows)
        print(f"[info] 合并输出完成：{combined_out}，总保留行数={len(combined_rows)}")

    print(f"[summary] 清洗汇总：总={total_stats['total']} 保留={total_stats['kept']} 重复丢弃={total_stats['dropped_dupe']} 空内容丢弃={total_stats['dropped_empty']}")


if __name__ == '__main__':
    main()
