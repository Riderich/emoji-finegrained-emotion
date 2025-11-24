# -*- coding: utf-8 -*-
# 将 cookies.json 规范化为最简单字符串形式的辅助脚本
# 中文行间注释：该脚本读取 cookies.json（支持字符串/字典/数组三种格式），
# 然后输出或重写为 "name=value; name=value" 的 Cookie 头字符串，便于脚本直接使用

import json
import os
import argparse
from typing import Any, Dict, List, Tuple


def _read_json(path: str) -> Any:
    # 中文行间注释：读取 JSON 文件，返回 Python 对象
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    # 中文行间注释：写入 JSON（带缩进），使用临时文件确保原子性
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _backup_json(path: str, original_obj: Any) -> str:
    # 中文行间注释：备份原始 cookies.json，以便需要时恢复
    backup_path = path.replace('.json', '.original.json')
    if not os.path.exists(backup_path):
        _write_json(backup_path, original_obj)
    return backup_path


def _pairs_from_string(s: str) -> List[Tuple[str, str]]:
    # 中文行间注释：从字符串形式解析出 (name, value) 列表
    pairs: List[Tuple[str, str]] = []
    for seg in s.split(';'):
        seg = seg.strip()
        if not seg or '=' not in seg:
            continue
        name, val = seg.split('=', 1)
        pairs.append((name.strip(), val.strip()))
    return pairs


def _pairs_from_dict(d: Dict[str, Any]) -> List[Tuple[str, str]]:
    # 中文行间注释：从字典形式提取键值对
    pairs: List[Tuple[str, str]] = []
    for k, v in d.items():
        pairs.append((str(k), str(v)))
    return pairs


def _pairs_from_list(lst: List[Any]) -> List[Tuple[str, str]]:
    # 中文行间注释：从数组（浏览器导出的 cookie 项）中提取 name/value
    pairs: List[Tuple[str, str]] = []
    for item in lst:
        if not isinstance(item, dict):
            continue
        # 常见字段名：name/value 或 Name/Value
        name = item.get('name') or item.get('Name') or item.get('key') or item.get('Key')
        value = item.get('value') or item.get('Value') or item.get('val') or item.get('Val')
        if name is None or value is None:
            # 中文行间注释：若缺少 name/value，则跳过该项（避免误解析元数据，如 domain/path/expires）
            continue
        pairs.append((str(name), str(value)))
    return pairs


def normalize_cookie(obj: Any, minimal: bool = False) -> str:
    # 中文行间注释：将任意支持的格式规范化为 Cookie 头字符串
    pairs: List[Tuple[str, str]] = []
    if isinstance(obj, str):
        pairs = _pairs_from_string(obj.strip())
    elif isinstance(obj, dict):
        pairs = _pairs_from_dict(obj)
    elif isinstance(obj, list):
        pairs = _pairs_from_list(obj)
    else:
        raise TypeError('不支持的 cookies.json 类型，仅支持 字符串/字典/数组')

    # 中文行间注释：去重（保留最后一次出现的值）
    merged: Dict[str, str] = {}
    for name, value in pairs:
        merged[name] = value

    # 中文行间注释：最小化模式仅保留关键登录态和高价值指纹字段
    if minimal:
        whitelist = [
            'SESSDATA', 'bili_jct', 'DedeUserID', 'DedeUserID__ckMd5',
            '_uuid', 'sid', 'buvid3', 'buvid4', 'buvid_fp', 'b_nut'
        ]
        merged = {k: merged[k] for k in whitelist if k in merged}

    # 中文行间注释：稳定的输出顺序（优先关键登录态），其余按名称字母排序
    priority = ['SESSDATA', 'bili_jct', 'DedeUserID', 'DedeUserID__ckMd5']
    keys = list(merged.keys())
    keys.sort(key=lambda k: (0 if k in priority else 1, priority.index(k) if k in priority else k.lower()))

    header = '; '.join(f'{k}={merged[k]}' for k in keys)
    return header


def main():
    # 中文行间注释：命令行参数解析
    default_path = os.path.join('emoji-finegrained-emotion', 'src', 'data', 'cookies.json')
    parser = argparse.ArgumentParser(description='将 cookies.json 处理为最简单的字符串形式')
    parser.add_argument('--path', default=default_path, help='cookies.json 文件路径')
    parser.add_argument('--minimal', action='store_true', help='仅保留关键字段，生成最小化字符串')
    parser.add_argument('--write', action='store_true', help='重写 cookies.json（JSON 字符串），并备份原始文件')
    parser.add_argument('--output', default=None, help='将字符串写入到指定 txt 路径（不改 cookies.json）')
    args = parser.parse_args()

    obj = _read_json(args.path)
    header = normalize_cookie(obj, minimal=args.minimal)

    if not header:
        print('警告：生成的 Cookie 字符串为空，请检查 cookies.json 格式与内容')
        return

    # 中文行间注释：根据参数执行写出逻辑
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(header)
        print(f'已写出简化 Cookie 字符串到: {args.output}')

    if args.write:
        _backup_json(args.path, obj)
        _write_json(args.path, header)
        print('已重写 cookies.json 为最简单的 JSON 字符串形式（原始文件已备份为 cookies.original.json）')

    # 中文行间注释：同时将结果打印，方便用户快速复制
    print('简化后的 Cookie 字符串：')
    print(header)


if __name__ == '__main__':
    main()

