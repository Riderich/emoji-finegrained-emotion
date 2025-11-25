r"""
B 站评论区表情爬取脚本（仅按 AID 抓取，不做句子切分）

功能概述：
- 通过接口 `https://api.bilibili.com/x/v2/reply` 以 `oid=<aid>` 抓取评论；
- 仅筛选“带有表情”的评论（`content.emote` 不为空），不进行文本句子切分；
- 输出每个表情一行，包含 BV、评论文本、表情占位与图片链接等。

新增：
- 同时识别评论文本中的 Unicode 表情（如 🙏、😂 等），即使 `content.emote` 为空也会写出对应行；
- 对 Unicode 表情，`emoji_alt` 与 `emoji_name` 使用该表情字符，`emote_url` 为空。

接口参数说明（简述）：
- `pn`: 页码，从 1 开始；
- `type`: 资源类型，这里使用 `1` 表示“视频”。B 站评论接口需要区分对象类型，
  例如视频、专栏、动态等，不同类型具有不同的 `type` 值；`type=1` 对应视频。
- `oid`: 对象 ID，这里统一传数字 `aid`（例如 `170001`）。

说明：脚本仍接受 BV 列表输入，但会先将 BV 转换为 AID，再按 AID 分页抓取评论（仅保留 AID 路径，已移除 BV 直传与自动模式相关代码）。

输出 CSV：`data/vendor/crawl/bilibili_emoji_sentences.csv`
字段：
- `bvid`: 视频 BV 号
- `rpid`: 评论 ID
- `emoji_alt`: 表情占位（如 `[笑哭]`）
- `emoji_name`: 平台显示名称（如 `笑哭`）
- `message`: 整条评论文本（不做切分）
- `sentence`: 与 `message` 相同（保持与你期望的字段结构一致）
- `emote_url`: 表情图片链接
- `mid`: 用户 ID
- `uname`: 用户名
- `ctime_iso`: 评论时间（ISO 格式）

运行示例（Windows）：
python -m emoji-finegrained-emotion.src.data.crawl_bilibili --root "E:\OneDrive - The Chinese University of Hong Kong\College\Course Content\y3\AIST4010\project\emoji-finegrained-emotion" --bvids BV1fV4y1N7Rc --max-pages 2 --sleep-seconds 0.6
"""

import csv
import os
import time
import argparse
from datetime import datetime, timezone
import re
import json
import requests
import random
from typing import Optional


# 构造一个通用的 Session，用于设置 UA 和复用连接
def build_session(sessdata: Optional[str] = None) -> requests.Session:
    """构建 requests 会话，设置常见的 UA 等；可选地加入 SESSDATA Cookie。

    - 某些视频评论或排序可能需要登录态；如提供 `SESSDATA` 可提升可访问性。
    """
    s = requests.Session()
    s.headers.update({
        # 伪装常见浏览器 UA，减少被限的概率
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.bilibili.com/",
        "Accept": "application/json, text/plain, */*",
        # 中文行间注释：补充常见请求头，尽量贴近浏览器环境，降低 412 概率
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Origin": "https://www.bilibili.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "X-Requested-With": "XMLHttpRequest",
    })
    # 中文行间注释：优先尝试从同目录 cookies.json 读取“完整Cookie”，其内容由浏览器导出
    try:
        default_cookie_path = os.path.join(os.path.dirname(__file__), 'cookies.json')
        header = load_cookie_header_from_json(default_cookie_path)
        if header:
            s.headers.update({"Cookie": header})
            # 中文行间注释：简单诊断 Cookie 完整性，提示是否缺少关键登录态
            lower = header.lower()
            if ("sessdata=" not in lower) or ("bili_jct=" not in lower):
                print("[warn] Cookie 可能不完整：缺少 SESSDATA/bili_jct，易触发 412。请更新 src/data/cookies.json。")
        elif sessdata:
            # 中文行间注释：若文件不可用，则退回到只设置 SESSDATA
            s.headers.update({"Cookie": f"SESSDATA={sessdata}"})
    except Exception:
        # 中文行间注释：读取 cookies.json 失败时不影响流程；如提供了 sessdata 仍可使用
        if sessdata:
            s.headers.update({"Cookie": f"SESSDATA={sessdata}"})
    return s


# 中文行间注释：从浏览器导出的 cookies.json 构建 Cookie 头部字符串
def load_cookie_header_from_json(path: str) -> Optional[str]:
    """读取 `cookies.json` 并生成 Cookie 头字符串，兼容多种格式。

    支持的格式：
    - 浏览器导出列表：`[{"name": "SESSDATA", "value": "..."}, ...]`
    - 纯字符串：`"SESSDATA=...; bili_jct=..."`（直接作为 Cookie 使用）
    - 字典形式：`{"Cookie": "SESSDATA=...; ..."}` 或 `{"cookie": "..."}`
    """
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # 情况一：纯字符串，直接返回
        if isinstance(data, str):
            s = data.strip()
            return s if s else None
        # 情况二：字典，先尝试常见头字段，再回退为键值映射
        if isinstance(data, dict):
            # 中文行间注释：优先处理常见键，兼容不同大小写写法
            for k in ('Cookie', 'cookie', 'header', 'Header', 'value', 'Value'):
                v = str(data.get(k) or '').strip()
                if v:
                    return v
            # 中文行间注释：若无上述键，且不像带有浏览器元数据的对象，则按“name=value”拼接
            meta_keys = {'domain', 'path', 'sameSite', 'secure', 'httpOnly', 'expirationDate', 'hostOnly', 'storeId', 'session'}
            if not any(k in data for k in meta_keys):
                pairs = []
                for k, v in data.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if not ks or not vs:
                        continue
                    pairs.append(f"{ks}={vs}")
                if pairs:
                    return '; '.join(pairs)
            return None
        # 情况三：列表（浏览器导出），兼容多种字段名
        if isinstance(data, list):
            pairs = []
            for it in data:
                if not isinstance(it, dict):
                    continue
                # 中文行间注释：兼容 name/value 的大小写及别名
                name = (it.get('name') or it.get('Name') or it.get('key') or it.get('Key') or '').strip()
                value = (it.get('value') or it.get('Value') or it.get('val') or it.get('Val') or '').strip()
                if not name or not value:
                    continue
                pairs.append(f"{name}={value}")
            if not pairs:
                return None
            return '; '.join(pairs)
        return None
    except Exception:
        return None


# 中文行间注释：安全预览 Cookie（仅打印关键名与部分值，避免泄露完整隐私）
def _preview_cookie(session: requests.Session) -> str:
    # 中文行间注释：改进安全预览逻辑——始终包含关键登录态（SESSDATA/bili_jct/DedeUserID 等），再附加少量其他项
    try:
        raw = session.headers.get("Cookie", "")
        if not raw:
            return ""
        entries = []  # [(name, value)]
        for seg in raw.split(';'):
            seg = seg.strip()
            if not seg or '=' not in seg:
                continue
            name, val = seg.split('=', 1)
            entries.append((name.strip(), val.strip()))

        # 建立索引便于查找关键项
        idx = {name.lower(): (name, val) for name, val in entries}
        preview = []

        def mask(name: str, val: str) -> str:
            # 中文行间注释：对敏感值进行打码，只保留首尾少量字符
            if len(val) > 14:
                return f"{name}={val[:8]}...{val[-6:]}"
            return f"{name}={val}"

        # 始终包含关键登录态（若存在）
        for key in ("sessdata", "bili_jct", "dedeuserid", "dedeuserid__ckmd5"):
            if key in idx:
                name, val = idx[key]
                # 其中 SESSDATA/bili_jct 视为敏感，打码显示
                if key in ("sessdata", "bili_jct"):
                    preview.append(mask(name, val))
                else:
                    preview.append(f"{name}={val}")

        # 补充部分普通项，只显示名称，避免日志过长与隐私泄露
        shown_names = {p.split('=')[0].lower() for p in preview}
        for name, _ in entries:
            if len(preview) >= 10:
                break
            if name.lower() in shown_names:
                continue
            preview.append(name)

        return '; '.join(preview)
    except Exception:
        return ""


def iso_from_ctime(ctime: int) -> str:
    """将评论的 ctime（秒级时间戳）转为 ISO 格式字符串。"""
    dt = datetime.fromtimestamp(ctime, tz=timezone.utc)
    return dt.isoformat()


def extract_unicode_emojis(text: str) -> list:
    """从文本中提取出现的 Unicode 表情字符（简单覆盖常见范围）。

    - 仅用于判定“带有表情的文本”，不做严格的全量覆盖；
    - 返回去重后的顺序列表（按首次出现顺序）。
    """
    if not text:
        return []

    # 中文行间注释：定义若干常见 Emoji 的 Unicode 范围（并不穷尽）
    ranges = [
        (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
        (0x1F600, 0x1F64F),  # Emoticons
        (0x1F680, 0x1F6FF),  # Transport and Map Symbols
        (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        (0x2600, 0x26FF),    # Misc symbols
        (0x2700, 0x27BF),    # Dingbats
    ]

    seen = set()
    out = []
    for ch in text:
        cp = ord(ch)
        for lo, hi in ranges:
            if lo <= cp <= hi:
                if ch not in seen:
                    seen.add(ch)
                    out.append(ch)
                break
    return out


# （已移除）按 BV 号抓取评论页：统一改为 BV→AID 后按 AID 抓取


# 中文行间注释：根据 BV 号获取数字 AID（视频 ID），用于评论接口的兜底回退
def get_aid_by_bvid(session: requests.Session, bvid: str) -> int:
    """调用 `x/web-interface/view` 获取视频的 aid（失败返回 0）。"""
    try:
        headers = {"Referer": f"https://www.bilibili.com/video/{bvid}/"}
        resp = session.get("https://api.bilibili.com/x/web-interface/view", params={"bvid": bvid}, headers=headers, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        if j.get("code") != 0:
            return 0
        data = j.get("data") or {}
        # 中文行间注释：为调试 aid 异常，打印关键字段结构与类型
        try:
            print(f"[debug] view.data keys={list(data.keys())[:8]} aid_raw={data.get('aid')} type={type(data.get('aid'))}")
        except Exception:
            pass
        aid = data.get("aid")
        try:
            return int(aid)
        except Exception:
            return 0
    except Exception:
        return 0


# 中文行间注释：按 AID 抓取评论页（用于 BV 失败时的兜底）
def fetch_replies_page_by_aid(session: requests.Session, aid: int, page: int, bvid_referer: Optional[str] = None) -> dict:
    """以 `oid=<aid>` 抓取评论页（失败返回空字典）。

    中文行间注释：可选传入 `bvid_referer`，用于设置更贴近真实视频页的 Referer，
    例如 `https://www.bilibili.com/video/BVxxxx/`，在某些场景下有助于降低 412 风险。
    """
    url = "https://api.bilibili.com/x/v2/reply"
    params = {
        "pn": page,
        "type": 1,
        "oid": aid,
        "sort": 0,
    }
    try:
        # 中文行间注释：优先使用视频页 Referer；若无则兜底主页
        headers = {"Referer": f"https://www.bilibili.com/video/{bvid_referer}/"} if bvid_referer else {"Referer": "https://www.bilibili.com/"}
        resp = session.get(url, params=params, headers=headers, timeout=10)
        status = resp.status_code
        # 中文行间注释：限流/风控（412/429）时，返回带有 code 的占位结构，便于上层识别并自适应退避
        if status in (412, 429):
            preview = ""
            try:
                preview = resp.text[:120]
            except Exception:
                pass
            print(f"[rate-limit] reply限流 status={status} aid={aid} pn={page}，将触发自适应退避。片段: {preview}")
            return {"code": status, "message": "rate_limited"}
        if status != 200:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}
    except Exception:
        return {}

    
# （已移除）按模式抓取（aid/bvid/auto）：统一改为仅按 AID 抓取


# 中文行间注释：新版评论接口，基于 cursor/next 分页；常用于网页端
def fetch_replies_main_by_aid(session: requests.Session, aid: int, next_cursor: int = 0, ps: int = 20, mode: int = 3) -> dict:
    """调用 `x/v2/reply/main` 接口以 AID 抓取评论（失败返回空字典）。

    参数含义（经验值）：
    - `next`: 游标分页起点，首页通常为 0；返回 JSON 中会包含新的 next。
    - `ps`: 每页大小，常见为 20 或 30；这里默认 20。
    - `mode`: 排序/展示模式（网页端通常用 3）。
    """
    url = "https://api.bilibili.com/x/v2/reply/main"
    params = {
        "oid": aid,
        "type": 1,
        "next": next_cursor,
        "ps": ps,
        "mode": mode,
    }
    try:
        resp = session.get(url, params=params, timeout=10)
        status = resp.status_code
        if status in (412, 429):
            preview = ""
            try:
                preview = resp.text[:120]
            except Exception:
                pass
            print(f"[rate-limit] reply.main 限流 status={status} aid={aid} next={next_cursor} ps={ps}，20秒后重试。片段: {preview}")
            return {"code": status, "message": "rate_limited"}
        if status != 200:
            return {}
        try:
            return resp.json()
        except Exception:
            return {}
    except Exception:
        return {}


def extract_rows_from_reply_item(bvid: str, item: dict) -> list:
    """从单条评论（含楼中楼）结构中抽取“带有表情”的行记录。

    - 仅当 `content.emote` 存在时，认为该评论“带有表情”。
    - 每个表情一行，`message` 保留整条文本，不做句子切分；
    - `sentence` 字段与 `message` 相同，以保持字段结构一致。
    """
    rows = []

    # 顶层评论的基本信息
    rpid = item.get("rpid")
    content = item.get("content") or {}
    message = content.get("message") or ""
    emote = content.get("emote") or {}

    # 用户信息与时间
    member = item.get("member") or {}
    mid = member.get("mid")
    uname = member.get("uname")
    ctime = item.get("ctime") or 0
    ctime_iso = iso_from_ctime(ctime) if ctime else ""

    # 中文行间注释：为避免重复，维护已写出的占位集合
    emitted = set()

    # 如果顶层评论带有表情（emote 字典非空），则按每个表情写一行
    if isinstance(emote, dict) and emote:
        for alt, info in emote.items():
            # alt 形如 "[笑哭]"；info 包含 name/url 等
            emoji_alt = alt or ""
            emoji_name = (info or {}).get("name") or ""
            emote_url = (info or {}).get("url") or ""

            rows.append({
                "bvid": bvid,
                "rpid": rpid,
                "emoji_alt": emoji_alt,
                "emoji_name": emoji_name,
                "message": message,
                "sentence": message,  # 不切分，保持和 message 一致
                "emote_url": emote_url,
                "mid": mid,
                "uname": uname,
                "ctime_iso": ctime_iso,
            })
            emitted.add(emoji_alt)

    # 额外：识别文本中的 Unicode 表情（即使没有 emote 字典）
    uni_emojis = extract_unicode_emojis(message)
    for uni in uni_emojis:
        rows.append({
            "bvid": bvid,
            "rpid": rpid,
            "emoji_alt": uni,       # 直接使用该字符
            "emoji_name": uni,      # 同字符
            "message": message,
            "sentence": message,    # 不切分
            "emote_url": "",       # Unicode 无平台图片 URL
            "mid": mid,
            "uname": uname,
            "ctime_iso": ctime_iso,
        })

    # 额外：识别文本中的方括号占位（如 [doge]），即使 emote 字典为空也写出
    # 中文行间注释：通过正则匹配 [xxx] 形式，占位名称为去掉方括号后的内容
    bracket_markers = re.findall(r"\[[^\[\]]+\]", message or "")
    for marker in bracket_markers:
        if marker in emitted:
            continue
        info = (emote or {}).get(marker) or {}
        emoji_name_guess = (info.get("name") or marker.strip("[]")) if isinstance(info, dict) else marker.strip("[]")
        emote_url_guess = (info.get("url") or "") if isinstance(info, dict) else ""

        rows.append({
            "bvid": bvid,
            "rpid": rpid,
            "emoji_alt": marker,
            "emoji_name": emoji_name_guess,
            "message": message,
            "sentence": message,
            "emote_url": emote_url_guess,
            "mid": mid,
            "uname": uname,
            "ctime_iso": ctime_iso,
        })
        emitted.add(marker)

    # 处理楼中楼（如果有），结构一般为 item["replies"] 列表
    replies = item.get("replies") or []
    for sub in replies:
        sub_rpid = sub.get("rpid")
        sub_content = sub.get("content") or {}
        sub_message = sub_content.get("message") or ""
        sub_emote = sub_content.get("emote") or {}

        sub_member = sub.get("member") or {}
        sub_mid = sub_member.get("mid")
        sub_uname = sub_member.get("uname")
        sub_ctime = sub.get("ctime") or 0
        sub_ctime_iso = iso_from_ctime(sub_ctime) if sub_ctime else ""

        sub_emitted = set()
        if isinstance(sub_emote, dict) and sub_emote:
            for alt, info in sub_emote.items():
                emoji_alt = alt or ""
                emoji_name = (info or {}).get("name") or ""
                emote_url = (info or {}).get("url") or ""

                rows.append({
                    "bvid": bvid,
                    "rpid": sub_rpid,
                    "emoji_alt": emoji_alt,
                    "emoji_name": emoji_name,
                    "message": sub_message,
                    "sentence": sub_message,
                    "emote_url": emote_url,
                    "mid": sub_mid,
                    "uname": sub_uname,
                    "ctime_iso": sub_ctime_iso,
                })
                sub_emitted.add(emoji_alt)

        # 楼中楼的 Unicode 表情识别
        sub_uni_emojis = extract_unicode_emojis(sub_message)
        for uni in sub_uni_emojis:
            rows.append({
                "bvid": bvid,
                "rpid": sub_rpid,
                "emoji_alt": uni,
                "emoji_name": uni,
                "message": sub_message,
                "sentence": sub_message,
                "emote_url": "",
                "mid": sub_mid,
                "uname": sub_uname,
                "ctime_iso": sub_ctime_iso,
            })

        # 楼中楼的方括号占位识别（如 [doge]）
        sub_bracket_markers = re.findall(r"\[[^\[\]]+\]", sub_message or "")
        for marker in sub_bracket_markers:
            if marker in sub_emitted:
                continue
            info = (sub_emote or {}).get(marker) or {}
            emoji_name_guess = (info.get("name") or marker.strip("[]")) if isinstance(info, dict) else marker.strip("[]")
            emote_url_guess = (info.get("url") or "") if isinstance(info, dict) else ""

            rows.append({
                "bvid": bvid,
                "rpid": sub_rpid,
                "emoji_alt": marker,
                "emoji_name": emoji_name_guess,
                "message": sub_message,
                "sentence": sub_message,
                "emote_url": emote_url_guess,
                "mid": sub_mid,
                "uname": sub_uname,
                "ctime_iso": sub_ctime_iso,
            })
            sub_emitted.add(marker)

    return rows


def crawl_bilibili_for_bvid(session: requests.Session, bvid: str, max_pages: int, sleep_seconds: float) -> list:
    """按 BV 抓取，但内部统一 BV→AID 后以 `oid=aid` 分页抓取。

    中文行间注释：仅保留 AID 路径，移除所有 BV/auto 模式与相关回退逻辑。
    """
    all_rows = []
    # 中文行间注释：先将 BV 转换为 AID，失败则直接返回空结果
    aid = get_aid_by_bvid(session, bvid)
    if not aid:
        print(f"[warn] 无法获取 AID，跳过 BV={bvid}")
        return all_rows

    for pn in range(1, max_pages + 1):
        # 中文行间注释：为每一页抓取增加“限流自适应”重试机制——遇到 412/429 等限流状态，等待20秒后重试当前页
        stop_this_bv = False
        attempts = 0
        while True:
            # 中文行间注释：以 AID 调用旧版分页接口，Referer 设置为视频页以降低 412 风险
            data = fetch_replies_page_by_aid(session, aid, pn, bvid_referer=bvid)
            code_prim = (data or {}).get("code")
            # 中文行间注释：-400（超出最大偏移量）表示页码超过实际页数，应提前停止该 BV
            if code_prim == -400:
                print(f"[info] bvid={bvid} aid={aid} pn={pn} 返回-400：超过最大偏移量，停止该BV。")
                stop_this_bv = True
                break
            if not data or code_prim != 0:
                # 中文行间注释：限流状态（412/429），等待20秒后重试当前页；最多重试3次
                if code_prim in (412, 429):
                    print(f"[rate-limit] bvid={bvid} aid={aid} pn={pn} code={code_prim}，20秒后重试当前页。")
                    time.sleep(20)
                    attempts += 1
                    if attempts < 3:
                        continue
                    else:
                        print(f"[warn] bvid={bvid} aid={aid} pn={pn} 限流重试超过上限，跳过该BV。")
                        stop_this_bv = True
                        break
                # 中文行间注释：其他错误，轻度退避后跳过该页
                jitter_prim = random.uniform(0.0, sleep_seconds * 0.5)
                time.sleep(sleep_seconds + jitter_prim)
                break
            # 中文行间注释：成功拿到数据，退出重试循环
            break
        if stop_this_bv:
            break

        if stop_this_bv:
            break
        payload = data.get("data") or {}
        replies = payload.get("replies") or []
        page_info = payload.get("page") or {}
        try:
            print(f"[debug] bvid={bvid} aid={aid} pn={pn} page.count={page_info.get('count', 0)} replies.len={len(replies)}")
        except Exception:
            pass

        if pn == 1 and not replies:
            print(f"[info] bvid={bvid} 首页回复为0，跳过该BV")
            break
        if not replies:
            print(f"[info] bvid={bvid} pn={pn} 当前页无回复，提前停止该BV")
            break

        for item in replies:
            rows = extract_rows_from_reply_item(bvid, item)
            all_rows.extend(rows)

        # 中文行间注释：分页间歇，降低限流风险；加入随机抖动
        jitter = random.uniform(0.0, sleep_seconds * 0.5)
        time.sleep(sleep_seconds + jitter)

    return all_rows


# 中文行间注释：从映射 JSON 加载可识别的表情占位名称集合（如 "[doge]"、"[笑哭]" 等）
def load_bilibili_emoji_name_set(root: str, rel_path: str) -> set:
    """读取 `data/vendor/bilibili_emojiall_map.json`，返回包含 `name` 字段的集合。

    - 映射 JSON 的每个对象含 `name`，值形如 `[doge]`；直接将这些值作为合法占位集合。
    - 若文件不存在或解析失败，返回空集合。
    """
    try:
        path = rel_path if os.path.isabs(rel_path) else os.path.join(root, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        names = set()
        if isinstance(data, list):
            for obj in data:
                name = (obj or {}).get("name")
                if isinstance(name, str) and name:
                    names.add(name)
        return names
    except Exception:
        return set()


# 中文行间注释：仅按映射集合过滤并抽取“带表情”的行（顶层 + 楼中楼）
def extract_rows_from_reply_item_mapped(bvid: str, item: dict, name_set: set) -> list:
    """基于映射集合 `name_set` 抽取评论中出现的表情占位，仅写出集合内的表情。

    - 识别两种来源：
      1) `content.emote` 字典中的键（形如 `[笑哭]`），仅当键在 `name_set` 中时写出；
      2) 文本中的方括号占位（正则匹配），仅当占位在 `name_set` 中时写出；
    - 不再写出纯 Unicode 表情（如 😂、🙏），以贴合“表情形式为 [表情名]”的需求。
    """
    rows = []

    # 顶层评论基础信息
    rpid = item.get("rpid")
    content = item.get("content") or {}
    message = content.get("message") or ""
    emote = content.get("emote") or {}

    member = item.get("member") or {}
    mid = member.get("mid")
    uname = member.get("uname")
    ctime = item.get("ctime") or 0
    ctime_iso = iso_from_ctime(ctime) if ctime else ""

    emitted = set()

    # 1) 顶层 emote 字典过滤写出
    if isinstance(emote, dict) and emote:
        for alt, info in emote.items():
            if alt in name_set:
                emoji_alt = alt or ""
                emoji_name = (info or {}).get("name") or emoji_alt.strip("[]")
                emote_url = (info or {}).get("url") or ""
                rows.append({
                    "bvid": bvid,
                    "rpid": rpid,
                    "emoji_alt": emoji_alt,
                    "emoji_name": emoji_name,
                    "message": message,
                    "sentence": message,
                    "emote_url": emote_url,
                    "mid": mid,
                    "uname": uname,
                    "ctime_iso": ctime_iso,
                })
                emitted.add(emoji_alt)

    # 2) 顶层消息文本中的方括号占位过滤写出
    for marker in re.findall(r"\[[^\[\]]+\]", message or ""):
        if marker in emitted:
            continue
        if marker in name_set:
            info = (emote or {}).get(marker) or {}
            emoji_name_guess = (info.get("name") or marker.strip("[]")) if isinstance(info, dict) else marker.strip("[]")
            emote_url_guess = (info.get("url") or "") if isinstance(info, dict) else ""
            rows.append({
                "bvid": bvid,
                "rpid": rpid,
                "emoji_alt": marker,
                "emoji_name": emoji_name_guess,
                "message": message,
                "sentence": message,
                "emote_url": emote_url_guess,
                "mid": mid,
                "uname": uname,
                "ctime_iso": ctime_iso,
            })
            emitted.add(marker)

    # 楼中楼处理
    replies = item.get("replies") or []
    for sub in replies:
        sub_rpid = sub.get("rpid")
        sub_content = sub.get("content") or {}
        sub_message = sub_content.get("message") or ""
        sub_emote = sub_content.get("emote") or {}

        sub_member = sub.get("member") or {}
        sub_mid = sub_member.get("mid")
        sub_uname = sub_member.get("uname")
        sub_ctime = sub.get("ctime") or 0
        sub_ctime_iso = iso_from_ctime(sub_ctime) if sub_ctime else ""

        sub_emitted = set()
        if isinstance(sub_emote, dict) and sub_emote:
            for alt, info in sub_emote.items():
                if alt in name_set:
                    emoji_alt = alt or ""
                    emoji_name = (info or {}).get("name") or emoji_alt.strip("[]")
                    emote_url = (info or {}).get("url") or ""
                    rows.append({
                        "bvid": bvid,
                        "rpid": sub_rpid,
                        "emoji_alt": emoji_alt,
                        "emoji_name": emoji_name,
                        "message": sub_message,
                        "sentence": sub_message,
                        "emote_url": emote_url,
                        "mid": sub_mid,
                        "uname": sub_uname,
                        "ctime_iso": sub_ctime_iso,
                    })
                    sub_emitted.add(emoji_alt)

        for marker in re.findall(r"\[[^\[\]]+\]", sub_message or ""):
            if marker in sub_emitted:
                continue
            if marker in name_set:
                info = (sub_emote or {}).get(marker) or {}
                emoji_name_guess = (info.get("name") or marker.strip("[]")) if isinstance(info, dict) else marker.strip("[]")
                emote_url_guess = (info.get("url") or "") if isinstance(info, dict) else ""
                rows.append({
                    "bvid": bvid,
                    "rpid": sub_rpid,
                    "emoji_alt": marker,
                    "emoji_name": emoji_name_guess,
                    "message": sub_message,
                    "sentence": sub_message,
                    "emote_url": emote_url_guess,
                    "mid": sub_mid,
                    "uname": sub_uname,
                    "ctime_iso": sub_ctime_iso,
                })
                sub_emitted.add(marker)

    return rows


# 中文行间注释：分页抓取并基于映射集合过滤输出
def crawl_bilibili_for_bvid_mapped(session: requests.Session, bvid: str, max_pages: int, sleep_seconds: float, name_set: set) -> list:
    """按 BV 抓取，但内部统一 BV→AID 后以 `oid=aid` 分页抓取，并按映射过滤。"""
    all_rows = []
    aid = get_aid_by_bvid(session, bvid)
    if not aid:
        print(f"[warn] 无法获取 AID，跳过 BV={bvid}")
        return all_rows

    for pn in range(1, max_pages + 1):
        attempts = 0
        stop_this_bv = False
        while True:
            data = fetch_replies_page_by_aid(session, aid, pn, bvid_referer=bvid)
            code = (data or {}).get("code")
            # 中文说明：-400（超出最大偏移量）表示页码超过实际页数，应提前停止该 BV
            if code == -400:
                print(f"[info] bvid={bvid} aid={aid} pn={pn} 返回-400：超过最大偏移量，停止该BV。")
                attempts = 0
                break
            if not data or code != 0:
                if code in (412, 429):
                    print(f"[rate-limit] bvid={bvid} aid={aid} pn={pn} code={code}，20秒后重试当前页。")
                    time.sleep(20)
                    attempts += 1
                    if attempts < 3:
                        continue
                    else:
                        print(f"[warn] bvid={bvid} aid={aid} pn={pn} 限流重试超过上限，跳过该BV。")
                        stop_this_bv = True
                        break
                jitter = random.uniform(0.0, sleep_seconds * 0.5)
                time.sleep(sleep_seconds + jitter)
                break
            break

        if stop_this_bv:
            break
        payload = data.get("data") or {}
        replies = payload.get("replies") or []

        # 中文说明：打印当前页的计数与回复条数，便于调试
        page_info = payload.get("page") or {}
        try:
            print(f"[debug] bvid={bvid} aid={aid} pn={pn} page.count={page_info.get('count', 0)} replies.len={len(replies)}")
        except Exception:
            pass

        if pn == 1 and not replies:
            print(f"[info] bvid={bvid} 首页回复为0，跳过该BV")
            break
        if not replies:
            print(f"[info] bvid={bvid} pn={pn} 当前页无回复，提前停止该BV")
            break

        for item in replies:
            rows = extract_rows_from_reply_item_mapped(bvid, item, name_set)
            all_rows.extend(rows)

        jitter = random.uniform(0.0, sleep_seconds * 0.5)
        time.sleep(sleep_seconds + jitter)

    return all_rows


def ensure_output_dir(root: str, output_rel: str) -> str:
    """确保输出目录存在，并返回输出文件绝对路径。"""
    # 中文说明：兼容相对路径 root 与 output_rel；统一按项目根解析为绝对路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    root_abs = root
    if not os.path.isabs(root_abs):
        root_abs = os.path.normpath(os.path.join(project_root, root_abs))
    if os.path.isabs(output_rel):
        out_path = os.path.normpath(output_rel)
    else:
        out_path = os.path.normpath(os.path.join(root_abs, output_rel))
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    return out_path


def write_csv(rows: list, out_path: str) -> None:
    """写出 CSV，所有字段强制加引号，编码为 UTF-8。"""
    fieldnames = [
        "bvid",
        "rpid",
        "emoji_alt",
        "emoji_name",
        "message",
        "sentence",
        "emote_url",
        "mid",
        "uname",
        "ctime_iso",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# （已移除）仅按 BV 抓取一页评论：统一改为 BV→AID 后按 AID 抓取


# 中文行间注释：收集指定 BV 的前若干条评论文本（顶层评论）
def collect_first_messages(session: requests.Session, bvid: str, max_pages: int, limit: int) -> list:
    """遍历页码收集顶层评论的 `content.message`，返回前 `limit` 条（BV→AID后抓取）。

    中文行间注释：仅保留 AID 路径；为便于排查，打印每页的 page.count 与 replies 数量。
    """
    out: list[str] = []
    # 中文行间注释：统一将 BV 转换为 AID；若失败则直接返回空
    aid = get_aid_by_bvid(session, bvid)
    if not aid:
        print(f"[warn] collect_first_messages: 无法获取 AID，跳过 BV={bvid}")
        return out

    for pn in range(1, max_pages + 1):
        data = fetch_replies_page_by_aid(session, aid, pn, bvid_referer=bvid)
        api_code = data.get("code") if data else None
        # 中文说明：-400（超出最大偏移量）表示页码超过实际页数，应提前停止该 BV
        if api_code == -400:
            print(f"[info] collect_first_messages: bvid={bvid} aid={aid} pn={pn} 返回-400，超过最大偏移量，停止该BV。")
            break
        if not data or api_code != 0:
            print(f"[debug] pn={pn} aid={aid} code={api_code}")
            break
        payload = data.get("data") or {}
        replies = payload.get("replies") or []
        page_info = payload.get("page") or {}
        print(f"[debug] pn={pn} aid={aid} page.count={page_info.get('count', 0)} replies.len={len(replies)}")
        if not replies:
            break
        for item in replies:
            content = item.get("content") or {}
            msg = content.get("message") or ""
            out.append(msg)
            if len(out) >= limit:
                return out
    return out


def dump_raw_for_bvid(session: requests.Session, bvid: str, max_pages: int, out_path: Optional[str]) -> None:
    """按页转储原始接口响应到控制台或 TXT 文件（BV→AID后按 `oid=aid`）。

    中文行间注释：
    - 统一执行 BV→AID 转换；若失败则输出告警并返回。
    - 出于隐私考虑，不打印 SESSDATA 的具体值，只提示是否存在。
    """
    sink_file = None
    # 中文行间注释：定义写入助手，统一写控制台或文件
    def sink_write(text: str):
        if sink_file:
            sink_file.write(text)
        else:
            print(text, end="")

    try:
        # 中文行间注释：如果提供了输出路径，则创建目录并打开文件
        if out_path:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            sink_file = open(out_path, "w", encoding="utf-8")

        # 中文行间注释：输出当前会话头信息（不展示 Cookie 值）
        has_sess = False
        try:
            ck = session.headers.get("Cookie", "")
            has_sess = ("SESSDATA=" in ck)
        except Exception:
            pass
        sink_write(f"=== Dump for BV={bvid} (via AID) ===\n")
        sink_write(f"User-Agent: {session.headers.get('User-Agent', '')}\n")
        sink_write(f"Referer: {session.headers.get('Referer', '')}\n")
        sink_write(f"SESSDATA present: {has_sess}\n\n")

        # 中文行间注释：先进行 BV→AID 转换；失败则不再尝试
        aid = get_aid_by_bvid(session, bvid)
        if not aid:
            sink_write(f"[warn] dump_raw: 无法获取 AID，跳过 BV={bvid}\n")
            return

        base_url = "https://api.bilibili.com/x/v2/reply"
        for pn in range(1, max_pages + 1):
            # 中文行间注释：按 AID 访问旧版分页接口，Referer 设置为视频页
            params_aid = {"pn": pn, "type": 1, "oid": aid, "sort": 0}
            try:
                headers = {"Referer": f"https://www.bilibili.com/video/{bvid}/"}
                resp = session.get(base_url, params=params_aid, headers=headers, timeout=10)
                sink_write(f"--- Page {pn} via AID (BV→AID) ---\n")
                sink_write(f"URL: {resp.url}\n")
                sink_write(f"Status: {resp.status_code}\n")
                try:
                    j = resp.json()
                    sink_write(json.dumps(j, ensure_ascii=False, indent=2) + "\n")
                except Exception:
                    sink_write("[error] Failed to parse JSON, raw text follows:\n")
                    sink_write(resp.text + "\n")
            except Exception as e:
                sink_write(f"[error] AID request failed: {e}\n")

        sink_write("=== End of Dump ===\n")
    finally:
        if sink_file:
            sink_file.close()


def main():
    """命令行入口：按 BV 抓取并输出 CSV。"""
    parser = argparse.ArgumentParser(description="Crawl Bilibili comments with emojis by BV, no sentence splitting.")
    parser.add_argument("--root", type=str, required=True, help="项目根路径")
    parser.add_argument("--bvids", type=str, nargs="*", default=[], help="要抓取的 BV 列表")
    # 中文行间注释：新增从文件读取 BV 列表（CSV或TXT）；CSV需包含列名 `bvid`
    parser.add_argument("--bvids-file", type=str, default="", help="BV 列表文件（CSV含bvid列，或TXT逐行一个BV）")
    parser.add_argument("--max-pages", type=int, default=3, help="每个 BV 抓取的评论页数上限")
    # 中文行间注释：新增分页请求间歇秒数（含随机抖动），避免限流
    parser.add_argument("--sleep-seconds", type=float, default=1.5, help="分页间歇秒数（含随机抖动，建议≥0.5）；若触发限流将自动等待20秒后重试该页")
    parser.add_argument("--output", type=str, default=os.path.join("data", "vendor", "crawl", "bilibili_emoji_sentences.csv"), help="输出 CSV 相对路径")
    parser.add_argument("--print-first-messages", type=int, default=0, help="打印首批评论文本条数（仅打印，不写CSV）")
    parser.add_argument("--sessdata", type=str, default=None, help="可选：B站登录态 SESSDATA，用于提升接口可访问性")
    parser.add_argument("--dump-raw", type=str, default="", help="转储原始响应到指定 TXT（相对 root 或绝对路径）。留空则不转储")
    # 中文行间注释：新增映射 JSON 路径参数；提供后将按映射过滤，仅写出 '[表情名]' 形式且存在于映射的评论
    parser.add_argument("--emoji-map", type=str, default=os.path.join("data", "vendor", "bilibili_emojiall_map.json"), help="表情映射 JSON 相对/绝对路径，用于过滤 [表情名]")
    # 中文行间注释：新增每个 BV 至少写出的记录数要求；脚本会在达到该阈值后停止该 BV 的抓取
    parser.add_argument("--min-per-bvid", type=int, default=0, help="每个 BV 至少写出多少条（仅在启用映射过滤时生效）")
    # 中文行间注释：新增按 BV 单独输出的目录；若提供则每个 BV 写独立 CSV，同时仍可写合并总表
    parser.add_argument("--per-bvid-output-dir", type=str, default=os.path.join("data", "vendor", "crawl", "by_bvid"), help="按 BV 单独输出的目录（相对 root 或绝对路径）")
    # 中文行间注释：新增分页器选择与每页大小；main 接口支持 ps（常见为 20 或 30）
    parser.add_argument("--pager", type=str, default="legacy", choices=["legacy", "main"], help="评论分页器：legacy=旧版 x/v2/reply；main=新版 x/v2/reply/main")
    parser.add_argument("--ps", type=int, default=30, help="每页评论条数（仅在 pager=main 时生效，典型值：20 或 30）")
    # （已移除）OID 模式参数：统一改为仅使用 AID 路径抓取

    args = parser.parse_args()

    session = build_session(sessdata=args.sessdata)

    # 中文行间注释：若提供 --bvids-file，则优先从文件加载 BV 列表
    if args.bvids_file:
        bvids_loaded: list[str] = []
        path = args.bvids_file
        if not os.path.isabs(path):
            # 兼容相对项目根路径写法
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            path = os.path.join(project_root, path)
        try:
            import csv
            # 中文行间注释：根据扩展名判断解析方式；CSV 用 utf-8-sig 以兼容 BOM
            lower = path.lower()
            if lower.endswith('.csv'):
                with open(path, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames or []
                    if 'bvid' in fieldnames:
                        for row in reader:
                            b = (row.get('bvid') or '').strip()
                            if b:
                                bvids_loaded.append(b)
                    else:
                        # 中文行间注释：若无表头，尝试逐行取首列作为 BV（跳过首行若包含 bvid）
                        f.seek(0)
                        for i, line in enumerate(f):
                            line = line.strip()
                            if not line:
                                continue
                            if i == 0 and line.lower().startswith('bvid'):
                                continue
                            # 取逗号前的片段
                            b = line.split(',')[0].strip().strip('"')
                            if b:
                                bvids_loaded.append(b)
            else:
                # 中文行间注释：当作逐行 TXT 读取，每行一个 BV
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        b = line.strip()
                        if b:
                            bvids_loaded.append(b)
        except Exception:
            print(f"[error] 读取 BV 文件失败：{path}")

        # 合并命令行与文件 BV，去重
        args.bvids = list({*(args.bvids or []), *bvids_loaded})
        print(f"[info] 从文件加载 BV 数量：{len(bvids_loaded)} | 合并后总数：{len(args.bvids)}")

    # 原始响应转储：若提供 --dump-raw 路径，则为每个 BV 执行转储并退出
    if hasattr(args, "dump_raw") and args.dump_raw:
        dump_path = args.dump_raw
        for bvid in args.bvids:
            out_path = dump_path
            if not os.path.isabs(out_path):
                out_path = os.path.join(args.root, out_path)
            # 中文行间注释：为该 BV 执行原始响应转储
            dump_raw_for_bvid(session, bvid, max_pages=args.max_pages, out_path=out_path)
        return

    # 如果需要打印首批消息，则仅打印并退出
    if args.print_first_messages and args.bvids:
        bvid = args.bvids[0]
        msgs = collect_first_messages(session, bvid, max_pages=args.max_pages, limit=args.print_first_messages)
        # 中文行间注释：按序打印前 N 条消息
        print(f"BV={bvid} 前{len(msgs)}条message：")
        for i, m in enumerate(msgs, 1):
            print(f"[{i}] {m}")
        return

    # 常规抓取并写 CSV：支持“每个 BV 至少 N 条”与“按 BV 单独输出”
    all_rows = []
    name_set = set()
    if args.emoji_map:
        name_set = load_bilibili_emoji_name_set(args.root, args.emoji_map)
        print(f"[info] loaded emoji name set size={len(name_set)} from {args.emoji_map}")

    # 中文行间注释：内部辅助——抓取直到达到最小条数或用完页数
    def crawl_until_min(bvid: str) -> list:
        rows_acc: list = []
        # 中文行间注释：统一将 BV 转换为 AID；若失败则跳过该 BV
        aid = get_aid_by_bvid(session, bvid)
        if not aid:
            print(f"[warn] crawl_until_min: 无法获取 AID，跳过 BV={bvid}")
            return rows_acc

        # 中文行间注释：当选择新版分页器（reply/main）时，使用游标与 ps 进行分页
        if args.pager == "main":
            next_cursor = 0
            skip_this_bv = False
            reached_min = False
            for pn in range(1, args.max_pages + 1):
                attempts = 0
                while True:
                    j = fetch_replies_main_by_aid(session, aid, next_cursor=next_cursor, ps=max(1, min(args.ps, 30)), mode=3)
                    api_code = j.get("code") if isinstance(j, dict) else None
                    if not j or (api_code is not None and api_code != 0):
                        if api_code in (412, 429):
                            print(f"[rate-limit] bvid={bvid} aid={aid} pn={pn} next={next_cursor} code={api_code}，20秒后重试当前页。")
                            time.sleep(20)
                            attempts += 1
                            if attempts < 3:
                                continue
                            else:
                                print(f"[warn] bvid={bvid} aid={aid} pn={pn} 限流重试超过上限，跳过该BV。")
                                skip_this_bv = True
                                break
                        time.sleep(args.sleep_seconds)
                        break
                if skip_this_bv or reached_min:
                    break
                    data_main = j.get("data") or {}
                    replies = data_main.get("replies") or []
                    cursor = data_main.get("cursor") or {}
                    page_info = data_main.get("page") or {}
                    try:
                        print(f"[debug] bvid={bvid} aid={aid} pn={pn} page.count={page_info.get('count', 0)} replies.len={len(replies)} ps={args.ps}")
                    except Exception:
                        pass
                    if pn == 1 and not replies:
                        print(f"[info] bvid={bvid} 首页回复为0，跳过该BV")
                        break
                    if not replies:
                        print(f"[info] bvid={bvid} pn={pn} 当前页无回复，提前停止该BV")
                        break

                    page_rows: list = []
                    for item in replies:
                        rows = extract_rows_from_reply_item_mapped(bvid, item, name_set) if name_set else extract_rows_from_reply_item(bvid, item)
                        if rows:
                            page_rows.extend(rows)
                    rows_acc.extend(page_rows)
                    print(f"[info] bvid={bvid} pn={pn} 筛选后行数={len(page_rows)} 累计={len(rows_acc)}")

                    if args.min_per_bvid and len(rows_acc) >= args.min_per_bvid:
                        print(f"[info] bvid={bvid} 达到阈值 min-per-bvid={args.min_per_bvid}（pn={pn}）当前累计={len(rows_acc)}")
                        reached_min = True
                        break

                    next_new = cursor.get("next") if isinstance(cursor, dict) else None
                    is_end = cursor.get("is_end") if isinstance(cursor, dict) else None
                    if is_end or next_new in (None, next_cursor):
                        break
                    next_cursor = int(next_new)
                    time.sleep(args.sleep_seconds)
            return rows_acc

        # 中文行间注释：旧版分页接口（按 pn 遍历）
        skip_this_bv = False
        for pn in range(1, args.max_pages + 1):
            attempts = 0
            while True:
                data = fetch_replies_page_by_aid(session, aid, pn, bvid_referer=bvid)
                api_code = data.get("code") if data else None
                if not data or api_code != 0:
                    try:
                        ua = session.headers.get("User-Agent", "")
                        cookie_preview = _preview_cookie(session)
                        dbg_params = {"pn": pn, "type": 1, "oid": aid}
                        api_msg = data.get("message") if isinstance(data, dict) else ""
                        keys = list(data.keys())[:6] if isinstance(data, dict) else []
                        if api_code == -400:
                            print(f"[info] bvid={bvid} aid={aid} pn={pn} 返回-400：超过最大偏移量，停止该BV。")
                        else:
                            print(f"[warn] bvid={bvid} aid={aid} pn={pn} 请求失败或返回码不为0，准备退避/重试")
                        print(f"[debug] params={dbg_params} ua='{ua[:36]}...' cookie='{cookie_preview}' api.code={api_code} api.msg='{api_msg}' json.keys={keys}")
                    except Exception:
                        pass
                    if api_code == -400:
                        break
                    if api_code in (412, 429):
                        print(f"[rate-limit] bvid={bvid} aid={aid} pn={pn} code={api_code}，20秒后重试当前页。")
                        time.sleep(20)
                        attempts += 1
                        if attempts < 3:
                            continue
                        else:
                            print(f"[warn] bvid={bvid} aid={aid} pn={pn} 限流重试超过上限，跳过该BV。")
                            skip_this_bv = True
                            break
                    time.sleep(args.sleep_seconds)
                    break
                break
            if skip_this_bv:
                break
            if api_code == -400:
                break

            payload = data.get("data") or {}
            replies = payload.get("replies") or []
            page_info = payload.get("page") or {}
            try:
                print(f"[debug] bvid={bvid} aid={aid} pn={pn} page.count={page_info.get('count', 0)} replies.len={len(replies)}")
            except Exception:
                pass

            if pn == 1 and not replies:
                print(f"[info] bvid={bvid} 首页回复为0，跳过该BV")
                break
            if not replies:
                print(f"[info] bvid={bvid} pn={pn} 当前页无回复，提前停止该BV")
                break

            page_rows: list = []
            for item in replies:
                rows = extract_rows_from_reply_item_mapped(bvid, item, name_set) if name_set else extract_rows_from_reply_item(bvid, item)
                if rows:
                    page_rows.extend(rows)

            rows_acc.extend(page_rows)
            print(f"[info] bvid={bvid} pn={pn} 筛选后行数={len(page_rows)} 累计={len(rows_acc)}")

            if args.min_per_bvid and len(rows_acc) >= args.min_per_bvid:
                print(f"[info] bvid={bvid} 达到阈值 min-per-bvid={args.min_per_bvid}（pn={pn}）当前累计={len(rows_acc)}")
                break
            time.sleep(args.sleep_seconds)
        return rows_acc

    # 中文行间注释：逐个 BV 抓取，分别写文件，并汇总到总表
    # 统一解析根路径（兼容相对项目根）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    root = args.root
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(project_root, root))
    # 统一解析每BV输出目录
    per_dir = args.per_bvid_output_dir or ''
    if per_dir and not os.path.isabs(per_dir):
        per_dir = os.path.join(root, per_dir)
    per_dir = os.path.normpath(per_dir)
    os.makedirs(per_dir, exist_ok=True)

    # 预扫描：统计哪些 BV 已存在输出文件，将在抓取阶段被跳过；打印概要以便可见
    pre_skipped: list[tuple[str, str]] = []
    pre_pending: list[str] = []
    for bvid in args.bvids:
        safe_bvid = re.sub(r"[^A-Za-z0-9._-]", "_", bvid).strip("_")
        per_out_probe = os.path.join(per_dir, f"bv_{safe_bvid}_emoji_mapped.csv")
        try:
            if os.path.exists(per_out_probe) and os.path.getsize(per_out_probe) > 0:
                pre_skipped.append((bvid, per_out_probe))
            else:
                pre_pending.append(bvid)
        except Exception:
            pre_pending.append(bvid)
    print(f"[info] 预扫描：BV总数={len(args.bvids)} 已完成={len(pre_skipped)} 待抓取={len(pre_pending)} 输出目录={per_dir}")
    if pre_skipped:
        max_show = 10
        print(f"[info] 将跳过的已完成BV（示例最多{max_show}条）：")
        for i, (b, p) in enumerate(pre_skipped[:max_show], 1):
            print(f"  [{i}] BV={b} -> {p}")

    for bvid in args.bvids:
        # 中文行间注释：先计算该 BV 的目标输出文件，用于“已存在则跳过”
        safe_bvid = re.sub(r"[^A-Za-z0-9._-]", "_", bvid).strip("_")
        per_out = os.path.join(per_dir, f"bv_{safe_bvid}_emoji_mapped.csv")

        # 中文行间注释：若目标文件已存在且非空，则认为该 BV 已爬取完成，直接跳过（避免重复抓取）
        try:
            if os.path.exists(per_out) and os.path.getsize(per_out) > 0:
                print(f"[info] BV={bvid} 检测到已存在的输出文件，跳过抓取：{per_out}")
                continue
        except Exception:
            # 中文行间注释：容错处理——若访问文件出现异常，忽略并继续正常抓取
            pass

        # 中文行间注释：执行抓取与写出
        rows = crawl_until_min(bvid)
        all_rows.extend(rows)
        write_csv(rows, per_out)
        print(f"[info] BV={bvid} 完成，总计匹配行数={len(rows)}；已写出到 {per_out}")

    out_path = ensure_output_dir(root, args.output)
    write_csv(all_rows, out_path)
    print(f"写出 {len(all_rows)} 行到: {out_path}")


if __name__ == "__main__":
    main()
