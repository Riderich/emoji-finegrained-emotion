r"""
中文说明：批量获取 B 站视频 BV 号的脚本（支持“关键词搜索”与“热门榜”两种模式）。

功能概述：
- 模式一（search）：基于接口 `https://api.bilibili.com/x/web-interface/search/type` 按关键词分页搜索视频；
- 模式二（popular）：基于接口 `https://api.bilibili.com/x/web-interface/popular` 按热门榜分页拉取视频；
- 可选按“视频总评论数（reply）”进行过滤，优先获取讨论更充分的视频；
- 输出至 CSV（默认：`data/vendor/crawl/seed_bvids.csv`），供后续 `crawl_bilibili.py` 按 BV 抓取评论使用。

使用建议：
- 若希望覆盖更广的热门视频，使用 `--mode popular` 并结合 `--min-reply`；
- 若希望聚焦某类话题，使用 `--mode search` 配合关键词文件；
- 请求间歇建议 `sleep-seconds>=0.6` 并随机抖动，避免触发限流；
- 可选传入 `--sessdata`（登录态），在部分场景下能提升可访问性与配额稳定性；

运行示例（Windows）：
python -m src.data.fetch_bilibili_bv --mode popular --max-pages 5 --min-reply 50 \
  --out data\vendor\crawl\seed_bvids.csv --sleep-seconds 0.8
"""

import os
import csv
import time
import random
import argparse
from typing import Iterable, List, Dict, Tuple, Optional

import requests
import json


# 中文说明：构建带 UA 与可选 Cookie 的会话；UA 模拟桌面浏览器，避免被识别为爬虫
def build_session(sessdata: Optional[str] = None) -> requests.Session:
    # 中文说明：创建会话并设置常见浏览器请求头，尽量贴近真实访问，降低 412 风险
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.bilibili.com",
        "Referer": "https://www.bilibili.com/",
        # 中文说明：补充常见头，保持与评论抓取脚本一致
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Connection": "keep-alive",
        "Sec-Fetch-Site": "same-site",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "X-Requested-With": "XMLHttpRequest",
    })
    # 中文说明：优先从同目录 cookies.json 读取完整 Cookie；若不可用则回退到 SESSDATA
    try:
        cookie_path = os.path.join(os.path.dirname(__file__), "cookies.json")
        cookie_header = load_cookie_header_from_json(cookie_path)
        if cookie_header:
            s.headers.update({"Cookie": cookie_header})
            lower = cookie_header.lower()
            if ("sessdata=" not in lower) or ("bili_jct=" not in lower):
                print("[warn] Cookie 可能不完整：缺少 SESSDATA/bili_jct，搜索接口可能更容易限流。")
        elif sessdata:
            s.headers.update({"Cookie": f"SESSDATA={sessdata}"})
    except Exception:
        if sessdata:
            s.headers.update({"Cookie": f"SESSDATA={sessdata}"})
    return s


# 中文说明：从浏览器导出的 cookies.json 组装为 Cookie 头字符串
def load_cookie_header_from_json(path: str) -> Optional[str]:
    # 中文说明：兼容字符串、字典与列表三种 cookies.json 结构；更宽容处理字段名大小写
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 情况一：纯字符串
        if isinstance(data, str):
            s = data.strip()
            return s if s else None
        # 情况二：字典
        if isinstance(data, dict):
            # 优先处理常见头字段
            for k in ("Cookie", "cookie", "header", "Header", "value", "Value"):
                v = str(data.get(k) or "").strip()
                if v:
                    return v
            # 若不像带元数据的对象，则按映射拼接
            meta_keys = {"domain", "path", "sameSite", "secure", "httpOnly", "expirationDate", "hostOnly", "storeId", "session"}
            if not any(k in data for k in meta_keys):
                pairs = []
                for k, v in data.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if not ks or not vs:
                        continue
                    pairs.append(f"{ks}={vs}")
                if pairs:
                    return "; ".join(pairs)
            return None
        # 情况三：列表（浏览器导出），兼容大小写与别名
        if isinstance(data, list):
            pairs = []
            for it in data:
                if not isinstance(it, dict):
                    continue
                name = (it.get("name") or it.get("Name") or it.get("key") or it.get("Key") or "").strip()
                value = (it.get("value") or it.get("Value") or it.get("val") or it.get("Val") or "").strip()
                if not name or not value:
                    continue
                pairs.append(f"{name}={value}")
            if not pairs:
                return None
            return "; ".join(pairs)
        return None
    except Exception:
        return None


# 中文说明：按关键词分页搜索视频，返回“视频条目”的列表
def search_video_by_keyword(
    session: requests.Session,
    keyword: str,
    max_pages: int,
    order: str,
    sleep_seconds: float,
) -> List[Dict]:
    out: List[Dict] = []
    base_url = "https://api.bilibili.com/x/web-interface/search/type"
    for page in range(1, max_pages + 1):
        params = {
            "search_type": "video",   # 中文说明：限定为视频搜索
            "keyword": keyword,
            "page": page,
            "order": order,            # 可选：pubdate/view 等
            "duration": 0,             # 0=全部时长
        }
        try:
            resp = session.get(base_url, params=params, timeout=10)
            if resp.status_code != 200:
                # 中文说明：调试输出失败时的关键信息，便于定位（仅异常时打印，避免刷屏）
                body_preview = ""
                try:
                    body_preview = resp.text[:200]
                except Exception:
                    pass
                print(f"[warn] 搜索接口HTTP状态异常 status={resp.status_code} ua={session.headers.get('User-Agent')} url={resp.url}")
                print(f"[warn] 响应片段: {body_preview}")
                time.sleep(sleep_seconds)
                continue
            j = resp.json()
            data = j.get("data") or {}
            result = data.get("result") or []
            # 中文说明：若接口返回码不为0或结果为空，进行轻度调试输出
            code = j.get("code")
            if code not in (None, 0) or not result:
                print(f"[debug] 搜索返回 code={code} page={page} keyword={keyword} keys={list(j.keys())}")
            # 中文说明：逐条提取必要字段；缺失字段做容错处理
            for it in result:
                bvid = (it.get("bvid") or "").strip()
                if not bvid:
                    continue
                title = (it.get("title") or "").strip()
                author = (it.get("author") or "").strip()
                pubdate = it.get("pubdate") or 0   # Unix 秒
                play = it.get("play") or 0
                danmaku = it.get("danmaku") or 0
                duration = (it.get("duration") or "").strip()  # 形如 "3:45"
                out.append({
                    "bvid": bvid,
                    "title": title,
                    "author": author,
                    "pubdate": pubdate,
                    "play": play,
                    "danmaku": danmaku,
                    "duration": duration,
                    "url": f"https://www.bilibili.com/video/{bvid}",
                    "keyword": keyword,
                    "page": page,
                })
        except Exception:
            # 中文说明：网络或解析失败，跳过该页并继续下一页
            time.sleep(sleep_seconds)
            continue
        # 中文说明：加入轻微随机抖动，降低被限流风险
        jitter = random.uniform(0.0, sleep_seconds * 0.3)
        time.sleep(sleep_seconds + jitter)
    return out


# 中文说明：按热门榜分页拉取视频，返回“视频条目”的列表
def fetch_popular_videos(
    session: requests.Session,
    max_pages: int,
    ps: int,
    sleep_seconds: float,
) -> List[Dict]:
    out: List[Dict] = []
    base_url = "https://api.bilibili.com/x/web-interface/popular"
    for page in range(1, max_pages + 1):
        params = {
            "pn": page,   # 中文说明：页码，从1开始
            "ps": ps,     # 中文说明：每页数量，常见值为20
        }
        try:
            resp = session.get(base_url, params=params, timeout=10)
            if resp.status_code != 200:
                time.sleep(sleep_seconds)
                continue
            j = resp.json()
            data = j.get("data") or {}
            # 中文说明：popular接口的列表字段通常为 data.list
            result = data.get("list") or []
            for it in result:
                bvid = (it.get("bvid") or "").strip()
                if not bvid:
                    continue
                title = (it.get("title") or "").strip()
                # 中文说明：作者信息位于 owner.name
                owner = it.get("owner") or {}
                author = (owner.get("name") or "").strip()
                # 中文说明：发布时间可能为 pubdate 或 ctime，做容错处理
                pubdate = it.get("pubdate") or it.get("ctime") or 0
                stat = it.get("stat") or {}
                view = stat.get("view") or 0
                reply = stat.get("reply") or 0
                danmaku = stat.get("danmaku") or 0
                duration = it.get("duration") or 0
                out.append({
                    "bvid": bvid,
                    "title": title,
                    "author": author,
                    "pubdate": pubdate,
                    "view": view,
                    "reply": reply,
                    "danmaku": danmaku,
                    "duration": duration,
                    "url": f"https://www.bilibili.com/video/{bvid}",
                    "keyword": "",   # 中文说明：热门模式不含关键词
                    "page": page,
                    "source": "popular",
                })
        except Exception:
            time.sleep(sleep_seconds)
            continue
        jitter = random.uniform(0.0, sleep_seconds * 0.3)
        time.sleep(sleep_seconds + jitter)
    return out


# 中文说明：获取单个BV的视频统计信息（view、reply等），用于按评论数过滤
def fetch_video_stat_by_bvid(session: requests.Session, bvid: str) -> Dict[str, int]:
    base_url = "https://api.bilibili.com/x/web-interface/view"
    try:
        resp = session.get(base_url, params={"bvid": bvid}, timeout=10)
        if resp.status_code != 200:
            return {"view": 0, "reply": 0}
        j = resp.json()
        data = j.get("data") or {}
        stat = data.get("stat") or {}
        return {
            "view": stat.get("view") or 0,
            "reply": stat.get("reply") or 0,
        }
    except Exception:
        return {"view": 0, "reply": 0}


# 中文说明：写出 BV 列表到 CSV，UTF-8 BOM 便于 Excel 直接打开
def write_bvid_csv(rows: Iterable[Dict], out_path: str) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen: set[str] = set()
    total = 0
    unique = 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                # 中文说明：新增 view/reply 字段；保留 play 兼容搜索结果
                "bvid", "title", "author", "pubdate", "play", "view", "reply", "danmaku", "duration", "url", "keyword", "page", "source"
            ]
        )
        w.writeheader()
        for r in rows:
            total += 1
            bvid = r.get("bvid")
            if not bvid:
                continue
            if bvid not in seen:
                unique += 1
                seen.add(bvid)
                w.writerow(r)
    return total, unique


def load_keywords_from_file(path: str) -> List[str]:
    # 中文说明：从文本文件加载关键词，按行读取并去除空白行
    out: List[str] = []
    if not path:
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if kw:
                out.append(kw)
    return out


def main():
    parser = argparse.ArgumentParser(description="批量获取B站视频BV号（支持关键词搜索与热门榜）")
    # 中文说明：模式选择：search（关键词搜索） / popular（热门榜）
    parser.add_argument("--mode", type=str, default="search", choices=["search", "popular"], help="获取模式：search=关键词搜索，popular=热门榜")
    # 中文说明：关键词相关配置（仅在 mode=search 时生效）
    parser.add_argument("--keywords", type=str, nargs="*", default=[], help="关键词列表（空格分隔）")
    parser.add_argument("--keywords-file", type=str, default="", help="关键词文件（逐行一个关键词，UTF-8）")
    parser.add_argument("--order", type=str, default="pubdate", choices=["pubdate", "view"], help="排序方式：时间或播放量（search模式）")
    # 中文说明：分页与节流
    parser.add_argument("--max-pages", type=int, default=5, help="分页页数上限（search为每关键词的页数；popular为热门榜页数）")
    parser.add_argument("--ps", type=int, default=20, help="popular模式每页数量（建议20）")
    parser.add_argument("--sleep-seconds", type=float, default=0.8, help="分页请求间歇秒数（含随机抖动，建议≥0.6；过低易限流）")  # 中文行间注释：默认从 1.2s 下调到 0.8s，若频繁失败请调回≥1.0
    parser.add_argument("--sessdata", type=str, default=None, help="可选：B站登录态 SESSDATA")
    # 中文说明：过滤条件
    parser.add_argument("--min-reply", type=int, default=0, help="按总评论数过滤（>=该值保留；0表示不过滤）")
    # 中文说明：输出路径
    parser.add_argument("--out", type=str, default=os.path.join("data", "vendor", "crawl", "seed_bvids.csv"), help="输出CSV相对路径")
    args = parser.parse_args()

    session = build_session(sessdata=args.sessdata)

    all_rows: List[Dict] = []
    if args.mode == "search":
        # 中文说明：汇总关键词来源：命令行 + 文件
        kws: List[str] = list(args.keywords)
        if args.keywords_file:
            kws += load_keywords_from_file(args.keywords_file)
        # 中文说明：去重并过滤空白
        kws = [k.strip() for k in kws if k and k.strip()]
        if not kws:
            print("[error] 未提供关键词（--keywords 或 --keywords-file），无法搜索！")
            return
        for kw in kws:
            print(f"[info] 搜索关键词：{kw}")
            rows = search_video_by_keyword(session, kw, max_pages=args.max_pages, order=args.order, sleep_seconds=args.sleep_seconds)
            # 中文说明：标注来源，便于后续分析
            for r in rows:
                r["source"] = "search"
            print(f"[info] 关键词 {kw} 收到 {len(rows)} 条视频结果")
            all_rows.extend(rows)
    elif args.mode == "popular":
        print(f"[info] 按热门榜获取视频：pages={args.max_pages} ps={args.ps}")
        rows = fetch_popular_videos(session, max_pages=args.max_pages, ps=args.ps, sleep_seconds=args.sleep_seconds)
        print(f"[info] popular 模式收到 {len(rows)} 条视频结果")
        all_rows.extend(rows)

    # 中文说明：若设置了按评论数过滤，则补全 reply/view 并进行过滤
    if args.min_reply and args.min_reply > 0:
        print(f"[info] 按评论数过滤：min-reply={args.min_reply}")
        filtered: List[Dict] = []
        for r in all_rows:
            bvid = r.get("bvid") or ""
            if not bvid:
                continue
            reply = r.get("reply")
            view = r.get("view")
            if reply is None:
                stat = fetch_video_stat_by_bvid(session, bvid)
                reply = stat.get("reply", 0)
                view = stat.get("view", view or 0)
                r["reply"] = reply
                r["view"] = view
                # 中文说明：每次stat查询后加入轻微抖动，降低限流风险
                jitter = random.uniform(0.0, args.sleep_seconds * 0.3)
                time.sleep(args.sleep_seconds + jitter)
            if int(reply or 0) >= int(args.min_reply):
                filtered.append(r)
        all_rows = filtered
        print(f"[info] 过滤后保留 {len(all_rows)} 条视频")

    # 中文说明：统一根路径解析，兼容相对路径写出
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(project_root, out_path)
    total, unique = write_bvid_csv(all_rows, out_path)
    print(f"[done] 写出 seed BV CSV：{out_path} | total={total} unique={unique}")


if __name__ == "__main__":
    main()
