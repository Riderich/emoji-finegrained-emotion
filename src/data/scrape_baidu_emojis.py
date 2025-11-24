"""
Scrape Baidu (Tieba) vendor emoji mapping from Emojiall.

This script fetches the Baidu platform page on Emojiall and extracts a mapping
between Baidu's vendor emoji images (PNG) and the underlying Unicode emoji.
It collects for each entry: detail URL, vendor image URL, decoded emoji char,
codepoints, and any shortname/alt if present. Optionally downloads images.

Outputs JSON and CSV in the project's data directory for downstream use.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PLATFORM_URL = "https://www.emojiall.com/zh-hans/platform-baidu"
PLATFORM_URLS = {
    "baidu": "https://www.emojiall.com/zh-hans/platform-baidu",
    "weibo": "https://www.emojiall.com/zh-hans/platform-weibo",
    "bilibili": "https://www.emojiall.com/zh-hans/platform-bilibili",
    "douyin": "https://www.emojiall.com/zh-hans/platform-douyin",
}

PLATFORM_PHRASES = {
    # 中文说明：用于定位页面中“平台表情对应的Emoji”章节标题的关键短语集合
    "baidu": [
        "Baidu表情对应的Emoji",
        "Baidu 表情对应的 Emoji",
        "以下Baidu表情对应的Emoji",
    ],
    "weibo": [
        "Weibo表情对应的Emoji",
        "Weibo 表情对应的 Emoji",
        "以下Weibo表情对应的Emoji",
    ],
    "bilibili": [
        "Bilibili表情对应的Emoji",
        "Bilibili 表情对应的 Emoji",
        "以下Bilibili表情对应的Emoji",
    ],
    "douyin": [
        "Douyin表情对应的Emoji",
        "Douyin 表情对应的 Emoji",
        "以下Douyin表情对应的Emoji",
    ],
}


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist.

    Notes
    -----
    Uses `Path.mkdir` with `parents=True` and `exist_ok=True` to guarantee
    that nested directories are created when needed.
    """
    # 中文说明：若目录不存在则创建，保证后续写文件不会失败
    path.mkdir(parents=True, exist_ok=True)


def build_session() -> requests.Session:
    """Create a configured HTTP session with headers and timeouts.

    Returns
    -------
    requests.Session
        A session with a desktop user-agent and sane defaults.
    """
    # 中文说明：构造带有常见浏览器 UA 的会话，降低被拒风险
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )
    # 中文说明：为网络故障准备重试适配器（连接/读/状态码重试）
    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        backoff_factor=0.6,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def fetch_html(
    sess: requests.Session,
    url: str,
    timeout: int = 30,
    attempts: int = 5,
    backoff: float = 1.5,
) -> Optional[str]:
    """Fetch HTML content from the given URL with retries.

    Parameters
    ----------
    sess : requests.Session
        Configured HTTP session.
    url : str
        Target URL to fetch.
    timeout : int
        Timeout in seconds for the request.
    attempts : int
        Maximum retry attempts when network or server issues occur.
    backoff : float
        Exponential backoff base (sleep increases linearly by factor).

    Returns
    -------
    Optional[str]
        Raw HTML text if successful, otherwise None.
    """
    # 中文说明：带重试的抓取逻辑；对 429/5xx 等状态码进行退避重试
    for i in range(max(1, attempts)):
        try:
            r = sess.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.text
            # 遇到限流或服务端错误时等待后重试
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (i + 1))
            else:
                # 其它状态码直接失败
                return None
        except Exception:
            # 连接错误等，等待后重试
            time.sleep(backoff * (i + 1))
    return None


def decode_emoji_from_href(href: str) -> Optional[str]:
    """Attempt to decode emoji character from an emoji detail URL.

    Parameters
    ----------
    href : str
        The anchor href pointing to a per-emoji detail page.

    Returns
    -------
    Optional[str]
        Emoji character string or None if it cannot be parsed.
    """
    # 中文说明：有些详情链接形如 /zh-hans/emoji/%F0%9F%98%80，可从路径段解码出字符
    try:
        # 优先处理包含 '/emoji/' 的形式
        import urllib.parse as _up

        if "/emoji/" in href:
            # 取最后一个路径段并进行 URL 解码
            last = href.split("/")[-1]
            return _up.unquote(last)
        # 某些页面可能是 '/emoji-<char>' 形式，尝试从最后的 '-' 后截取
        if "/emoji-" in href:
            tail = href.split("/emoji-")[-1]
            # 直接返回尾部（通常就是实际 emoji 字符）
            return tail
        return None
    except Exception:
        return None


def codepoints_of(emoji: str) -> List[str]:
    """Convert an emoji string to a list of U+XXXX codepoint labels.

    Parameters
    ----------
    emoji : str
        Emoji character(s), possibly comprising multiple code points.

    Returns
    -------
    List[str]
        A list of codepoint labels like ["U+1F600", "U+FE0F"].
    """
    # 中文说明：将 emoji 字符串逐字符转为 Unicode 码点标签
    cps = []
    for ch in emoji:
        cps.append(f"U+{ord(ch):04X}")
    return cps


def parse_platform_page(html: str) -> List[Dict]:
    """Parse the Baidu platform page and collect item anchors and images.

    Parameters
    ----------
    html : str
        Raw HTML of the platform page.

    Returns
    -------
    List[Dict]
        A list of preliminary entries with keys: detail_url, img_url, alt.
    """
    # 中文说明：从平台页上抓取每个 emoji 的详情链接与对应的百度图片 URL
    soup = BeautifulSoup(html, "html.parser")

    entries: List[Dict] = []
    # 经验做法：平台页上每个条目通常是一个 <a>，内部包含 <img>（vendor 图片）
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # 仅保留指向 emoji 详情页的链接
        if "/emoji/" in href or "/emoji-" in href:
            # 查找内部图片（百度供应商图）
            img = a.find("img")
            if not img:
                continue
            img_url = img.get("src") or img.get("data-src")
            alt = img.get("alt")
            # 将相对链接补全为绝对链接
            if href.startswith("/"):
                detail_url = f"https://www.emojiall.com{href}"
            else:
                detail_url = href
            entries.append({"detail_url": detail_url, "img_url": img_url, "alt": alt})

    # 去重：根据 detail_url 进行去重，防止重复条目
    uniq: Dict[str, Dict] = {}
    for e in entries:
        uniq[e["detail_url"]] = e
    return list(uniq.values())


def find_vendor_list_section(soup: BeautifulSoup, platform: str) -> Optional[BeautifulSoup]:
    """Locate the vendor image list section by heading nearby.

    English
    -------
    Try to find a section under headings like:
    - "Bilibili提供的Emoji表情符号图片列表"
    - "Douyin提供的Emoji表情符号图片列表"
    - "Weibo提供的Emoji表情符号图片列表"
    - "Baidu提供的Emoji表情符号图片列表"
    Also supports English fallbacks like "Emoji images list".

    中文说明：根据“<平台>提供的Emoji表情符号图片列表”这样的标题，定位下方图片列表区域；
    若中文不可达，则尝试英文“emoji images list”等关键词。

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed page soup.
    platform : str
        Platform key (baidu/weibo/douyin/bilibili), used for phrase matching.

    Returns
    -------
    Optional[BeautifulSoup]
        The container element (table or div) that holds vendor images.
    """
    # 中文说明：构造平台相关的中文短语和英文回退关键词
    phrases_cn = [f"{platform.capitalize()}提供的Emoji表情符号图片列表", f"{platform}提供的Emoji表情符号图片列表", "提供的Emoji表情符号图片列表"]
    phrases_en = ["emoji images list", "emoji image list", "images list"]

    # 搜索标题节点
    candidates = []
    for tag_name in ("h1", "h2", "h3", "h4"):
        for h in soup.find_all(tag_name):
            text = (h.get_text() or "").strip()
            low = text.lower()
            if any(p in text for p in phrases_cn) or any(p in low for p in phrases_en):
                # 在该标题之后查找第一个 table 或包含大量图片的 div
                nxt = h
                for _ in range(200):
                    nxt = nxt.find_next_sibling()
                    if not nxt:
                        break
                    if nxt.name == "table":
                        return nxt
                    # 若是容器 div，且内部有较多图片，认为是候选
                    imgs = nxt.find_all("img") if hasattr(nxt, "find_all") else []
                    if imgs and len(imgs) >= 10:
                        return nxt
                candidates.append(h)

    # 兜底：扫描包含大量图片的区域，排除已识别的映射表
    for s in soup.find_all(["section", "div"]):
        imgs = s.find_all("img")
        if imgs and len(imgs) >= 20:
            return s

    return None


def parse_vendor_list(container: BeautifulSoup) -> List[Dict]:
    """Parse vendor image list container into vendor_only entries.

    English
    -------
    Extract `img_url` and `name` from images within the container and mark
    entries as `kind == "vendor_only"`.

    中文说明：遍历容器中的图片，提取图片地址与名称（alt），生成 `vendor_only` 条目。

    Parameters
    ----------
    container : BeautifulSoup
        Container element that holds vendor images.

    Returns
    -------
    List[Dict]
        Entries with keys: `img_url`, `name`, `emoji` (None), `codepoints` ([]), `detail_url` (None), `kind`.
    """
    rows: List[Dict] = []
    # 中文说明：查找所有 img；若没有 alt，用空字符串代替
    imgs = container.find_all("img")
    for img in imgs:
        src = img.get("src") or img.get("data-src")
        if not src:
            continue
        # 规范化为绝对 URL
        if src.startswith("/"):
            src = f"https://www.emojiall.com{src}"
        name_text = (img.get("alt") or "").strip()
        rows.append(
            {
                "img_url": src,
                "name": name_text,
                "emoji": None,
                "codepoints": [],
                "detail_url": None,
                "kind": "vendor_only",
            }
        )
    # 去重：按 img_url 去重
    uniq: Dict[str, Dict] = {}
    for e in rows:
        uniq[e["img_url"]] = e
    return list(uniq.values())


def find_mapping_table(soup: BeautifulSoup, target_phrases: List[str]) -> Optional[BeautifulSoup]:
    """Locate the specific mapping table under the 'Baidu表情对应的Emoji' section.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed HTML soup of the platform page.

    Returns
    -------
    Optional[BeautifulSoup]
        The `<table>` element if found, otherwise None.
    """
    # 中文说明：优先根据标题文本查找对应的表格；若失败则退化为查找包含表头“图片/名称/对应Emoji”的表格
    # 先通过标题就近查找
    for tag_name in ("h1", "h2", "h3", "h4"):
        for h in soup.find_all(tag_name):
            text = (h.get_text() or "").strip()
            if any(p in text for p in target_phrases):
                # 在该标题之后查找第一个 table
                # 通过遍历兄弟节点直到发现 <table>
                nxt = h
                for _ in range(200):  # 防御性限制，避免无限循环
                    nxt = nxt.find_next_sibling()
                    if not nxt:
                        break
                    if nxt.name == "table":
                        return nxt
                    # 有些结构中表格可能嵌套在 div 中
                    t = nxt.find("table")
                    if t:
                        return t

    # 兜底：遍历所有 table，查找是否含有预期的表头（支持中英文）
    for t in soup.find_all("table"):
        # 尝试提取第一行作为表头
        thead = t.find("thead")
        header_cells: List[str] = []
        if thead:
            header_cells = [c.get_text(strip=True) for c in thead.find_all("th")]
        else:
            first_row = t.find("tr")
            if first_row:
                header_cells = [c.get_text(strip=True) for c in first_row.find_all(["th", "td"])]
        header_text = "|".join(header_cells)
        header_text_l = header_text.lower()
        # 中文说明：列头匹配规则：图片(image/img/icon)、名称(name)、emoji（对应/对应emoji/emoji）至少包含对应关键词
        has_image = ("图片" in header_text) or ("图像" in header_text) or ("image" in header_text_l) or ("img" in header_text_l) or ("icon" in header_text_l)
        has_name = ("名称" in header_text) or ("name" in header_text_l)
        has_emoji = ("对应Emoji" in header_text) or ("对应 emoji" in header_text) or ("emoji" in header_text_l)
        if has_image and has_name and has_emoji:
            return t

    return None


def parse_mapping_table(table: BeautifulSoup) -> List[Dict]:
    """Parse rows of the mapping table into structured entries.

    Parameters
    ----------
    table : BeautifulSoup
        The `<table>` element representing the mapping section.

    Returns
    -------
    List[Dict]
        Entries with keys: `img_url`, `name`, `emoji`, `codepoints`, `detail_url`, `kind`.
    """
    # 中文说明：遍历数据行，提取三列信息：图片URL、名称文本、对应的emoji字符与详情链接
    rows = []
    # 跳过表头，从 tbody 开始；若无 tbody，则跳过第一行
    tbody = table.find("tbody")
    tr_list = tbody.find_all("tr") if tbody else table.find_all("tr")
    # 如果第一行是表头（含 th），则跳过
    if tr_list and tr_list[0].find("th"):
        tr_list = tr_list[1:]

    for tr in tr_list:
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        # 第1列：图片
        img_td = tds[0]
        img = img_td.find("img")
        img_url = img.get("src") if img else None
        # 规范化为绝对 URL
        if img_url and img_url.startswith("/"):
            img_url = f"https://www.emojiall.com{img_url}"
        # 第2列：名称
        name_td = tds[1]
        name_text = (name_td.get_text() or "").strip()
        # 第3列：对应Emoji（可能包含链接与实际字符）
        emoji_td = tds[2]
        detail_a = emoji_td.find("a", href=True)
        detail_url = None
        if detail_a:
            href = detail_a["href"]
            detail_url = f"https://www.emojiall.com{href}" if href.startswith("/") else href
        emoji_text = (emoji_td.get_text() or "").strip()

        # 现在支持两类：
        # - kind == "mapped": 三列非空
        # - kind == "vendor_only": 仅图片与名称非空，emoji 为空
        if img_url and name_text:
            if emoji_text:
                kind = "mapped"
                cps = codepoints_of(emoji_text)
                rows.append(
                    {
                        "img_url": img_url,
                        "name": name_text,
                        "emoji": emoji_text,
                        "codepoints": cps,
                        "detail_url": detail_url,
                        "kind": kind,
                    }
                )
            else:
                kind = "vendor_only"
                rows.append(
                    {
                        "img_url": img_url,
                        "name": name_text,
                        "emoji": None,
                        "codepoints": [],
                        "detail_url": detail_url,
                        "kind": kind,
                    }
                )
    return rows


def enrich_with_detail(
    sess: requests.Session,
    entry: Dict,
    timeout: int = 30,
    attempts: int = 3,
    backoff: float = 1.2,
) -> Dict:
    """Fetch and enrich a platform entry with emoji char and codepoints.

    Parameters
    ----------
    sess : requests.Session
        HTTP session for requests.
    entry : Dict
        Preliminary entry with at least `detail_url`.

    Returns
    -------
    Dict
        Entry enriched with `emoji` and `codepoints` if discoverable.
    """
    # 中文说明：尝试从链接本身或详情页解析出 emoji 字符及其码点
    href = entry.get("detail_url")
    emoji = decode_emoji_from_href(href or "")

    # 若从链接解析失败，则请求详情页并尝试从页面内容提取
    if not emoji:
        html = fetch_html(sess, href, timeout=timeout, attempts=attempts, backoff=backoff)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            # 兜底策略：从标题中寻找实际 emoji（页面标题常含有 emoji 字符）
            title = soup.find("title")
            if title and title.text:
                # 取标题中的非 ASCII 字符作为候选（通常就是 emoji）
                candidates = [c for c in title.text if ord(c) > 0x80]
                if candidates:
                    emoji = "".join(candidates)

    if emoji:
        entry["emoji"] = emoji
        entry["codepoints"] = codepoints_of(emoji)
    else:
        entry["emoji"] = None
        entry["codepoints"] = []
    return entry


def write_json(path: Path, data: List[Dict]) -> None:
    """Write data as JSON to the given path with UTF-8 encoding.

    Parameters
    ----------
    path : Path
        Output JSON file path.
    data : List[Dict]
        Data entries to serialize.
    """
    # 中文说明：将抓取到的数据保存为 JSON 文件，便于后续程序读取
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, data: List[Dict]) -> None:
    """Write data as CSV to the given path with UTF-8 encoding.

    Parameters
    ----------
    path : Path
        Output CSV file path.
    data : List[Dict]
        Data entries to serialize.
    """
    # 中文说明：保存为 CSV，便于用 Excel / pandas 直接查看或处理
    ensure_dir(path.parent)
    cols = ["emoji", "codepoints", "name", "img_url", "detail_url", "kind"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in data:
            # 将 codepoints 列表合并为以空格分隔的字符串
            cps = " ".join(e.get("codepoints", []))
            w.writerow([
                e.get("emoji"),
                cps,
                e.get("name"),
                e.get("img_url"),
                e.get("detail_url"),
                e.get("kind"),
            ])


def write_image_name_csv(path: Path, data: List[Dict]) -> None:
    """写出图片-表情名映射的CSV（含本地保存路径）。

    中文说明：该CSV专注于“图片URL ↔ 表情名称”的对应关系，同时保留
    可用的emoji字符、码点、详情页链接与kind，并在下载图片时记录本地路径。

    参数
    ----
    path : Path
        输出CSV文件路径。
    data : List[Dict]
        富集的条目列表，可能包含 'local_path' 字段。
    """
    ensure_dir(path.parent)
    cols = ["name", "img_url", "local_path", "emoji", "codepoints", "detail_url", "kind"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for e in data:
            cps = " ".join(e.get("codepoints", []))
            w.writerow([
                e.get("name"),
                e.get("img_url"),
                e.get("local_path"),  # 中文说明：如有下载则记录本地文件路径
                e.get("emoji"),
                cps,
                e.get("detail_url"),
                e.get("kind"),
            ])


def download_image(sess: requests.Session, url: str, out_dir: Path, name_hint: Optional[str]) -> Optional[Path]:
    """Download a single image to `out_dir` with optional name hint.

    Parameters
    ----------
    sess : requests.Session
        HTTP session for requests.
    url : str
        Image URL to download.
    out_dir : Path
        Destination directory.
    name_hint : Optional[str]
        A string hint for filename (e.g., emoji or alt text).

    Returns
    -------
    Optional[Path]
        Saved file path if success, else None.
    """
    # 中文说明：下载图片并以可读的文件名保存；失败时返回 None
    try:
        ensure_dir(out_dir)
        # 若为相对链接，补全为绝对链接
        full_url = url
        if url.startswith("/"):
            full_url = f"https://www.emojiall.com{url}"
        r = sess.get(full_url, timeout=30)
        if r.status_code != 200:
            return None
        # 自动推断扩展名（默认为 .png）
        ext = ".png"
        # 尝试从 URL 中提取扩展名
        for x in [".png", ".jpg", ".jpeg", ".webp"]:
            if x in full_url.lower():
                ext = x
                break
        # 构造文件名：优先使用 URL 尾段（更唯一），若缺失则使用 name_hint
        tail = full_url.split("/")[-1].split("?")[0]
        fname = tail or (name_hint or "emoji")
        # 去除文件名中的不安全字符
        safe = "".join(c for c in fname if c.isalnum() or c in ["_", "-"])
        if not safe:
            safe = "emoji"
        out = out_dir / f"{safe}{ext}"
        with out.open("wb") as f:
            f.write(r.content)
        return out
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the scraper.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including output paths and flags.
    """
    # 中文说明：命令行参数包括输出文件、是否下载图片、抓取条目上限等
    p = argparse.ArgumentParser()
    p.add_argument("--platform", type=str, choices=["baidu", "weibo", "bilibili", "douyin"], default="baidu", help="Target platform vendor (baidu, weibo, bilibili, or douyin)")
    p.add_argument("--platform_url", type=str, default=None, help="Override platform page URL")
    p.add_argument("--out_json", type=str, default=None, help="Output JSON path")
    p.add_argument("--out_csv", type=str, default=None, help="Output CSV path")
    p.add_argument("--out_name_csv", type=str, default=None, help="Output image-name mapping CSV path")
    p.add_argument("--download_images", action="store_true", help="Download vendor images to local directory")
    p.add_argument("--img_dir", type=str, default=None, help="Image output directory if downloading")
    p.add_argument("--limit", type=int, default=0, help="Limit number of entries to process (0 means no limit)")
    p.add_argument("--sleep", type=float, default=0.4, help="Sleep seconds between detail requests to be polite")
    p.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    p.add_argument("--attempts", type=int, default=5, help="Max attempts for fetching pages")
    p.add_argument("--backoff", type=float, default=1.5, help="Backoff factor for retry sleeps")
    return p.parse_args()


def try_fetch_with_locales(
    sess: requests.Session,
    url: str,
    locales: List[str],
    timeout: int,
    attempts: int,
    backoff: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Attempt fetching platform page with locale and protocol fallbacks.

    Parameters
    ----------
    sess : requests.Session
        Configured HTTP session.
    url : str
        Base platform URL (typically zh-hans).
    locales : List[str]
        Locale candidates to try in order, e.g., ["zh-hans", "zh-hant", "en"].
    timeout : int
        Request timeout.
    attempts : int
        Max attempts per URL.
    backoff : float
        Backoff factor.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        (html, final_url) where html is the fetched content or None
        and final_url is the URL that succeeded.
    """
    # 中文说明：当主页面获取失败时，尝试不同语言路径与 http 协议
    def replace_locale(u: str, loc: str) -> str:
        return u.replace("/zh-hans/", f"/{loc}/") if "/zh-hans/" in u else u

    # 先尝试原 URL
    html = fetch_html(sess, url, timeout=timeout, attempts=attempts, backoff=backoff)
    if html:
        return html, url

    # 尝试不同语言
    for loc in locales:
        alt = replace_locale(url, loc)
        html = fetch_html(sess, alt, timeout=timeout, attempts=attempts, backoff=backoff)
        if html:
            return html, alt

    # 尝试 http 协议（某些网络环境下 https 证书或 SNI 可能异常）
    if url.startswith("https://"):
        http_url = "http://" + url[len("https://") :]
        html = fetch_html(sess, http_url, timeout=timeout, attempts=attempts, backoff=backoff)
        if html:
            return html, http_url

    # 尝试 http + 语言回退
    for loc in locales:
        alt = replace_locale(url, loc)
        if alt.startswith("https://"):
            alt_http = "http://" + alt[len("https://") :]
        else:
            alt_http = alt
        html = fetch_html(sess, alt_http, timeout=timeout, attempts=attempts, backoff=backoff)
        if html:
            return html, alt_http

    return None, None


def main() -> None:
    """Entry point: scrape Baidu vendor mapping and export JSON/CSV.

    Workflow
    --------
    - Fetch platform page and parse all emoji entries.
    - Enrich each entry with decoded emoji character and codepoints.
    - Optionally download vendor images locally.
    - Write consolidated outputs to JSON and CSV.
    """
    # 中文说明：整体流程：抓平台页 -> 解析条目 -> 解析详情 -> 可选下载图片 -> 保存数据
    args = parse_args()
    sess = build_session()

    # 计算平台 URL 与默认输出位置
    platform = args.platform
    platform_url = args.platform_url or PLATFORM_URLS.get(platform, PLATFORM_URL)
    # 中文说明：将默认输出目录设置为仓库根目录下的 data/vendor 与 data/emoji_images
    repo_root = Path(__file__).resolve().parents[2]  # emoji-finegrained-emotion 根目录
    default_vendor_dir = repo_root / "data" / "vendor"
    default_images_dir = repo_root / "data" / "emoji_images"
    # 英文注释：Default paths now target repo's data folders to avoid CWD issues
    out_json = Path(args.out_json) if args.out_json else default_vendor_dir / f"{platform}_emojiall_map.json"
    out_csv = Path(args.out_csv) if args.out_csv else default_vendor_dir / f"{platform}_emojiall_map.csv"
    # 中文说明：新增图片-名称专用映射CSV的默认输出位置
    out_name_csv = Path(args.out_name_csv) if args.out_name_csv else default_vendor_dir / f"{platform}_image_name_map.csv"
    img_dir = Path(args.img_dir) if args.img_dir else default_images_dir / platform

    html, final_url = try_fetch_with_locales(
        sess,
        platform_url,
        locales=["zh-hans", "zh-hant", "en"],
        timeout=args.timeout,
        attempts=args.attempts,
        backoff=args.backoff,
    )
    if not html:
        print(f"[Error] Failed to fetch platform page with fallbacks: {platform_url}")
        return
    if final_url and final_url != platform_url:
        print(f"[Info] Fallback used, fetching from: {final_url}")

    soup = BeautifulSoup(html, "html.parser")
    table = find_mapping_table(soup, PLATFORM_PHRASES.get(platform, PLATFORM_PHRASES["baidu"]))
    enriched: List[Dict] = []

    if table:
        # 优先解析“Baidu表情对应的Emoji”表格
        table_rows = parse_mapping_table(table)
        if args.limit and args.limit > 0:
            table_rows = table_rows[: args.limit]
        # 统计两类数量，用于日志
        mapped_count = sum(1 for r in table_rows if r.get("kind") == "mapped")
        vendor_only_count = sum(1 for r in table_rows if r.get("kind") == "vendor_only")
        print(f"[{platform.capitalize()}] Parsed mapping table rows: {len(table_rows)} | mapped={mapped_count} | vendor_only={vendor_only_count}")
        # 表格中已拿到 emoji 与图片、名称；如果有详情链接则可补充
        for e in table_rows:
            if e.get("kind") == "mapped":
                # 若 emoji 缺失但存在详情链接（极端情况），尝试补充
                if not e.get("emoji") and e.get("detail_url"):
                    enriched_e = enrich_with_detail(
                        sess,
                        {"detail_url": e["detail_url"], "img_url": e.get("img_url"), "alt": e.get("name")},
                        timeout=args.timeout,
                        attempts=max(1, args.attempts // 2),
                        backoff=max(0.8, args.backoff * 0.8),
                    )
                    e.update({"emoji": enriched_e.get("emoji"), "codepoints": enriched_e.get("codepoints")})
            enriched.append(e)
    else:
        # 兜底：使用平台页通用解析（不限定三列）
        entries = parse_platform_page(html)
        if args.limit and args.limit > 0:
            entries = entries[: args.limit]
        print(f"[{platform.capitalize()}] Parsed platform entries: {len(entries)}")
        for i, e in enumerate(entries, 1):
            enriched_e = enrich_with_detail(sess, e)
            enriched.append(enriched_e)
            # 礼貌休眠，避免请求过于频繁
            if args.sleep > 0:
                time.sleep(args.sleep)
            if i % 50 == 0:
                print(f"  processed {i} entries...")

    # 新增：尝试定位“<平台>提供的Emoji表情符号图片列表”并解析为 vendor_only
    vendor_section = find_vendor_list_section(soup, platform)
    if vendor_section:
        vendor_only_rows = parse_vendor_list(vendor_section)
        # 去除与已映射重复的图片（按 img_url 匹配）
        mapped_img_urls = {e.get("img_url") for e in enriched if e.get("kind") == "mapped"}
        added = 0
        for e in vendor_only_rows:
            if e.get("img_url") not in mapped_img_urls:
                enriched.append(e)
                added += 1
        print(f"[{platform.capitalize()}] Vendor-only list entries added: {added}")

    # 可选下载图片
    if args.download_images:
        # 将未映射的图片放入单独目录，区分存储
        img_dir_unmapped = Path(str(img_dir) + "_unmapped")
        saved_m, failed_m = 0, 0
        saved_u, failed_u = 0, 0
        for e in enriched:
            url = e.get("img_url")
            if not url:
                if e.get("kind") == "vendor_only":
                    failed_u += 1
                else:
                    failed_m += 1
                continue
            hint = e.get("emoji") or e.get("name")
            # 根据类型选择输出目录
            target_dir = img_dir_unmapped if e.get("kind") == "vendor_only" else img_dir
            out = download_image(sess, url, target_dir, hint)
            if out:
                # 中文说明：若下载成功，记录本地文件路径，便于后续建立图片与名称映射
                e["local_path"] = str(out)
                if e.get("kind") == "vendor_only":
                    saved_u += 1
                else:
                    saved_m += 1
            else:
                if e.get("kind") == "vendor_only":
                    failed_u += 1
                else:
                    failed_m += 1
        print(f"[Images] Mapped Saved: {saved_m} | Failed: {failed_m} | Dir: {img_dir}")
        print(f"[Images] Unmapped Saved: {saved_u} | Failed: {failed_u} | Dir: {img_dir_unmapped}")

    # 输出 JSON 与 CSV
    write_json(out_json, enriched)
    write_csv(out_csv, enriched)
    # 中文说明：额外写出图片-表情名映射CSV，满足对“图片与表情名对应关系”的保存需求
    write_image_name_csv(out_name_csv, enriched)
    print(f"[Output] JSON: {out_json} | CSV: {out_csv} | NAME_CSV: {out_name_csv}")


if __name__ == "__main__":
    main()