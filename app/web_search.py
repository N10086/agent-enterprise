"""web_search 的后端实现。

对外只暴露 `run_web_search(query, max_results)`，工具定义在 `tools.py`。

后端按查询语言排序后**并行**取优先级最高的两个引擎，再轮询交替取结果、
按 URL 去重合并——单一引擎在本机网络下经常给出跑偏或空结果。

每个后端统一返回 ``[(title, url, snippet), ...]``；任一后端可用即可，
全部失败时返回带说明的错误文本，提示模型如实说明而不是编造。
"""
from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser

_SEARCH_TIMEOUT = 20
_SNIPPET_LIMIT = 300
_OUTPUT_LIMIT = 3000
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
#: 搜索页里的非结果块标题
_SKIP_TITLES = {"其他人还搜了", "相关搜索", "大家还在搜", "相关推荐"}


def _http_get(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    )
    with urllib.request.urlopen(request, timeout=_SEARCH_TIMEOUT) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, "ignore")


class _ResultParser(HTMLParser):
    """把搜索结果页解析成 [(title, url, snippet)]，只用标准库。

    两种引擎结构可以归纳成同一套规则：标题都在 h2/h3 里的 <a> 上，
    摘要文本跟在标题之后。Bing 是
    ``<li class="b_algo"><h2><a>标题</a></h2><p>摘要</p>``，
    360/搜狗是 ``<h3><a>标题</a></h3>`` 后跟一段文本。

    之所以不用 BeautifulSoup：它并非到处都装了，缺了会让联网搜索整块失效；
    标准库的 HTMLParser 足够应付这两种结构，还少一个依赖。
    """

    _HEADINGS = {"h2", "h3"}
    _SNIPPET_LIMIT = 300

    def __init__(self, max_results: int, base_url: str = ""):
        super().__init__(convert_charrefs=True)
        self.max_results = max_results
        self.base_url = base_url
        self.results: list[tuple[str, str, str]] = []
        self._title: list[str] = []
        self._url: str | None = None
        self._snippet: list[str] = []
        self._in_heading = False
        self._finished = False

    def handle_starttag(self, tag, attrs):
        if self._finished or tag in ("script", "style"):
            return
        if tag in self._HEADINGS:
            self._close_result()
            self._in_heading, self._title, self._url = True, [], None
        elif tag == "a" and self._in_heading and self._url is None:
            self._url = dict(attrs).get("href") or None

    def handle_endtag(self, tag):
        if tag in self._HEADINGS:
            self._in_heading = False

    def handle_data(self, data):
        if self._finished:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_heading:
            self._title.append(text)
        elif sum(len(part) for part in self._snippet) < self._SNIPPET_LIMIT:
            self._snippet.append(text)

    def _close_result(self):
        """一条结果结束（遇到下一个标题或文档末尾），整理并收进结果列表。"""
        title = " ".join(self._title).strip()
        url = self._url
        snippet = " ".join(self._snippet).strip()[: self._SNIPPET_LIMIT]
        self._title, self._url, self._snippet = [], None, []

        if not title or not url or title in _SKIP_TITLES:
            return
        if url.startswith("/"):
            url = self.base_url + url
        elif not url.startswith("http"):
            return

        self.results.append((title, url, snippet))
        if len(self.results) >= self.max_results:
            self._finished = True

    def close(self):
        super().close()
        if not self._finished:
            self._close_result()


def _parse_results(page: str, max_results: int, base_url: str = ""):
    parser = _ResultParser(max_results, base_url)
    parser.feed(page)
    parser.close()
    return parser.results


def _search_bing(query: str, max_results: int):
    url = "https://www.bing.com/search?" + urllib.parse.urlencode(
        {"q": query, "count": max(max_results * 2, 10), "setlang": "zh-CN", "mkt": "zh-CN"}
    )
    return _parse_results(_http_get(url), max_results)


def _search_sogou(query: str, max_results: int):
    url = "https://www.sogou.com/web?" + urllib.parse.urlencode({"query": query})
    return _parse_results(_http_get(url), max_results, "https://www.sogou.com")


def _search_so360(query: str, max_results: int):
    url = "https://www.so.com/s?" + urllib.parse.urlencode({"q": query})
    return _parse_results(_http_get(url), max_results, "https://www.so.com")


def _search_duckduckgo(query: str, max_results: int):
    from duckduckgo_search import DDGS

    hits = DDGS().text(query, max_results=max_results) or []
    return [(h.get("title", ""), h.get("href", ""), h.get("body", "")) for h in hits]


def _search_tavily(query: str, max_results: int, api_key: str):
    from tavily import TavilyClient

    data = TavilyClient(api_key=api_key, timeout=_SEARCH_TIMEOUT).search(
        query=query, max_results=max_results
    )
    return [
        (r.get("title", ""), r.get("url", ""), r.get("content", ""))
        for r in data.get("results", [])
    ]


def _backend_order(query: str) -> list[str]:
    """中文查询优先用国内引擎（Bing 对这类查询经常跑偏），英文查询优先用 Bing/360。"""
    if re.search(r"[\u4e00-\u9fff]", query):
        return ["so360", "sogou", "bing", "duckduckgo"]
    return ["bing", "so360", "sogou", "duckduckgo"]


def _merge_round_robin(outcomes: dict, order: list[str], max_results: int):
    """按引擎优先级轮询交替取结果，去重后合并。"""
    available = [name for name in order if outcomes.get(name)]
    merged, seen, index = [], set(), 0
    while len(merged) < max_results and available:
        progressed = False
        for name in available:
            items = outcomes[name]
            if index >= len(items):
                continue
            title, url, snippet = items[index]
            key = (url or title).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append((title, url, snippet))
            progressed = True
            if len(merged) >= max_results:
                break
        if not progressed:
            break
        index += 1
    return merged, available


def _format_results(source: str, items) -> str:
    lines = [f"（来源：{source}，共 {len(items)} 条）"]
    used = 0
    for index, (title, url, snippet) in enumerate(items, 1):
        snippet = snippet.strip()
        if len(snippet) > _SNIPPET_LIMIT:
            snippet = snippet[: _SNIPPET_LIMIT] + "…"
        entry = f"{index}. {title or '(无标题)'}\n   {url}\n   {snippet or '(无摘要)'}"
        if used + len(entry) > _OUTPUT_LIMIT:
            lines.append("…（结果过多，已截断）")
            break
        lines.append(entry)
        used += len(entry)
    return "\n".join(lines)


def run_web_search(query: str, max_results: int = 5) -> str:
    """并行查询优先级最高的两个引擎；都失败再依次试其余后端。"""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    order = _backend_order(query)
    if api_key:
        order = ["tavily"] + order

    registry = {
        "tavily": lambda: _search_tavily(query, max_results, api_key),
        "bing": lambda: _search_bing(query, max_results),
        "so360": lambda: _search_so360(query, max_results),
        "sogou": lambda: _search_sogou(query, max_results),
        "duckduckgo": lambda: _search_duckduckgo(query, max_results),
    }

    failures: list[str] = []
    for group in (order[:2], order[2:]):
        if not group:
            continue

        outcomes: dict = {}
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            futures = {pool.submit(registry[name]): name for name in group}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    outcomes[name] = future.result()
                except Exception as exc:
                    failures.append(f"{name}({type(exc).__name__})")

        for name in group:
            if name in outcomes and not outcomes[name]:
                failures.append(f"{name}(被拦截或无结果)")

        merged, sources = _merge_round_robin(outcomes, group, max_results)
        if merged:
            return _format_results("+".join(sources), merged)

    detail = "、".join(failures) if failures else "所有后端均无结果"
    return f"Error: 联网搜索失败（{detail}）。请如实告知用户你无法获取实时信息，不要凭记忆编造。"


__all__ = ["run_web_search"]
