from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException


@dataclass(frozen=True)
class PlatformAgent:
    """
    MVP 版本的“平台视角/来源”：
    - 使用 DuckDuckGo 做通用网页搜索，拿到标题/摘要/链接作为可引用来源
    - 通过 `site:` 做轻量“平台倾向”过滤（不保证完全命中）
    """

    max_results_per_query: int = 5
    retry_times: int = 2
    retry_sleep_s: float = 1.2

    PLATFORM_SITES: Dict[str, List[str]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.PLATFORM_SITES is None:
            object.__setattr__(
                self,
                "PLATFORM_SITES",
                {
                    "通用网页": [],
                    "Reddit": ["reddit.com"],
                    "YouTube": ["youtube.com"],
                    "知乎": ["zhihu.com"],
                    "小红书": ["xiaohongshu.com"],
                    "电商": ["amazon.com", "jd.com", "taobao.com", "tmall.com"],
                    "品牌官网": [],
                },
            )

    def _search(self, query: str, site: str | None = None) -> List[dict]:
        q = query if not site else f"{query} site:{site}"
        last_err: Exception | None = None

        for attempt in range(self.retry_times + 1):
            try:
                results: List[dict] = []
                with DDGS() as ddgs:
                    # backend="lite" 通常更稳、更不容易触发限制
                    for r in ddgs.text(q, max_results=self.max_results_per_query, backend="lite"):
                        results.append(
                            {
                                "title": r.get("title") or "",
                                "url": r.get("href") or "",
                                "snippet": r.get("body") or "",
                            }
                        )
                return results
            except DuckDuckGoSearchException as e:
                last_err = e
                # 常见：rate limit / 429。做退避重试
                if attempt < self.retry_times:
                    time.sleep(self.retry_sleep_s * (attempt + 1))
                    continue
                return []
            except Exception as e:
                last_err = e
                if attempt < self.retry_times:
                    time.sleep(self.retry_sleep_s * (attempt + 1))
                    continue
                return []

    def run(self, queries: List[str]) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {k: [] for k in self.PLATFORM_SITES.keys()}
        for q in queries:
            # 通用网页
            out["通用网页"].extend(self._search(q))

            # 带站点约束的平台
            for platform, sites in self.PLATFORM_SITES.items():
                if not sites or platform == "通用网页":
                    continue
                for site in sites[:2]:
                    out[platform].extend(self._search(q, site=site))

        # 简单去重（按 url）
        for platform, items in out.items():
            seen = set()
            deduped = []
            for it in items:
                url = (it.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                deduped.append(it)
            out[platform] = deduped
        return out

