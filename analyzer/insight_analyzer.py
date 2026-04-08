from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlparse


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower().replace("www.", "")
    except Exception:
        return ""


@dataclass
class GeoReport:
    keyword: str
    answer: str
    queries: List[str] = field(default_factory=list)
    cited_sources: List[str] = field(default_factory=list)
    top_domains: List[str] = field(default_factory=list)
    platform_domain_breakdown: Dict[str, List[str]] = field(default_factory=dict)
    takeaways: List[str] = field(default_factory=list)
    prompt_library: List[str] = field(default_factory=list)
    synthetic_answers: Dict[str, str] = field(default_factory=dict)
    synthetic_citations: Dict[str, List[str]] = field(default_factory=dict)
    report_lang: str = ""
    source_region: str = ""
    geo_playbook: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class InsightAnalyzer:
    """
    将“模型回答 + 搜索来源”整理成一个结构化 GEO 报告（MVP）。
    """

    def run(
        self,
        *,
        keyword: str,
        queries: List[str],
        answer: str,
        platform_sources: Dict[str, List[dict]],
        cited_sources: Optional[List[str]] = None,
        lang: str = "zh",
        prompt_library: Optional[List[str]] = None,
        synthetic_answers: Optional[Dict[str, str]] = None,
        synthetic_citations: Optional[Dict[str, List[str]]] = None,
        source_region: str = "",
        geo_playbook: str = "",
    ) -> GeoReport:
        cited_sources = cited_sources or []
        lang = (lang or "zh").strip().lower()
        prompt_library = prompt_library if prompt_library is not None else list(queries)
        synthetic_answers = synthetic_answers or {}
        synthetic_citations = synthetic_citations or {}

        # 统计域名
        all_domains: List[str] = []
        platform_domains: Dict[str, List[str]] = {}
        for platform, items in platform_sources.items():
            ds = []
            for it in items:
                d = _domain(it.get("url") or "")
                if d:
                    ds.append(d)
                    all_domains.append(d)
            platform_domains[platform] = ds

        top_domains = [d for d, _ in Counter(all_domains).most_common(10)]

        # 简单要点（可替换为更强的 LLM 分析）
        takeaways = []
        if lang == "en":
            if top_domains:
                takeaways.append(f"Most frequent source domains: {', '.join(top_domains[:5])}")
            if cited_sources:
                takeaways.append(f"The answer cites {len(cited_sources)} URLs (useful for GEO attribution).")
            if queries:
                takeaways.append(f"Generated {len(queries)} simulated user queries for intent coverage.")
        else:
            if top_domains:
                takeaways.append(f"最常出现的来源域名：{', '.join(top_domains[:5])}")
            if cited_sources:
                takeaways.append(f"回答中引用了 {len(cited_sources)} 个来源（可用于 GEO 归因）。")
            if queries:
                takeaways.append(f"共生成 {len(queries)} 条用户查询，用于模拟不同搜索意图。")

        return GeoReport(
            keyword=keyword,
            queries=queries,
            answer=answer,
            cited_sources=cited_sources,
            top_domains=top_domains,
            platform_domain_breakdown={
                p: [d for d, _ in Counter(ds).most_common(10)]
                for p, ds in platform_domains.items()
            },
            takeaways=takeaways,
            prompt_library=list(prompt_library),
            synthetic_answers=dict(synthetic_answers),
            synthetic_citations={k: list(v or []) for k, v in dict(synthetic_citations).items()},
            report_lang=lang,
            source_region=source_region,
            geo_playbook=geo_playbook,
        )

