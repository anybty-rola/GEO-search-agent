from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class QueryAgent:
    """
    生成“用户会怎么搜/问”的查询列表（用于 GEO 分析的输入）。
    这是 MVP：用规则 + 少量模板，避免过度依赖模型生成导致不稳定。
    """

    max_queries: int = 8

    def run(self, keyword: str, *, lang: str = "zh") -> List[str]:
        kw = (keyword or "").strip()
        if not kw:
            return []

        lang = (lang or "zh").strip().lower()
        if lang == "en":
            templates = [
                "{kw} best picks",
                "{kw} buying guide",
                "{kw} worth it",
                "{kw} vs alternatives",
                "{kw} review",
                "{kw} pros and cons",
                "{kw} price",
                "{kw} reddit",
                "{kw} for travel",
                "{kw} safety",
            ]
        else:
            templates = [
                "{kw} 推荐",
                "{kw} 怎么选",
                "{kw} 值得买吗",
                "{kw} 对比",
                "{kw} 评测",
                "{kw} 优缺点",
                "{kw} 价格",
                "{kw} 哪个牌子好",
                "{kw} 适合什么人",
                "{kw} 使用体验",
            ]
        queries = [t.format(kw=kw) for t in templates]
        return queries[: self.max_queries]

