from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import requests


_URL_RE = re.compile(r"https?://[^\s\)\]]+")


def extract_urls(text: str) -> List[str]:
    urls = _URL_RE.findall(text or "")
    # 轻度清洗去重
    out = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(".,;")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


@dataclass(frozen=True)
class DeepSeekClient:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_s: int = 60

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")
        base_url = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").strip()
        model = (os.getenv("DEEPSEEK_MODEL") or "deepseek-chat").strip()
        return cls(api_key=api_key, base_url=base_url, model=model)

    def chat(self, prompt: str) -> Tuple[str, List[str]]:
        url = self.base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个GEO（Generative Engine Optimization）分析助手。你必须在回答末尾给出Sources列表，并尽量引用给定材料里的URL。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        if r.status_code >= 400:
            # 让上层更容易做“友好降级”
            raise RuntimeError(f"DEEPSEEK_HTTP_{r.status_code}: {r.text[:800]}")
        data = r.json()
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        urls = extract_urls(content)
        return content, urls

