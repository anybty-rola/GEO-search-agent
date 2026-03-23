from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


_URL_RE = re.compile(r"https?://[^\s\)\]]+")


def extract_urls(text: str) -> List[str]:
    urls = _URL_RE.findall(text or "")
    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(".,;")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _coerce_text_content(content: Any) -> str:
    """
    兼容不同 provider/model 的返回格式：
    - str
    - list[{"type":"text","text":"..."}]
    - 其他可转字符串对象
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt)
        return "\n".join([p for p in parts if p]).strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_assistant_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            content = _coerce_text_content(msg.get("content"))
            if content:
                return content
        # 某些实现把文本放在 choices[0].text
        txt = _coerce_text_content(choices[0].get("text"))
        if txt:
            return txt

    # 少数兼容层可能返回 output_text
    txt = _coerce_text_content(data.get("output_text"))
    if txt:
        return txt
    return ""


@dataclass(frozen=True)
class OpenAICompatClient:
    """
    OpenAI-compatible Chat Completions client.
    Works with DeepSeek / OpenRouter and other compatible providers.
    """

    api_key: str
    base_url: str
    model: str
    timeout_s: int = 60
    extra_headers: Optional[Dict[str, str]] = None

    def chat(self, prompt: str) -> Tuple[str, List[str]]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.extra_headers:
            headers.update(self.extra_headers)

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
            raise RuntimeError(f"LLM_HTTP_{r.status_code}: {r.text[:800]}")
        data = r.json()
        content = _extract_assistant_text(data)
        if not content:
            # 给出可读错误，避免 silent empty response
            raise RuntimeError(f"LLM_EMPTY_RESPONSE: {str(data)[:500]}")
        return content, extract_urls(content)


def build_client(
    *,
    provider: str,
    api_key: str,
    model: str,
    base_url: str,
    site_url: Optional[str] = None,
    app_name: Optional[str] = None,
) -> OpenAICompatClient:
    provider = (provider or "").strip().lower()

    extra_headers: Dict[str, str] = {}
    if provider == "openrouter":
        # OpenRouter 推荐带上这两个 header（可选，但有助于额度/风控）
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name

    return OpenAICompatClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        extra_headers=extra_headers or None,
    )


def env_get(*names: str) -> str:
    for n in names:
        v = (os.getenv(n) or "").strip()
        if v:
            return v
    return ""

