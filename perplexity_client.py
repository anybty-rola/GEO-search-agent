from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


def _coerce_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content") or ""
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())
        return "\n".join([p for p in parts if p]).strip()
    if content is None:
        return ""
    return str(content).strip()


def _extract_assistant_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            c = _coerce_text_content(msg.get("content"))
            if c:
                return c
        t = _coerce_text_content(choices[0].get("text"))
        if t:
            return t
    return _coerce_text_content(data.get("output_text"))


@dataclass(frozen=True)
class PerplexityClient:
    """
    Perplexity 官方 Chat Completions API（/v1/sonar）轻量封装。
    文档：https://docs.perplexity.ai/api-reference/chat-completions-post
    """

    api_key: str
    base_url: str = "https://api.perplexity.ai"
    model: str = "sonar-pro"
    timeout_s: int = 60

    def chat(
        self,
        *,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        language_preference: Optional[str] = None,
        search_mode: Optional[str] = "web",
    ) -> Tuple[str, List[str]]:
        url = self.base_url.rstrip("/") + "/v1/sonar"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        if language_preference:
            payload["language_preference"] = language_preference
        if search_mode:
            payload["search_mode"] = search_mode

        r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
        if r.status_code >= 400:
            raise RuntimeError(f"PERPLEXITY_HTTP_{r.status_code}: {r.text[:800]}")
        data = r.json()
        text = _extract_assistant_text(data)
        if not text:
            raise RuntimeError(f"PERPLEXITY_EMPTY_RESPONSE: {str(data)[:600]}")
        citations = data.get("citations") or []
        if not isinstance(citations, list):
            citations = []
        citations = [str(u).strip() for u in citations if str(u).strip()]
        # 去重保序
        seen = set()
        out: List[str] = []
        for u in citations:
            if u in seen:
                continue
            seen.add(u)
            out.append(u)
        return text, out

