from __future__ import annotations

import json
import os
import sys
import datetime as _dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).parent
# 兼容“从不同工作目录启动 streamlit”的情况：确保本地模块可被导入
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from agents import PlatformAgent, QueryAgent
from analyzer import InsightAnalyzer
from llm_client import build_client, env_get
from perplexity_client import PerplexityClient
from region_sites import get_platform_sites, get_region_label, platform_choices_for_region


def _eff_model(override: str, main: str) -> str:
    """侧栏「留空=主模型」：override 非空则覆盖主 Model。"""
    o = (override or "").strip()
    return o if o else (main or "").strip()


def load_keywords_csv() -> List[str]:
    p = APP_DIR / "data" / "keywords.csv"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    # 第一行 header
    out = []
    for line in lines[1:]:
        line = line.strip()
        if line:
            out.append(line)
    return out


def format_sources(platform_sources: Dict[str, List[dict]], max_each: int = 6) -> str:
    chunks = []
    for platform, items in platform_sources.items():
        if not items:
            continue
        chunks.append(f"## {platform}")
        for it in items[:max_each]:
            title = (it.get("title") or "").strip()
            url = (it.get("url") or "").strip()
            snippet = (it.get("snippet") or "").strip()
            chunks.append(f"- {title}\n  - {url}\n  - {snippet}")
    return "\n".join(chunks)


def _resolve_report_lang(choice: str, keyword: str) -> str:
    c = (choice or "").strip()
    if c.startswith("中文"):
        return "zh"
    if c == "English":
        return "en"
    # 自动（随关键词）：含中日韩统一表意文字则中文，否则英文
    kw = keyword or ""
    for ch in kw:
        if "\u4e00" <= ch <= "\u9fff":
            return "zh"
    return "en"


def build_prompt(
    keyword: str,
    queries: List[str],
    platform_sources: Dict[str, List[dict]],
    *,
    lang: str,
    source_region_label: str,
    synthetic_block: str = "",
) -> str:
    sources_md = format_sources(platform_sources)
    q_md = "\n".join([f"- {q}" for q in queries])
    lang = (lang or "zh").strip().lower()
    syn = (synthetic_block or "").strip()
    syn_en = (
        f"\n\nSynthetic multi-\"platform\" style sketches (via API; not real product crawl):\n{syn}\n"
        if syn
        else ""
    )
    syn_zh = (
        f"\n\n以下为「多平台风格模拟」摘要（通过 API 并行生成，非各产品真实网页结果）：\n{syn}\n"
        if syn
        else ""
    )
    if lang == "en":
        return f"""Keyword: {keyword}
Target source region profile: {source_region_label}

Simulated user queries:
{q_md}

Below are web snippets aggregated via search (title / URL / snippet). Write a helpful consumer-facing comparison/buying guide grounded in these materials. End with a Sources section listing URLs.

Materials:
{sources_md}
{syn_en}

Requirements:
1) Output entirely in English (headings + bullets).
2) Mention notable brands and decision criteria when supported by materials.
3) End with:

Sources:
- https://...
- https://...
"""
    return f"""关键词：{keyword}
信源地区：{source_region_label}

用户可能会这样搜/问：
{q_md}

下面是从网页搜索聚合到的材料（标题/URL/摘要）。你需要基于这些材料写一个面向用户的推荐/对比回答，并在最后输出Sources（URL列表）。

材料：
{sources_md}
{syn_zh}

输出要求：
1) 全文使用中文，结构清晰（小标题 + 要点）
2) 明确提到常见品牌/选购要点（如果材料支持）
3) 末尾必须有：
Sources:
- https://...
- https://...
"""


def build_mock_answer(
    keyword: str,
    queries: List[str],
    platform_sources: Dict[str, List[dict]],
    *,
    lang: str,
) -> str:
    lang = (lang or "zh").strip().lower()
    # 不调用模型的降级方案：把来源摘要整理成“可读”的回答骨架
    total = sum(len(v) for v in platform_sources.values())
    top_sources = []
    for platform, items in platform_sources.items():
        for it in items[:3]:
            if it.get("url"):
                top_sources.append(it["url"])
    top_sources = list(dict.fromkeys(top_sources))[:8]

    q_md = "\n".join([f"- {q}" for q in queries[:8]])
    s_md = "\n".join([f"- {u}" for u in top_sources]) if top_sources else "- (none yet — retry)"
    if lang == "en":
        return f"""## (Mock) First-pass notes on "{keyword}"

**No LLM / Mock mode.** Draft from queries + aggregated links only.

### Queries users might run
{q_md}

### Next steps
- Inspect raw sources for repeated domains (reviews, retail, forums, official sites).
- Re-run with fewer platforms / lower `max results` if you hit rate limits.

### Sources (aggregated {total})
{s_md}
"""
    s_md_zh = "\n".join([f"- {u}" for u in top_sources]) if top_sources else "- （暂无，可稍后重试）"
    return f"""## （Mock）关于「{keyword}」的初步结论

当前为**无模型/Mock 模式**（未调用 LLM），因此以下内容为基于“查询意图 + 来源列表”的结构化草案。

### 你可以这样搜/这样问
{q_md}

### 下一步怎么做（建议）
- 先看 “查看聚合来源（原材料）” 里出现频率最高的域名与内容类型（评测/电商/论坛/官方）
- 针对 1-2 个子问题（比如价格、对比、优缺点）重新点击 Run Analysis，以减少限流与噪声

### Sources（聚合到 {total} 条）
{s_md_zh}
"""

def _explain_llm_error(err: Exception) -> str:
    """
    将 llm_client.py 抛出的 RuntimeError('LLM_HTTP_xxx: ...') 转成更友好的提示。
    注意：不会包含任何 API Key。
    """
    if isinstance(err, requests.exceptions.Timeout):
        return (
            "请求超时：本机到 OpenRouter 的网络较慢或被代理拦截。"
            "可稍后重试、换网络/VPN，或在 `llm_client.py` 里适当增大 `timeout_s`。"
        )
    if isinstance(err, requests.exceptions.ConnectionError):
        return (
            "无法建立连接：请确认能访问 https://openrouter.ai（浏览器试一下），"
            "并检查系统代理/VPN/公司防火墙是否拦截出站 HTTPS。"
        )

    msg = str(err)
    if msg.startswith("LLM_EMPTY_RESPONSE"):
        return "模型返回了空响应或非标准文本格式，已自动降级为 Mock。建议更换模型后重试。"
    if msg.startswith("LLM_HTTP_"):
        try:
            code = int(msg.split("LLM_HTTP_")[1].split(":")[0])
        except Exception:
            code = None
        body = msg.split(":", 1)[1].strip() if ":" in msg else msg

        if code == 401:
            return "401 未授权：API Key 无效/未开通，或 Provider 选错（比如用 DeepSeek Key 调 OpenRouter）。"
        if code == 402:
            return "402 余额/额度不足：需要在对应平台充值/开通额度（有些“free”模型也需要账号有可用额度/风控通过）。"
        if code == 403:
            return "403 被拒绝：可能是风控/地区/权限限制。"
        if code == 404:
            return "404 模型或接口不存在：请检查 Model 名称是否在该 Provider 可用。"
        if code == 429:
            return "429 频率限制：请求太快或触发限流，稍等 30-120 秒再试，或换一个模型。"
        return f"{code} 调用失败：{body[:200]}"

    return msg[:200]

def _fetch_openrouter_models(*, api_key: str, base_url: str) -> List[str]:
    """
    拉取 OpenRouter 可用模型列表（返回 model id 列表）。
    需要 OpenRouter API Key；不会打印/回显 key。
    """
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter 模型列表拉取失败 {r.status_code}: {r.text[:300]}")
    data = r.json()
    items = data.get("data") or data.get("models") or []
    out = []
    for it in items:
        mid = (it.get("id") or "").strip()
        if mid:
            out.append(mid)
    out = sorted(set(out))
    return out


def _try_register_cjk_font() -> str | None:
    """
    尝试注册一个可显示中文的字体（Windows 常见：微软雅黑/宋体）。
    返回已注册字体名；失败返回 None（PDF 可能出现乱码）。
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return None

    candidates = [
        ("MSYH", r"C:\Windows\Fonts\msyh.ttc"),
        ("SIMSUN", r"C:\Windows\Fonts\simsun.ttc"),
        ("MSYH", str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "msyh.ttc")),
        ("SIMSUN", str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "simsun.ttc")),
    ]

    for font_name, font_path in candidates:
        try:
            if Path(font_path).exists():
                # 避免重复注册报错
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
        except Exception:
            continue

    # 兜底：使用 ReportLab 自带的中文 CID 字体（不依赖系统字体文件）
    # 注：这能避免 Helvetica 在输出中文时直接报 UnicodeEncodeError。
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        fallback = "STSong-Light"
        if fallback not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(UnicodeCIDFont(fallback))
        return fallback
    except Exception:
        return None
    return None


def _wrap_text(text: str, *, max_chars: int) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines: List[str] = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            lines.append("")
            continue
        while len(s) > max_chars:
            lines.append(s[:max_chars])
            s = s[max_chars:]
        lines.append(s)
    return lines


def build_pdf_bytes(
    *,
    keyword: str,
    answer_md: str,
    report: object,
    source_region_label: str = "",
    prompt_library: List[str] | None = None,
    synthetic_answers: Dict[str, str] | None = None,
    synthetic_citations: Dict[str, List[str]] | None = None,
    geo_playbook: str = "",
) -> bytes:
    """
    生成 PDF 报告（bytes），用于 streamlit download_button。
    """
    from io import BytesIO

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    font_name = _try_register_cjk_font() or "Helvetica"
    title_font = font_name
    body_font = font_name

    margin_x = 48
    y = height - 56

    def draw_line(s: str, size: int = 11, leading: int = 15) -> None:
        nonlocal y
        if y < 56:
            c.showPage()
            y = height - 56
        c.setFont(body_font, size)
        c.drawString(margin_x, y, s)
        y -= leading

    # 标题
    c.setFont(title_font, 18)
    c.drawString(margin_x, y, "GEO Multi-Agent 报告")
    y -= 24

    c.setFont(body_font, 11)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin_x, y, f"关键词 / Keyword: {keyword}")
    y -= 16
    c.drawString(margin_x, y, f"生成时间：{ts}")
    y -= 16
    if source_region_label:
        c.drawString(margin_x, y, f"信源地区：{source_region_label}")
        y -= 22
    else:
        y -= 6

    # report 是 GeoReport dataclass，带 to_dict()
    try:
        rdict = report.to_dict()  # type: ignore[attr-defined]
    except Exception:
        rdict = {}

    takeaways = rdict.get("takeaways") or []
    cited_sources = rdict.get("cited_sources") or []
    top_domains = rdict.get("top_domains") or []
    prompt_library = prompt_library if prompt_library is not None else (rdict.get("prompt_library") or [])
    synthetic_answers = synthetic_answers if synthetic_answers is not None else (rdict.get("synthetic_answers") or {})
    synthetic_citations = synthetic_citations if synthetic_citations is not None else (rdict.get("synthetic_citations") or {})
    geo_playbook = geo_playbook or (rdict.get("geo_playbook") or "")

    draw_line("Prompt 库（消费者拓展查询）：", size=13, leading=18)
    if prompt_library:
        for t in prompt_library[:20]:
            for ln in _wrap_text(f"- {t}", max_chars=70):
                draw_line(ln)
    else:
        draw_line("- （暂无）")
    y -= 8

    draw_line("多平台模拟回答（节选）：", size=13, leading=18)
    if synthetic_answers:
        for pname, ptxt in list(synthetic_answers.items())[:6]:
            draw_line(f"[{pname}]", size=11, leading=14)
            clip = (ptxt or "").replace("**", "")
            for ln in _wrap_text(clip[:1200], max_chars=78):
                draw_line(ln)
            cits = (synthetic_citations or {}).get(pname) or []
            if cits:
                draw_line("Citations:", size=10, leading=12)
                for u in cits[:8]:
                    for ln in _wrap_text(f"- {u}", max_chars=86):
                        draw_line(ln, size=9, leading=11)
            y -= 4
    else:
        draw_line("- （未启用或无结果）")
    y -= 8

    draw_line("GEO 行动建议（Playbook）：", size=13, leading=18)
    if geo_playbook:
        gp = geo_playbook.replace("**", "").replace("### ", "").replace("## ", "")
        for ln in _wrap_text(gp, max_chars=78):
            draw_line(ln)
    else:
        draw_line("- （暂无）")
    y -= 8

    draw_line("洞察要点：", size=13, leading=18)
    if takeaways:
        for t in takeaways:
            for ln in _wrap_text(f"- {t}", max_chars=70):
                draw_line(ln)
    else:
        draw_line("- （暂无）")
    y -= 8

    draw_line("AI 回答（节选/全文）：", size=13, leading=18)
    # PDF 里用纯文本展示，简单去掉 markdown 的多余符号
    answer_txt = (answer_md or "").replace("**", "").replace("### ", "").replace("## ", "").replace("# ", "")
    for ln in _wrap_text(answer_txt, max_chars=80):
        draw_line(ln)
    y -= 8

    draw_line("Sources（模型引用）：", size=13, leading=18)
    if cited_sources:
        for u in cited_sources[:30]:
            for ln in _wrap_text(f"- {u}", max_chars=90):
                draw_line(ln)
    else:
        draw_line("- （暂无）")
    y -= 8

    draw_line("Top Domains：", size=13, leading=18)
    if top_domains:
        draw_line(", ".join(top_domains[:15]))
    else:
        draw_line("（暂无）")

    if font_name == "Helvetica":
        y -= 18
        draw_line("注意：未检测到中文字体，PDF 可能出现中文乱码。可安装/启用“微软雅黑/宋体”后重试。", size=9, leading=12)

    c.showPage()
    c.save()
    return buf.getvalue()


def _parse_queries_from_text(text: str, *, max_queries: int) -> List[str]:
    lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        # 去掉常见的序号/项目符号
        for prefix in ("- ", "* ", "• ", "1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. "):
            if s.startswith(prefix):
                s = s[len(prefix) :].strip()
                break
        if not s:
            continue
        if s not in out:
            out.append(s)
        if len(out) >= max_queries:
            break
    return out[:max_queries]


def _parse_takeaways_from_text(text: str, *, max_items: int = 8) -> List[str]:
    items = _parse_queries_from_text(text, max_queries=max_items)
    return items


def _inject_streamlit_secrets_into_environ() -> None:
    """
    Streamlit Community Cloud：在 App Settings → Secrets 中配置的键，同步到 os.environ，
    便于与本地 .env / 侧边栏逻辑共用 env_get。
    """
    # 注意：在本地未配置 `.streamlit/secrets.toml` 时，`st.secrets` 的“取值/判空”
    # 可能触发 Streamlit 去解析 secrets 并抛 FileNotFoundError。
    # 这里显式兜底：没有 secrets 就当空，不影响本地使用环境变量或 Mock。
    try:
        sec = st.secrets
        try:
            sec_dict = sec.to_dict()  # type: ignore[attr-defined]
        except Exception:
            # 某些版本/实现不提供 to_dict，则尝试直接转 dict
            sec_dict = dict(sec)
    except Exception:
        return
    if not sec_dict:
        return
    keys = (
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_MODEL",
        "OPENROUTER_SITE_URL",
        "OPENROUTER_APP_NAME",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    )
    for k in keys:
        if k in sec_dict and str(sec_dict[k]).strip():
            if not (os.getenv(k) or "").strip():
                os.environ[k] = str(sec_dict[k]).strip()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _cached_platform_sources(
    *,
    queries: tuple[str, ...],
    max_results_per_query: int,
    enabled_platforms: tuple[str, ...],
    max_sites_per_platform: int,
    region: str,
) -> Dict[str, List[dict]]:
    agent = PlatformAgent(max_results_per_query=max_results_per_query, platform_sites=get_platform_sites(region))
    return agent.run(
        list(queries),
        enabled_platforms=list(enabled_platforms),
        max_sites_per_platform=max_sites_per_platform,
    )


SYNTH_PLATFORM_LABELS = ("DeepSeek-style", "ChatGPT-style", "Gemini-style", "Perplexity-style")


def _synthetic_system_prompt(name: str, lang: str) -> str:
    lang = (lang or "zh").strip().lower()
    if lang == "en":
        base = "You simulate answers as if from a specific AI product. Be concise, structured, and honest about uncertainty."
        if "Perplexity" in name:
            return base + " Lead with a direct answer, then bullets; prefer citing URLs present in the materials block."
        if "Gemini" in name:
            return base + " Prefer cautious wording; call out conflicting claims when materials disagree."
        if "ChatGPT" in name:
            return base + " Use clear headings and balanced pros/cons."
        if "DeepSeek" in name:
            return base + " Favor practical comparisons and price/value framing when materials allow."
        return base
    base = "你在模拟不同「AI 问答产品」的回答风格。简洁、结构化，不确定时要说明。"
    if "Perplexity" in name:
        return base + " 先给结论，再分要点；材料里出现的 URL 尽量在回答中引用。"
    if "Gemini" in name:
        return base + " 材料冲突时要指出分歧，不要武断下结论。"
    if "ChatGPT" in name:
        return base + " 用小标题、利弊对比，语气中立。"
    if "DeepSeek" in name:
        return base + " 偏实用与性价比/参数对比（在材料支持时）。"
    return base


def _run_synthetic_platforms_parallel(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model_default: str,
    site_url: str | None,
    app_name: str | None,
    model_overrides: Dict[str, str],
    enabled_labels: List[str],
    keyword: str,
    materials_md: str,
    lang: str,
) -> Dict[str, str]:
    """用同一兼容 API（通常为 OpenRouter）并行调用多个 model id，模拟不同平台风格。"""
    lang = (lang or "zh").strip().lower()
    label_set = set(enabled_labels)
    to_run = [n for n in SYNTH_PLATFORM_LABELS if n in label_set]
    if not to_run:
        return {}

    user_tail = (
        "Answer entirely in English."
        if lang == "en"
        else "全文使用中文。"
    )

    def job(name: str) -> Tuple[str, str]:
        mid = (model_overrides.get(name) or "").strip() or model_default
        cli = build_client(
            provider=provider.lower(),
            api_key=api_key,
            base_url=base_url,
            model=mid,
            site_url=site_url,
            app_name=app_name,
        )
        up = (
            f"Keyword / topic: {keyword}\n\n"
            f"Materials (may be incomplete):\n{materials_md}\n\n"
            f"Task: Act in the style implied by your system role. Give a short answer a shopper would want.\n"
            f"{user_tail}\n"
            f"End with a line 'Sources:' then bullet URLs copied from the materials when possible."
        )
        txt, _ = cli.chat(up, system_prompt=_synthetic_system_prompt(name, lang))
        return name, txt

    out: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(to_run)))) as ex:
        futs = {ex.submit(job, name): name for name in to_run}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                k, v = fut.result()
                out[k] = v
            except Exception as e:
                out[name] = f"(failed) {e}"
    return out


def main() -> None:
    # 允许本地放一个 .env（但仓库不提交）
    load_dotenv()

    st.set_page_config(page_title="GEO Multi-Agent MVP", layout="wide")
    _inject_streamlit_secrets_into_environ()
    st.title("GEO Multi-Agent MVP（Streamlit）")
    st.caption(
        "Agent A：拓展 Prompt 库 → Agent B：按地区站点聚合网页来源 → Agent C：并行模拟多平台风格回答 → "
        "Agent D：综合生成带引用的回答 → Agent E：GEO Playbook + 导出 PDF"
    )

    with st.sidebar:
        st.subheader("配置")
        report_language = st.selectbox(
            "报告语言",
            ("自动（随关键词）", "中文", "English"),
            index=0,
            help="英文关键词 + 自动 = 英文报告；含中文则输出中文。也可强制指定。",
        )
        region_ui = st.selectbox(
            "信源地区（DuckDuckGo site: 池）",
            ("国内（中文站群为主）", "海外（英文站群为主）", "混合（国内+海外）"),
            index=2,
        )
        region_key = {"国内（中文站群为主）": "cn", "海外（英文站群为主）": "global", "混合（国内+海外）": "mixed"}[
            region_ui
        ]
        st.caption(
            "说明：Perplexity / ChatGPT 网页等无法在本地替你“登录使用”；"
            "Agent C 是用 OpenAI 兼容 API（推荐 OpenRouter）多模型 + 人设模拟不同产品风格。"
        )

        provider = st.selectbox("LLM Provider", options=["OpenRouter", "DeepSeek"], index=0)

        # API Key：优先使用输入框，其次用环境变量
        st.caption("API Key 不会写入代码/仓库；仅用于当前会话。")
        ui_key = st.text_input("API Key（密码框）", value="", type="password", placeholder="粘贴你的 API Key（不要发到聊天里）")

        if provider == "OpenRouter":
            default_base = "https://openrouter.ai/api/v1"
            # 说明：免费模型会变动，默认值仅作占位；建议点“拉取模型列表”后下拉选择
            default_model = env_get("OPENROUTER_MODEL") or "（建议点下方按钮拉取模型列表）"
            base_url = st.text_input("Base URL", value=env_get("OPENROUTER_BASE_URL") or default_base)
            # 拉取可用模型列表（避免手填导致 404）
            if "openrouter_models" not in st.session_state:
                st.session_state.openrouter_models = []
            c1, c2 = st.columns([1, 1])
            with c1:
                load_models = st.button("拉取模型列表", use_container_width=True)
            with c2:
                st.caption("需要 OpenRouter Key")

            if load_models:
                if not ((ui_key or "").strip() or env_get("OPENROUTER_API_KEY")):
                    st.warning("请先在上方输入 OpenRouter API Key，再拉取模型列表。")
                else:
                    try:
                        st.session_state.openrouter_models = _fetch_openrouter_models(
                            api_key=((ui_key or "").strip() or env_get("OPENROUTER_API_KEY")),
                            base_url=base_url,
                        )
                        st.success(f"已加载 {len(st.session_state.openrouter_models)} 个模型。")
                    except Exception as e:
                        st.error(str(e))

            if st.session_state.openrouter_models:
                model = st.selectbox("Model（从列表选择）", options=st.session_state.openrouter_models)
            else:
                model = st.text_input("Model（手动输入）", value=default_model)

            site_url = st.text_input("Site URL（可选）", value=env_get("OPENROUTER_SITE_URL"))
            st.caption(
                "仅用于 OpenRouter 的 HTTP-Referer（可选）。**不是**「要分析的材料网址」；"
                "网页材料来自下方 DuckDuckGo 搜索聚合。"
            )
            app_name = st.text_input("App Name（可选）", value=env_get("OPENROUTER_APP_NAME") or "geo-search-agent")
            env_key = env_get("OPENROUTER_API_KEY")
        else:
            default_base = "https://api.deepseek.com/v1"
            default_model = "deepseek-chat"
            base_url = st.text_input("Base URL", value=env_get("DEEPSEEK_BASE_URL") or default_base)
            model = st.text_input("Model", value=env_get("DEEPSEEK_MODEL") or default_model)
            site_url = ""
            app_name = "geo-search-agent"
            env_key = env_get("DEEPSEEK_API_KEY")

        api_key = (ui_key or "").strip() or env_key
        has_key = bool(api_key)
        st.write("API Key 状态：", "✅ 已检测到" if has_key else "❌ 未检测到（请在上方输入 Key，或设置环境变量）")

        st.divider()
        st.subheader("各 Agent 模型（可选）")
        st.caption(
            "以下留空表示 **与上方「主 Model」一致**（同一 Provider / 同一 API Key / 同一 Base URL）。"
            "仅填写 **model id** 即可切换各 Agent 使用的模型。"
        )
        agent_a_model_ov = st.text_input(
            "Agent A（Prompt 拓展 / LLM 拓库）",
            value="",
            placeholder="留空=主模型",
            key="agent_a_model_ov",
        )
        st.caption("Agent B：仅 DuckDuckGo 聚合，**不涉及 LLM**，无需选择模型。")
        agent_c_or_model_ov = st.text_input(
            "Agent C（OpenRouter 多风格 · 默认 model）",
            value="",
            placeholder="留空=主模型；各风格仍可单独覆盖",
            key="agent_c_or_model_ov",
        )
        agent_c_pplx_model = st.selectbox(
            "Agent C（Perplexity 官方 API · model）",
            options=("sonar", "sonar-pro", "sonar-deep-research", "sonar-reasoning-pro"),
            index=1,
            key="agent_c_pplx_model",
        )
        agent_d_model_ov = st.text_input(
            "Agent D（主回答）",
            value="",
            placeholder="留空=主模型",
            key="agent_d_model_ov",
        )
        agent_e_model_ov = st.text_input(
            "Agent E（GEO Playbook）",
            value="",
            placeholder="留空=主模型",
            key="agent_e_model_ov",
        )

        st.divider()
        st.subheader("Agents / 模型一览")
        st.caption("开启下方选项后，LLM 调用次数会增加；Agent B 始终是 DuckDuckGo（不走大模型）。")
        with st.expander("查看各 Agent 使用的模型/服务", expanded=False):
            st.markdown(
                "\n".join(
                    [
                        f"- **Agent A**：LLM 拓展 Prompt 库 → 有效 Model=`{_eff_model(agent_a_model_ov, str(model))}`",
                        f"- **Agent B**：DuckDuckGo，`region={region_key}`（无 LLM）",
                        f"- **Agent C**：OpenRouter 默认=`{_eff_model(agent_c_or_model_ov, str(model))}`；Perplexity=`{agent_c_pplx_model}`",
                        f"- **Agent D**：主回答 → 有效 Model=`{_eff_model(agent_d_model_ov, str(model))}`",
                        f"- **Agent E**：Playbook → 有效 Model=`{_eff_model(agent_e_model_ov, str(model))}`",
                        "- **InsightAnalyzer**：结构化统计 + 可选 LLM 要点（见下方独立配置）",
                    ]
                )
            )

        st.divider()
        st.subheader("参数")
        max_queries = st.slider("生成查询数", min_value=3, max_value=12, value=8, step=1)
        max_results = st.slider("每条查询抓取结果数", min_value=2, max_value=10, value=5, step=1)
        max_sites_per_platform = st.slider("每个平台 site: 数（越小越快）", min_value=0, max_value=3, value=1, step=1)

        _plat_choices, _plat_default = platform_choices_for_region(region_key)
        tmp_platform_agent = PlatformAgent(platform_sites=get_platform_sites(region_key))
        platform_options = [
            p for p, sites in tmp_platform_agent.PLATFORM_SITES.items() if p != "通用网页" and bool(sites)
        ]
        enabled_platforms = st.multiselect(
            "启用的平台来源（越少越快）",
            options=platform_options,
            default=[p for p in _plat_default if p in platform_options],
        )
        st.caption("建议提速：先只勾选 1-3 个平台，并把“每条查询抓取结果数/每个平台 site 数”调低。")
        force_mock = st.toggle("无模型/Mock 模式（不调用 LLM）", value=False)

        st.divider()
        st.subheader("工作流选项")
        use_prompt_lib_llm = st.toggle("Agent A：用 LLM 拓展 Prompt 库（消费者视角）", value=False)
        synth_enabled = st.toggle("Agent C：并行模拟多平台风格回答", value=False)
        synth_mode = "openrouter"
        perplexity_cfg = {"enabled": False}
        synth_platforms: List[str] = []
        synth_model_overrides: Dict[str, str] = {}
        if synth_enabled:
            synth_mode = st.radio(
                "Agent C 模式",
                options=("OpenRouter 多模型 + 人设模拟（默认）", "Perplexity 官方 API（真实 citations，可选）"),
                index=0,
                horizontal=False,
            )
            synth_mode = "perplexity" if synth_mode.startswith("Perplexity") else "openrouter"
            if synth_mode == "perplexity":
                with st.expander("Perplexity 官方 API 配置", expanded=True):
                    p_key = st.text_input("PERPLEXITY_API_KEY（密码框）", value="", type="password")
                    st.caption(f"Model 使用侧栏 **Agent C（Perplexity）**：`{agent_c_pplx_model}`")
                    p_search_mode = st.selectbox("Search mode", options=("web", "academic", "sec"), index=0)
                    p_max_q = st.slider("Perplexity 采样问题数（从 Prompt 库取前 N 条）", min_value=1, max_value=12, value=6, step=1)
                    p_workers = st.slider("Perplexity 并发数（过高可能 429）", min_value=1, max_value=6, value=3, step=1)
                    perplexity_cfg = {
                        "enabled": bool((p_key or "").strip()),
                        "api_key": (p_key or "").strip(),
                        "model": agent_c_pplx_model,
                        "search_mode": p_search_mode,
                        "max_q": p_max_q,
                        "workers": p_workers,
                    }
                    if not perplexity_cfg["enabled"]:
                        st.warning("未提供 PERPLEXITY_API_KEY：将回退到默认 OpenRouter 模式。")
            synth_platforms = list(
                st.multiselect(
                    "选择要模拟的风格标签",
                    list(SYNTH_PLATFORM_LABELS),
                    default=list(SYNTH_PLATFORM_LABELS),
                )
            )
            with st.expander("各风格可选 Model ID（留空=Agent C 默认 / 主 Model）", expanded=False):
                for lab in SYNTH_PLATFORM_LABELS:
                    synth_model_overrides[lab] = st.text_input(lab, value="", key=f"synth_ov_{lab}")

        st.divider()
        st.subheader("按 Agent 分开用模型（可选）")
        st.caption("开启后：QueryAgent/InsightAnalyzer 也会各自调用一次 LLM；可分别选择 Provider/Key/Model。")

        qa_use_llm = st.toggle("QueryAgent：用 LLM 生成查询（覆盖规则模板）", value=False)
        ia_use_llm = st.toggle("InsightAnalyzer：用 LLM 生成洞察要点（更像报告）", value=False)

        def _agent_llm_config(*, label: str):
            with st.expander(f"{label}：LLM 配置", expanded=False):
                use_same = st.checkbox(f"{label} 复用 Answer 的配置", value=True, key=f"{label}_same")
                if use_same:
                    return {
                        "enabled": True,
                        "provider": provider,
                        "api_key": api_key,
                        "base_url": base_url,
                        "model": model,
                        "site_url": site_url,
                        "app_name": app_name,
                        "has_key": has_key,
                    }
                p2 = st.selectbox(f"{label} Provider", options=["OpenRouter", "DeepSeek"], index=0, key=f"{label}_prov")
                k2 = st.text_input(f"{label} API Key（密码框）", value="", type="password", key=f"{label}_key")
                if p2 == "OpenRouter":
                    b2 = st.text_input(f"{label} Base URL", value="https://openrouter.ai/api/v1", key=f"{label}_base")
                    m2 = st.text_input(f"{label} Model", value=env_get("OPENROUTER_MODEL") or "", key=f"{label}_model")
                    s2 = st.text_input(f"{label} Site URL（可选）", value="", key=f"{label}_site")
                    a2 = st.text_input(f"{label} App Name（可选）", value="geo-search-agent", key=f"{label}_app")
                    envk = env_get("OPENROUTER_API_KEY")
                else:
                    b2 = st.text_input(f"{label} Base URL", value="https://api.deepseek.com/v1", key=f"{label}_base")
                    m2 = st.text_input(f"{label} Model", value="deepseek-chat", key=f"{label}_model")
                    s2 = ""
                    a2 = "geo-search-agent"
                    envk = env_get("DEEPSEEK_API_KEY")

                ak = (k2 or "").strip() or envk
                hk = bool(ak)
                st.write(f"{label} Key 状态：", "✅ 已检测到" if hk else "❌ 未检测到")
                return {
                    "enabled": True,
                    "provider": p2,
                    "api_key": ak,
                    "base_url": b2,
                    "model": m2,
                    "site_url": s2,
                    "app_name": a2,
                    "has_key": hk,
                }

        qa_cfg = _agent_llm_config(label="QueryAgent") if qa_use_llm else {"enabled": False}
        ia_cfg = _agent_llm_config(label="InsightAnalyzer") if ia_use_llm else {"enabled": False}

    col1, col2 = st.columns([2, 1])
    with col1:
        keyword = st.text_input("输入关键词", value="")
    with col2:
        samples = [""] + load_keywords_csv()
        sample_kw = st.selectbox("示例关键词（可选）", options=samples, index=0)
        if sample_kw:
            keyword = sample_kw

    run = st.button("Run Analysis", type="primary", use_container_width=True)

    if not run:
        if has_key:
            st.info("提示：已检测到 API Key，可以直接点击 Run Analysis 开始分析。")
        else:
            st.info("提示：先在左侧输入 API Key（推荐），或用环境变量设置 Key，然后点击 Run Analysis。")
        return

    keyword = (keyword or "").strip()
    if not keyword:
        st.error("请输入关键词。")
        return

    if not force_mock and not has_key:
        st.error("未检测到 API Key：请在侧边栏输入 Key（推荐），或设置环境变量。")
        return

    report_lang = _resolve_report_lang(report_language, keyword)
    source_region_label = get_region_label(region_key)
    st.info(f"本次报告语言：**{report_lang}** ｜ 信源地区：**{source_region_label}**（`{region_key}`）")

    client = None
    if not force_mock:
        try:
            client = build_client(
                provider=provider.lower(),
                api_key=api_key,
                base_url=base_url,
                model=model,
                site_url=site_url or None,
                app_name=app_name or None,
            )
        except Exception as e:
            st.error(f"LLM 配置错误：{e}")
            return

    def _make_eff_client(model_override: str):
        """同一 Provider/Key/Base URL，仅替换 model id（与主 Model 一致）。"""
        return build_client(
            provider=provider.lower(),
            api_key=api_key,
            base_url=base_url,
            model=_eff_model(model_override, model),
            site_url=site_url or None,
            app_name=app_name or None,
        )

    query_agent = QueryAgent(max_queries=max_queries)
    platform_agent = PlatformAgent(max_results_per_query=max_results, platform_sites=get_platform_sites(region_key))
    analyzer = InsightAnalyzer()

    with st.status("Agent A：生成 / 拓展 Prompt 库…", expanded=False) as s1:
        queries = query_agent.run(keyword, lang=report_lang)
        if qa_use_llm and qa_cfg.get("enabled"):
            if not qa_cfg.get("has_key"):
                st.warning("QueryAgent 开启了 LLM，但未检测到该 Agent 的 API Key：已回退到规则生成。")
            else:
                try:
                    qa_client = build_client(
                        provider=str(qa_cfg["provider"]).lower(),
                        api_key=str(qa_cfg["api_key"]),
                        base_url=str(qa_cfg["base_url"]),
                        model=str(qa_cfg["model"]),
                        site_url=(qa_cfg.get("site_url") or None),
                        app_name=(qa_cfg.get("app_name") or None),
                    )
                    if report_lang == "en":
                        q_prompt = (
                            "You simulate realistic user queries for AI search/chat.\n"
                            f"Keyword: {keyword}\n\n"
                            f"Output {max_queries} short lines, one query per line. Cover compare/recommend/price/pros-cons/pitfalls/who is it for/alternatives.\n"
                            "No explanation—list only.\n"
                        )
                        q_sys = "You output concise English search queries only."
                    else:
                        q_prompt = (
                            "你是一个搜索意图模拟器。请针对给定关键词，生成用户可能会在 AI 搜索/问答里输入的查询句。\n"
                            f"关键词：{keyword}\n\n"
                            f"要求：\n- 生成 {max_queries} 条\n- 每条尽量短、像真人\n"
                            "- 覆盖：对比/推荐/价格/优缺点/避坑/适用人群/替代品\n"
                            "- 只输出列表，每行一条，不要解释。\n"
                        )
                        q_sys = "你是一个中文搜索意图模拟器，只输出列表。"
                    q_text, _ = qa_client.chat(q_prompt, system_prompt=q_sys)
                    q_llm = _parse_queries_from_text(q_text, max_queries=max_queries)
                    if q_llm:
                        queries = q_llm
                        st.caption("✅ QueryAgent 已使用 LLM 生成查询。")
                except Exception as e:
                    st.warning(f"QueryAgent LLM 生成失败（已回退规则）：{e}")

        prompt_library: List[str] = list(queries)
        if use_prompt_lib_llm:
            if force_mock or client is None:
                st.warning("Agent A 的「LLM 拓展 Prompt 库」需要可用 API：当前为 Mock 或未初始化 Client，已跳过拓展。")
            else:
                try:
                    if report_lang == "en":
                        ex_user = (
                            f"Keyword: {keyword}\n"
                            f"Write {max_queries} consumer-style prompts people would paste into ChatGPT/Perplexity/"
                            f"Gemini to research this topic. One per line, English only, no numbering prologue.\n"
                        )
                        ex_sys = "You expand prompts; output English lines only."
                    else:
                        ex_user = (
                            f"关键词：{keyword}\n"
                            f"请输出 {max_queries} 条「消费者会怎样向 AI 提问」的短句，每行一条；覆盖对比、避坑、价格、适用人群等。\n"
                            "不要解释。\n"
                        )
                        ex_sys = "你只输出中文列表。"
                    ex_text, _ = _make_eff_client(agent_a_model_ov).chat(ex_user, system_prompt=ex_sys)
                    ex_list = _parse_queries_from_text(ex_text, max_queries=max_queries)
                    if ex_list:
                        prompt_library = ex_list
                        st.caption("✅ Agent A：已用 LLM 拓展 Prompt 库。")
                except Exception as e:
                    st.warning(f"Agent A LLM 拓展失败（保留上一步查询）：{e}")

        st.write(prompt_library)
        s1.update(label="Prompt 库就绪", state="complete")

    with st.status("Agent B：按地区站点聚合网页来源（DuckDuckGo）…", expanded=False) as s2:
        try:
            platform_sources = _cached_platform_sources(
                queries=tuple(prompt_library),
                max_results_per_query=max_results,
                enabled_platforms=tuple(enabled_platforms),
                max_sites_per_platform=max_sites_per_platform,
                region=region_key,
            )
            total = sum(len(v) for v in platform_sources.values())
            st.write(f"共聚合到 {total} 条来源。")
            if total == 0:
                st.warning(
                    "当前**未聚合到任何网页来源**（常见原因：DuckDuckGo 限流/网络不稳定/地区访问受限）。"
                    "因此发给模型的「材料」段落为空，AI 可能会写「未提供具体材料 URL」——**这不是让你填侧边栏 Site URL**，"
                    "而是指没有可用的搜索链接可引用。可稍后重试、换网络或降低「每条查询抓取结果数」再试。"
                )
            s2.update(label="来源聚合完成", state="complete")
        except Exception as e:
            platform_sources = {k: [] for k in platform_agent.PLATFORM_SITES.keys()}
            st.warning(f"来源聚合失败（将降级继续）：{e}")
            s2.update(label="来源聚合失败（已降级）", state="error")

    materials_md = format_sources(platform_sources, max_each=6)
    synthetic: Dict[str, str] = {}
    synthetic_citations: Dict[str, List[str]] = {}
    if synth_enabled and (not force_mock):
        if synth_mode == "perplexity" and perplexity_cfg.get("enabled"):
            with st.status("Agent C：Perplexity 官方 API（含真实 citations）…", expanded=False) as sc:
                try:
                    pc = PerplexityClient(
                        api_key=str(perplexity_cfg["api_key"]),
                        model=str(perplexity_cfg.get("model") or "sonar-pro"),
                    )
                    sys_prompt = _synthetic_system_prompt("Perplexity-style", report_lang)
                    lang_pref = "en" if report_lang == "en" else "zh"
                    search_mode = str(perplexity_cfg.get("search_mode") or "web")

                    max_q = int(perplexity_cfg.get("max_q") or 6)
                    workers = int(perplexity_cfg.get("workers") or 3)
                    selected = list(prompt_library)[: max(1, min(len(prompt_library), max_q))]

                    def p_job(q: str) -> Tuple[str, str, List[str]]:
                        if report_lang == "en":
                            user_prompt = (
                                f"User question: {q}\n\n"
                                "Return a concise, shopper-friendly answer grounded in web search.\n"
                                "End with a Sources: URL list.\n"
                            )
                        else:
                            user_prompt = (
                                f"用户问题：{q}\n\n"
                                "请基于 web 搜索给出简洁、面向消费者的回答。\n"
                                "末尾用 Sources: 列出 URL。\n"
                            )
                        txt, cits = pc.chat(
                            user_prompt=user_prompt,
                            system_prompt=sys_prompt,
                            language_preference=lang_pref,
                            search_mode=search_mode,
                        )
                        return q, txt, cits

                    with ThreadPoolExecutor(max_workers=min(max(1, workers), 6)) as ex:
                        futs = {ex.submit(p_job, q): q for q in selected}
                        for fut in as_completed(futs):
                            q = futs[fut]
                            try:
                                q2, txt, cits = fut.result()
                                key = f"Perplexity (official) · {q2}"
                                synthetic[key] = txt
                                synthetic_citations[key] = cits
                            except Exception as e:
                                key = f"Perplexity (official) · {q}"
                                synthetic[key] = f"(failed) {e}"
                                synthetic_citations[key] = []

                    sc.update(label="Perplexity 完成", state="complete")
                except Exception as e:
                    st.warning(f"Perplexity 调用失败（已回退默认模式）：{e}")
                    sc.update(label="Perplexity 失败（回退）", state="error")
                    synth_mode = "openrouter"

        if synth_mode == "openrouter" and synth_platforms and client is not None:
            with st.status("Agent C：OpenRouter 多模型 + 人设并行模拟…", expanded=False) as sc:
                try:
                    synthetic = _run_synthetic_platforms_parallel(
                        provider=provider,
                        api_key=api_key,
                        base_url=base_url,
                        model_default=_eff_model(agent_c_or_model_ov, model),
                        site_url=site_url or None,
                        app_name=app_name or None,
                        model_overrides=synth_model_overrides,
                        enabled_labels=synth_platforms,
                        keyword=keyword,
                        materials_md=materials_md,
                        lang=report_lang,
                    )
                    sc.update(label="多平台模拟完成", state="complete")
                except Exception as e:
                    st.warning(f"Agent C 失败：{e}")
                    sc.update(label="多平台模拟失败", state="error")
        elif synth_mode == "openrouter" and client is None:
            st.warning("Agent C（OpenRouter）需要可用主 LLM：当前未初始化 Client，已跳过。")
    elif synth_enabled and force_mock:
        st.warning("Agent C 需要可用 LLM：当前为 Mock，已跳过。")

    synth_block = "\n".join([f"### {k}\n{(v or '')[:2200]}" for k, v in synthetic.items()])

    prompt = build_prompt(
        keyword,
        prompt_library,
        platform_sources,
        lang=report_lang,
        source_region_label=source_region_label,
        synthetic_block=synth_block,
    )
    answer_sys = (
        "You are a GEO analyst. Always end with a Sources: URL list grounded in provided materials."
        if report_lang == "en"
        else "你是 GEO 分析助手，必须在材料范围内引用 URL，并以 Sources: 列表结尾。"
    )

    cited_urls = []
    llm_error: Exception | None = None
    if force_mock:
        answer = build_mock_answer(keyword, prompt_library, platform_sources, lang=report_lang)
        st.info("当前使用 Mock 模式：不会消耗额度。")
    else:
        with st.status("Agent D：主模型生成回答（带引用）…", expanded=False) as s3:
            try:
                assert client is not None
                answer, cited_urls = _make_eff_client(agent_d_model_ov).chat(prompt, system_prompt=answer_sys)
                s3.update(label="主回答生成完成", state="complete")
            except Exception as e:
                llm_error = e
                msg = str(e)
                if "LLM_HTTP_402" in msg and ("Insufficient Balance" in msg or "insufficient" in msg.lower()):
                    st.error("模型返回 402：余额不足（Insufficient Balance）。请到对应平台充值/开通额度，或开启侧边栏的“无模型/Mock 模式”。")
                    answer = build_mock_answer(keyword, prompt_library, platform_sources, lang=report_lang)
                    s3.update(label="模型调用失败（余额不足，已降级为 Mock）", state="error")
                else:
                    st.error(f"模型调用失败：{e}")
                    answer = build_mock_answer(keyword, prompt_library, platform_sources, lang=report_lang)
                    s3.update(label="模型调用失败（已降级为 Mock）", state="error")

    if synthetic:
        with st.expander("Agent C：多平台风格模拟（原文摘录）", expanded=False):
            for k, v in synthetic.items():
                st.markdown(f"#### {k}")
                st.markdown(v or "(empty)")
                cits = (synthetic_citations or {}).get(k) or []
                if cits:
                    st.caption("Citations（官方返回）")
                    st.write(cits)

    st.subheader("Agent D：AI 回答（含引用）")
    st.markdown(answer)

    if llm_error is not None:
        with st.expander("查看模型调用失败原因（不含 Key）", expanded=True):
            st.write(_explain_llm_error(llm_error))
            with st.container():
                st.caption("服务端返回摘要（便于排查，可能含英文错误码）：")
                st.code(str(llm_error)[:1200], language="text")
            st.caption(
                "建议：① OpenRouter 控制台确认 Key 有效、有余额；② 免费模型常出现 429/排队，可换模型或稍后重试；"
                "③ 侧边栏点「拉取模型列表」从下拉框选模型，避免手填错 id；④ 若在国内网络，确认能直连 OpenRouter。"
            )

    with st.expander("查看聚合来源（原材料）", expanded=False):
        for platform, items in platform_sources.items():
            if not items:
                continue
            st.markdown(f"#### {platform}（{len(items)}）")
            for it in items[:20]:
                st.markdown(f"- [{it.get('title')}]({it.get('url')})")
                snippet = (it.get("snippet") or "").strip()
                if snippet:
                    st.caption(snippet)

    report = analyzer.run(
        keyword=keyword,
        queries=prompt_library,
        answer=answer,
        platform_sources=platform_sources,
        cited_sources=cited_urls,
        lang=report_lang,
        prompt_library=prompt_library,
        synthetic_answers=synthetic,
        synthetic_citations=synthetic_citations,
        source_region=source_region_label,
        geo_playbook="",
    )

    playbook_md = ""
    if not force_mock and client is not None:
        with st.status("Agent E：生成 GEO Playbook（拓展 prompt / 竞品 / 关键词建议）…", expanded=False) as spb:
            try:
                if report_lang == "en":
                    pb_user = (
                        f"Keyword: {keyword}\n"
                        f"Source region profile: {source_region_label}\n\n"
                        "Produce markdown with these sections:\n"
                        "## Extended consumer prompts (10 bullets)\n"
                        "## Competitor & brand signals (from evidence)\n"
                        "## Landing pages & citation targets to monitor\n"
                        "## Keyword clusters & on-page angles for GEO\n\n"
                        f"Draft answer (excerpt):\n{(answer or '')[:3500]}\n\n"
                        f"Top domains from crawl: {', '.join(report.top_domains[:12])}\n"
                    )
                    pb_sys = "You are a GEO strategist. Be specific; if evidence is thin, say what data is missing."
                else:
                    pb_user = (
                        f"关键词：{keyword}\n信源地区：{source_region_label}\n\n"
                        "请用 Markdown 输出以下章节：\n"
                        "## 拓展后的消费者 Prompt（10 条）\n"
                        "## 竞品与品牌信号（需基于上文证据，避免臆造）\n"
                        "## 值得跟进的落地页与可引用来源\n"
                        "## 关键词簇与内容角度（面向 GEO）\n\n"
                        f"主回答节选：\n{(answer or '')[:3500]}\n\n"
                        f"抓取 Top 域名：{', '.join(report.top_domains[:12])}\n"
                    )
                    pb_sys = "你是 GEO 策略顾问；缺证据时要明确写「材料不足」，不要编造数据。"
                playbook_md, _ = _make_eff_client(agent_e_model_ov).chat(pb_user, system_prompt=pb_sys)
                spb.update(label="Playbook 已生成", state="complete")
            except Exception as e:
                st.warning(f"Agent E Playbook 生成失败：{e}")
                spb.update(label="Playbook 失败", state="error")

    report = replace(report, geo_playbook=playbook_md or "")

    st.subheader("结构化统计 + 洞察（InsightAnalyzer）")
    takeaways = list(report.takeaways or [])
    if ia_use_llm and ia_cfg.get("enabled"):
        if not ia_cfg.get("has_key"):
            st.warning("InsightAnalyzer 开启了 LLM，但未检测到该 Agent 的 API Key：已使用统计版洞察。")
        else:
            try:
                ia_client = build_client(
                    provider=str(ia_cfg["provider"]).lower(),
                    api_key=str(ia_cfg["api_key"]),
                    base_url=str(ia_cfg["base_url"]),
                    model=str(ia_cfg["model"]),
                    site_url=(ia_cfg.get("site_url") or None),
                    app_name=(ia_cfg.get("app_name") or None),
                )
                if report_lang == "en":
                    ia_prompt = (
                        "You are a GEO analyst. Output 5-8 insight bullets (one per line), no fluff.\n\n"
                        f"Keyword: {keyword}\n\n"
                        "Draft answer (excerpt):\n"
                        f"{(answer or '')[:2000]}\n\n"
                        "Top domains:\n"
                        f"{', '.join(getattr(report, 'top_domains', [])[:10])}\n\n"
                        "Platform domain breakdown (JSON fragment):\n"
                        f"{json.dumps(getattr(report, 'platform_domain_breakdown', {}), ensure_ascii=False)[:2000]}\n"
                    )
                    ia_sys = "You write concise English bullets only."
                else:
                    ia_prompt = (
                        "你是 GEO（Generative Engine Optimization）洞察分析师。基于输入信息，输出 5-8 条要点洞察。\n"
                        "只输出列表，每行一条，不要解释。\n\n"
                        f"关键词：{keyword}\n\n"
                        "模型回答摘要（可引用）：\n"
                        f"{(answer or '')[:2000]}\n\n"
                        "Top Domains（搜索材料出现频率最高）：\n"
                        f"{', '.join(getattr(report, 'top_domains', [])[:10])}\n\n"
                        "平台域名分布（Top）：\n"
                        f"{json.dumps(getattr(report, 'platform_domain_breakdown', {}), ensure_ascii=False)[:2000]}\n"
                    )
                    ia_sys = "你只输出中文要点列表。"
                ia_text, _ = ia_client.chat(ia_prompt, system_prompt=ia_sys)
                ia_takeaways = _parse_takeaways_from_text(ia_text, max_items=8)
                if ia_takeaways:
                    takeaways = ia_takeaways
                    st.caption("✅ InsightAnalyzer 已使用 LLM 生成洞察要点。")
            except Exception as e:
                st.warning(f"InsightAnalyzer LLM 生成失败（已回退统计）：{e}")

    report = replace(report, takeaways=takeaways)

    st.write(takeaways)

    if playbook_md:
        st.subheader("Agent E：GEO Playbook")
        st.markdown(playbook_md)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top Domains")
        st.write(report.top_domains[:10])
    with c2:
        st.markdown("#### 平台域名分布（Top）")
        st.json(report.platform_domain_breakdown)

    st.markdown("#### JSON 导出")
    st.download_button(
        label="下载 GEO 报告 JSON",
        data=json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        file_name=f"geo_report_{keyword}.json".replace(" ", "_"),
        mime="application/json",
        use_container_width=True,
    )

    st.markdown("#### PDF 导出")
    try:
        pdf_bytes = build_pdf_bytes(
            keyword=keyword,
            answer_md=answer,
            report=report,
            source_region_label=source_region_label,
            prompt_library=prompt_library,
            synthetic_answers=synthetic,
            synthetic_citations=synthetic_citations,
            geo_playbook=playbook_md or report.geo_playbook,
        )
        st.download_button(
            label="下载 GEO 报告 PDF",
            data=pdf_bytes,
            file_name=f"geo_report_{keyword}.pdf".replace(" ", "_"),
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        msg = str(e)
        if "No module named 'reportlab'" in msg:
            st.warning("PDF 生成失败：缺少依赖 `reportlab`。")
            st.caption(f"当前 Python：`{sys.executable}`")
            st.caption("请在该解释器下安装：`python -m pip install reportlab`，然后重启应用。")
        else:
            st.warning(f"PDF 生成失败：{e}")


if __name__ == "__main__":
    main()

