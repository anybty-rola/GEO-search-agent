from __future__ import annotations

import json
import os
import sys
import datetime as _dt
from pathlib import Path
from typing import Dict, List

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


def build_prompt(keyword: str, queries: List[str], platform_sources: Dict[str, List[dict]]) -> str:
    sources_md = format_sources(platform_sources)
    q_md = "\n".join([f"- {q}" for q in queries])
    return f"""关键词：{keyword}

用户可能会这样搜/问：
{q_md}

下面是从网页搜索聚合到的材料（标题/URL/摘要）。你需要基于这些材料写一个面向用户的推荐/对比回答，并在最后输出Sources（URL列表）。

材料：
{sources_md}

输出要求：
1) 用中文输出，结构清晰（小标题 + 要点）
2) 明确提到常见品牌/选购要点（如果材料支持）
3) 末尾必须有：
Sources:
- https://...
- https://...
"""

def build_mock_answer(keyword: str, queries: List[str], platform_sources: Dict[str, List[dict]]) -> str:
    # 不调用模型的降级方案：把来源摘要整理成“可读”的回答骨架
    total = sum(len(v) for v in platform_sources.values())
    top_sources = []
    for platform, items in platform_sources.items():
        for it in items[:3]:
            if it.get("url"):
                top_sources.append(it["url"])
    top_sources = list(dict.fromkeys(top_sources))[:8]

    q_md = "\n".join([f"- {q}" for q in queries[:8]])
    s_md = "\n".join([f"- {u}" for u in top_sources]) if top_sources else "- （暂无，可稍后重试）"
    return f"""## （Mock）关于「{keyword}」的初步结论

当前为**无模型/Mock 模式**（未调用 LLM），因此以下内容为基于“查询意图 + 来源列表”的结构化草案。

### 你可以这样搜/这样问
{q_md}

### 下一步怎么做（建议）
- 先看 “查看聚合来源（原材料）” 里出现频率最高的域名与内容类型（评测/电商/论坛/官方）
- 针对 1-2 个子问题（比如价格、对比、优缺点）重新点击 Run Analysis，以减少限流与噪声

### Sources（聚合到 {total} 条）
{s_md}
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


def build_pdf_bytes(*, keyword: str, answer_md: str, report: object) -> bytes:
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
    c.drawString(margin_x, y, f"关键词：{keyword}")
    y -= 16
    c.drawString(margin_x, y, f"生成时间：{ts}")
    y -= 22

    # report 是 GeoReport dataclass，带 to_dict()
    try:
        rdict = report.to_dict()  # type: ignore[attr-defined]
    except Exception:
        rdict = {}

    takeaways = rdict.get("takeaways") or []
    cited_sources = rdict.get("cited_sources") or []
    top_domains = rdict.get("top_domains") or []

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


def _inject_streamlit_secrets_into_environ() -> None:
    """
    Streamlit Community Cloud：在 App Settings → Secrets 中配置的键，同步到 os.environ，
    便于与本地 .env / 侧边栏逻辑共用 env_get。
    """
    try:
        sec = st.secrets
    except Exception:
        return
    if not sec:
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
        if k in sec and str(sec[k]).strip():
            if not (os.getenv(k) or "").strip():
                os.environ[k] = str(sec[k]).strip()


def main() -> None:
    # 允许本地放一个 .env（但仓库不提交）
    load_dotenv()

    st.set_page_config(page_title="GEO Multi-Agent MVP", layout="wide")
    _inject_streamlit_secrets_into_environ()
    st.title("GEO Multi-Agent MVP（Streamlit）")
    st.caption("输入关键词 → 模拟用户查询 → 聚合来源 → LLM 生成回答并引用 → 输出 GEO 洞察报告")

    with st.sidebar:
        st.subheader("配置")
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
        st.subheader("参数")
        max_queries = st.slider("生成查询数", min_value=3, max_value=12, value=8, step=1)
        max_results = st.slider("每条查询抓取结果数", min_value=2, max_value=10, value=5, step=1)
        force_mock = st.toggle("无模型/Mock 模式（不调用 DeepSeek）", value=False)

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

    query_agent = QueryAgent(max_queries=max_queries)
    platform_agent = PlatformAgent(max_results_per_query=max_results)
    analyzer = InsightAnalyzer()

    with st.status("正在生成用户查询…", expanded=False) as s1:
        queries = query_agent.run(keyword)
        st.write(queries)
        s1.update(label="用户查询已生成", state="complete")

    with st.status("正在聚合搜索来源（DuckDuckGo）…", expanded=False) as s2:
        try:
            platform_sources = platform_agent.run(queries)
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

    prompt = build_prompt(keyword, queries, platform_sources)

    cited_urls = []
    llm_error: Exception | None = None
    if force_mock:
        answer = build_mock_answer(keyword, queries, platform_sources)
        st.info("当前使用无模型/Mock 模式：不会消耗额度，也不会调用 DeepSeek。")
    else:
        with st.status("正在调用模型生成回答（带引用）…", expanded=False) as s3:
            try:
                assert client is not None
                answer, cited_urls = client.chat(prompt)
                s3.update(label="模型回答生成完成", state="complete")
            except Exception as e:
                llm_error = e
                msg = str(e)
                if "LLM_HTTP_402" in msg and ("Insufficient Balance" in msg or "insufficient" in msg.lower()):
                    st.error("模型返回 402：余额不足（Insufficient Balance）。请到对应平台充值/开通额度，或开启侧边栏的“无模型/Mock 模式”。")
                    answer = build_mock_answer(keyword, queries, platform_sources)
                    s3.update(label="模型调用失败（余额不足，已降级为 Mock）", state="error")
                else:
                    st.error(f"模型调用失败：{e}")
                    answer = build_mock_answer(keyword, queries, platform_sources)
                    s3.update(label="模型调用失败（已降级为 Mock）", state="error")

    st.subheader("AI 回答（含引用）")
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
        queries=queries,
        answer=answer,
        platform_sources=platform_sources,
        cited_sources=cited_urls,
    )

    st.subheader("GEO 洞察报告（结构化）")
    st.write(report.takeaways)

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
        pdf_bytes = build_pdf_bytes(keyword=keyword, answer_md=answer, report=report)
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

