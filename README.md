# GEO Multi-Agent MVP（Streamlit）

**GEO Multi-Agent** is an experimental AI agent that analyzes how AI search engines recommend brands and products. With tools like ChatGPT and Perplexity, traditional SEO is evolving into **GEO (Generative Engine Optimization)**. This project simulates user queries, aggregates web sources, and surfaces brand/citation insights.

---

这是一个最小可运行的 **GEO Multi-Agent** 演示站点：输入关键词 -> 模拟用户搜索意图 -> 调用大模型生成回答 -> 抽取引用来源 -> 输出结构化 GEO 洞察报告。

## 目录结构

```
geo-search-agent
├── app.py                 # Streamlit 网站入口
├── agents
│   ├── __init__.py
│   ├── query_agent.py     # 生成“用户会怎么搜/问”
│   └── platform_agent.py  # 生成“各平台视角/来源”
├── analyzer
│   ├── __init__.py
│   └── insight_analyzer.py # 汇总为结构化 GEO 报告
├── data
│   └── keywords.csv       # 示例关键词（可选）
├── requirements.txt      # 运行依赖
├── requirements-dev.txt # 开发依赖（ruff、basedpyright）
├── env.example
├── DEPLOY.md              # GitHub 同步 + 公网部署（Streamlit Cloud / Docker / Vercel 落地页）
└── vercel-landing/        # 可选：静态页，部署在 Vercel 跳转到 Streamlit 公网地址
```

## 部署到公网（GitHub + 对外访问）

完整步骤见 **[DEPLOY.md](./DEPLOY.md)**。简要说明：

- **Streamlit 应用**适合用 **[Streamlit Community Cloud](https://streamlit.io/cloud)** 连接 GitHub，获得 `https://xxx.streamlit.app`。
- **Vercel** 不适合直接跑 Streamlit 主进程；若需要 **vercel.app 域名**，可用 `vercel-landing/` 部署静态跳转页指向上述地址。

## 运行（Windows / PowerShell）

进入子项目目录：

```bash
cd E:\Projects\gemini-weapp\geo-search-agent
```

建议创建虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

安装依赖：

```bash
pip install -r requirements.txt
```

可选，安装开发依赖（lint/类型检查）：

```bash
pip install -r requirements-dev.txt
```

配置环境变量（LLM：DeepSeek / OpenRouter）：

- 参考 `env.example`（本仓库不放 `.env`）
- 或直接在终端设置环境变量（推荐，不落盘）：

```bash
# DeepSeek
$env:DEEPSEEK_API_KEY="你的key"

# OpenRouter
$env:OPENROUTER_API_KEY="你的key"
```

启动：

```bash
streamlit run app.py
```

## DeepSeek 说明

本项目使用 **OpenAI 兼容**的调用方式，通过 `DEEPSEEK_BASE_URL` + `DEEPSEEK_API_KEY` 访问 DeepSeek。

## 常见问题

- 如果启动时报缺包：确认已激活虚拟环境并重新 `pip install -r requirements.txt`
- 如果 API 调用 401：确认 `DEEPSEEK_API_KEY` 正确且已导出到环境变量
