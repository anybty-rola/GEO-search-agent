# 部署到 GitHub 与公网访问

## 1. 同步到 GitHub（本地已有代码时）

在仓库根目录（例如 `gemini-weapp`）执行：

```bash
git init
git add .
git status   # 确认没有 .env / .venv
git commit -m "feat: geo-search-agent MVP"
```

在 GitHub 新建空仓库（不要勾选自动添加 README），然后：

```bash
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git branch -M main
git push -u origin main
```

**务必不要提交：**

- `.env`、真实 API Key  
- `.venv/`、`__pycache__/`

根目录 `.gitignore` 已包含常见 Python/虚拟环境规则。

---

## 2. 为什么「主应用」不适合直接放在 Vercel？

本项目是 **Streamlit**（Python 长连接 + WebSocket）。  
**Vercel** 以 Serverless/边缘为主，不适合托管标准 Streamlit 进程，官方也不推荐把 Streamlit 当主服务部署在 Vercel。

若你希望 **对外可访问的公网地址**，推荐下面两种方式之一。

---

## 3. 推荐：Streamlit Community Cloud（与 GitHub 联动）

1. 打开 [Streamlit Community Cloud](https://streamlit.io/cloud)，用 GitHub 登录。  
2. **New app** → 选择你的仓库与分支。  
3. **Main file path** 填：`geo-search-agent/app.py`（若仓库根目录下只有 `geo-search-agent` 子目录，按实际路径填写）。  
4. **App URL** 会得到类似：`https://<app-name>.streamlit.app`，外部即可访问。

### 密钥（不要在仓库里写死）

在 Streamlit Cloud 的 **Settings → Secrets** 中配置，例如：

```toml
OPENROUTER_API_KEY = "sk-or-..."
# 可选
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
```

应用已支持：启动后把 **Secrets** 中的键同步到 `os.environ`（见 `app.py` 中 `_inject_streamlit_secrets_into_environ`），与本地 `.env` 使用同一套变量名即可。

---

## 4. 备选：Docker → Render / Railway / Fly.io

仓库内已提供 `Dockerfile`（在 `geo-search-agent` 目录）。

构建示例（在 `geo-search-agent` 目录下）：

```bash
docker build -t geo-search-agent .
docker run -p 8501:8501 -e OPENROUTER_API_KEY=你的key geo-search-agent
```

在托管平台设置 **端口 8501** 与 **环境变量**，绑定公网域名即可。

---

## 5. 若必须使用「Vercel 域名」对外展示

用本仓库中的 **静态落地页**：`geo-search-agent/vercel-landing/`  

- 部署该目录到 Vercel，得到 `https://xxx.vercel.app`  
- 在 `index.html` 里把跳转链接改为你的 Streamlit 公网地址（见该目录下 `README.md`）

这样：**入口在 Vercel**，**应用在 Streamlit Cloud（或其它主机）**。

---

## 6. API Key 与安全

- 公网演示建议使用 **限额/只读 Key**，并定期轮换。  
- 不要在仓库、Issue、截图中泄露 Key。  
- DuckDuckGo 搜索在部分网络环境下可能不稳定，属正常现象。
