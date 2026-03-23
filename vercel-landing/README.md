# Vercel 静态落地页（可选）

Streamlit 应用**不能**像普通 Node 站点一样直接部署在 Vercel 上长期运行（需要常驻进程与 WebSocket）。  
若你希望对外使用 **Vercel 提供的域名**，可以用本目录部署一个**静态说明页**，按钮跳转到：

- [Streamlit Community Cloud](https://streamlit.io/cloud) 给出的 `https://xxx.streamlit.app`，或  
- 你在 Render / Railway 等用 Docker 跑出来的公网地址。

## 部署步骤（Vercel）

1. 修改 `index.html` 里的 `STREAMLIT_URL` 为你的真实 Streamlit 地址。
2. 在 [Vercel](https://vercel.com) 新建项目，导入同一 GitHub 仓库。
3. **Root Directory** 设为：`geo-search-agent/vercel-landing`（若仓库根目录是 `gemini-weapp`，按你实际路径调整）。
4. Framework Preset 选 **Other** 或 **Static**，部署即可。

外部用户访问的是 Vercel 域名，点击按钮进入 Streamlit 应用。
