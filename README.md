# LuminaRAG

基于 LangChain Agent + Milvus 的 RAG 问答应用。FastAPI 后端 + Vue 3 前端，支持混合检索、Auto-merging、精排与流式输出。

## 技术栈
- 后端：FastAPI、LangChain/LangGraph、SQLAlchemy、PostgreSQL、Redis
- 检索：Milvus 2.5+（Dense + 原生 BM25 混合检索 + RRF）、Embedding API、Rerank API
- 文档解析：MinerU（PDF，可选）、python-docx（Word 按章节）、openpyxl（Excel 按 sheet）
- 前端：Vite + Vue 3 + TypeScript + Pinia

## 本地部署

### 1) 安装依赖
```bash
python -m venv .venv
# Windows 激活
.venv\Scripts\activate
# Linux / macOS 激活
source .venv/bin/activate

pip install -r requirements.txt
```

> MinerU 用于 PDF 结构化解析，会连带安装 torch；不装则 PDF 自动降级 PyPDFLoader 纯文本提取。可从 requirements.txt 中注释掉 `mineru==3.4.5` 跳过。

### 2) 配置环境变量
```bash
cp .env.example .env
```
按需填写 `.env` 中的 API Key、模型名与连接地址，变量说明见 `.env.example` 内注释。

### 3) 启动基础设施
```bash
docker compose up -d

# Milvus standalone 启动约需 90 秒，可查看状态
docker compose ps
```

端口：PostgreSQL `5432`、Redis `6379`、Milvus `19530`、MinIO `9000`/`9001`、Attu `8080`。

### 4) 构建前端
```bash
cd frontend
npm install
npm run build
```
构建产物输出至 `frontend/dist/`，由后端静态托管。

### 5) 启动应用
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

访问：
- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

### 前端开发调试（可选）
```bash
cd frontend
npm run dev   # 运行于 http://localhost:3000，代理到后端 8000
```

## 核心能力
- **混合检索**：L3 叶子分块入 Milvus，Dense + 稀疏双路召回 + RRF 融合，Auto-merging 回父块，Rerank 精排。
- **自适应复杂度规划**：简单问题本地规则直接检索；复杂问题由 FAST_MODEL 一次完成判断与 2-4 个子问题规划，LangGraph `Send` 并行执行检索 → 证据评判。
- **纠错型 RAG**：GRADE_MODEL 结构化判断证据相关性/可回答性，证据不足时在 Step-back / HyDE 中单选一种重写，只执行一次二次检索。
- **流式输出**：SSE 逐 token 推送，实时展示 RAG 检索步骤，支持随时中断。
- **账号体系**：JWT 鉴权 + RBAC（admin 管理知识库，user 聊天）。
- **会话记忆**：消息持久化 PostgreSQL，超长自动摘要，Redis 缓存热点会话与父文档。
- **文档入库**：上传 → 结构化拆页（PDF 章节/Word Heading/Excel sheet）→ 三级滑窗分块 → 叶子块向量化入 Milvus，重复上传自动清理旧数据。

## API 速览
- 鉴权：`POST /auth/register`、`POST /auth/login`、`GET /auth/me`
- 聊天：`POST /chat`（非流式）、`POST /chat/stream`（SSE 流式）
- 会话：`GET /sessions`、`GET /sessions/{id}`、`DELETE /sessions/{id}`
- 文档（管理员）：`GET /documents`、`POST /documents/upload`、`DELETE /documents/{filename}`

## 目录结构
```
backend/
  api/routes/     HTTP 路由（auth / chat / sessions / documents）
  chat/           对话服务（service / runtime / request_context / storage）
  rag/            检索管线（LangGraph 工作流 / 混合检索 / Auto-merging / Rerank）
  indexing/       文档解析、分块、向量化与 Milvus 写入
  infra/          数据库 / Redis 缓存 / 鉴权
  db/             SQLAlchemy ORM 模型
  tools/          Agent 工具（天气、知识库检索）
  jobs/           后台任务（上传进度）
frontend/         Vite + Vue 3 前端（stores + components）
tests/            unittest 单测（mock 隔离外部服务）
docker-compose.yml / docker-compose.prod.yml
```

## 运行测试
```bash
python -m unittest discover -s tests
```
