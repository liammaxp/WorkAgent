# WorkAgent

- [English](#english)
- [中文](#中文)

## English

WorkAgent is a local, single-user AI workspace for job applications. It connects a resume, job description, personal background, GitHub evidence, generated documents, interview preparation, and application records into one workflow.

The project is designed for truthful, conservative job-search writing. It helps organize and tailor real experience; it should not invent credentials, metrics, company experience, awards, ownership, APIs, deployment details, or unsupported technologies.

## What It Does

- Analyze a saved job description and summarize requirements, skills, responsibilities, expectations, and fit.
- Edit a base resume and generate a tailored LaTeX resume for the current role.
- Let the agent select the strongest truthful project mix for a role by removing weaker resume projects, updating bullets, or adding projects stored in memory.
- Update Chroma-backed vector memory from resume material, with similarity checks before insert or update.
- Generate and edit cover letters based on the tailored resume, with fallback to the base resume.
- Generate and edit interview preparation notes.
- Configure model providers, models, Base URLs, and API keys from the Web UI.
- Configure GitHub usernames, commit author names, commit emails, and GitHub token from the Web UI.
- Start from an example system prompt and customize the agent prompt from the Web UI.
- Scan GitHub repository links from the resume and vector memory, then fetch README, languages, commits, file changes, and diff signals after confirmation.
- Use GitHub evidence conservatively to support project descriptions without overstating contribution.
- Track applications in a local SQLite database.
- Provide both a local Web UI and the original CLI workflow.
- Switch the Web UI between Chinese and English.

## Architecture

```text
.
|-- backend/
|   |-- main.py              # Core CLI agent, model adapters, tools, GitHub logic
|   |-- api_server.py        # FastAPI HTTP layer for the frontend
|   |-- memory_store.py      # Chroma persistence, local embeddings, semantic retrieval
|   `-- requirements.txt     # Python dependencies
|-- frontend/
|   |-- src/                 # React app source
|   |-- package.json         # Frontend scripts and dependencies
|   `-- vite.config.js       # Vite dev server and /api proxy
|-- information/             # Local private working files, Chroma vectors, and SQLite database
|-- background/              # Prompts and background notes
|-- logs/                    # Development/runtime logs
|-- outputs/
|   |-- backend/             # Generated analysis, letters, resumes, and legacy GitHub JSON
|   `-- frontend/            # Frontend production build output
|-- start_workagent.bat      # Windows one-click launcher
|-- start_workagent.ps1      # Windows launcher script
`-- README.md
```

The system has four main layers:

1. `backend/main.py`: local agent logic, model adapters, file tools, GitHub context extraction, and SQLite application tracking.
2. `backend/memory_store.py`: Chroma collections, deterministic local embeddings, similarity-aware writes, semantic retrieval, and legacy JSON migration.
3. `backend/api_server.py`: FastAPI endpoints used by the Web UI.
4. `frontend/`: React + Vite workspace with dashboard, job description, resume, cover letter, applications, interview prep, GitHub evidence, prompt settings, and chat pages.

## Model Providers

Supported providers:

- OpenAI
- OpenAI-compatible APIs
- DeepSeek
- Claude / Anthropic
- Gemini / Google

Default models include `deepseek-v4-pro` for DeepSeek when `DEEPSEEK_MODEL` is not explicitly configured.

You can configure providers from the Dashboard:

1. Select the API provider.
2. Paste the API key.
3. Add or adjust Base URL when needed.
4. Save and enable the provider.
5. Adjust the active model separately in model settings.

The backend writes provider settings into `information/.env`, including variables such as:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_COMPATIBLE_API_KEY`
- `OPENAI_COMPATIBLE_BASE_URL`
- `OPENAI_COMPATIBLE_MODEL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `GEMINI_MODEL`
- `MODEL_PROVIDER`

## GitHub Evidence Setup

The GitHub Evidence page lets you configure:

- GitHub username
- Commit author name
- Commit author email
- GitHub token, optional but recommended for private repositories and higher rate limits

The backend writes:

- GitHub identities to `information/github_accounts.txt`
- GitHub token to `information/.env` as `GITHUB_TOKEN`

After saving GitHub settings, scan the tailored resume, base resume, and vector-memory projects, confirm access, and WorkAgent will fetch repository context for use in resume tailoring, cover letters, and interview prep. The combined source is selected by default, ignores a missing tailored resume, and deduplicates repository links. Memory projects can therefore be considered before they appear in a tailored resume.

Approved repository metadata, verified identities, matched commits, changed files, diff patches, and extracted diff signals are written to the `github_evidence` Chroma collection. GitHub evidence remains separate from durable profile facts.

## Vector Memory

WorkAgent stores durable profile memory and approved GitHub evidence in separate collections inside a local Chroma vector database:

```text
information/chroma/
```

The `profile_facts` collection stores durable user facts. The `github_evidence` collection stores approved repository and commit evidence.

New facts are embedded locally, compared with similar stored records, and then inserted, updated, or deduplicated. Retrieval also uses vector search when the agent provides a task, skill, or project query. The local embedder is deterministic and works offline without downloading an embedding model or sending private profile data to an external embedding API.

Existing `information/memory.json` and older `outputs/backend/github_context/*.json` files are imported automatically when the Chroma collections are empty. They remain migration sources only; Chroma is the active store after migration.

The Resume page can merge durable facts from the base resume into Chroma. The backend also supports merging from the tailored resume through `POST /api/resume/update-memory`. Chroma records are reconstructed as JSON when profile memory is read through the backend.

## Prompt Customization

WorkAgent reads its system prompt from:

```text
background/prompt.txt
```

A reusable starter prompt is included at:

```text
background/prompt.example.txt
```

Use the Prompt Settings page to:

1. Edit the current system prompt.
2. Load the example prompt as a starting point.
3. Save changes without restarting the backend.

The example prompt includes placeholders for name, background, target roles, skills, projects, constraints, truthfulness rules, resume rules, scoring rules, and response style.

## Web UI Pages

- Dashboard: provider/model status, API key setup, file readiness, recent outputs, and quick-start links.
- Job Description: edit, save, and analyze the current job description.
- Resume: edit the base resume, edit the tailored resume, update Chroma vector memory, and generate a tailored LaTeX resume with optional JD-based project selection.
- Cover Letter: choose a writing style, generate a cover letter, and edit the saved draft.
- Applications: add records, filter by status, update records, and delete records.
- Interview Prep: generate and edit interview preparation notes.
- GitHub Evidence: configure GitHub identity/token, scan repositories from the tailored resume, base resume, and vector memory by default, and fetch approved context into Chroma.
- Prompt Settings: edit the system prompt and load the reusable example prompt.
- Agent Chat: free-form chat interface for the same agent workflow.
- Language Switch: change the Web UI between Chinese and English.

## API Endpoints

Main FastAPI endpoints:

- `GET /api/status`
- `POST /api/shutdown`
- `POST /api/session/open`
- `POST /api/provider`
- `GET /api/provider-configs`
- `POST /api/provider-configs`
- `POST /api/model`
- `GET /api/files/{name}`
- `PUT /api/files/{name}`
- `GET /api/prompt`
- `PUT /api/prompt`
- `POST /api/agent/ask`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/resume/update-memory`
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `POST /api/github/scan`
- `GET /api/github/config`
- `POST /api/github/config`
- `POST /api/github/context`
- `GET /api/applications`
- `POST /api/applications`
- `PATCH /api/applications/{record_id}`
- `DELETE /api/applications/{record_id}`

## Local Files And Privacy

WorkAgent intentionally uses local files as working state. These files can contain private information and should not be committed:

- `information/.env`
- `information/resume.txt`
- `information/tailored_resume.txt`
- `information/job_description.txt`
- `information/cover_letter.txt`
- `information/interview_prep.txt`
- `information/memory.json`
- `information/chroma/`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `background/prompt.example.txt`
- `outputs/`

Do not commit API keys, resumes, job descriptions, GitHub identities, generated documents, application records, or personal background notes.

## Setup

### One-Click Start On Windows

Double-click:

```text
start_workagent.bat
```

It starts the backend API, starts the frontend dev server, waits for both to become ready, and opens:

```text
http://localhost:5173
```

The Web UI opens a local session when loaded and notifies the backend when the page closes.

### Manual Backend Start

From `backend/`:

```powershell
pip install -r requirements.txt
python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8001
```

The API runs at:

```text
http://127.0.0.1:8001
```

### Manual Frontend Start

From `frontend/`:

```powershell
npm install
npm run dev
```

The Web UI runs at:

```text
http://localhost:5173
```

During development, Vite proxies `/api` requests to `http://127.0.0.1:8001`.

## CLI Usage

The original CLI workflow is still available:

```powershell
cd backend
python main.py
```

Useful CLI commands:

- `provider`: show current provider.
- `provider PROVIDER_NAME`: switch provider.
- `model`: show current model.
- `model MODEL_NAME`: switch model.
- `github diff`: fetch GitHub repository context.
- `exit` or `quit`: close the CLI.

## Development Checks

Backend syntax check:

```powershell
python -m py_compile backend\memory_store.py backend\api_server.py backend\main.py
```

Frontend production build:

```powershell
cd frontend
npm run build
```

Production frontend output is written to `outputs/frontend/`.

## Current Limitations

- Generation tasks are synchronous; there is no streaming output or cancellation yet.
- GitHub evidence is still displayed mostly as JSON rather than a polished visual report.
- Resume and cover letter editing is plain text; there is no built-in PDF/DOCX preview or export.
- The app is local-first and single-user; it has no login, multi-user isolation, or cloud deployment model.

## Roadmap

- Add a one-click application package flow: analysis, tailored resume, cover letter, interview prep, and application record.
- Add task queues, progress updates, cancellation, and WebSocket/SSE streaming.
- Add structured GitHub evidence visualization.
- Add application dashboards, statistics, batch actions, and richer search.
- Add PDF/DOCX preview and export.
- Improve mobile layout and add dark mode.

## 中文

WorkAgent 是一个本地运行、面向单用户的 AI 求职工作台。它把简历、职位描述、个人背景、GitHub 证据、生成文档、面试准备和投递记录串成一个完整流程。

项目的目标是生成真实、保守、可验证的求职材料。它帮助你组织和定制已有经历，不应该编造学历、指标、公司经历、奖项、项目所有权、API、部署细节或来源材料中没有的技术。

## 功能概览

- 分析已保存的职位描述，提取岗位要求、技能、职责、隐含期望和匹配度。
- 编辑基础简历，并为当前岗位生成定制版 LaTeX 简历。
- 根据简历材料更新 Chroma 向量记忆；新增或更新前会先检索并对比相似记录。
- 基于定制简历生成和编辑求职信，定制简历不可用时回退到基础简历。
- 生成和编辑面试准备笔记。
- 直接在 Web UI 中配置模型供应商、模型、Base URL 和 API Key。
- 直接在 Web UI 中配置 GitHub 用户名、提交作者名、提交邮箱和 GitHub Token。
- 提供可直接试用的示例系统 Prompt，并支持在 Web UI 中编辑个性化 Prompt。
- 从简历和向量记忆中扫描 GitHub 仓库链接，并在确认后读取 README、语言、提交记录、文件变更和 diff 信号。
- 保守使用 GitHub 证据支持项目描述，避免夸大个人贡献。
- 使用本地 SQLite 数据库追踪求职申请。
- 同时提供本地 Web UI 和原始 CLI 流程。
- 在 Web UI 中切换中文和英文。

## 架构

```text
.
|-- backend/
|   |-- main.py              # 核心 CLI agent、模型适配、工具、GitHub 逻辑
|   |-- api_server.py        # 面向前端的 FastAPI HTTP 层
|   |-- memory_store.py      # Chroma 持久化、本地向量化和语义检索
|   `-- requirements.txt     # Python 依赖
|-- frontend/
|   |-- src/                 # React 应用源码
|   |-- package.json         # 前端脚本和依赖
|   `-- vite.config.js       # Vite 开发服务器和 /api 代理
|-- information/             # 本地私有工作文件、Chroma 向量和 SQLite 数据库
|-- background/              # Prompt 和背景说明
|-- logs/                    # 开发和运行日志
|-- outputs/
|   |-- backend/             # 生成的分析、求职信、简历和旧版 GitHub JSON
|   `-- frontend/            # 前端生产构建输出
|-- start_workagent.bat      # Windows 一键启动入口
|-- start_workagent.ps1      # Windows 启动脚本
`-- README.md
```

系统主要分为四层：

1. `backend/main.py`：本地 agent 逻辑、模型适配器、文件工具、GitHub 上下文提取和 SQLite 投递记录。
2. `backend/memory_store.py`：Chroma collections、确定性的本地向量化、写入前相似度对比、语义检索和旧 JSON 自动迁移。
3. `backend/api_server.py`：Web UI 使用的 FastAPI 接口。
4. `frontend/`：React + Vite 前端，包含仪表盘、职位描述、简历、求职信、投递记录、面试准备、GitHub 证据、Prompt 设置和聊天页面。

## 模型配置

支持的供应商：

- OpenAI
- OpenAI-compatible APIs
- DeepSeek
- Claude / Anthropic
- Gemini / Google

如果没有显式配置 `DEEPSEEK_MODEL`，DeepSeek 默认使用 `deepseek-v4-pro`。

可以在 Dashboard 中配置：

1. 选择 API 供应商。
2. 粘贴 API Key。
3. 必要时填写或修改 Base URL。
4. 保存并启用供应商。
5. 在模型设置中单独调整当前模型。

后端会把配置写入 `information/.env`，常见变量包括：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_COMPATIBLE_API_KEY`
- `OPENAI_COMPATIBLE_BASE_URL`
- `OPENAI_COMPATIBLE_MODEL`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_MODEL`
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_MODEL`
- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `GEMINI_MODEL`
- `MODEL_PROVIDER`

## GitHub 证据配置

GitHub Evidence 页面可以配置：

- GitHub 用户名
- 提交作者名
- 提交邮箱
- GitHub Token，可选，但建议用于私有仓库和更高 API 限额

后端会把 GitHub 身份写入 `information/github_accounts.txt`，把 Token 写入 `information/.env` 的 `GITHUB_TOKEN`。

保存后，扫描定制简历、基础简历和向量记忆中的仓库链接并确认授权，即可读取仓库上下文，用于简历定制、求职信和面试准备。页面默认选择这个完整组合；定制简历尚不存在时会自动忽略，并对仓库链接去重。记忆中的项目即使还没有出现在当前简历里，也可以进入候选范围。

已授权的仓库元数据、已验证身份、匹配到的 commits、文件变更、diff patch 和提取出的 diff 信号会写入 Chroma 的 `github_evidence` collection。GitHub 证据与长期画像事实分开存储。

## 向量记忆

WorkAgent 使用本地 Chroma 向量数据库保存长期画像记忆和已授权的 GitHub 证据：

```text
information/chroma/
```

`profile_facts` collection 保存稳定的个人画像事实。`github_evidence` collection 保存已授权的仓库和 commit 证据。

新增信息会先在本地完成向量化，再与已有记录进行相似度对比，最后决定新增、更新或去重。提取信息时，agent 也可以根据任务、技能或项目关键词进行语义检索。内置向量化器是确定性的本地实现，不会下载 embedding 模型，也不会把个人资料发送给外部 embedding API。

旧版 `information/memory.json` 和 `outputs/backend/github_context/*.json` 会在 Chroma collection 为空时自动导入。导入后，它们只作为迁移来源保留；日常读写以 Chroma 为准。

Resume 页面可以把基础简历中的长期事实合并到 Chroma。后端也支持通过 `POST /api/resume/update-memory` 从定制简历合并长期事实。通过后端读取画像记忆时，Chroma 记录会重新组织为 JSON。

## Prompt 个性化

WorkAgent 的系统 Prompt 来自：

```text
background/prompt.txt
```

仓库内提供了一个可复用示例：

```text
background/prompt.example.txt
```

在 Prompt Settings 页面可以：

1. 编辑当前系统 Prompt。
2. 一键载入示例 Prompt 作为起点。
3. 保存后立即生效，不需要重启后端。

示例 Prompt 包含姓名、背景、目标岗位、技能、项目、限制条件、真实性规则、简历规则、评分规则和回复风格等占位内容。

## Web UI 页面

- Dashboard：查看 provider/model 状态、配置 API Key、检查文件状态、查看最近输出和快速入口。
- Job Description：编辑、保存并分析当前职位描述。
- Resume：编辑基础简历和定制简历，更新 Chroma 向量记忆，生成定制版 LaTeX 简历。
- Cover Letter：选择写作风格，生成求职信，并编辑保存草稿。
- Applications：新增、筛选、更新和删除投递记录。
- Interview Prep：生成并编辑面试准备笔记。
- GitHub Evidence：配置 GitHub 身份/Token，默认从定制简历、基础简历和向量记忆扫描仓库，并把已确认的上下文写入 Chroma。
- Prompt Settings：编辑系统 Prompt，并载入可复用示例 Prompt。
- Agent Chat：与核心 agent 自由对话。
- 语言切换：在中文和英文界面之间切换。

## API 接口

主要 FastAPI 接口：

- `GET /api/status`
- `POST /api/shutdown`
- `POST /api/session/open`
- `POST /api/provider`
- `GET /api/provider-configs`
- `POST /api/provider-configs`
- `POST /api/model`
- `GET /api/files/{name}`
- `PUT /api/files/{name}`
- `GET /api/prompt`
- `PUT /api/prompt`
- `POST /api/agent/ask`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/resume/update-memory`
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `POST /api/github/scan`
- `GET /api/github/config`
- `POST /api/github/config`
- `POST /api/github/context`
- `GET /api/applications`
- `POST /api/applications`
- `PATCH /api/applications/{record_id}`
- `DELETE /api/applications/{record_id}`

## 本地文件与隐私

WorkAgent 会使用本地文件作为工作状态。以下文件可能包含个人信息或密钥，不应提交到 git：

- `information/.env`
- `information/resume.txt`
- `information/tailored_resume.txt`
- `information/job_description.txt`
- `information/cover_letter.txt`
- `information/interview_prep.txt`
- `information/memory.json`
- `information/chroma/`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `background/prompt.example.txt`
- `outputs/`

不要提交 API Key、简历、职位描述、GitHub 身份、生成文档、投递记录或个人背景资料。

## 启动方式

### Windows 一键启动

双击：

```text
start_workagent.bat
```

脚本会启动后端、启动前端、等待服务就绪，并打开：

```text
http://localhost:5173
```

Web UI 加载时会打开本地会话，页面关闭时会通知后端。

### 手动启动后端

在 `backend/` 目录中运行：

```powershell
pip install -r requirements.txt
python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8001
```

API 地址：

```text
http://127.0.0.1:8001
```

### 手动启动前端

在 `frontend/` 目录中运行：

```powershell
npm install
npm run dev
```

Web UI 地址：

```text
http://localhost:5173
```

开发模式下，Vite 会把 `/api` 请求代理到 `http://127.0.0.1:8001`。

## CLI 用法

原始 CLI 流程仍然可用：

```powershell
cd backend
python main.py
```

常用 CLI 命令：

- `provider`：查看当前供应商。
- `provider PROVIDER_NAME`：切换供应商。
- `model`：查看当前模型。
- `model MODEL_NAME`：切换模型。
- `github diff`：获取 GitHub 仓库上下文。
- `exit` 或 `quit`：退出 CLI。

## 开发检查

后端语法检查：

```powershell
python -m py_compile backend\memory_store.py backend\api_server.py backend\main.py
```

前端生产构建：

```powershell
cd frontend
npm run build
```

前端生产构建输出会写入 `outputs/frontend/`。

## 当前限制

- 生成任务仍是同步请求，暂时没有流式输出或取消功能。
- GitHub 证据目前主要以 JSON 展示，还没有完整的结构化可视化报告。
- 简历和求职信目前是纯文本编辑，没有内置 PDF/DOCX 预览或导出。
- 项目是本地优先、单用户设计，没有登录、多用户隔离或云端部署模型。

## Roadmap

- 增加一键求职材料包流程：职位分析、定制简历、求职信、面试准备和投递记录。
- 增加任务队列、进度更新、取消功能和 WebSocket/SSE 流式输出。
- 增加结构化 GitHub 证据可视化。
- 增加投递统计、批量操作和更丰富的搜索。
- 增加 PDF/DOCX 预览和导出。
- 改进移动端布局并增加深色模式。
