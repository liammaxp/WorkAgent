# WorkAgent

- [English](#english)
- [中文](#中文)

## English

WorkAgent is a local, single-user AI workspace for job applications. It connects a resume, job description, personal background, GitHub evidence, generated documents, and application records into one workflow.

The project is designed for truthful, conservative job-search writing. It helps organize and tailor real experience; it should not invent credentials, metrics, company experience, awards, ownership, APIs, deployment details, or unsupported technologies.

## What It Does

- Analyze a saved job description and summarize requirements, skills, responsibilities, expectations, and fit.
- Edit a base resume and generate a tailored LaTeX resume for the current role.
- Generate and edit cover letters based on the tailored resume, with fallback to the base resume.
- Configure model providers and API keys directly in the Web UI.
- Configure GitHub usernames, commit author names, commit emails, and GitHub token directly in the Web UI.
- Start from an example system prompt and customize the agent prompt directly in the Web UI.
- Scan GitHub repository links from the resume and fetch README, languages, commits, file changes, and diff signals after confirmation.
- Use GitHub evidence conservatively to support project descriptions without overstating contribution.
- Generate and edit interview preparation notes.
- Track applications in a local SQLite database.
- Provide both a local Web UI and the original CLI workflow.

## Architecture

```text
.
|-- backend/
|   |-- main.py              # Core CLI agent, model adapters, tools, GitHub logic
|   |-- api_server.py        # FastAPI HTTP layer for the frontend
|   `-- requirements.txt     # Python dependencies
|-- frontend/
|   |-- src/                 # React app source
|   |-- package.json         # Frontend scripts and dependencies
|   `-- vite.config.js       # Vite dev server and /api proxy
|-- information/             # Local private working files and database
|-- background/              # Prompts and background notes
|-- logs/                    # Project logs
|-- outputs/
|   |-- backend/             # Generated analysis, letters, resumes, GitHub context
|   `-- frontend/            # Frontend production build output
|-- start_workagent.bat      # Windows one-click launcher
|-- start_workagent.ps1      # Windows launcher script
`-- README.md
```

The system has three main layers:

1. `backend/main.py`: local agent logic, model adapters, file tools, GitHub context extraction, and SQLite application tracking.
2. `backend/api_server.py`: FastAPI endpoints used by the Web UI.
3. `frontend/`: React + Vite workspace with dashboard, job description, resume, cover letter, applications, interview prep, GitHub evidence, and chat pages.

## Model Providers

Supported providers:

- OpenAI
- OpenAI-compatible APIs
- DeepSeek
- Claude / Anthropic
- Gemini / Google

You can configure providers from the Dashboard:

1. Select the API provider.
2. Paste the API key.
3. Add or adjust Base URL when needed.
4. Click `Save and enable`.

The backend writes the correct environment variables into `information/.env`, such as:

- `OPENAI_API_KEY`
- `OPENAI_COMPATIBLE_API_KEY`
- `OPENAI_COMPATIBLE_BASE_URL`
- `DEEPSEEK_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `MODEL_PROVIDER`

The active model is edited separately in the Dashboard model settings.

## GitHub Evidence Setup

The GitHub Evidence page lets you configure:

- GitHub username
- Commit author name
- Commit author email
- GitHub token, optional but recommended for private repositories and higher rate limits

The backend writes:

- GitHub identities to `information/github_accounts.txt`
- GitHub token to `information/.env` as `GITHUB_TOKEN`

After saving GitHub settings, scan the resume source, confirm access, and WorkAgent will fetch repository context for use in resume tailoring, cover letters, and interview prep.

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
- Resume: edit the base resume, edit the tailored resume, and generate a tailored LaTeX resume.
- Cover Letter: choose a writing style, generate a cover letter, and edit the saved draft.
- Applications: add records, filter by status, update records, and delete records.
- Interview Prep: generate and edit interview preparation notes.
- GitHub Evidence: configure GitHub identity/token, scan repositories, and fetch approved context.
- Prompt Settings: edit the system prompt and load the reusable example prompt.
- Agent Chat: free-form chat interface for the same agent workflow.

## API Endpoints

Main FastAPI endpoints:

- `GET /api/status`
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
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `GET /api/github/config`
- `POST /api/github/config`
- `POST /api/github/scan`
- `POST /api/github/context`
- `GET /api/applications`
- `POST /api/applications`
- `PATCH /api/applications/{id}`
- `DELETE /api/applications/{id}`

## Local Files And Privacy

WorkAgent intentionally uses local files as working state. These files can contain private information and should not be committed:

- `information/.env`
- `information/resume.txt`
- `information/tailored_resume.txt`
- `information/job_description.txt`
- `information/cover_letter.txt`
- `information/interview_prep.txt`
- `information/memory.json`
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
python -m py_compile backend\api_server.py backend\main.py
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

WorkAgent 是一个本地运行、面向单用户的 AI 求职工作台。它把简历、职位描述、个人背景、GitHub 证据、生成文档和申请记录串成一个完整流程。

这个项目的目标是生成真实、保守、可验证的求职材料。它帮助你更清楚地组织和定制已有经历，不应该编造学历、指标、公司经历、奖项、项目所有权、API、部署细节或来源材料中没有的技术。

## 功能概览

- 分析已保存的职位描述，提取岗位要求、技能、职责、隐含期待和匹配度。
- 编辑基础简历，并为当前岗位生成定制版 LaTeX 简历。
- 基于定制简历生成和编辑求职信，定制简历不可用时回退到基础简历。
- 直接在前端配置模型供应商和 API Key。
- 直接在前端配置 GitHub 用户名、提交作者名称、提交邮箱和 GitHub Token。
- 提供可直接试用的示例系统 Prompt，并支持在前端编辑个性化 Prompt。
- 从简历链接中扫描 GitHub 仓库，并在确认后读取 README、语言、提交记录、文件变更和 diff 信号。
- 保守使用 GitHub 证据支持项目描述，避免夸大个人贡献。
- 生成和编辑面试准备笔记。
- 使用本地 SQLite 数据库追踪求职申请。
- 同时提供本地 Web UI 和原始 CLI 流程。

## 模型配置

支持的供应商：

- OpenAI
- OpenAI-compatible APIs
- DeepSeek
- Claude / Anthropic
- Gemini / Google

在 Dashboard 中可以直接配置：

1. 选择 API 厂商。
2. 粘贴 API Key。
3. 必要时填写或修改 Base URL。
4. 点击“保存并启用”。

后端会自动把正确变量写入 `information/.env`，例如 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`ANTHROPIC_API_KEY`、`GEMINI_API_KEY` 和 `MODEL_PROVIDER`。当前模型在 Dashboard 的“模型设置”中单独修改。

## GitHub 配置

GitHub Evidence 页面可以直接配置：

- GitHub 用户名
- 提交作者名称
- 提交邮箱
- GitHub Token，可选，但建议用于私有仓库和更高 API 限额

后端会把 GitHub 身份写入 `information/github_accounts.txt`，把 Token 写入 `information/.env` 的 `GITHUB_TOKEN`。

保存后，选择简历来源并扫描仓库，确认授权后即可读取仓库上下文，用于简历定制、求职信和面试准备。

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

## 页面

- Dashboard：查看 provider/model 状态、配置 API Key、查看文件状态、最近输出和快速入口。
- Job Description：编辑、保存并分析当前职位描述。
- Resume：编辑基础简历和定制简历，生成定制版 LaTeX 简历。
- Cover Letter：选择写作风格，生成求职信，并编辑保存草稿。
- Applications：新增、筛选、更新和删除申请记录。
- Interview Prep：生成并编辑面试准备笔记。
- GitHub Evidence：配置 GitHub 身份/Token，扫描仓库并获取已确认的上下文。
- Prompt Settings：编辑系统 Prompt，并载入可复用示例 Prompt。
- Agent Chat：与核心 Agent 自由对话。

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

### 手动启动后端

```powershell
cd backend
pip install -r requirements.txt
python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8001
```

### 手动启动前端

```powershell
cd frontend
npm install
npm run dev
```

前端地址：

```text
http://localhost:5173
```

开发模式下，Vite 会把 `/api` 请求代理到 `http://127.0.0.1:8001`。

## 本地文件与隐私

以下文件可能包含个人信息或密钥，不要提交到 git：

- `information/.env`
- `information/resume.txt`
- `information/tailored_resume.txt`
- `information/job_description.txt`
- `information/cover_letter.txt`
- `information/interview_prep.txt`
- `information/memory.json`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `background/prompt.example.txt`
- `outputs/`

## 当前限制

- 生成任务仍是同步请求，暂时没有流式输出或取消功能。
- GitHub 证据目前主要以 JSON 展示，还没有完整的结构化可视化报告。
- 简历和求职信目前是纯文本编辑，没有内置 PDF/DOCX 预览或导出。
- 项目是本地优先、单用户设计，没有登录、多用户隔离或云端部署模型。
