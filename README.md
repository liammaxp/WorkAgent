# WorkAgent

- [English](#english)
- [中文](#中文)

## English

WorkAgent is a local, single-user AI workspace for job applications. It helps turn a resume, a job description, project evidence, and application notes into a connected workflow for job analysis, tailored resumes, cover letters, interview preparation, and application tracking.

The project is currently focused on Liam's internship search workflow, with an emphasis on truthful, conservative resume and application writing. It is designed to organize real experience more clearly, not to invent credentials, metrics, company experience, awards, or project ownership.

## What It Does

- Analyzes a saved job description and summarizes role requirements, skills, responsibilities, hidden expectations, and fit.
- Reads a base LaTeX resume and generates a tailored LaTeX resume for the current role.
- Generates a cover letter based primarily on the tailored resume, with fallback to the base resume.
- Extracts GitHub repository context from resume links, including README content, languages, commits, file changes, and diff signals.
- Uses GitHub evidence conservatively to support project descriptions without overstating contribution.
- Generates interview preparation notes from the job description, resume, background, and optional GitHub context.
- Tracks applications in a local SQLite database.
- Provides both a CLI workflow and a local Web UI.

## Current Architecture

```text
.
├── my-agent/
│   ├── main.py              # Core CLI agent and tool logic
│   ├── api_server.py        # FastAPI HTTP layer for the frontend
│   ├── requirements.txt     # Python dependencies
│   └── outputs/             # Generated analysis and GitHub context files
├── frontend/
│   ├── src/                 # React app source
│   ├── package.json         # Frontend scripts and dependencies
│   └── vite.config.js       # Vite dev server and /api proxy
├── prompt.txt               # Local system prompt, ignored by git
└── README.md
```

The system has three main layers:

1. `my-agent/main.py`: the core local agent, model adapters, file tools, GitHub context logic, and SQLite application tracking.
2. `my-agent/api_server.py`: a FastAPI service that wraps the agent into REST endpoints for the frontend.
3. `frontend/`: a React + Vite web workspace with pages for dashboard, job description, resume, cover letter, applications, interview prep, GitHub evidence, and chat.

## Backend Features

The backend supports multiple model providers through adapter classes:

- OpenAI
- OpenAI-compatible APIs
- DeepSeek
- Claude / Anthropic
- Gemini / Google

The FastAPI service exposes endpoints for:

- `GET /api/status`
- `POST /api/provider`
- `POST /api/model`
- `GET /api/files/{name}`
- `PUT /api/files/{name}`
- `POST /api/agent/ask`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `POST /api/github/scan`
- `POST /api/github/context`
- `GET /api/applications`
- `POST /api/applications`
- `PATCH /api/applications/{id}`

## Frontend Features

The frontend is a local web workspace built with React 19, React Router 7, Vite 6, and plain CSS.

Implemented pages:

- Dashboard: provider/model status, file readiness, recent outputs, and quick-start flow.
- Job Description: edit, save, and analyze the current job description.
- Resume: edit the base resume, edit the tailored resume, and generate a tailored LaTeX resume.
- Cover Letter: choose a writing style, generate a cover letter, and edit the saved draft.
- Applications: add records, filter by status, and update application state.
- Interview Prep: generate and edit interview preparation notes.
- GitHub Evidence: scan resume links and fetch repository context after confirmation.
- Agent Chat: free-form web chat interface for the same agent workflow.

## Local Files

WorkAgent intentionally uses local files as the working state for a single user. These files can contain private information and are ignored by git:

- `my-agent/.env`
- `my-agent/resume.txt`
- `my-agent/tailored_resume.txt`
- `my-agent/job_description.txt`
- `my-agent/cover_letter.txt`
- `my-agent/interview_prep.txt`
- `my-agent/memory.json`
- `my-agent/github_accounts.txt`
- `my-agent/applications.sqlite3`
- `my-agent/outputs/`
- `prompt.txt`

Do not commit API keys, resumes, job descriptions, GitHub identities, generated letters, application records, or personal background notes.

## Setup

### Backend

From `my-agent/`:

```powershell
pip install -r requirements.txt
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

The API will run at:

```text
http://127.0.0.1:8000
```

### Frontend

From `frontend/`:

```powershell
npm install
npm run dev
```

The web app will run at:

```text
http://localhost:5173
```

During development, Vite proxies `/api` requests to `http://127.0.0.1:8000`.

## CLI Usage

The original CLI workflow is still available:

```powershell
cd my-agent
python main.py
```

Useful CLI commands:

- `provider`: show current provider.
- `provider PROVIDER_NAME`: switch provider.
- `model`: show current model.
- `model MODEL_NAME`: switch model.
- `github diff`: fetch GitHub repository context.
- `exit` or `quit`: close the CLI.

## Design Principles

- Keep the system local-first and single-user unless the backend is redesigned for multi-user isolation.
- Keep user claims grounded in the resume, memory file, job description, and approved GitHub evidence.
- Prefer conservative language for team projects, such as "contributed to", "supported", or "implemented parts of".
- Avoid inventing metrics, production impact, leadership, awards, company experience, APIs, deployment details, or technologies not supported by the source material.
- Keep model API keys and private working files on the backend/local machine only.

## Current Limitations

- Generation tasks are synchronous and can take time; there is no streaming output or cancellation yet.
- GitHub evidence is displayed mostly as JSON rather than a structured visual report.
- Application records do not yet support deletion, detail pages, pagination, or complex search.
- Resume and cover letter editing is plain text only; there is no built-in PDF/DOCX preview or export.
- The project has no login, multi-user isolation, or cloud deployment model.

## Roadmap

- Add a one-click application package workflow: analysis, tailored resume, cover letter, interview prep, and application record.
- Add task queues, progress updates, cancellation, and WebSocket/SSE streaming.
- Add structured GitHub evidence visualization.
- Add application dashboards, statistics, batch actions, and richer search.
- Add PDF/DOCX preview and export.
- Add dark mode and mobile interaction improvements.

## 中文

WorkAgent 是一个本地运行、面向单用户的 AI 求职工作台。它把简历、职位描述、项目证据和申请记录串成一个连续流程，用来完成职位分析、定制简历、求职信、面试准备和申请进度管理。

当前项目主要服务 Liam 的实习求职流程，重点是生成真实、保守、可验证的求职材料。它的目标不是编造更漂亮的经历，而是把已有经历组织得更清楚、更有针对性、更符合目标岗位。

## 功能概览

- 分析已保存的职位描述，提取岗位要求、技能、职责、隐藏期望和匹配度。
- 读取基础 LaTeX 简历，并根据当前岗位生成定制版 LaTeX 简历。
- 基于定制简历优先生成 cover letter，在定制简历不可用时回退到基础简历。
- 从简历中的 GitHub 链接提取仓库上下文，包括 README、语言、提交、文件变更和 diff 信号。
- 保守使用 GitHub 证据支撑项目描述，避免夸大个人贡献。
- 根据职位描述、简历、背景信息和可选 GitHub 证据生成面试准备笔记。
- 使用本地 SQLite 数据库追踪求职申请记录。
- 同时提供 CLI 工作流和本地 Web UI。

## 当前架构

```text
.
├── my-agent/
│   ├── main.py              # 核心 CLI Agent 与工具逻辑
│   ├── api_server.py        # 面向前端的 FastAPI HTTP 服务层
│   ├── requirements.txt     # Python 依赖
│   └── outputs/             # 生成的职位分析与 GitHub 上下文
├── frontend/
│   ├── src/                 # React 应用源码
│   ├── package.json         # 前端脚本与依赖
│   └── vite.config.js       # Vite 开发服务器与 /api 代理
├── prompt.txt               # 本地系统提示词，被 git 忽略
└── README.md
```

系统主要分为三层：

1. `my-agent/main.py`：核心本地 Agent，包含模型适配、本地文件工具、GitHub 上下文逻辑和 SQLite 申请记录管理。
2. `my-agent/api_server.py`：FastAPI 服务，将 Agent 能力封装为 REST 接口供前端调用。
3. `frontend/`：React + Vite Web 工作台，包含概览、职位描述、简历、求职信、申请记录、面试准备、GitHub 证据和 Agent 对话页面。

## 后端功能

后端通过适配器支持多个模型供应商：

- OpenAI
- OpenAI-compatible APIs
- DeepSeek
- Claude / Anthropic
- Gemini / Google

FastAPI 服务提供的主要接口：

- `GET /api/status`
- `POST /api/provider`
- `POST /api/model`
- `GET /api/files/{name}`
- `PUT /api/files/{name}`
- `POST /api/agent/ask`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `POST /api/github/scan`
- `POST /api/github/context`
- `GET /api/applications`
- `POST /api/applications`
- `PATCH /api/applications/{id}`

## 前端功能

前端是一个本地 Web 工作台，使用 React 19、React Router 7、Vite 6 和原生 CSS 实现。

已实现页面：

- 概览：查看 provider/model、文件就绪状态、最近输出和快速开始流程。
- 职位描述：编辑、保存并分析当前 JD。
- 简历：编辑基础简历和定制简历，并生成 tailored resume。
- 求职信：选择写作风格，生成 cover letter，并编辑保存草稿。
- 申请记录：新增记录、按状态筛选、更新申请状态。
- 面试准备：生成并编辑面试准备笔记。
- GitHub 证据：扫描简历链接，并在用户确认后抓取仓库上下文。
- Agent 对话：提供与核心 Agent 交互的自由对话入口。

## 本地文件与隐私

WorkAgent 有意使用本地文件作为单用户工作状态。以下文件可能包含个人信息，已被 git 忽略：

- `my-agent/.env`
- `my-agent/resume.txt`
- `my-agent/tailored_resume.txt`
- `my-agent/job_description.txt`
- `my-agent/cover_letter.txt`
- `my-agent/interview_prep.txt`
- `my-agent/memory.json`
- `my-agent/github_accounts.txt`
- `my-agent/applications.sqlite3`
- `my-agent/outputs/`
- `prompt.txt`

不要提交 API key、简历、职位描述、GitHub 身份、生成的求职信、申请记录或个人背景材料。

## 启动方式

### 后端

在 `my-agent/` 目录下运行：

```powershell
pip install -r requirements.txt
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

API 地址：

```text
http://127.0.0.1:8000
```

### 前端

在 `frontend/` 目录下运行：

```powershell
npm install
npm run dev
```

Web 应用地址：

```text
http://localhost:5173
```

开发模式下，Vite 会将 `/api` 请求代理到 `http://127.0.0.1:8000`。

## CLI 使用

原始 CLI 工作流仍然可用：

```powershell
cd my-agent
python main.py
```

常用命令：

- `provider`：查看当前模型供应商。
- `provider PROVIDER_NAME`：切换模型供应商。
- `model`：查看当前模型。
- `model MODEL_NAME`：切换模型。
- `github diff`：读取 GitHub 仓库上下文。
- `exit` 或 `quit`：退出 CLI。

## 设计原则

- 默认保持本地优先、单用户使用，除非后端专门改造为多用户隔离。
- 所有求职材料都应基于简历、记忆文件、职位描述和用户批准的 GitHub 证据。
- 团队项目优先使用保守措辞，例如 `contributed to`、`supported` 或 `implemented parts of`。
- 不虚构指标、生产影响、领导经历、奖项、公司经验、API、部署细节或来源材料中没有的技术。
- 模型 API key 和私人工作文件只保留在后端/本机。

## 当前限制

- 生成类任务仍是同步请求，耗时较长时只有 loading，没有流式输出或取消功能。
- GitHub 证据主要以 JSON 形式展示，尚未做结构化可视化报告。
- 申请记录暂不支持删除、详情页、分页或复杂搜索。
- 简历和求职信目前是纯文本编辑，没有内置 PDF/DOCX 预览或导出。
- 项目没有登录、多用户隔离或云端部署设计。

## 后续路线

- 增加一键申请包流程：职位分析、定制简历、cover letter、面试准备和申请记录一次完成。
- 增加任务队列、进度更新、取消生成和 WebSocket/SSE 流式输出。
- 增加结构化 GitHub 证据可视化。
- 增加申请看板、统计信息、批量操作和更丰富的搜索。
- 增加 PDF/DOCX 预览与导出。
- 增加深色模式和移动端体验优化。
