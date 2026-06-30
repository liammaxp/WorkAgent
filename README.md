# WorkAgent

- [English](#english)
- [中文](#中文)

## English

WorkAgent is a local, single-user AI workspace for job applications. It connects a resume, job description, personal background, GitHub evidence, generated documents, interview preparation, and application records into one workflow.

The project is designed for truthful, conservative job-search writing. It helps organize and tailor real experience; it should not invent credentials, metrics, company experience, awards, ownership, APIs, deployment details, or unsupported technologies.

## What It Does

- Analyze a saved job description and summarize requirements, skills, responsibilities, expectations, and fit.
- Edit a base resume and generate a tailored LaTeX resume for the current role.
- Let the agent select the strongest truthful project mix for a role by removing weaker resume projects, updating bullets, or adding projects stored in memory. The tailored Projects section now prefers 2 projects for a one-page resume, allows 3 only when the third is job-critical, and gives higher-ranked projects more bullet space.
- Let the agent tailor Project and Experience bullets through a stricter ReAct bullet writer that rejects stack-only, CRUD-only, UI-control-only, or broad-module-only wording unless the bullet names a concrete implementation method, substantive workflow capability, and value.
- Let the agent tailor Experience bullets for the job description by reordering, rewriting, or removing weak and redundant bullets while preserving factual meaning. Removing an entire Experience entry requires explicit user approval.
- Return a staged resume-tailoring summary with role profile, extracted JD requirements, and a gap report that highlights missing or weak evidence and unsupported keywords to avoid.
- Update Chroma-backed vector memory from resume material, with similarity checks before insert or update.
- Delete a specific durable-memory fact through Agent Chat; project deletion is synchronized across Chroma profile memory and Project Memory.
- Attach JPG, PNG, GIF, or WebP images in Agent Chat for supported vision models to inspect and act on.
- Ask Agent Chat to prepare application materials; it can generate the tailored resume and/or cover letter, pause for missing fresh JD or base resume input, and create an application record automatically.
- Generate and edit cover letters based on the tailored resume, with fallback to the base resume.
- Generate and edit interview preparation notes.
- Keep generated analysis, tailored resume, cover letter, and interview prep outputs as readable role-based history files, reusing an existing file when regenerated content is unchanged.
- Use the saved job description's predominant language for all generated application content and Agent Chat responses, independently of the Web UI language.
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
|-- script/                  # Shared Linux/macOS Bash implementations
|   |-- install_workagent.sh
|   |-- uninstall_workagent.sh
|   `-- start_workagent.sh
|-- windows/                 # Windows .bat and PowerShell entry points
|-- linux/                   # Linux .sh entry points
|-- macos/                   # macOS double-click .command entry points
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

The default provider is OpenAI (`MODEL_PROVIDER=openai`). DeepSeek remains available for text tasks, but the configured DeepSeek chat provider does not support Agent Chat image attachments.

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

Changing the active provider or model from the Web UI also updates `MODEL_PROVIDER` and the provider-specific model variable immediately, so the selection survives backend restarts.

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

Approved repository metadata, verified identities, matched commits, changed files, diff patches, and extracted diff signals are written to the `github_evidence` Chroma collection. In the same GitHub extraction flow, WorkAgent also analyzes repository metadata, README content, languages, root files, and lightweight code/file signals to write the separate `information/project_memory.json` file. GitHub evidence remains separate from durable profile facts and from Project Memory.

```text
GitHub extraction
-> evidence branch: commit diff / files / README / metadata -> Chroma github_evidence
-> project branch: README / repository metadata / languages / root files / code-file summary -> project_memory.json
-> resume generation: JD + original resume + project_memory.json
-> map each Project Memory project one-to-one to Chroma github_evidence for code/file/commit/diff details
-> classify role family, extract JD requirements, rank projects, and report evidence gaps
-> resume bullets
```

## Vector Memory

WorkAgent stores durable profile memory and approved GitHub evidence in separate collections inside a local Chroma vector database:

```text
information/chroma/
```

The `profile_facts` collection stores durable user and profile facts. The `github_evidence` collection stores approved repository and commit evidence. The `information/project_memory.json` file is a separate project-truth file generated from repository analysis, and resume tailoring uses it as the primary source before consulting Chroma evidence for supporting details.

New facts are embedded locally, compared with similar stored records, and then inserted, updated, or deduplicated. Retrieval also uses vector search when the agent provides a task, skill, or project query. The local embedder is deterministic and works offline without downloading an embedding model or sending private profile data to an external embedding API.

Existing `information/memory.json` and older `outputs/backend/github_context/*.json` files are imported automatically when the Chroma collections are empty. They remain migration sources only; Chroma is the active store after migration.

The Resume page can merge durable facts from the base resume into Chroma. The backend also supports merging from the tailored resume through `POST /api/resume/update-memory`. Chroma records are reconstructed as JSON when profile memory is read through the backend.

Agent Chat can also delete a requested durable-memory fact. The agent reads both Chroma profile memory and Project Memory first. For projects, it prefers an exact `project_id` or `project_name` and removes the matching project from both Chroma profile memory and `information/project_memory.json`; other list facts can be deleted by index, and a whole section is removed only when explicitly requested.

Agent Chat also accepts image attachments. Select up to 4 JPG, PNG, GIF, or WebP images per message, with a maximum size of 10 MB per image. You can add a text instruction or send images alone. The selected provider and model must support vision input; text-only models will reject image requests.

Agent Chat keeps conversation history, unsent text, and selected images while the app remains open, including when you navigate to other pages. Opening the app again starts a fresh chat session.

Agent Chat also saves readable transcript files to `outputs/backend/chat_sessions/`. The frontend autosaves after chat changes and sends a final snapshot when the page closes. Each `.txt` file contains the conversation history and unsent draft text in chronological order; attached images are saved in the matching session assets directory and referenced by path in the transcript.

Agent Chat can detect requests such as preparing a resume, cover letter, or full application material package. For text-only material requests it checks that the current app session has a saved job description and that a base resume exists. If either prerequisite is missing or stale, it pauses the flow and routes you to the needed page. After you save the missing material, return to Agent Chat and continue; WorkAgent will generate the requested documents and add a local application record using company, role, link, and notes extracted from the job description when possible.

Example Agent Chat requests:

```text
Forget the React skill in my profile memory.
Delete the WorkAgent project from my profile memory.
Delete the entire target_roles memory section.
Inspect the attached job-posting screenshot and summarize the role requirements.
Read the attached resume screenshot and suggest factual improvements.
Prepare a tailored resume and cover letter for this application.
```

The delete tool requires an exact section plus a zero-based list-item index, an exact project identifier for project deletion, or an explicit whole-section deletion flag. Legacy `information/memory.json` migration is marked after its first attempt so deleted facts are not restored from the old JSON source after a restart.

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
- Resume: edit the base resume, switch or delete text-output versions by readable file name, export and manage PDF versions, open PDFs with the desktop default application, update Chroma vector memory, and generate a tailored LaTeX resume with JD-based project selection, role/JD analysis metadata, and evidence-gap reporting.
- Cover Letter: choose a writing style, optionally use GitHub evidence, generate a cover letter, and edit the saved draft.
- Applications: add records, filter by status, update records, and delete records.
- Interview Prep: generate and edit interview preparation notes, with the GitHub-evidence toggle remembered locally.
- GitHub Evidence: configure GitHub identity/token, scan repositories from the tailored resume, base resume, and vector memory by default, and fetch approved context into Chroma.
- Prompt Settings: edit the system prompt and load the reusable example prompt.
- Agent Chat: free-form chat interface for the same agent workflow, including image attachments and deletion of specific profile-memory facts.
- Language Switch: change the Web UI between Chinese and English. This affects interface text only; generated content follows the saved job description's predominant language.

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
- `GET /api/output-file`
- `POST /api/output-file/launch`
- `DELETE /api/output-file`
- `GET /api/prompt`
- `PUT /api/prompt`
- `POST /api/agent/ask`
- `POST /api/chat/session`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/resume/update-memory`
- `POST /api/resume/pdf-to-latex`
- `POST /api/resume/tailored/pdf`
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

Agent Chat image requests use data URLs:

```json
{
  "message": "Inspect this screenshot and summarize the role requirements.",
  "language": "en",
  "images": [
    {
      "name": "job-posting.png",
      "mime_type": "image/png",
      "data_url": "data:image/png;base64,..."
    }
  ]
}
```

`POST /api/agent/ask` accepts up to 4 images per request and validates that each image is JPG, PNG, GIF, or WebP and no larger than 10 MB.

`GET /api/status` includes `file_metadata` timestamps for local working files. The frontend uses those timestamps to avoid showing stale generated resume, cover letter, and interview prep outputs from before the current app session.

`GET /api/output-file`, `POST /api/output-file/launch`, and `DELETE /api/output-file` let the frontend read, open with the desktop default application, or delete generated output files such as tailored resume text versions and exported PDF versions. Output history lists readable file names rather than timestamps and omits reserved current-working files. When analysis, tailored resume, cover letter, or interview prep generation produces unchanged content for the same company and role, WorkAgent reuses the matching history file instead of creating a duplicate numbered version.

`POST /api/resume/tailor` accepts `allow_project_selection`, `allow_experience_removal`, and `include_application_hint`. Experience bullet tailoring is enabled by default, while removing an entire Experience entry is disabled unless the user explicitly enables it. The staged tailoring path also returns `role_profile`, `jd_requirements`, and `gap_report` so the UI can show role classification, extracted requirements, missing evidence, weak bullet candidates, and unsafe unsupported keywords. When `include_application_hint` is true, the response can include extracted `company`, `role`, `link`, and `notes` values for creating an application record.

`POST /api/resume/pdf-to-latex` converts an uploaded PDF resume into editable LaTeX and saves it as the base resume. `POST /api/resume/tailored/pdf` compiles the current tailored LaTeX resume into a PDF output file. PDF export now guards `glyphtounicode`/`pdfgentounicode` commands for engine compatibility and tries PDF-oriented LaTeX commands first when the resume requests those settings.

`POST /api/cover-letter/generate` also accepts `include_application_hint` and can return the same extracted application fields.

## Local Files And Privacy

WorkAgent intentionally uses local files as working state. These files can contain private information and should not be committed:

- `information/.env`
- `information/resume.txt`
- `information/tailored_resume.txt`
- `information/job_description.txt`
- `information/cover_letter.txt`
- `information/interview_prep.txt`
- `information/memory.json`
- `information/project_memory.json`
- `information/chroma/`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `background/prompt.example.txt`
- `outputs/`

Do not commit API keys, resumes, job descriptions, GitHub identities, generated documents, application records, or personal background notes.

## Minimum Environment Requirements

The following baseline is the minimum supported environment for the included local setup scripts:

| Item | Minimum requirement | Notes |
| --- | --- | --- |
| Operating system | 64-bit Windows 10/11, a current Linux desktop distribution, or macOS 12 or newer | Windows uses `.bat`/`.ps1`; Linux and macOS use the portable `.sh` scripts; macOS also includes double-clickable `.command` launchers. Linux package-manager support includes `apt-get`, `dnf`, `yum`, `pacman`, and `zypper`; macOS uses Homebrew for optional LaTeX installation. |
| PowerShell | Windows PowerShell 5.1 | Required only by the Windows one-click scripts. |
| Python | Python 3.12 or newer | Required by the backend code and the packages in `backend/requirements.txt`. Make sure `python3` or `python` is available in `PATH` and can run `-m pip`. |
| Node.js | Node.js 18 or newer | Required by the React + Vite frontend. |
| npm | A version bundled with Node.js 18 or newer | Make sure `npm` is available in `PATH`. |
| Memory | 4 GB RAM | 8 GB or more is recommended when other development tools are open. |
| Free disk space | 2 GB | Used by Python packages, `node_modules`, local Chroma data, logs, and generated files. |
| Browser | A current version of Edge, Chrome, or Firefox | Required for the local Web UI. |
| LaTeX toolchain | MiKTeX on Windows or TeX Live/MacTeX on Linux and macOS | Optional for normal use, but required for tailored-resume PDF export. Installers use `winget` on Windows, the detected Linux package manager on Linux, and Homebrew `mactex-no-gui` on macOS. |

Backend packages installed from `backend/requirements.txt` include `openai`, `python-dotenv`, `requests`, `fastapi`, `uvicorn[standard]`, and `chromadb`. Frontend packages are installed from `frontend/package.json`.

An internet connection is required when installing dependencies and when calling a configured AI model provider. GitHub access is required only when using GitHub Evidence features. The local Web UI, SQLite application records, and local Chroma storage run on the local machine.

## Setup

### One-Click Dependency Installation On Windows

Before the first start, double-click:

```text
windows\install_workagent.bat
```

It checks that Python and npm are available, creates a project-local `.venv`, installs the backend and frontend dependencies, then installs MiKTeX and Strawberry Perl through `winget` when needed for tailored-resume PDF export. It also runs a small LaTeX warmup compile in `outputs/latex_install_warmup/` so MiKTeX can download common resume packages during installation instead of waiting until the first PDF export.

### One-Click Dependency Installation On Linux

Run:

```bash
chmod +x linux/install_workagent.sh
./linux/install_workagent.sh
```

The script installs backend and frontend dependencies into the project-local `.venv`, then installs the LaTeX and Perl packages needed for PDF export through `apt-get`, `dnf`, `yum`, `pacman`, or `zypper` when missing.

### One-Click Dependency Installation On macOS

Double-click `macos/install_workagent.command`, or run it from Terminal:

```bash
chmod +x macos/install_workagent.command *.sh
./macos/install_workagent.command
```

The macOS entry uses the shared implementation in `script/`, installs Python packages into `.venv`, and uses Homebrew `mactex-no-gui` when a LaTeX toolchain is missing. If macOS blocks the first launch, Control-click the file, choose **Open**, and approve it once.

After a ZIP download or a checkout that does not preserve executable bits, run this once before using the macOS double-click launchers:

```bash
chmod +x linux/*.sh script/*.sh macos/*.command
```

### One-Click Environment Uninstall On Windows

To remove the installed WorkAgent environment, double-click:

```text
windows\uninstall_workagent.bat
```

The script removes the local `frontend/node_modules`, `.venv`, and LaTeX warmup files, then asks before uninstalling MiKTeX or Strawberry Perl because those system tools may be shared with other projects.

### One-Click Environment Uninstall On Linux

Run:

```bash
chmod +x linux/uninstall_workagent.sh
./linux/uninstall_workagent.sh
```

The script removes `frontend/node_modules`, `.venv`, and LaTeX warmup files. After confirmation it can remove the Linux LaTeX/Perl packages used for PDF export.

### One-Click Environment Uninstall On macOS

Double-click `macos/uninstall_workagent.command`, or run:

```bash
./macos/uninstall_workagent.command
```

Removing Homebrew MacTeX is optional because it may be shared by other projects.

### One-Click Start On Windows

Double-click:

```text
windows\start_workagent.bat
```

It starts the backend API, starts the frontend dev server, waits for both to become ready, and opens:

```text
http://localhost:5173
```

The Web UI opens a local session when loaded and notifies the backend when the page closes.

### One-Click Start On Linux

Run:

```bash
chmod +x linux/start_workagent.sh
./linux/start_workagent.sh
```

The script selects the project `.venv`, starts both services in the background, writes logs and PID files under `logs/`, and opens the Web UI with the available Linux desktop opener.

### One-Click Start On macOS

Double-click `macos/start_workagent.command`, or run:

```bash
./macos/start_workagent.command
```

It starts both services and opens the Web UI through macOS `open`.

The PDF **Open** button follows the desktop operating system: Windows uses the Shell file association, macOS uses Launch Services through `open`, and Linux tries `xdg-open`, `gio open`, and common GNOME/KDE openers. The configured default PDF application is used, matching normal desktop file opening behavior.

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
- Resume and cover letter editing has no built-in document preview or DOCX export; tailored resumes can be exported to PDF when a LaTeX toolchain is installed.
- The app is local-first and single-user; it has no login, multi-user isolation, or cloud deployment model.

## Roadmap

- Expand the Agent Chat application-material flow to include job analysis and interview prep.
- Add task queues, progress updates, cancellation, and WebSocket/SSE streaming.
- Add structured GitHub evidence visualization.
- Add application dashboards, statistics, batch actions, and richer search.
- Add document preview and DOCX export.
- Improve mobile layout and add dark mode.

## 中文

WorkAgent 是一个本地运行、面向单用户的 AI 求职工作台。它把简历、职位描述、个人背景、GitHub 证据、生成文档、面试准备和投递记录串成一个完整流程。

项目的目标是生成真实、保守、可验证的求职材料。它帮助你组织和定制已有经历，不应该编造学历、指标、公司经历、奖项、项目所有权、API、部署细节或来源材料中没有的技术。

## 功能概览

- 分析已保存的职位描述，提取岗位要求、技能、职责、隐含期望和匹配度。
- 编辑基础简历，并为当前岗位生成定制版 LaTeX 简历。
- 允许 Agent 根据职位描述重排、改写或删除 Experience 中较弱和重复的 bullet，同时保持事实含义不变。删除整段 Experience 经历需要用户显式授权。
- 根据简历材料更新 Chroma 向量记忆；新增或更新前会先检索并对比相似记录。
- 通过 Agent Chat 删除指定的长期记忆；删除项目时会同步清理 Chroma 画像记忆和 Project Memory。
- 在 Agent Chat 中上传 JPG、PNG、GIF 或 WebP 图片，让支持视觉输入的模型识别图片并执行任务。
- 在 Agent Chat 中请求生成求职材料；它可以生成定制简历和/或求职信，在缺少本次会话内保存的 JD 或基础简历时暂停，并自动创建投递记录。
- 基于定制简历生成和编辑求职信，定制简历不可用时回退到基础简历。
- 生成和编辑面试准备笔记。
- 将职位分析、定制简历、求职信和面试准备保存为可读的岗位历史文件；同一公司和岗位下内容未变化时复用已有文件，避免重复编号版本。
- 所有生成的求职材料和 Agent Chat 回复均使用已保存职位描述的主要语言，不受 Web UI 界面语言影响。
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
|-- script/                  # Linux/macOS 共用 Bash 实现
|   |-- install_workagent.sh
|   |-- uninstall_workagent.sh
|   `-- start_workagent.sh
|-- windows/                 # Windows .bat 与 PowerShell 入口
|-- linux/                   # Linux .sh 入口
|-- macos/                   # macOS 可双击的 .command 入口
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

默认供应商为 OpenAI（`MODEL_PROVIDER=openai`）。DeepSeek 仍可用于文本任务，但当前配置的 DeepSeek chat provider 不支持 Agent Chat 图片附件。

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

在 Web UI 中切换当前供应商或模型时，后端也会立即更新 `MODEL_PROVIDER` 和对应供应商的模型变量，因此重启后端后仍会保留当前选择。

## GitHub 证据配置

GitHub Evidence 页面可以配置：

- GitHub 用户名
- 提交作者名
- 提交邮箱
- GitHub Token，可选，但建议用于私有仓库和更高 API 限额

后端会把 GitHub 身份写入 `information/github_accounts.txt`，把 Token 写入 `information/.env` 的 `GITHUB_TOKEN`。

保存后，扫描定制简历、基础简历和向量记忆中的仓库链接并确认授权，即可读取仓库上下文，用于简历定制、求职信和面试准备。页面默认选择这个完整组合；定制简历尚不存在时会自动忽略，并对仓库链接去重。记忆中的项目即使还没有出现在当前简历里，也可以进入候选范围。

已授权的仓库元数据、已验证身份、匹配到的 commits、文件变更、diff patch 和提取出的 diff 信号会写入 Chroma 的 `github_evidence` collection。同一次 GitHub 提取流程中，WorkAgent 也会分析仓库元数据、README、语言、根目录文件和轻量代码/文件信号，并写入单独的 `information/project_memory.json` 文件。GitHub 证据与长期画像事实、Project Memory 分开存储。

```text
GitHub extraction
-> evidence branch: commit diff / files / README / metadata -> Chroma github_evidence
-> project branch: README / repository metadata / languages / root files / code-file summary -> project_memory.json
-> resume generation: JD + original resume + project_memory.json
-> map each Project Memory project one-to-one to Chroma github_evidence for code/file/commit/diff details
-> classify role family, extract JD requirements, rank projects, and report evidence gaps
-> resume bullets
```

## 向量记忆

WorkAgent 使用本地 Chroma 向量数据库保存长期画像记忆和已授权的 GitHub 证据：

```text
information/chroma/
```

`profile_facts` collection 保存稳定的个人画像事实。`github_evidence` collection 保存已授权的仓库和 commit 证据。`information/project_memory.json` 是由仓库分析生成的独立项目事实文件，简历定制会先把它作为主来源，再读取 Chroma 证据补充代码、文件、commit 和 diff 细节。

新增信息会先在本地完成向量化，再与已有记录进行相似度对比，最后决定新增、更新或去重。提取信息时，agent 也可以根据任务、技能或项目关键词进行语义检索。内置向量化器是确定性的本地实现，不会下载 embedding 模型，也不会把个人资料发送给外部 embedding API。

旧版 `information/memory.json` 和 `outputs/backend/github_context/*.json` 会在 Chroma collection 为空时自动导入。导入后，它们只作为迁移来源保留；日常读写以 Chroma 为准。

Resume 页面可以把基础简历中的长期事实合并到 Chroma。后端也支持通过 `POST /api/resume/update-memory` 从定制简历合并长期事实。通过后端读取画像记忆时，Chroma 记录会重新组织为 JSON。

Agent Chat 也可以删除指定的长期记忆。agent 会先读取 Chroma 画像记忆和 Project Memory。删除项目时优先使用准确的 `project_id` 或 `project_name`，并同时从 Chroma 画像记忆和 `information/project_memory.json` 中移除匹配项目；其他列表事实可以按索引删除，只有用户明确要求时才删除整个 section。

Agent Chat 也支持上传图片。每条消息最多选择 4 张 JPG、PNG、GIF 或 WebP 图片，单张最大 10 MB。可以附带文字指令，也可以仅发送图片。当前选择的 provider 和模型必须支持视觉输入；纯文本模型会拒绝图片请求。

只要应用仍然打开，Agent Chat 就会保留对话记录、未发送的文字和已选择的图片，包括切换到其他页面再返回的情况。重新打开应用时会创建新的空白对话。

Agent Chat 也会把可直接阅读的转录文件保存到 `outputs/backend/chat_sessions/`。前端会在对话变化后自动保存，并在网页关闭时提交最后一次快照。每个 `.txt` 文件按时间顺序包含对话记录和未发送草稿；附件图片保存在对应的会话资源目录中，并在转录文本里标注路径。

Agent Chat 可以识别“准备简历”“生成求职信”或“准备申请材料”这类请求。对于不带图片的材料生成请求，它会先检查当前应用会话中是否保存了新的职位描述，以及基础简历是否存在。如果任一前置材料缺失或过期，流程会暂停并跳转到对应页面。补齐材料后回到 Agent Chat 继续，WorkAgent 会生成所需文档，并尽量从职位描述中提取公司、岗位、链接和备注来自动新增本地投递记录。

示例：

```text
忘记我的 React 技能。
删除画像记忆中的 WorkAgent 项目。
删除整个 target_roles 记忆 section。
分析我上传的职位截图，总结岗位要求。
阅读我上传的简历截图，并给出基于事实的改进建议。
为这个岗位准备定制简历和求职信。
```

删除工具必须接收准确的 section，并且需要列表项的从零开始索引、用于删除项目的准确项目标识，或显式的整段删除标记。旧版 `information/memory.json` 首次尝试迁移后会写入标记，避免已删除的记忆在重启后从旧 JSON 来源恢复。

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
- Resume：编辑基础简历，按可读文件名切换或删除文本输出版本，导出和管理 PDF 版本，使用桌面默认应用打开 PDF，更新 Chroma 向量记忆，并生成带 JD 项目选择、角色/JD 分析元数据和证据缺口报告的定制版 LaTeX 简历。
- Cover Letter：选择写作风格，可选择使用 GitHub 证据，生成求职信，并编辑保存草稿。
- Applications：新增、筛选、更新和删除投递记录。
- Interview Prep：生成并编辑面试准备笔记，并在本地记住是否使用 GitHub 证据。
- GitHub Evidence：配置 GitHub 身份/Token，默认从定制简历、基础简历和向量记忆扫描仓库，并把已确认的上下文写入 Chroma。
- Prompt Settings：编辑系统 Prompt，并载入可复用示例 Prompt。
- Agent Chat：与核心 agent 自由对话，可以上传图片，也可以删除指定的画像记忆。
- 语言切换：在中文和英文界面之间切换。该设置只影响界面文字；生成内容使用已保存职位描述的主要语言。

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
- `GET /api/output-file`
- `POST /api/output-file/launch`
- `DELETE /api/output-file`
- `GET /api/prompt`
- `PUT /api/prompt`
- `POST /api/agent/ask`
- `POST /api/chat/session`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/resume/update-memory`
- `POST /api/resume/pdf-to-latex`
- `POST /api/resume/tailored/pdf`
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

Agent Chat 图片请求使用 data URL：

```json
{
  "message": "分析这张职位截图并总结岗位要求。",
  "language": "zh",
  "images": [
    {
      "name": "job-posting.png",
      "mime_type": "image/png",
      "data_url": "data:image/png;base64,..."
    }
  ]
}
```

`POST /api/agent/ask` 每次最多接受 4 张图片，并校验每张图片是否为 JPG、PNG、GIF 或 WebP 格式且不超过 10 MB。

`GET /api/status` 会返回本地工作文件的 `file_metadata` 时间戳。前端用这些时间戳避免展示当前应用会话之前生成的旧版简历、求职信和面试准备内容。

`GET /api/output-file`、`POST /api/output-file/launch` 和 `DELETE /api/output-file` 让前端读取、用桌面默认应用打开或删除生成输出文件，例如定制简历文本版本和导出的 PDF 版本。输出历史按可读文件名展示，而不是按时间戳展示，并且会排除保留的当前工作文件。当职位分析、定制简历、求职信或面试准备在同一公司和岗位下生成出未变化的内容时，WorkAgent 会复用匹配的历史文件，而不是创建重复的编号版本。

`POST /api/resume/tailor` 接受 `allow_project_selection`、`allow_experience_removal` 和 `include_application_hint`。Experience bullet 默认允许定制，但整段 Experience 经历默认不会删除，只有用户显式开启后才允许移除。分阶段定制流程还会返回 `role_profile`、`jd_requirements` 和 `gap_report`，用于展示角色分类、提取出的 JD 要求、缺失证据、较弱 bullet 候选和应避免的 unsupported keywords。`include_application_hint` 为 true 时，响应可以包含用于创建投递记录的 `company`、`role`、`link` 和 `notes` 字段。

`POST /api/resume/pdf-to-latex` 会把上传的 PDF 简历转换为可编辑 LaTeX，并保存为基础简历。`POST /api/resume/tailored/pdf` 会把当前定制版 LaTeX 简历编译成 PDF 输出文件。PDF 导出会为 `glyphtounicode`/`pdfgentounicode` 命令加上引擎兼容保护，并在简历需要这些设置时优先尝试 PDF-oriented LaTeX 编译命令。

`POST /api/cover-letter/generate` 也接受 `include_application_hint`，并可以返回同样的投递记录字段。

## 本地文件与隐私

WorkAgent 会使用本地文件作为工作状态。以下文件可能包含个人信息或密钥，不应提交到 git：

- `information/.env`
- `information/resume.txt`
- `information/tailored_resume.txt`
- `information/job_description.txt`
- `information/cover_letter.txt`
- `information/interview_prep.txt`
- `information/memory.json`
- `information/project_memory.json`
- `information/chroma/`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `background/prompt.example.txt`
- `outputs/`

不要提交 API Key、简历、职位描述、GitHub 身份、生成文档、投递记录或个人背景资料。

## 最低环境配置要求

以下配置是仓库内本地安装和启动脚本支持的最低运行基线：

| 项目 | 最低要求 | 说明 |
| --- | --- | --- |
| 操作系统 | 64 位 Windows 10/11、当前仍受支持的 Linux 桌面发行版，或 macOS 12 及以上版本 | Windows 使用 `.bat`/`.ps1`；Linux 与 macOS 使用 `.sh`；macOS 还提供可双击的 `.command` 入口。Linux 支持 `apt-get`、`dnf`、`yum`、`pacman` 和 `zypper`，macOS 的可选 LaTeX 安装使用 Homebrew。 |
| PowerShell | Windows PowerShell 5.1 | 仅 Windows 一键脚本需要使用。 |
| Python | Python 3.12 或更高版本 | 后端代码以及 `backend/requirements.txt` 中的依赖需要使用。请确保 `python3` 或 `python` 已加入 `PATH`，并且可以执行 `-m pip`。 |
| Node.js | Node.js 18 或更高版本 | React + Vite 前端需要使用。 |
| npm | Node.js 18 或更高版本附带的 npm | 请确保 `npm` 已加入 `PATH`。 |
| 内存 | 4 GB RAM | 如果同时开启其他开发工具，建议使用 8 GB 或更多内存。 |
| 可用磁盘空间 | 2 GB | 用于 Python 依赖、`node_modules`、本地 Chroma 数据、日志和生成文件。 |
| 浏览器 | 当前版本的 Edge、Chrome 或 Firefox | 用于访问本地 Web UI。 |
| LaTeX 工具链 | Windows 使用 MiKTeX，Linux 使用 TeX Live，macOS 使用 MacTeX | 普通使用可不安装；导出定制简历 PDF 时需要。安装脚本分别使用 `winget`、检测到的 Linux 包管理器和 Homebrew `mactex-no-gui`。 |

后端会根据 `backend/requirements.txt` 安装 `openai`、`python-dotenv`、`requests`、`fastapi`、`uvicorn[standard]` 和 `chromadb`。前端依赖根据 `frontend/package.json` 安装。

安装依赖以及调用已配置的 AI 模型服务时需要联网。只有使用 GitHub Evidence 功能时才需要访问 GitHub。本地 Web UI、SQLite 投递记录和本地 Chroma 存储均在本机运行。

## 启动方式

### Windows 一键安装依赖

首次启动前，双击：

```text
windows\install_workagent.bat
```

脚本会检查 Python 和 npm 是否可用，创建项目本地 `.venv`，安装后端与前端依赖，并在需要时通过 `winget` 自动安装 MiKTeX 和 Strawberry Perl，用于定制简历 PDF 导出。脚本还会在 `outputs/latex_install_warmup/` 执行一次小型 LaTeX 预热编译，让 MiKTeX 在安装阶段下载常用简历宏包，而不是等到第一次导出 PDF 时再下载。

### Linux 一键安装依赖

运行：

```bash
chmod +x linux/install_workagent.sh
./linux/install_workagent.sh
```

脚本会把后端依赖安装到项目本地 `.venv`，并在缺少 PDF 导出组件时通过 `apt-get`、`dnf`、`yum`、`pacman` 或 `zypper` 安装 LaTeX/Perl。

### macOS 一键安装依赖

在 Finder 中双击 `macos/install_workagent.command`，或在终端运行：

```bash
chmod +x macos/install_workagent.command *.sh
./macos/install_workagent.command
```

macOS 入口会调用 `script/` 中的共用实现，把 Python 依赖安装到 `.venv`，并在缺少 LaTeX 时使用 Homebrew `mactex-no-gui`。若首次被系统拦截，可按住 Control 点击文件，选择“打开”并确认一次。

如果通过 ZIP 下载，或 Git 检出时没有保留可执行权限，请先执行一次：

```bash
chmod +x linux/*.sh script/*.sh macos/*.command
```

### Windows 一键卸载环境

如需移除 WorkAgent 安装的环境，双击：

```text
windows\uninstall_workagent.bat
```

脚本会删除项目本地的 `frontend/node_modules`、`.venv` 和 LaTeX 预热文件；卸载 MiKTeX 或 Strawberry Perl 前会先询问确认，因为这些系统工具可能被其他项目共用。

### Linux 一键卸载环境

运行：

```bash
chmod +x linux/uninstall_workagent.sh
./linux/uninstall_workagent.sh
```

脚本会删除项目本地的 `frontend/node_modules`、`.venv` 和 LaTeX 预热文件；确认后也可以卸载 Linux 中用于 PDF 导出的 LaTeX/Perl 系统包。

### macOS 一键卸载环境

在 Finder 中双击 `macos/uninstall_workagent.command`，或运行：

```bash
./macos/uninstall_workagent.command
```

是否卸载 Homebrew MacTeX 会单独询问，因为它可能被其他项目共用。

### Windows 一键启动

双击：

```text
windows\start_workagent.bat
```

脚本会启动后端、启动前端、等待服务就绪，并打开：

```text
http://localhost:5173
```

Web UI 加载时会打开本地会话，页面关闭时会通知后端。

### Linux 一键启动

运行：

```bash
chmod +x linux/start_workagent.sh
./linux/start_workagent.sh
```

脚本会使用项目 `.venv`，在后台启动后端和前端，把日志及 PID 文件写入 `logs/`，并使用 Linux 桌面打开器打开 Web UI。

### macOS 一键启动

在 Finder 中双击 `macos/start_workagent.command`，或运行：

```bash
./macos/start_workagent.command
```

脚本会启动前后端，并通过 macOS `open` 打开 Web UI。

PDF 的“打开”按钮会遵循桌面系统默认行为：Windows 使用 Shell 文件关联，macOS 通过 Launch Services 的 `open` 打开，Linux 依次尝试 `xdg-open`、`gio open` 以及常见 GNOME/KDE 打开器，并使用系统配置的默认 PDF 应用。

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
- 简历和求职信没有内置文档预览或 DOCX 导出；安装 LaTeX 工具链后可以把定制简历导出为 PDF。
- 项目是本地优先、单用户设计，没有登录、多用户隔离或云端部署模型。

## Roadmap

- 扩展 Agent Chat 求职材料流程，加入职位分析和面试准备。
- 增加任务队列、进度更新、取消功能和 WebSocket/SSE 流式输出。
- 增加结构化 GitHub 证据可视化。
- 增加投递统计、批量操作和更丰富的搜索。
- 增加文档预览和 DOCX 导出。
- 改进移动端布局并增加深色模式。
