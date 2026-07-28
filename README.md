# WorkAgent

- [English](#english)
- [中文](#中文)

## English

WorkAgent is a local, single-user AI workspace for job applications. It connects a resume, job description, personal background, GitHub evidence, generated documents, interview preparation, and application records into one workflow.

The project is designed for truthful, conservative job-search writing. It helps organize and tailor real experience; it should not invent credentials, metrics, company experience, awards, ownership, APIs, deployment details, or unsupported technologies.

## Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Model Providers](#model-providers)
- [GitHub Evidence Setup](#github-evidence-setup)
- [Vector Memory](#vector-memory)
- [Prompt Customization](#prompt-customization)
- [Web UI Pages](#web-ui-pages)
- [API Endpoints](#api-endpoints)
- [Local Files And Privacy](#local-files-and-privacy)
- [Minimum Environment Requirements](#minimum-environment-requirements)
- [Setup](#setup)
- [CLI Usage](#cli-usage)
- [Development Checks](#development-checks)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)

## What It Does

- Analyze a saved job description and summarize requirements, skills, responsibilities, expectations, and fit.
- Edit a base resume and generate a tailored LaTeX resume for the current role.
- Let the agent select the strongest truthful project mix for a role by removing weaker resume projects, updating bullets, or adding projects stored in memory. The tailored Projects section now prefers 2 projects for a one-page resume, allows 3 only when the third is job-critical, and gives higher-ranked projects more bullet space.
- Let the agent tailor Project and Experience bullets through a stricter ReAct bullet writer that rejects stack-only, CRUD-only, UI-control-only, or broad-module-only wording unless the bullet names a concrete implementation method, substantive workflow capability, and value.
- Let the agent tailor Experience bullets for the job description by reordering, rewriting, or removing weak and redundant bullets while preserving factual meaning. Removing an entire Experience entry requires explicit user approval.
- Preserve the base resume's compact Technical Skills LaTeX style, clean process/feature phrases out of generated skills, reclassify real user-backed skills into the expected categories, and reject visible bullets, placeholders, generic filler, duplicates, or unsupported sentence fragments.
- Use a compact technical ontology to recognize JD technologies, aliases, cautious resume wording, and unsupported-claim risks without treating ontology terms as proof of user experience.
- Validate final resume merges against project order, bullet budgets, Technical Skills evidence, unsupported skills, summary quality, and mechanism-rich bullet depth before saving the tailored LaTeX. Quality-gate results are returned as structured issues with source, severity, code, message, and repairability metadata.
- Return a staged resume-tailoring summary with role profile, extracted JD requirements, project ranking, project-section validation, and a gap report that highlights missing or weak evidence and unsupported keywords to avoid.
- Run long agent work as cancellable background tasks with status, messages, result retrieval, and live guidance capture for reruns.
- Check whether Project Memory has enough STAR evidence for resume tailoring and save missing project facts before generation.
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
- Process saved GitHub evidence through an internal project evidence pipeline that validates bounded inputs, normalizes and deduplicates records, synthesizes conservative evidence facts, scores evidence quality, and extracts capability signals without inferring unsupported claims. Project evidence remains a Python-internal backend layer and is not exposed through the Web UI or HTTP API.
- Track applications in a local SQLite database.
- Provide both a local Web UI and the original CLI workflow.
- Switch the Web UI between Chinese and English.

## Architecture

```text
.
|-- backend/
|   |-- main.py              # Core CLI agent, model adapters, tools, GitHub logic
|   |-- api_server.py        # FastAPI HTTP layer for the frontend
|   |-- evidence_memory.py            # GitHub evidence JSONL storage
|   |-- evidence_pipeline.py          # GitHub evidence build orchestration and inspection
|   |-- evidence_*.py                 # Evidence chunk, change-summary, and evidence-card helpers
|   |-- project_change_memory.py      # Project change schema, extraction, and persistence
|   |-- project_change_pipeline.py    # Project change build, inspect, and health helpers
|   |-- project_evidence_models.py    # Bounded project-evidence models and semantic stable IDs
|   |-- project_evidence_input.py     # Read-only evidence, change-memory, and project-fact adapters
|   |-- project_evidence_*.py         # Evidence normalization, synthesis, scoring, and persistence
|   |-- project_capability_*.py       # Capability taxonomy and deterministic extraction
|   |-- project_claim_boundaries.py   # Conservative allowed/forbidden claim boundaries
|   |-- capability_extractor.py
|   |-- tech_ontology.py     # Compact technology taxonomy and safe-claim helpers
|   |-- data/
|   |   `-- tech_ontology.jsonl
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
|-- tests/                   # Backend regression tests for resume, GitHub evidence, project memory, and privacy
`-- README.md
```

The system has seven main layers:

1. `backend/main.py`: local agent logic, model adapters, file tools, GitHub context extraction, and SQLite application tracking.
2. `backend/memory_store.py`: Chroma collections, deterministic local embeddings, similarity-aware writes, semantic retrieval, and legacy JSON migration.
3. `backend/tech_ontology.py` and `backend/data/tech_ontology.jsonl`: local technology-term matching, alias mapping, safe wording hints, and unsupported-claim guardrails for JD analysis, skills selection, and resume bullet validation.
4. `backend/evidence_memory.py`, `backend/evidence_pipeline.py`, and `backend/evidence_*.py`: GitHub evidence raw-source storage plus chunk, change-summary, evidence-card, and capability-fact processing for saved GitHub context.
5. `backend/project_change_memory.py` and `backend/project_change_pipeline.py`: project-change extraction from saved GitHub compare/file patches, with deterministic summaries, qualified evidence cards, capability facts, inspect views, and health checks.
6. `backend/project_evidence_models.py`, `backend/project_evidence_input.py`, `backend/project_evidence_*.py`, `backend/project_capability_*.py`, and `backend/project_claim_boundaries.py`: bounded project-evidence models, read-only adapters, deterministic normalization/deduplication, conservative fact synthesis, quality scoring, capability extraction, claim boundaries, and atomic project-evidence-memory persistence. This layer consumes GitHub evidence, optional project-change memory, and project facts without adding an HTTP route.
7. `backend/api_server.py` and `frontend/`: FastAPI plus the React + Vite Web UI for dashboard, job description, resume, cover letter, applications, interview prep, GitHub evidence, prompt settings, and chat pages.

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

Repository analysis can also cache compact per-project facts in `information/project_compact_facts.json`. The resume pipeline uses these cached facts to keep prompt payloads small while preserving key modules, technical stack, contribution signals, metric candidates, risk flags, and JD relevance notes.

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
- GitHub Evidence: configure GitHub identity/token, scan repositories from the tailored resume, base resume, and vector memory by default, fetch approved context into Chroma, review saved repository evidence status, and run the unified Evidence Processing Pipeline for GitHub evidence chunks, change summaries, evidence cards, and capability facts when `USE_GITHUB_EVIDENCE_MEMORY=1`. The backend also exposes bounded context preview/raw inspect APIs and project change memory diff-memory APIs, but the current page keeps the UI focused on summary views instead of rendering full raw content.
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
- `POST /api/agent/progress-guidance`
- `POST /api/agent/cancel`
- `POST /api/agent-tasks/start`
- `GET /api/agent-tasks/{task_id}/status`
- `POST /api/agent-tasks/{task_id}/message`
- `POST /api/agent-tasks/{task_id}/cancel`
- `GET /api/agent-tasks/{task_id}/result`
- `POST /api/chat/session`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/resume/star-check`
- `POST /api/resume/star-fact`
- `POST /api/resume/update-memory`
- `POST /api/resume/pdf-to-latex`
- `POST /api/resume/tailored/pdf`
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `POST /api/github/scan`
- `GET /api/github/config`
- `POST /api/github/config`
- `POST /api/github/context`
- `GET /api/github/context/status`
- `GET /api/github/evidence/status`
- `GET /api/github/evidence/preview`
- `POST /api/github/evidence/build`
- `GET /api/github/evidence/inspect`
- `GET /api/github/evidence/health`
- `POST /api/github/evidence/chunk`
- `GET /api/github/evidence/chunks/preview`
- `POST /api/github/evidence/summarize-changes`
- `GET /api/github/evidence/change-summaries/preview`
- `POST /api/github/evidence/build-evidence-cards`
- `GET /api/github/evidence/evidence-cards/preview`
- `POST /api/github/evidence/build-capability-facts`
- `GET /api/github/evidence/capability-facts/preview`
- `POST /api/github/change-memory/build`
- `GET /api/github/change-memory/inspect`
- `GET /api/github/change-memory/health`
- `GET /api/github/context/preview`
- `GET /api/github/context/raw`
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

`POST /api/agent/cancel` cancels an in-flight agent request by `agent_task_id`. The `/api/agent-tasks/*` endpoints start background agent work, poll stage/status messages, append user guidance while the task is active, cancel the task, and fetch the final result after completion.

`GET /api/status` includes `file_metadata` timestamps for local working files. The frontend uses those timestamps to avoid showing stale generated resume, cover letter, and interview prep outputs from before the current app session.

`GET /api/output-file`, `POST /api/output-file/launch`, and `DELETE /api/output-file` let the frontend read, open with the desktop default application, or delete generated output files such as tailored resume text versions and exported PDF versions. Output history lists readable file names rather than timestamps and omits reserved current-working files. When analysis, tailored resume, cover letter, or interview prep generation produces unchanged content for the same company and role, WorkAgent reuses the matching history file instead of creating a duplicate numbered version.

`POST /api/resume/tailor` accepts `allow_project_selection`, `allow_experience_removal`, and `include_application_hint`. Experience bullet tailoring is enabled by default, while removing an entire Experience entry is disabled unless the user explicitly enables it. The staged tailoring path also returns `role_profile`, `jd_requirements`, `project_ranking`, `project_section_validation`, and `gap_report` so the UI can show role classification, extracted requirements, selected/omitted projects, one-page allocation checks, missing evidence, weak bullet candidates, and unsafe unsupported keywords. The pipeline enriches JD requirements with the local tech ontology, but JD-only and ontology-only terms are moved to the gap report instead of Technical Skills unless user-specific evidence supports them. When `include_application_hint` is true, the response can include extracted `company`, `role`, `link`, and `notes` values for creating an application record.

`POST /api/resume/star-check` reviews Project Memory and staged project candidates for missing STAR facts before tailoring. `POST /api/resume/star-fact` saves a user-provided project fact so later resume generation can stay evidence-grounded instead of inventing unsupported claims.

`POST /api/resume/pdf-to-latex` converts an uploaded PDF resume into editable LaTeX and saves it as the base resume. `POST /api/resume/tailored/pdf` compiles the current tailored LaTeX resume into a PDF output file. PDF export now guards `glyphtounicode`/`pdfgentounicode` commands for engine compatibility and tries PDF-oriented LaTeX commands first when the resume requests those settings.

`POST /api/cover-letter/generate` also accepts `include_application_hint` and can return the same extracted application fields.

`GET /api/github/context/status` reports the structured GitHub context workspace state and source counts for the current GitHub context implementation. `GET /api/github/context/preview` returns bounded raw-source and derived-record previews, and `GET /api/github/context/raw` returns a bounded raw-source payload by `source_id`.

GitHub evidence memory is controlled by `USE_GITHUB_EVIDENCE_MEMORY`. When enabled, GitHub context sync persists raw sources into JSONL storage under `information/github_evidence_memory/`, and the GitHub evidence pipeline can turn those sources into chunks, raw change summaries, evidence cards, and capability facts. `GET /api/github/evidence/status`, `health`, and `inspect` report counts, project summaries, missing stages, safe samples, and the next recommended action. `POST /api/github/evidence/build` runs the full ordered pipeline, while the stage-specific endpoints run or preview individual stages.

Project change memory is controlled by `USE_PROJECT_CHANGE_MEMORY`. When enabled, its pipeline reads saved GitHub compare/file patches, extracts deterministic diff units, derives raw change summaries, filters evidence cards to qualified claims, aggregates capability facts, and persists the result to `information/project_change_memory.json`. `POST /api/github/change-memory/build` runs the pipeline, `GET /api/github/change-memory/inspect` returns bounded per-project samples and capability types, and `GET /api/github/change-memory/health` reports whether the saved memory is ready, empty, degraded, or disabled.

Project evidence memory is controlled by `USE_PROJECT_EVIDENCE_MEMORY` and persists to `information/project_evidence_memory.json` with schema `project_evidence_memory.v1`. Its Python pipeline reads GitHub evidence JSONL, optional project-change memory, `project_memory.json`, and `project_compact_facts.json`; then it normalizes, synthesizes, scores, extracts capabilities, builds claim boundaries, and atomically validates the resulting memory. It has no `/api/project-evidence/*` route or frontend trigger. Invalid or incomplete optional inputs become sorted warnings; unsupported metrics and capability inferences remain outside synthesized claims.

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
- `information/project_compact_facts.json`
- `information/github_evidence_memory/`
- `information/project_change_memory.json`
- `information/project_evidence_memory.json`
- `information/chroma/`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `outputs/`

Do not commit API keys, resumes, job descriptions, GitHub identities, generated documents, application records, or personal background notes.
Safe templates such as `.env.example` and `background/prompt.example.txt` remain trackable. `.gitignore` prevents future accidental tracking; it does not remove sensitive data from existing Git history.
The repository privacy policy also ignores credentials, uploads, generated exports, runtime databases, logs, local model caches, and frontend dependencies, while allowing documentation, safe templates, source code, and synthetic test fixtures. `tests/test_repository_privacy_policy.py` checks representative ignore rules and verifies that no tracked path is also ignored.

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

Resume quality-gate regression tests:

```powershell
python -m unittest tests\test_resume_quality_gates.py
```

Backend regression tests:

```powershell
python -m pytest tests -q
```

Project-change and project-evidence tests:

```powershell
$projectEvidenceTests = Get-ChildItem tests -File | Where-Object { $_.Name -match 'project_evidence|project_capability|project_claim|project_change' } | ForEach-Object { $_.FullName }
python -m pytest $projectEvidenceTests -q
```

Repository privacy-policy regression test:

```powershell
python -m pytest tests\test_repository_privacy_policy.py -q
```

Frontend production build:

```powershell
cd frontend
npm run build
```

Production frontend output is written to `outputs/frontend/`.

## Current Limitations

- Long generation tasks can run in the background and be cancelled, but there is still no token-by-token streaming output.
- GitHub evidence currently focuses on status summaries, bounded previews, and pipeline results rather than a fully polished visual explorer for all GitHub evidence and project change memory records.
- Project evidence memory is persisted and inspectable through Python backend interfaces, but it is not exposed through FastAPI or rendered in the GitHub Evidence page.
- Resume and cover letter editing has no built-in document preview or DOCX export; tailored resumes can be exported to PDF when a LaTeX toolchain is installed.
- The app is local-first and single-user; it has no login, multi-user isolation, or cloud deployment model.

## Roadmap

- Expand the Agent Chat application-material flow to include job analysis and interview prep.
- Add persistent task queues and WebSocket/SSE streaming.
- Add structured GitHub evidence visualization.
- Define an explicit production contract before connecting project evidence claim boundaries to generation or UI flows.
- Add application dashboards, statistics, batch actions, and richer search.
- Add document preview and DOCX export.
- Improve mobile layout and add dark mode.

## 中文

WorkAgent 是一个本地运行、面向单用户的 AI 求职工作台。它把简历、职位描述、个人背景、GitHub 证据、生成文档、面试准备和投递记录串成一个完整流程。

项目的目标是生成真实、保守、可验证的求职材料。它帮助你组织和定制已有经历，不应该编造学历、指标、公司经历、奖项、项目所有权、API、部署细节或来源材料中没有的技术。

## 目录

- [功能概览](#功能概览)
- [架构](#架构)
- [模型配置](#模型配置)
- [GitHub 证据配置](#github-证据配置)
- [向量记忆](#向量记忆)
- [Prompt 个性化](#prompt-个性化)
- [Web UI 页面](#web-ui-页面)
- [API 接口](#api-接口)
- [本地文件与隐私](#本地文件与隐私)
- [最低环境配置要求](#最低环境配置要求)
- [启动方式](#启动方式)
- [CLI 用法](#cli-用法)
- [开发检查](#开发检查)
- [当前限制](#当前限制)
- [Roadmap](#roadmap-1)

## 功能概览

- 分析已保存的职位描述，提取岗位要求、技能、职责、隐含期望和匹配度。
- 编辑基础简历，并为当前岗位生成定制版 LaTeX 简历。
- 允许 Agent 根据岗位选择最强且真实的项目组合：移除较弱项目、更新已有项目 bullet，或加入 Project Memory 中更匹配的项目。定制版 Projects 部分优先保持 2 个项目，只有第三个项目对岗位很关键时才使用 3 个，并给排名更高的项目更多 bullet 空间。
- 通过更严格的 ReAct bullet writer 定制 Project 和 Experience bullet；如果 bullet 只是在堆技术栈、描述 CRUD、UI 控件或宽泛模块，而没有具体实现方法、实质工作流能力和价值，会被拒绝。
- 允许 Agent 根据职位描述重排、改写或删除 Experience 中较弱和重复的 bullet，同时保持事实含义不变。删除整段 Experience 经历需要用户显式授权。
- 保留基础简历紧凑的 Technical Skills LaTeX 样式，清理生成技能中的流程/功能描述短语，把真实且有用户证据支持的技能重新归类到预期类别，并拒绝可见 bullet、占位符、泛泛填充内容、重复项或无证据支持的技能片段。
- 使用紧凑技术本体识别 JD 技术、别名、谨慎简历措辞和 unsupported claim 风险，但不会把本体术语当作用户经验的证据。
- 在保存定制版 LaTeX 前校验最终合并结果，检查项目顺序、bullet 预算、Technical Skills 证据、unsupported skills、summary 质量，以及 bullet 是否保留具体机制深度。质量闸门结果会以结构化 issue 返回，包含 source、severity、code、message 和 repairable 元数据。
- 返回分阶段简历定制摘要，包括角色画像、提取出的 JD 要求、项目排名、项目区块校验和 gap report，用于展示缺失或较弱证据以及应避免的 unsupported keywords。
- 把较长的 agent 工作作为可取消的后台任务运行，支持状态、消息、结果读取和重新运行时的实时指导补充。
- 在生成前检查 Project Memory 是否有足够 STAR 证据，并保存用户补充的项目事实。
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
|   |-- evidence_memory.py   # GitHub 证据 JSONL 存储
|   |-- evidence_pipeline.py # GitHub evidence 证据构建编排和检查
|   |-- evidence_*.py        # GitHub evidence 分块、变更摘要、证据卡辅助模块
|   |-- project_change_memory.py # project change memory schema、提取和持久化
|   |-- project_change_pipeline.py    # project change memory pipeline、inspect 和 health 辅助逻辑
|   |-- capability_extractor.py
|   |-- tech_ontology.py     # 紧凑技术 taxonomy 和安全声明辅助逻辑
|   |-- data/
|   |   `-- tech_ontology.jsonl
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

系统主要分为六层：

1. `backend/main.py`：本地 agent 逻辑、模型适配器、文件工具、GitHub 上下文提取和 SQLite 投递记录。
2. `backend/memory_store.py`：Chroma collections、确定性的本地向量化、写入前相似度对比、语义检索和旧 JSON 自动迁移。
3. `backend/tech_ontology.py` 和 `backend/data/tech_ontology.jsonl`：本地技术术语匹配、别名映射、安全措辞提示，以及 JD 分析、技能选择和 resume bullet 校验中的 unsupported claim 防护。
4. `backend/evidence_memory.py`、`backend/evidence_pipeline.py` 和 `backend/evidence_*.py`：保存 GitHub evidence raw source，并把已保存的 GitHub context 继续处理成 chunks、change summaries、evidence cards 和 capability facts。
5. `backend/project_change_memory.py` 和 `backend/project_change_pipeline.py`：从已保存的 GitHub compare/file patch 提取 project change memory，生成确定性的变更摘要、合格证据卡片、能力事实，以及 inspect 和 health 结果。
6. `backend/api_server.py` 和 `frontend/`：提供 Web UI 使用的 FastAPI 接口，以及包含仪表盘、职位描述、简历、求职信、投递记录、面试准备、GitHub 证据、Prompt 设置和聊天页面的 React + Vite 前端。

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

仓库分析也可以把每个项目的紧凑事实缓存到 `information/project_compact_facts.json`。简历流程会用这些缓存控制 prompt 大小，同时保留关键模块、技术栈、个人贡献信号、指标候选、风险提示和 JD 相关性说明。

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
- GitHub Evidence：配置 GitHub 身份/Token，默认从定制简历、基础简历和向量记忆扫描仓库，把已确认的上下文写入 Chroma，查看已保存仓库证据状态，并在 `USE_GITHUB_EVIDENCE_MEMORY=1` 时通过统一 Evidence Processing Pipeline 手动构建 GitHub evidence 的 chunks、change summaries、evidence cards 和 capability facts。后端同时保留受限的 context preview/raw inspect API 和 project change memory API，但当前页面以摘要结果为主，不直接渲染完整 raw 内容。
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
- `POST /api/agent/progress-guidance`
- `POST /api/agent/cancel`
- `POST /api/agent-tasks/start`
- `GET /api/agent-tasks/{task_id}/status`
- `POST /api/agent-tasks/{task_id}/message`
- `POST /api/agent-tasks/{task_id}/cancel`
- `GET /api/agent-tasks/{task_id}/result`
- `POST /api/chat/session`
- `POST /api/job-description`
- `POST /api/job-description/analyze`
- `POST /api/resume/tailor`
- `POST /api/resume/star-check`
- `POST /api/resume/star-fact`
- `POST /api/resume/update-memory`
- `POST /api/resume/pdf-to-latex`
- `POST /api/resume/tailored/pdf`
- `POST /api/cover-letter/generate`
- `POST /api/interview-prep/generate`
- `POST /api/github/scan`
- `GET /api/github/config`
- `POST /api/github/config`
- `POST /api/github/context`
- `GET /api/github/context/status`
- `GET /api/github/evidence/status`
- `GET /api/github/evidence/preview`
- `POST /api/github/evidence/build`
- `GET /api/github/evidence/inspect`
- `GET /api/github/evidence/health`
- `POST /api/github/evidence/chunk`
- `GET /api/github/evidence/chunks/preview`
- `POST /api/github/evidence/summarize-changes`
- `GET /api/github/evidence/change-summaries/preview`
- `POST /api/github/evidence/build-evidence-cards`
- `GET /api/github/evidence/evidence-cards/preview`
- `POST /api/github/evidence/build-capability-facts`
- `GET /api/github/evidence/capability-facts/preview`
- `POST /api/github/change-memory/build`
- `GET /api/github/change-memory/inspect`
- `GET /api/github/change-memory/health`
- `GET /api/github/context/preview`
- `GET /api/github/context/raw`
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

`POST /api/resume/tailor` 接受 `allow_project_selection`、`allow_experience_removal` 和 `include_application_hint`。Experience bullet 默认允许定制，但整段 Experience 经历默认不会删除，只有用户显式开启后才允许移除。分阶段定制流程还会返回 `role_profile`、`jd_requirements`、`project_ranking`、`project_section_validation` 和 `gap_report`，用于展示角色分类、提取出的 JD 要求、选中/省略的项目、一页简历分配检查、缺失证据、较弱 bullet 候选和应避免的 unsupported keywords。流程会用本地技术本体增强 JD 要求，但只有用户材料支持的技术才能进入 Technical Skills；仅来自 JD 或本体的术语会进入 gap report。`include_application_hint` 为 true 时，响应可以包含用于创建投递记录的 `company`、`role`、`link` 和 `notes` 字段。

`POST /api/resume/star-check` 会在定制前检查 Project Memory 和分阶段项目候选是否缺少 STAR 事实。`POST /api/resume/star-fact` 会保存用户补充的项目事实，让后续简历生成继续基于证据，而不是编造 unsupported claims。

`POST /api/resume/pdf-to-latex` 会把上传的 PDF 简历转换为可编辑 LaTeX，并保存为基础简历。`POST /api/resume/tailored/pdf` 会把当前定制版 LaTeX 简历编译成 PDF 输出文件。PDF 导出会为 `glyphtounicode`/`pdfgentounicode` 命令加上引擎兼容保护，并在简历需要这些设置时优先尝试 PDF-oriented LaTeX 编译命令。

`POST /api/cover-letter/generate` 也接受 `include_application_hint`，并可以返回同样的投递记录字段。

`GET /api/github/context/status` 会返回当前 GitHub context 结构化 GitHub context 工作区状态和来源计数。`GET /api/github/context/preview` 会返回受限数量的 raw sources 和派生记录预览，`GET /api/github/context/raw` 会按 `source_id` 返回受限长度的 raw source 内容。

GitHub evidence memory 由 `USE_GITHUB_EVIDENCE_MEMORY` 控制。启用后，GitHub context 同步会把 raw sources 持久化到 `information/github_evidence_memory/` 下的 JSONL 存储，GitHub evidence pipeline 可以继续把这些来源构建为 chunks、raw change summaries、evidence cards 和 capability facts。`GET /api/github/evidence/status`、`health` 和 `inspect` 会返回计数、项目摘要、缺失阶段、安全样本和下一步建议；`POST /api/github/evidence/build` 运行完整有序 pipeline，阶段专用接口则用于单独运行或预览某个阶段。

Project change memory 由 `USE_PROJECT_CHANGE_MEMORY` 控制。启用后，project change memory pipeline 会读取已保存的 GitHub compare/file patch，提取确定性的 diff units，生成 raw change summaries，过滤出合格的 evidence cards，聚合 capability facts，并把结果写入 `information/project_change_memory.json`。`POST /api/github/change-memory/build` 用于运行该 pipeline，`GET /api/github/change-memory/inspect` 返回按项目裁剪后的样例和 capability types，`GET /api/github/change-memory/health` 返回当前 project change memory 是否 ready、empty、degraded 或 disabled。

Project evidence memory 由 `USE_PROJECT_EVIDENCE_MEMORY` 控制，并按 `project_evidence_memory.v1` schema 持久化到 `information/project_evidence_memory.json`。它从 GitHub evidence JSONL、可选的 project-change memory、`project_memory.json` 和 `project_compact_facts.json` 只读读取输入，然后执行规范化、合成、质量评分、能力抽取、claim boundary 构建和原子校验；当前没有 `/api/project-evidence/*` 接口或前端触发入口。无效或不完整的可选输入会变成排序后的 warning，不支持的指标和能力推断不会进入合成声明。

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
- `information/project_compact_facts.json`
- `information/github_evidence_memory/`
- `information/project_change_memory.json`
- `information/project_evidence_memory.json`
- `information/chroma/`
- `information/github_accounts.txt`
- `information/applications.sqlite3`
- `background/prompt.txt`
- `outputs/`

不要提交 API Key、简历、职位描述、GitHub 身份、生成文档、投递记录或个人背景资料。
`.env.example` 和 `background/prompt.example.txt` 等安全模板仍可被 Git 跟踪。`.gitignore` 只能防止以后误提交，不能从已有 Git 历史中删除敏感数据。
仓库级隐私规则还会忽略凭据、上传文件、生成导出物、运行时数据库、日志、本地模型缓存和前端依赖，同时保留文档、安全模板、源代码和合成测试 fixture 的可跟踪性。`tests/test_repository_privacy_policy.py` 会检查代表性忽略规则，并确认没有“已跟踪且同时被忽略”的路径。

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

后端回归测试：

```powershell
python -m pytest tests -q
```

仓库隐私策略回归测试：

```powershell
python -m pytest tests\test_repository_privacy_policy.py -q
```

前端生产构建：

```powershell
cd frontend
npm run build
```

前端生产构建输出会写入 `outputs/frontend/`。

## 当前限制

- 较长生成任务可以在后台运行并取消，但还没有 token-by-token 流式输出。
- GitHub 证据目前主要以 JSON 展示，还没有完整的结构化可视化报告。
- 简历和求职信没有内置文档预览或 DOCX 导出；安装 LaTeX 工具链后可以把定制简历导出为 PDF。
- 项目是本地优先、单用户设计，没有登录、多用户隔离或云端部署模型。

## Roadmap

- 扩展 Agent Chat 求职材料流程，加入职位分析和面试准备。
- 增加持久任务队列和 WebSocket/SSE 流式输出。
- 增加结构化 GitHub 证据可视化。
- 增加投递统计、批量操作和更丰富的搜索。
- 增加文档预览和 DOCX 导出。
- 改进移动端布局并增加深色模式。
