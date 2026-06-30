"""FastAPI HTTP layer for WorkAgent frontend."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import APIStatusError
from pydantic import BaseModel, Field

import main as agent

app = FastAPI(title="WorkAgent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SHUTDOWN_GRACE_SECONDS = 2.0
shutdown_timer: Optional[threading.Timer] = None
shutdown_lock = threading.Lock()
CHAT_SESSION_OUTPUT_DIR = agent.OUTPUT_DIR / "chat_sessions"
JOB_ANALYSIS_HISTORY_PATH = agent.INFORMATION_DIR / "job_analysis_history.json"
MAX_JOB_ANALYSIS_HISTORY = 20
chat_session_lock = threading.Lock()
TAILORED_RESUME_PDF_OUTPUT_DIR = agent.OUTPUT_DIR / "tailored_resume_pdfs"
LATEX_BUILD_DIR = agent.OUTPUT_DIR / "latex_build"
current_agent_task_id: ContextVar[str] = ContextVar("current_agent_task_id", default="")
agent_task_lock = threading.Lock()
agent_task_cancellations: dict[str, threading.Event] = {}
agent_task_adapters: dict[str, list[Any]] = {}
background_task_lock = threading.Lock()
background_agent_tasks: dict[str, dict[str, Any]] = {}

FILE_MAP = {
    "resume": agent.RESUME_PATH,
    "tailored_resume": agent.OUTPUT_RESUME_PATH,
    "job_description": agent.JOB_DESCRIPTION_PATH,
    "cover_letter": agent.COVER_LETTER_PATH,
    "interview_prep": agent.INTERVIEW_PREP_PATH,
    "memory": agent.MEMORY_PATH,
    "project_memory": agent.PROJECT_MEMORY_PATH,
    "github_accounts": agent.GITHUB_ACCOUNTS_PATH,
}

INTERVIEW_PREP_PROMPT = """
Generate interview preparation notes for the saved job description.

Requirements:
- Read job_description.txt, tailored_resume.txt (fallback to resume.txt), and Chroma profile memory.
- Include likely technical questions, behavioral/STAR prompts, project talking points, and gaps to prepare for.
- Keep claims grounded in the resume and job description.
- Return the complete interview preparation notes directly.
"""

RESUME_TAILOR_PROMPT = """
Based on the saved job_description.txt, project_memory.json, resume.txt, and approved GitHub context if useful,
generate the modified complete LaTeX resume code.
Hard source hierarchy:
1. Project Memory is the primary source of project truth.
2. RAG GitHub evidence is an evidence library only.
3. Do not write resume bullets directly from retrieved GitHub evidence.
4. First use project_memory.json to decide what each project is, what problem it solves,
   the technical stack, core workflow, confirmed implemented features, and real metrics.
5. For each relevant Project Memory project, map it one-to-one to Chroma github_evidence via project_id,
   project_name, repository, rag_refs, and semantic query.
6. Use mapped GitHub evidence only for per-project code details, files, commits, diffs, and proof.
Compare the projects currently listed in resume.txt with the factual projects available in Project Memory.
Choose the strongest project mix for the saved job description: you may remove a weaker resume project,
update an existing project's bullets, or add a better-matching memory project that is not currently in the resume.
If WorkAgent is relevant to the saved job description, treat it as a high-priority project because Project Memory
identifies it as a local AI job application workspace connecting job analysis, tailored resumes, cover letters,
interview preparation, memory, GitHub evidence, model configuration, and application tracking.
Prefer a one-page resume. The Projects section should normally keep 2 projects, may keep 3 only when the
third project is clearly job-critical, and must not keep more than 3. Higher-ranked projects should receive
more bullets than lower-ranked projects: about 3 bullets for rank 1, 2 bullets for rank 2, and 1 concise
bullet for rank 3.
Tailor the Experience section for the saved job description: you may reorder factual bullets, rewrite bullets
for relevance and clarity, and remove weaker or redundant bullets. Preserve the factual meaning of the source resume.
Keep every existing Experience entry unless the user explicitly allows removing entire Experience entries.
Repository links in Chroma profile memory may be used only as candidates for approved GitHub evidence.
Do not invent claims, technologies, metrics, responsibilities, employers, roles, dates, or repository facts.

Resume Bullet Writing Rules:
- Project and Experience bullet wording must come from the mandatory ReAct bullet writer tool/process.
- Every Project and Experience bullet must be STAR-grounded: Situation/problem or context, Task/personal ownership,
  Action/technical implementation, and Result/verified impact. If all four cannot fit naturally, preserve the
  strongest supported parts and never invent the missing parts.
- The writer must first reason why the fact can be written, why it belongs in the project or experience,
  what business capability it demonstrates, and what technical capability it demonstrates.
- Final bullets should use a strong action verb plus concrete technical method, substantive
  logic/business capability, and result or value. Natural phrasing is allowed; do not force
  every sentence into the same "to ..." template.
- The technical method must be a concrete implementation approach, algorithm, data flow, debugging
  approach, or system mechanism. A technology stack alone is not a technical method.
- Never invent business scale, ownership level, QPS, latency, cost, users, accuracy, deployment,
  production status, or before/after metrics. Use quantified results only when supported by resume,
  Project Memory, GitHub evidence, logs/code/README, or explicit user guidance.
- If metrics are missing and the user said no data/unknown, write conservative concrete results
  such as reducing repetitive manual review or improving maintainability without fake percentages.
- Before writing any project bullet, read the selected project's Project Memory fields in this order:
  1. identity.positioning
  2. identity.core_problem
  3. identity.core_value
  4. workflows
  5. tech_stack and confirmed_features as supporting details
- The first bullet for each selected project must explain what the project is and what workflow or problem it addresses.
- Bullets should follow this order of importance:
  1. problem / purpose
  2. workflow or system behavior
  3. technical implementation
  4. metrics, only if explicitly provided
- Do not write bullets that only describe storage, CRUD, file handling, framework usage, or generic implementation.
- Do not describe shallow UI or broad module work as the achievement, such as "added buttons",
  "built a page", or "implemented a dashboard", unless the bullet explains the underlying
  workflow, decision logic, validation, routing, matching, recovery, or quality-control capability.
- Avoid empty stack-only phrasing such as "with FastAPI, React, and SQLite"; name the actual method,
  such as vector retrieval, schema validation, state comparison, cache invalidation, chunked processing,
  retry handling, dependency parsing, ranking/matching logic, or error recovery.
- Prefer Used, Implemented, Automated, or Debugged as the bullet's main verb.
- Avoid vague verbs such as leveraged, utilized, facilitated, enabled, supported unless necessary.
- Technology names should support the story, not become the whole story.
- The Projects section should usually contain 2 projects and at most 3 projects.
- Allocate bullets by project rank instead of giving every project equal length: strongest project about 3 bullets,
  second project about 2 bullets, optional third project about 1 concise bullet.
- Each bullet should be concise, factual, and ATS-friendly.
- Never invent metrics, technologies, deployment, users, business impact, ownership, or performance claims.

Resume bullet examples:
Bad:
- Used SQLite to organize application records.
- Implemented file-based resume and job-description handling.
- Developed a FastAPI backend.
- Implemented a FastAPI, React, and SQLite application to manage resumes.
- Implemented resume buttons and pages to let users edit content.
- Built a resume generation module to create resumes.

Good:
- Automated chunked resume-tailoring pipelines with vector memory retrieval and LaTeX block merging, reducing repetitive resume generation while preserving factual project evidence.
- Implemented job-description keyword extraction and project-ranking logic for evidence-backed project selection and reduced unsupported resume claims.
- Used GitHub evidence compression and Project Memory mapping to prevent generic resume bullets and connect implementation details with ATS-friendly application positioning.

Return only LaTeX code with no Markdown fences and no analysis text.
Save with save_tailored_resume when complete.
"""

RESUME_BULLET_WRITER_PROMPT = """
You are the mandatory ReAct resume bullet writer for WorkAgent.

Use this mode for Project-section bullets and Experience-section bullets:
1. STAR Check: identify Situation, Task, Action, and Result evidence for each candidate bullet.
2. Think: why this fact can be written in the resume without exaggeration.
3. Reason: why this point belongs under this project or experience for the target job.
4. Act: identify the business capability or product/workflow value shown.
5. Act: identify the technical capability, stack, implementation method, bug fix, or feature delivery shown.
6. Write: produce final resume bullets.

Final bullet shape:
Use Used / Implemented / Automated / Debugged when natural, then name the concrete technical
method, the substantive logic/workflow/business capability, and the result or value. Natural
phrasing is allowed; do not force every bullet into an identical "to ..." sentence.
Prefer: action verb + project/module + technical action + verified result/impact.

Rules:
- Use only supported facts from the original resume, Project Memory, staged candidates, and approved evidence.
- Do not invent metrics, technologies, files, commits, dates, ownership, deployment, users, business impact, or performance claims.
- STAR is mandatory for analysis. Each final bullet should be backed by:
  Situation: problem, workflow, users, data, files, requests, applications, or business context.
  Task: the user's personal module/ownership level such as led, independently built, collaborated on, or maintained.
  Action: concrete technical design or implementation method.
  Result: verified quantitative result when supported; otherwise a conservative qualitative result.
- If Situation, Task, Action, or Result is missing, do not fabricate it. Use available user guidance if present.
  If the user explicitly said no data/unknown, write without fake numbers and keep the result conservative.
- Avoid final bullets that rely on vague phrases such as responsible for, worked on, helped with,
  participated in, familiar with, used technology to develop system, or improved performance without support.
- Prefer business-relevant technical writing: technology stack, technical solution, bug or feature implemented, and the business requirement or workflow value it supports.
- Do not write generic bullets that only say storage, CRUD, framework usage, or file handling.
- Do not make the implemented feature a shallow UI control or broad module. Write the underlying
  logic or capability, such as preventing monotonous resume generation, selecting relevant projects,
  validating supported claims, recovering from context-window overflow, or preserving factual evidence.
- The technical method must explain how the work was implemented, not just which stack was used.
- Prefer concrete methods such as algorithms, retrieval/ranking logic, parsing, schema validation,
  state comparison, cache invalidation, chunked processing, retry/fallback handling, data flow
  orchestration, or debugging/root-cause isolation when supported by evidence.
- Each bullet should include at least 3 of these 5 elements: concrete method/tool, implemented
  function/process, technical or process challenge, user/business/engineering value, and role-relevant keyword.
- Use the provided role profile, JD requirements, evidence card, role lens, allowed claims, and forbidden claims;
  do not infer stronger claims from vague evidence.
- Reject stack-only bullets like "Implemented X with FastAPI, React, and SQLite" unless the bullet
  also names the implementation method or system mechanism.
- If a result is qualitative, phrase it as workflow value, reliability, clarity, maintainability, relevance, or user/application-preparation value without inventing numbers.
- Return ONLY valid JSON.
"""

PROJECT_MEMORY_FROM_REPO_ANALYSIS_PROMPT = """
Update project_memory.json from repository analysis.

Architecture rule:
- Project Memory is the primary source of project truth.
- RAG GitHub evidence is an evidence library only.
- Resume bullets must not be written directly from GitHub evidence.

During GitHub extraction, the repository data splits into two outputs:
- Chroma github_evidence keeps the existing evidence records unchanged.
- project_memory.json receives a separate project analysis written from README, repository metadata,
  root files, languages, dependency/config filenames, and lightweight code/file summaries.

Do not summarize Chroma retrieval results into Project Memory. Use the repository analysis payload provided here.

Project Memory must answer, per project:
- What is this project?
- What background or user problem motivated it?
- What problem does it solve?
- What technical stack is confirmed?
- What is the core workflow?
- Which features are confirmed implemented?
- Which real metrics are confirmed?

GitHub evidence should answer only:
- Which commits, files, README sections, or diffs prove those facts?
- What changed recently?
- Whether a specific technical point has evidence.

Rules:
- Only add or update durable project facts supported by README, repository metadata, languages,
  root files, dependency/config filenames, and code/file summaries in the repository analysis payload.
- Commit messages and diff details may be referenced as evidence_notes or recent_changes, but they are not the source
  for resume bullet wording.
- Preserve existing Project Memory unless the repository analysis provides a clearer or more complete version of the same project fact.
- Keep unsupported metrics empty or omitted. Never invent product impact, scale, users, performance, or business results.
- Store project facts in project_memory["projects"] as objects using this schema when possible:
  {
    "project_id": stable lowercase id such as "workagent",
    "project_name": display name,
    "identity": {
      "project_type": confirmed category,
      "positioning": concise positioning,
      "target_user": confirmed or clearly implied user,
      "background": why the project exists, if supported,
      "core_problem": user/problem statement,
      "core_value": value delivered by the project
    },
    "tech_stack": array of confirmed technologies,
    "workflows": array of confirmed workflows,
    "confirmed_features": array of implemented features,
    "real_metrics": object or array of metrics only when evidence explicitly supports them,
    "recent_changes": array of recent evidence-backed changes,
    "rag_refs": {
      "collection": "github_evidence",
      "filter": {"project_id": same stable project id}
    },
    "evidence_notes": compact pointers to repositories, commits, files, README, or diffs
  }
- Keep the result compact and useful for future resume, cover letter, and interview prep tasks.
- Return only valid JSON with exactly these keys:
  "changed": boolean,
  "additions": array of short strings describing newly added or strengthened project facts,
  "project_memory": object
"""

PREFERRED_RESUME_PROJECTS = 2
MAX_STAGED_PROJECTS = 3
MAX_STAGED_TEXT_CHARS = 12000
MAX_PROMPT_FILES_PER_REPO = 12
MAX_PROMPT_DIFF_SIGNALS = 20
MAX_PROMPT_CLAIMS = 12
MAX_PROMPT_SIGNAL_CHARS = 240
MAX_PROMPT_FILE_SUMMARY_CHARS = 500
MAX_PROMPT_EVIDENCE_CHARS = 9000


class ProviderBody(BaseModel):
    provider: str


class ProviderConfigBody(BaseModel):
    provider: str
    api_key: str
    base_url: str = ""
    model: str = ""


class ModelBody(BaseModel):
    model: str


class FileBody(BaseModel):
    content: str


class PromptBody(BaseModel):
    content: str


class AgentImageBody(BaseModel):
    name: str = ""
    mime_type: str
    data_url: str


class ChatHistoryEntryBody(BaseModel):
    role: str
    text: str = ""
    images: list[AgentImageBody] = Field(default_factory=list)


class ChatSessionBody(BaseModel):
    session_id: str
    created_at: str = ""
    language: str = "zh"
    message: str = ""
    images: list[AgentImageBody] = Field(default_factory=list)
    attachment_error: str = ""
    history: list[ChatHistoryEntryBody] = Field(default_factory=list)


class ShutdownBody(BaseModel):
    chat_session: Optional[ChatSessionBody] = None


class AgentAskBody(BaseModel):
    message: str
    images: list[AgentImageBody] = Field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    language: str = "zh"
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class AgentProgressGuidanceBody(BaseModel):
    title: str = ""
    stage_label: str = ""
    user_message: str
    prior_messages: list[dict[str, Any]] = Field(default_factory=list)
    language: str = "zh"
    agent_task_id: str = ""


class AgentCancelBody(BaseModel):
    agent_task_id: str


class AgentTaskStartBody(BaseModel):
    task_id: str = ""
    taskType: str = ""
    task_type: str = ""
    body: dict[str, Any] = Field(default_factory=dict)
    language: str = "zh"


class AgentTaskMessageBody(BaseModel):
    content: str


class JobDescriptionBody(BaseModel):
    content: str


class AnalyzeBody(BaseModel):
    use_github_context: bool = False
    language: str = "zh"
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class TailorBody(BaseModel):
    use_github_context: bool = True
    allow_project_selection: bool = True
    allow_experience_removal: bool = False
    include_application_hint: bool = False
    language: str = "zh"
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class ResumeStarCheckBody(BaseModel):
    allow_project_selection: bool = True
    asked_question_keys: list[str] = Field(default_factory=list)
    language: str = "zh"
    agent_task_id: str = ""


class ResumeStarFactBody(BaseModel):
    project_id: str = ""
    project_name: str = ""
    field_type: str
    missing_info_type: str = ""
    question_key: str = ""
    raw_answer: str
    normalized_fact: str = ""
    confidence: str = "high"
    language: str = "zh"
    agent_task_id: str = ""


class ResumeMemoryBody(BaseModel):
    resume_source: str = "resume"
    project_name: str = ""
    project_id: str = ""
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class ResumePdfToLatexBody(BaseModel):
    filename: str = "resume.pdf"
    data_base64: str
    language: str = "zh"
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class TailoredResumePdfBody(BaseModel):
    content: str = ""


class CoverLetterBody(BaseModel):
    use_tailored_resume: bool = True
    use_github_context: bool = False
    style: str = "concise"
    include_application_hint: bool = False
    language: str = "zh"
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class InterviewPrepBody(BaseModel):
    use_github_context: bool = True
    language: str = "zh"
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class GitHubScanBody(BaseModel):
    resume_source: str = "resume"
    project_name: str = ""
    project_id: str = ""
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class GitHubContextBody(BaseModel):
    approved: bool = True
    resume_source: str = "resume"
    project_name: str = ""
    project_id: str = ""
    force_refresh: bool = False
    reanalyze_cached: bool = False
    agent_progress_messages: list[dict[str, Any]] = Field(default_factory=list)
    agent_task_id: str = ""


class GitHubConfigBody(BaseModel):
    usernames: list[str] = Field(default_factory=list)
    author_names: list[str] = Field(default_factory=list)
    author_emails: list[str] = Field(default_factory=list)
    token: str = ""


class ApplicationCreateBody(BaseModel):
    company: str
    role: str
    link: str = ""
    status: str = "Interested"
    applied_date: str = ""
    resume_version: str = ""
    cover_letter_version: str = ""
    notes: str = ""


class ApplicationUpdateBody(BaseModel):
    company: Optional[str] = None
    role: Optional[str] = None
    link: Optional[str] = None
    status: Optional[str] = None
    applied_date: Optional[str] = None
    resume_version: Optional[str] = None
    cover_letter_version: Optional[str] = None
    notes: Optional[str] = None


PROVIDER_CONFIGS = {
    "openai": {
        "label": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "default_base_url": "",
        "default_model": "gpt-5.5",
        "requires_base_url": False,
    },
    "openai-compatible": {
        "label": "OpenAI Compatible",
        "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
        "base_url_env": "OPENAI_COMPATIBLE_BASE_URL",
        "model_env": "OPENAI_COMPATIBLE_MODEL",
        "default_base_url": "",
        "default_model": os.getenv("OPENAI_MODEL", "gpt-5.5"),
        "requires_base_url": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
        "model_env": "DEEPSEEK_MODEL",
        "default_base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-pro",
        "requires_base_url": False,
    },
    "claude": {
        "label": "Claude / Anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "model_env": "ANTHROPIC_MODEL",
        "default_base_url": "https://api.anthropic.com",
        "default_model": "claude-sonnet-4-5",
        "requires_base_url": False,
    },
    "gemini": {
        "label": "Gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url_env": "GEMINI_BASE_URL",
        "model_env": "GEMINI_MODEL",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-2.5-flash",
        "requires_base_url": False,
    },
}


def normalize_provider(provider: str) -> str:
    normalized = provider.lower().strip()
    aliases = {"compatible": "openai-compatible", "anthropic": "claude", "google": "gemini"}
    return aliases.get(normalized, normalized)


def provider_supports_images(provider: str) -> bool:
    return normalize_provider(provider) != "deepseek"


def extract_model_api_error_message(error: Exception) -> str:
    if isinstance(error, APIStatusError):
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            nested = body.get("error")
            if isinstance(nested, dict) and nested.get("message"):
                return str(nested["message"])
        message = getattr(error, "message", None)
        if message:
            return str(message)
        return str(error)

    if isinstance(error, urllib.error.HTTPError):
        raw_body = ""
        if error.fp is not None:
            try:
                raw_body = error.read().decode("utf-8", errors="replace")
            except OSError:
                raw_body = ""
        if raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                return raw_body.strip() or error.reason or str(error)
            if isinstance(payload, dict):
                nested = payload.get("error")
                if isinstance(nested, dict) and nested.get("message"):
                    return str(nested["message"])
                if payload.get("message"):
                    return str(payload["message"])
                if payload.get("error"):
                    return str(payload["error"])
            return raw_body.strip()
        return error.reason or str(error)

    return str(error)


def raise_model_api_http_exception(error: Exception) -> None:
    if isinstance(error, APIStatusError):
        upstream_status = error.status_code or 502
        http_status = upstream_status if 400 <= upstream_status < 600 else 502
    elif isinstance(error, urllib.error.HTTPError):
        http_status = error.code if 400 <= error.code < 600 else 502
    else:
        raise error

    message = extract_model_api_error_message(error)
    raise HTTPException(
        status_code=http_status,
        detail=f"Model API error: {message}",
    ) from error


def is_context_window_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPException):
        detail = getattr(exc, "detail", "")
        status_code = getattr(exc, "status_code", None)
        text = str(detail).lower()
        return status_code == 400 and any(
            marker in text
            for marker in [
                "context_length_exceeded",
                "context_window_exceeded",
                "input exceeds the context window",
                "maximum context length",
                "too many tokens",
                "context window",
                "context length",
            ]
        )
    message = str(exc).lower()
    return any(
        marker in message
        for marker in [
            "context_length_exceeded",
            "context_window_exceeded",
            "input exceeds the context window",
            "maximum context length",
            "too many tokens",
        ]
    )


def quote_env_value(value: str) -> str:
    if not value:
        return ""
    if any(char.isspace() for char in value) or any(char in value for char in ['"', "'", "#", "="]):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def write_env_values(values: dict[str, str]) -> None:
    env_path = agent.INFORMATION_DIR / ".env"
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    pending = dict(values)
    updated_lines = []

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue

        key = line.split("=", 1)[0].strip()
        if key in pending:
            updated_lines.append(f"{key}={quote_env_value(pending.pop(key).strip())}")
        else:
            updated_lines.append(line)

    for key, value in pending.items():
        updated_lines.append(f"{key}={quote_env_value(value.strip())}")

    env_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")


def clean_string_list(values: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for raw_value in values:
        value = raw_value.strip()
        if not value:
            continue
        normalized = value.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(value)
    return cleaned


def write_github_identities(
    usernames: list[str],
    author_names: list[str],
    author_emails: list[str],
) -> None:
    lines = []
    for username in clean_string_list(usernames):
        lines.append(f"username: {username}")
    for author_name in clean_string_list(author_names):
        lines.append(f"name: {author_name}")
    for author_email in clean_string_list(author_emails):
        lines.append(f"email: {author_email}")
    agent.GITHUB_ACCOUNTS_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_github_config_status() -> dict[str, Any]:
    project_memory_mtime = (
        agent.PROJECT_MEMORY_PATH.stat().st_mtime
        if agent.PROJECT_MEMORY_PATH.exists()
        else None
    )
    return {
        "identities": agent.read_github_identities(),
        "token_configured": agent.github_token_is_configured(),
        "memory_repositories": agent.MEMORY_STORE.list_github_repositories(),
        "project_memory_updated_at": project_memory_mtime,
    }


def build_provider_config_status() -> dict[str, Any]:
    providers = []
    for key, config in PROVIDER_CONFIGS.items():
        providers.append(
            {
                "provider": key,
                "label": config["label"],
                "configured": bool(os.getenv(config["api_key_env"])),
                "base_url": os.getenv(config["base_url_env"], config["default_base_url"]),
                "model": os.getenv(config["model_env"], config["default_model"]),
                "default_base_url": config["default_base_url"],
                "default_model": config["default_model"],
                "requires_base_url": config["requires_base_url"],
                "supports_images": provider_supports_images(key),
            }
        )
    return {"providers": providers}


def get_adapter(provider: Optional[str] = None):
    name = normalize_provider(provider or agent.current_provider)
    try:
        return agent.create_model_adapter(name), name
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def output_application_metadata(path: Path, suffix: str) -> dict[str, str]:
    stem = path.name
    if suffix and stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    stem = re.sub(r"_(\d+)$", "", stem).strip()
    if re.match(r"^(tailored_resume|cover_letter|interview_prep|chat_session)(?:_|$)", stem):
        return {"company": "", "role": ""}
    parts = [part.strip() for part in stem.split("_") if part.strip()]
    if len(parts) < 2:
        return {"company": "", "role": ""}
    return {
        "company": parts[0],
        "role": " ".join(parts[1:]).strip(),
    }


def list_output_files(directory: Path, suffix: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    files = sorted(
        (
            path
            for path in directory.glob(f"*{suffix}")
            if path.stat().st_size > 0
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [
        {
            "name": path.name,
            "path": str(path),
            "generated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "generated_at_ms": int(path.stat().st_mtime * 1000),
            **output_application_metadata(path, suffix),
        }
        for path in (files[:limit] if limit is not None else files)
    ]


def resolve_output_file(raw_path: str) -> Path:
    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
        output_root = agent.OUTPUT_DIR.resolve(strict=True)
        path.relative_to(output_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Output file not found.") from error
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return path


def remove_analysis_history_path(path: Path) -> None:
    resolved_path = str(path.resolve())
    entries = [
        entry
        for entry in read_job_analysis_history()
        if str(Path(str(entry.get("analysis_path", ""))).resolve()) != resolved_path
    ]
    write_job_analysis_history(entries)


def clean_history_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def job_hash(job_description: str) -> str:
    normalized = re.sub(r"\s+", " ", job_description).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def job_history_key(company: str, role: str, description_hash: str) -> str:
    if company and role:
        normalized = re.sub(r"\s+", " ", f"{company}:{role}").strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"description:{description_hash}"


def job_history_display_name(entry: dict[str, Any]) -> str:
    company = clean_history_text(entry.get("company"), "未知公司")
    role = clean_history_text(entry.get("role"), "未知职位")
    updated_at = clean_history_text(entry.get("updated_at_display"), "")
    return f"{company}：{role}，{updated_at}" if updated_at else f"{company}：{role}"


def read_job_analysis_history() -> list[dict[str, Any]]:
    if not JOB_ANALYSIS_HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(JOB_ANALYSIS_HISTORY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def write_job_analysis_history(entries: list[dict[str, Any]]) -> None:
    JOB_ANALYSIS_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOB_ANALYSIS_HISTORY_PATH.write_text(
        json.dumps(entries[:MAX_JOB_ANALYSIS_HISTORY], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_job_analysis_history(
    company: str,
    role: str,
    job_description: str,
    analysis_path: Path,
) -> dict[str, Any]:
    description_hash = job_hash(job_description)
    key = job_history_key(company, role, description_hash)
    saved_at = datetime.now().astimezone()
    updated_at = saved_at.isoformat(timespec="seconds")
    updated_at_display = saved_at.strftime("%Y-%m-%d %H:%M")
    entries = read_job_analysis_history()

    existing = None
    remaining = []
    for entry in entries:
        same_key = entry.get("key") == key
        same_job = bool(company and role and entry.get("company") == company and entry.get("role") == role)
        same_description = entry.get("job_hash") == description_hash
        if existing is None and (same_key or same_job or same_description):
            existing = entry
        else:
            remaining.append(entry)

    entry = {
        **(existing or {}),
        "key": key,
        "job_hash": description_hash,
        "company": clean_history_text(company, "未知公司"),
        "role": clean_history_text(role, "未知职位"),
        "updated_at": updated_at,
        "updated_at_display": updated_at_display,
        "analysis_path": str(analysis_path),
    }
    entry.setdefault("created_at", updated_at)
    updated_entries = sorted([entry, *remaining], key=lambda item: item.get("updated_at", ""), reverse=True)
    write_job_analysis_history(updated_entries)
    return entry


def list_job_analysis_history(limit: int = MAX_JOB_ANALYSIS_HISTORY) -> list[dict[str, str]]:
    return [
        {
            "name": job_history_display_name(entry),
            "path": str(entry.get("analysis_path", "")),
            "company": clean_history_text(entry.get("company"), "未知公司"),
            "role": clean_history_text(entry.get("role"), "未知职位"),
            "updated_at": clean_history_text(entry.get("updated_at"), ""),
        }
        for entry in read_job_analysis_history()[:limit]
    ]


def read_file_content(name: str) -> tuple[bool, str]:
    if name == "memory":
        content = agent.read_memory()
        return bool(agent.MEMORY_STORE.profile_count()), content

    if name == "project_memory":
        content = agent.read_project_memory()
        return agent.file_is_ready(agent.PROJECT_MEMORY_PATH), content

    if name == "tailored_resume":
        try:
            content = agent.read_tailored_resume()
        except (FileNotFoundError, ValueError):
            return False, ""
        return bool(content.strip()), content

    path = FILE_MAP[name]
    if not path.exists():
        return False, ""
    content = path.read_text(encoding="utf-8")
    ready = agent.file_is_ready(path)
    return ready, content


def file_ready(name: str, path: Path) -> bool:
    if name == "memory":
        return bool(agent.MEMORY_STORE.profile_count())
    if name == "project_memory":
        return agent.file_is_ready(path)
    if name == "tailored_resume":
        return agent.file_is_ready(agent.latest_tailored_resume_path())
    return agent.file_is_ready(path)


def save_file_content(name: str, content: str) -> None:
    if name == "memory":
        try:
            memory = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Memory must be a valid JSON object.") from error
        if not isinstance(memory, dict):
            raise ValueError("Memory must be a valid JSON object.")
        agent.replace_profile_memory(memory, source="web-memory-editor")
        return
    if name == "project_memory":
        try:
            project_memory = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Project Memory must be a valid JSON object.") from error
        if not isinstance(project_memory, dict):
            raise ValueError("Project Memory must be a valid JSON object.")
        if "projects" not in project_memory and not project_memory.get("project_id"):
            raise ValueError("Project Memory must include a projects array or one project object.")
        if project_memory.get("project_id"):
            project_memory = {
                "version": 1,
                "source": "manual-project-memory-editor",
                "projects": [project_memory],
            }
        agent.write_project_memory_file(project_memory)
        return
    if name == "tailored_resume":
        hint = resolve_saved_application_hint()
        agent.save_tailored_resume(content, company=hint["company"], role=hint["role"])
        return
    if name == "cover_letter":
        hint = resolve_saved_application_hint()
        agent.save_cover_letter(content, company=hint["company"], role=hint["role"])
        return
    if name == "interview_prep":
        hint = resolve_saved_application_hint()
        agent.save_interview_prep(content, company=hint["company"], role=hint["role"])
        return
    agent.write_text_file(FILE_MAP[name], content)


def read_prompt_example() -> str:
    example_path = agent.BACKGROUND_DIR / "prompt.example.txt"
    if not example_path.exists():
        return ""
    return example_path.read_text(encoding="utf-8")


def pids_listening_on_port(port: int) -> set[int]:
    if os.name != "nt":
        return set()

    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    pids = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address = parts[1]
        state = parts[3].upper()
        if state != "LISTENING" or not local_address.endswith(f":{port}"):
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return pids


def frontend_process_pids() -> set[int]:
    if os.name != "nt":
        return set()

    return process_pids_for_workspace_command(
        str(agent.ROOT_DIR / "frontend"),
        ["*npm run dev*", "*vite*5173*"],
    )


def backend_process_pids() -> set[int]:
    if os.name != "nt":
        return set()

    return process_pids_for_workspace_command(
        str(agent.ROOT_DIR / "backend"),
        ["*uvicorn*api_server*", "*api_server:app*8001*"],
    )


def process_pids_for_workspace_command(workspace_dir: str, patterns: list[str]) -> set[int]:
    if os.name != "nt":
        return set()

    escaped_workspace_dir = workspace_dir.replace("'", "''")
    escaped_patterns = (pattern.replace("'", "''") for pattern in patterns)
    pattern_checks = " -or ".join(
        f"$_.CommandLine -like '{pattern}'" for pattern in escaped_patterns
    )
    command = (
        f"$workspaceDir = '{escaped_workspace_dir}'; "
        "$currentPid = $PID; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { "
        "$_.ProcessId -ne $currentPid -and "
        "($_.Name -in @('powershell.exe','pwsh.exe','cmd.exe','node.exe','npm.cmd')) -and "
        "$_.CommandLine -and "
        f"($_.CommandLine -like \"*$workspaceDir*\" -or {pattern_checks}) "
        "} | ForEach-Object { $_.ProcessId }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    pids = set()
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        try:
            pids.add(int(value))
        except ValueError:
            continue
    return pids


def kill_process_tree(pid: int) -> None:
    if pid == os.getpid():
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=8,
                check=False,
            )
        else:
            os.kill(pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        return


def stop_frontend_dev_server() -> None:
    for pid in pids_listening_on_port(5173) | frontend_process_pids():
        kill_process_tree(pid)


def shutdown_application() -> None:
    stop_frontend_dev_server()
    time.sleep(0.2)
    backend_pids = backend_process_pids()
    for pid in backend_pids:
        kill_process_tree(pid)
    if backend_pids:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
            capture_output=True,
            timeout=8,
            check=False,
        )
    else:
        os.kill(os.getpid(), signal.SIGTERM)


def cancel_pending_shutdown() -> bool:
    global shutdown_timer
    with shutdown_lock:
        if not shutdown_timer:
            return False
        shutdown_timer.cancel()
        shutdown_timer = None
        return True


def schedule_shutdown() -> None:
    global shutdown_timer
    with shutdown_lock:
        if shutdown_timer:
            shutdown_timer.cancel()
        shutdown_timer = threading.Timer(SHUTDOWN_GRACE_SECONDS, shutdown_application)
        shutdown_timer.daemon = True
        shutdown_timer.start()


def normalize_language(language: str) -> str:
    return "en" if (language or "").lower().strip().startswith("en") else "zh"


def output_language_instruction(language: str) -> str:
    job_language_requirement = agent.job_description_output_language_instruction()
    if job_language_requirement:
        return job_language_requirement
    if normalize_language(language) != "en":
        return "\n\nOutput language requirement: respond entirely in Chinese."
    return (
        "\n\nOutput language requirement: respond entirely in English. "
        "All user-facing headings, analysis, recommendations, cover letters, "
        "interview preparation notes, and chat responses must be English."
    )


def original_resume_language_instruction(output_type: str) -> str:
    # The job-description language takes precedence for application artifacts.
    if agent.job_description_output_language_instruction():
        return ""
    try:
        resume = agent.read_resume()
    except FileNotFoundError:
        return ""

    chinese_character_count = len(re.findall(r"[\u4e00-\u9fff]", resume))
    if chinese_character_count < 20:
        return ""

    if output_type == "tailored_resume":
        output_requirement = (
            "Generate the tailored resume entirely in Chinese. Preserve the required LaTeX commands and "
            "factual proper nouns, but write all user-facing section headings, summaries, experience descriptions, "
            "project descriptions, and bullet points in Chinese."
        )
    else:
        output_requirement = "Write the complete cover letter entirely in Chinese."

    return (
        "\n\nOriginal resume language requirement: resume.txt is a Chinese resume. "
        f"{output_requirement} This requirement overrides any conflicting UI language setting."
    )


def original_resume_language_instruction_for_request(message: str) -> str:
    if agent.is_likely_cover_letter_request(message):
        return original_resume_language_instruction("cover_letter")
    if agent.is_likely_resume_edit_request(message):
        return original_resume_language_instruction("tailored_resume")
    return ""


def job_analysis_language_instruction(language: str) -> str:
    return output_language_instruction(language)


def interview_prep_language_instruction(language: str) -> str:
    return output_language_instruction(language)


MAX_AGENT_IMAGES = 4
MAX_AGENT_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_AGENT_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
MAX_RESUME_PDF_BYTES = 20 * 1024 * 1024
MAX_RESUME_PDF_TEXT_CHARS = 45000


def validate_resume_pdf(body: ResumePdfToLatexBody) -> bytes:
    try:
        decoded = base64.b64decode(body.data_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="Invalid base64 PDF data.") from error

    if not decoded:
        raise HTTPException(status_code=400, detail="PDF file is empty.")
    if len(decoded) > MAX_RESUME_PDF_BYTES:
        raise HTTPException(status_code=400, detail="PDF file must be 20 MB or smaller.")
    if not decoded.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Selected file is not a valid PDF.")

    return decoded


def extract_pdf_resume_content(pdf_bytes: bytes) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF parsing dependency is missing. Install backend requirements again "
                "so the 'pypdf' package is available."
            ),
        ) from error

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {error}") from error

    page_texts: list[str] = []
    links: list[dict[str, Any]] = []
    seen_links: set[tuple[int, str]] = set()
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            page_texts.append(f"--- Page {page_index} ---\n{text.strip()}")

        annotations = page.get("/Annots") or []
        for annotation_ref in annotations:
            try:
                annotation = annotation_ref.get_object()
                action = annotation.get("/A") or {}
                uri = str(action.get("/URI") or "").strip()
            except Exception:
                uri = ""
            if not uri:
                continue
            key = (page_index, uri)
            if key in seen_links:
                continue
            seen_links.add(key)
            links.append({"page": page_index, "url": uri})

    text_content = "\n\n".join(page_texts).strip()
    if not text_content:
        raise HTTPException(
            status_code=400,
            detail=(
                "No selectable text was found in the PDF. If this is an image-only scanned "
                "resume, run OCR first and upload the OCR/searchable PDF."
            ),
        )

    return {
        "text": text_content[:MAX_RESUME_PDF_TEXT_CHARS],
        "truncated": len(text_content) > MAX_RESUME_PDF_TEXT_CHARS,
        "links": links,
        "pages": len(reader.pages),
    }


def build_pdf_to_latex_prompt(
    filename: str,
    extracted: dict[str, Any],
    language: str,
    agent_progress_messages: Optional[list[dict[str, Any]]] = None,
) -> str:
    links_text = "\n".join(
        f"- Page {link['page']}: {link['url']}" for link in extracted["links"]
    ) or "- None detected"
    job_language_requirement = agent.job_description_output_language_instruction()
    language_requirement = job_language_requirement or (
        "Write user-facing resume content in English."
        if normalize_language(language) == "en"
        else (
            "Preserve the resume's original language as detected from the extracted PDF text. "
            "If the source is Chinese, write user-facing resume content in Chinese; if it is English, write in English."
        )
    )
    truncation_note = (
        "\nThe extracted text was truncated for model limits; convert all visible high-signal resume content."
        if extracted["truncated"]
        else ""
    )

    prompt = f"""
Convert the extracted PDF resume into complete, compile-ready LaTeX resume code.

Requirements:
- Return only LaTeX code. Do not wrap it in Markdown fences and do not add analysis text.
- Include a full document beginning with \\documentclass and ending with \\end{{document}}.
- Preserve factual names, schools, employers, roles, dates, projects, skills, metrics, and contact details from the PDF.
- Preserve detected hyperlinks using \\href{{url}}{{label}} where the label is supported by nearby PDF text or clearly implied by the URL.
- Do not invent content that is not supported by the extracted text or detected links.
- Use a clean ATS-friendly resume structure with compact sections.
- Escape LaTeX special characters correctly.
- {language_requirement}

Source filename: {filename}
Page count: {extracted["pages"]}{truncation_note}

Detected PDF hyperlinks:
{links_text}

Extracted PDF text:
{extracted["text"]}
""".strip()
    return append_agent_progress_guidance(prompt, agent_progress_messages or [])


def latex_commands_for_resume(tex_path: Path, build_dir: Path) -> list[list[str]]:
    commands = []
    try:
        tex_content = tex_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        tex_content = ""
    prefers_pdftex = (
        "glyphtounicode" in tex_content
        or "\\pdfgentounicode" in tex_content
        or "\\pdfglyphtounicode" in tex_content
    )

    xelatex = shutil.which("xelatex")
    pdflatex = shutil.which("pdflatex")
    latexmk = shutil.which("latexmk")

    if prefers_pdftex and latexmk:
        commands.append(
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build_dir}",
                str(tex_path),
            ]
        )
    if prefers_pdftex and pdflatex:
        commands.append(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                str(tex_path),
            ]
        )
    if xelatex:
        commands.append(
            [
                xelatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                str(tex_path),
            ]
        )
    if not prefers_pdftex and pdflatex:
        commands.append(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                str(tex_path),
            ]
        )
    if latexmk:
        commands.append(
            [
                latexmk,
                "-xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-outdir={build_dir}",
                str(tex_path),
            ]
        )

    if commands:
        return commands

    raise HTTPException(
        status_code=500,
        detail=(
            "No LaTeX compiler was found. Install MiKTeX or TeX Live and make sure "
            "latexmk, xelatex, or pdflatex is available on PATH."
        ),
    )


def latex_document_for_pdf_export(document: str) -> str:
    export_document = document
    if "\\input{glyphtounicode}" in export_document and "\\ifdefined\\pdfglyphtounicode" not in export_document:
        export_document = export_document.replace(
            "\\input{glyphtounicode}",
            "\\ifdefined\\pdfglyphtounicode\n\\input{glyphtounicode}\n\\fi",
        )
    if "\\pdfgentounicode=1" in export_document and "\\ifdefined\\pdfgentounicode" not in export_document:
        export_document = export_document.replace(
            "\\pdfgentounicode=1",
            "\\ifdefined\\pdfgentounicode\n\\pdfgentounicode=1\n\\fi",
        )
    return export_document


def compile_tailored_resume_pdf(latex: str, company: str = "", role: str = "") -> Path:
    document = agent.extract_latex_document(latex)
    if not document:
        raise HTTPException(status_code=400, detail="No complete LaTeX document found.")
    document = latex_document_for_pdf_export(document)

    TAILORED_RESUME_PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = agent.unique_application_output_path(
        TAILORED_RESUME_PDF_OUTPUT_DIR,
        company=company,
        role=role,
        suffix=".pdf",
        fallback_prefix="tailored_resume",
    )
    stem = output_pdf.stem
    tex_path = LATEX_BUILD_DIR / f"{stem}.tex"
    tex_path.write_text(document + "\n", encoding="utf-8")

    failures = []
    result = None
    for command in latex_commands_for_resume(tex_path, LATEX_BUILD_DIR):
        runs = 1 if Path(command[0]).name.lower().startswith("latexmk") else 2
        for _ in range(runs):
            result = subprocess.run(
                command,
                cwd=LATEX_BUILD_DIR,
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if result.returncode != 0:
                break

        if result.returncode == 0:
            break

        output = (result.stdout + "\n" + result.stderr).strip()
        log_tail = "\n".join(output.splitlines()[-20:])
        failures.append(f"{Path(command[0]).name} failed:\n{log_tail or 'No compiler output.'}")

    if result is None or result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail="LaTeX PDF export failed. Compiler log:\n" + "\n\n".join(failures),
        )

    built_pdf = LATEX_BUILD_DIR / f"{stem}.pdf"
    if not built_pdf.exists():
        raise HTTPException(status_code=500, detail="LaTeX compiler finished but no PDF was produced.")

    shutil.copyfile(built_pdf, output_pdf)
    return output_pdf


def validate_agent_images(images: list[AgentImageBody]) -> list[dict[str, str]]:
    if len(images) > MAX_AGENT_IMAGES:
        raise HTTPException(status_code=400, detail=f"Attach at most {MAX_AGENT_IMAGES} images.")

    validated = []
    for image in images:
        mime_type = image.mime_type.lower().strip()
        if mime_type not in ALLOWED_AGENT_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {image.mime_type}")

        prefix = f"data:{mime_type};base64,"
        if not image.data_url.startswith(prefix):
            raise HTTPException(status_code=400, detail="Invalid image data URL.")

        base64_data = image.data_url[len(prefix) :]
        try:
            decoded = base64.b64decode(base64_data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise HTTPException(status_code=400, detail="Invalid base64 image data.") from error

        if not decoded:
            raise HTTPException(status_code=400, detail="Attached image is empty.")
        if len(decoded) > MAX_AGENT_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Each image must be 10 MB or smaller.")

        validated.append(
            {
                "name": image.name.strip(),
                "mime_type": mime_type,
                "data_url": image.data_url,
                "base64_data": base64_data,
            }
        )

    return validated


def save_chat_session(body: ChatSessionBody) -> dict[str, Any]:
    safe_session_id = re.sub(r"[^A-Za-z0-9_-]", "", body.session_id)[:80]
    if not safe_session_id:
        raise HTTPException(status_code=400, detail="Chat session id is required.")

    CHAT_SESSION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets_dir = CHAT_SESSION_OUTPUT_DIR / f"chat_session_{safe_session_id}_assets"
    image_extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }

    def save_images(images: list[AgentImageBody]) -> list[dict[str, Any]]:
        saved_images = []
        for image in validate_agent_images(images):
            decoded = base64.b64decode(image["base64_data"], validate=True)
            digest = hashlib.sha256(decoded).hexdigest()[:16]
            assets_dir.mkdir(parents=True, exist_ok=True)
            image_path = assets_dir / f"{digest}{image_extensions[image['mime_type']]}"
            if not image_path.exists():
                image_path.write_bytes(decoded)
            saved_images.append(
                {
                    "name": image["name"],
                    "mime_type": image["mime_type"],
                    "path": str(image_path.relative_to(agent.ROOT_DIR)),
                    "size_bytes": len(decoded),
                }
            )
        return saved_images

    saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    draft_images = save_images(body.images)
    transcript_lines = [
        "WorkAgent Chat Session",
        f"Session ID: {safe_session_id}",
        f"Created at: {body.created_at or '-'}",
        f"Saved at: {saved_at}",
        f"Language: {normalize_language(body.language)}",
        "",
        "Draft",
        "-----",
        body.message.strip() or "(no unsent text)",
    ]
    if draft_images:
        transcript_lines.append("")
        transcript_lines.append("Draft attachments:")
        for image in draft_images:
            transcript_lines.append(
                f"- {image['name'] or 'image'} ({image['mime_type']}, {image['size_bytes']} bytes): {image['path']}"
            )
    if body.attachment_error:
        transcript_lines.extend(["", f"Attachment error: {body.attachment_error}"])

    transcript_lines.extend(["", "Conversation", "------------"])
    if not body.history:
        transcript_lines.append("(no sent messages)")
    for index, entry in enumerate(body.history, start=1):
        role = entry.role.strip() or "unknown"
        transcript_lines.extend(["", f"[{index}] {role}", "-" * (len(role) + len(str(index)) + 3)])
        transcript_lines.append(entry.text.strip() or "(no text)")
        entry_images = save_images(entry.images)
        if entry_images:
            transcript_lines.append("")
            transcript_lines.append("Attachments:")
            for image in entry_images:
                transcript_lines.append(
                    f"- {image['name'] or 'image'} ({image['mime_type']}, {image['size_bytes']} bytes): {image['path']}"
                )

    path = CHAT_SESSION_OUTPUT_DIR / f"chat_session_{safe_session_id}.txt"
    with chat_session_lock:
        path.write_text("\n".join(transcript_lines).rstrip() + "\n", encoding="utf-8")
    return {"saved": True, "path": str(path)}


def ensure_provider_supports_images(provider: str, images: list[dict[str, str]]) -> None:
    if images and not provider_supports_images(provider):
        raise HTTPException(
            status_code=400,
            detail="The configured DeepSeek provider does not support image attachments. Switch to OpenAI, Claude, Gemini, or a vision-capable OpenAI-compatible provider.",
        )


def agent_progress_guidance_text(messages: list[dict[str, Any]], max_messages: int = 8) -> str:
    cleaned = []
    for item in messages[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role not in {"user", "agent", "system"}:
            role = "user"
        content = str(item.get("content") or item.get("text") or "").strip()
        if not content:
            continue
        cleaned.append(f"- {role}: {truncate_text(content, 1200)}")
    if not cleaned:
        return ""
    return (
        "\n\nLive user guidance from the Agent progress modal:\n"
        "Use these messages as additional user constraints for this stage and later stages. "
        "Apply them only when they do not conflict with factuality, evidence, or safety rules. "
        "Do not mention this guidance section in the final artifact unless the user explicitly asked for that.\n"
        + "\n".join(cleaned)
        + "\n"
    )


def append_agent_progress_guidance(prompt: str, messages: list[dict[str, Any]]) -> str:
    return prompt + agent_progress_guidance_text(messages)


class AgentTaskCancelled(RuntimeError):
    pass


def normalize_agent_task_id(task_id: str = "") -> str:
    return re.sub(r"[^a-zA-Z0-9_.:-]", "", str(task_id or "").strip())[:120]


def agent_task_event(task_id: str) -> Optional[threading.Event]:
    task_id = normalize_agent_task_id(task_id)
    if not task_id:
        return None
    with agent_task_lock:
        event = agent_task_cancellations.get(task_id)
        if event is None:
            event = threading.Event()
            agent_task_cancellations[task_id] = event
        return event


def register_agent_task_adapter(task_id: str, adapter: Any) -> None:
    task_id = normalize_agent_task_id(task_id)
    if not task_id:
        return
    with agent_task_lock:
        agent_task_adapters.setdefault(task_id, []).append(adapter)


def unregister_agent_task_adapter(task_id: str, adapter: Any) -> None:
    task_id = normalize_agent_task_id(task_id)
    if not task_id:
        return
    with agent_task_lock:
        adapters = agent_task_adapters.get(task_id, [])
        if adapter in adapters:
            adapters.remove(adapter)
        if not adapters:
            agent_task_adapters.pop(task_id, None)


def cancel_agent_task_id(task_id: str) -> bool:
    task_id = normalize_agent_task_id(task_id)
    if not task_id:
        return False
    with agent_task_lock:
        event = agent_task_cancellations.setdefault(task_id, threading.Event())
        event.set()
    return True


def assert_agent_task_not_cancelled() -> None:
    task_id = current_agent_task_id.get("")
    event = agent_task_event(task_id) if task_id else None
    if event and event.is_set():
        raise AgentTaskCancelled("Agent task was cancelled.")


def utc_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def background_message(role: str, content: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": utc_now_iso(),
    }


def snapshot_background_task(task_id: str) -> dict[str, Any]:
    task_id = normalize_agent_task_id(task_id)
    with background_task_lock:
        task = background_agent_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Agent task not found.")
        return {
            key: value
            for key, value in task.items()
            if key not in {"thread"}
        }


def update_background_task(task_id: str, **updates: Any) -> None:
    task_id = normalize_agent_task_id(task_id)
    with background_task_lock:
        task = background_agent_tasks.get(task_id)
        if not task:
            return
        task.update(updates)
        task["updated_at"] = utc_now_iso()


def append_background_task_message(task_id: str, role: str, content: str) -> None:
    task_id = normalize_agent_task_id(task_id)
    with background_task_lock:
        task = background_agent_tasks.get(task_id)
        if not task:
            return
        task.setdefault("messages", []).append(background_message(role, content))
        task["updated_at"] = utc_now_iso()


def set_background_stage(task_id: str, stage_id: str, status: str, detail: str = "") -> None:
    task_id = normalize_agent_task_id(task_id)
    with background_task_lock:
        task = background_agent_tasks.get(task_id)
        if not task:
            return
        task["stages"] = [
            {**stage, "status": status, "detail": detail or stage.get("detail", "")}
            if stage.get("id") == stage_id
            else stage
            for stage in task.get("stages", [])
        ]
        task["currentStage"] = stage_id
        task["updated_at"] = utc_now_iso()


def task_error_detail(error: Exception) -> str:
    if isinstance(error, HTTPException):
        detail = str(error.detail)
    else:
        detail = str(error)
    if "524" in detail or "origin_response_timeout" in detail:
        return "模型请求超时，可以稍后重试。Cloudflare 524 表示上游模型响应超过代理读取窗口。"
    return detail or "Agent task failed."


def run_background_agent_task(task_id: str) -> None:
    task_snapshot = snapshot_background_task(task_id)
    task_type = task_snapshot.get("taskType") or task_snapshot.get("task_type")
    body = task_snapshot.get("body") or {}
    try:
        with agent_task_context(task_id, clear_existing=False):
            if task_type == "resume_tailor":
                set_background_stage(task_id, "generate", "running", "后台正在生成并合并简历内容")
                append_background_task_message(task_id, "system", "后台任务已开始；HTTP start 请求已释放，不会等待完整模型结果。")
                assert_agent_task_not_cancelled()
                result = tailor_resume_task(TailorBody(**body))
                assert_agent_task_not_cancelled()
                set_background_stage(task_id, "generate", "done", "简历生成完成")
                update_background_task(
                    task_id,
                    status="done",
                    result=result,
                    resultAvailable=True,
                    currentStage="generate",
                )
                append_background_task_message(task_id, "agent", "定制简历已生成，可以读取最终结果。")
                return
            raise HTTPException(status_code=400, detail=f"Unsupported agent task type: {task_type}")
    except HTTPException as error:
        if error.status_code == 499:
            set_background_stage(task_id, "generate", "cancelled", "任务已取消，结果已丢弃")
            update_background_task(task_id, status="cancelled", result=None, resultAvailable=False, error="")
            append_background_task_message(task_id, "system", "任务已取消。")
            return
        detail = task_error_detail(error)
        set_background_stage(task_id, "generate", "error", detail)
        update_background_task(task_id, status="error", error=detail, result=None, resultAvailable=False)
        append_background_task_message(task_id, "agent", detail)
    except Exception as error:
        detail = task_error_detail(error)
        set_background_stage(task_id, "generate", "error", detail)
        update_background_task(task_id, status="error", error=detail, result=None, resultAvailable=False)
        append_background_task_message(task_id, "agent", detail)


def create_background_agent_task(body: AgentTaskStartBody) -> dict[str, Any]:
    task_id = normalize_agent_task_id(body.task_id) or str(uuid.uuid4())
    task_type = body.taskType or body.task_type
    if not task_type:
        raise HTTPException(status_code=400, detail="taskType is required.")
    now = utc_now_iso()
    with background_task_lock:
        existing = background_agent_tasks.get(task_id)
        if existing and existing.get("status") in {"started", "running", "waiting_for_user"}:
            raise HTTPException(status_code=409, detail="Agent task is already running.")
        background_agent_tasks[task_id] = {
            "taskId": task_id,
            "taskType": task_type,
            "body": body.body,
            "status": "started",
            "stages": [
                {"id": "generate", "label": "后台生成简历", "status": "pending", "detail": ""},
            ],
            "messages": [background_message("system", "后台 Agent 任务已创建。")],
            "currentStage": "generate",
            "result": None,
            "resultAvailable": False,
            "error": "",
            "created_at": now,
            "updated_at": now,
        }
    thread = threading.Thread(target=run_background_agent_task, args=(task_id,), daemon=True)
    with background_task_lock:
        background_agent_tasks[task_id]["thread"] = thread
        background_agent_tasks[task_id]["status"] = "running"
    thread.start()
    return {"taskId": task_id, "status": "started"}


@contextmanager
def agent_task_context(task_id: str = "", clear_existing: bool = True):
    task_id = normalize_agent_task_id(task_id)
    event = agent_task_event(task_id) if task_id else None
    if clear_existing and event and event.is_set():
        event.clear()
    token = current_agent_task_id.set(task_id)
    try:
        assert_agent_task_not_cancelled()
        yield
        assert_agent_task_not_cancelled()
    except AgentTaskCancelled as error:
        raise HTTPException(status_code=499, detail=str(error)) from error
    finally:
        current_agent_task_id.reset(token)
        if task_id:
            with agent_task_lock:
                if not agent_task_adapters.get(task_id):
                    agent_task_adapters.pop(task_id, None)
                    agent_task_cancellations.pop(task_id, None)


def run_agent_task(
    message: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    images: Optional[list[dict[str, str]]] = None,
) -> str:
    adapter, _ = get_adapter(provider)
    chosen_model = model or adapter.default_model()
    task_id = current_agent_task_id.get("")
    assert_agent_task_not_cancelled()
    register_agent_task_adapter(task_id, adapter)
    try:
        result = agent.ask_agent(message, adapter=adapter, model=chosen_model, images=images)
        assert_agent_task_not_cancelled()
        return result
    except AgentTaskCancelled as error:
        raise HTTPException(status_code=499, detail=str(error)) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (APIStatusError, urllib.error.HTTPError) as error:
        assert_agent_task_not_cancelled()
        raise_model_api_http_exception(error)
    except agent.transient_network_errors() as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        unregister_agent_task_adapter(task_id, adapter)


def run_text_task(message: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    adapter, _ = get_adapter(provider)
    chosen_model = model or adapter.default_model()
    task_id = current_agent_task_id.get("")
    assert_agent_task_not_cancelled()
    register_agent_task_adapter(task_id, adapter)
    try:
        response = adapter.create_response(
            model=chosen_model,
            instructions=agent.SYSTEM_PROMPT,
            tools=[],
            input_items=[{"role": "user", "content": message}],
        )
        assert_agent_task_not_cancelled()
        return adapter.output_text(response)
    except AgentTaskCancelled as error:
        raise HTTPException(status_code=499, detail=str(error)) from error
    except (APIStatusError, urllib.error.HTTPError) as error:
        assert_agent_task_not_cancelled()
        raise_model_api_http_exception(error)
    except agent.transient_network_errors() as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        unregister_agent_task_adapter(task_id, adapter)


APPLICATION_HINT_EXTRACTION_PROMPT = """
Read the job description below and extract fields for an internal application-tracking form.
Return ONLY valid JSON with exactly these string keys: "company", "role", "link", "notes".

Rules:
- company: hiring employer / company name only (not a person, team, or location).
- role: job title / position name only (not seniority fluff alone, not the company name).
- link: the single best URL to view or apply for this job posting; use "" if none is clearly a job link.
- notes: optional brief context such as location or employment type (max 120 characters), or "".
- Use the same language as the job description for company and role when the JD mixes languages.
- Do not invent employers, titles, URLs, or facts that are not supported by the text.
- Ignore resume bullets, benefits marketing, and equal-opportunity boilerplate when choosing company and role.

Job description:
"""


def resolve_application_hint(job_description: str) -> dict[str, str]:
    empty = {"company": "", "role": "", "link": "", "notes": ""}
    trimmed = job_description.strip()
    if not trimmed:
        return empty

    prompt = APPLICATION_HINT_EXTRACTION_PROMPT + trimmed
    try:
        response = run_text_task(prompt)
        payload = extract_json_object(response)
    except HTTPException:
        return empty

    def as_string(key: str) -> str:
        value = payload.get(key, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.strip()

    return {
        "company": as_string("company"),
        "role": as_string("role"),
        "link": as_string("link"),
        "notes": as_string("notes")[:120],
    }


def resolve_saved_application_hint(job_description: str = "") -> dict[str, str]:
    if not job_description:
        try:
            job_description = (
                agent.read_text_file(agent.JOB_DESCRIPTION_PATH)
                if agent.file_is_ready(agent.JOB_DESCRIPTION_PATH)
                else ""
            )
        except (FileNotFoundError, ValueError):
            job_description = ""
    hint = resolve_application_hint(job_description)
    agent.set_application_output_metadata(company=hint["company"], role=hint["role"])
    return hint


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(status_code=500, detail="Agent did not return valid JSON.")
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail="Agent did not return valid JSON.") from error

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=500, detail="Agent JSON response must be an object.")
    return parsed


def load_memory_for_merge() -> Any:
    return agent.MEMORY_STORE.read_profile()


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def normalize_match_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"https?://(?:www\.)?github\.com/", "", text)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    text = text.removesuffix(".git")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_match_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_match_text(value))


def list_memory_projects(memory: dict[str, Any]) -> list[Any]:
    projects = memory.get("projects", [])
    if isinstance(projects, dict):
        return [projects]
    if isinstance(projects, list):
        return projects
    return []


def project_field_values(project: Any) -> list[str]:
    if not isinstance(project, dict):
        return [str(project)]
    values = [
        project.get("project_id"),
        project.get("project_name"),
        project.get("name"),
        project.get("title"),
        project.get("repository"),
    ]
    identity = project.get("identity")
    if isinstance(identity, dict):
        values.extend(
            [
                identity.get("project_id"),
                identity.get("project_name"),
                identity.get("name"),
                identity.get("positioning"),
            ]
        )
    return [str(value) for value in values if str(value or "").strip()]


def project_matches(project: Any, project_name: str = "", project_id: str = "") -> bool:
    target_id = normalize_match_text(project_id)
    target_name = normalize_match_text(project_name)
    compact_target_id = compact_match_text(project_id)
    compact_target_name = compact_match_text(project_name)
    field_values = [normalize_match_text(value) for value in project_field_values(project)]
    compact_field_values = [compact_match_text(value) for value in project_field_values(project)]
    if target_id and any(value == target_id for value in field_values):
        return True
    if compact_target_id and any(value == compact_target_id for value in compact_field_values):
        return True
    if not target_name:
        return False
    if any(value == target_name or target_name in value or value in target_name for value in field_values):
        return True
    return any(
        value == compact_target_name or compact_target_name in value or value in compact_target_name
        for value in compact_field_values
    )


def scoped_project_memory(current_memory: dict[str, Any], project_name: str = "", project_id: str = "") -> dict[str, Any]:
    projects = [
        project
        for project in list_memory_projects(current_memory)
        if project_matches(project, project_name=project_name, project_id=project_id)
    ]
    return {"projects": projects}


def merge_scoped_project_memory(
    current_memory: dict[str, Any],
    scoped_memory: dict[str, Any],
    project_name: str = "",
    project_id: str = "",
) -> dict[str, Any]:
    scoped_projects = list_memory_projects(scoped_memory)
    if not scoped_projects:
        return current_memory

    merged_memory = dict(current_memory)
    current_projects = list_memory_projects(current_memory)
    retained_projects = [
        project
        for project in current_projects
        if not project_matches(project, project_name=project_name, project_id=project_id)
    ]

    seen = set()
    merged_projects = []
    for project in retained_projects + scoped_projects:
        key_values = project_field_values(project)
        key = normalize_match_text(key_values[0] if key_values else normalized_json(project))
        if key in seen:
            continue
        seen.add(key)
        merged_projects.append(project)

    merged_memory["projects"] = merged_projects
    return merged_memory


def update_memory_from_resume_source(
    resume_source: str,
    project_name: str = "",
    project_id: str = "",
    agent_progress_messages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    if resume_source == "tailored_resume":
        resume = agent.read_tailored_resume()
        source_label = "tailored_resume.txt"
    elif resume_source == "resume":
        resume = agent.read_resume()
        source_label = "resume.txt"
    else:
        raise HTTPException(status_code=400, detail="resume_source must be 'resume' or 'tailored_resume'.")

    current_memory = load_memory_for_merge()
    scoped_update = bool(project_name.strip() or project_id.strip())
    memory_for_prompt = (
        scoped_project_memory(current_memory, project_name=project_name, project_id=project_id)
        if scoped_update and isinstance(current_memory, dict)
        else current_memory
    )
    target_project_text = ""
    if scoped_update:
        requested = project_name.strip() or project_id.strip()
        target_project_text = f"""
Target project:
- project_name: {project_name.strip() or "(not specified)"}
- project_id: {project_id.strip() or "(not specified)"}

Scoped update rules:
- Update only the requested project's item inside memory["projects"].
- If the current scoped memory has no matching project but the resume clearly contains the requested project, add one compact project item.
- Do not add, modify, summarize, or remove unrelated projects or non-project memory sections.
- Return only the scoped memory object containing the requested project's projects list.
"""
    prompt = f"""
Update the user's long-term memory JSON from the resume below.

Rules:
- Only add factual, durable information explicitly present in the resume.
- Preserve all existing memory unless the resume clearly gives a more complete version of the same fact.
- Do not add job-specific tailoring, unsupported claims, guesses, or generic wording.
- Keep the result compact and useful for future resume, cover letter, and interview prep tasks.
- Return only valid JSON with exactly these keys:
  "changed": boolean,
  "additions": array of short strings describing newly added facts,
  "memory": object

{target_project_text}
Current memory JSON:
{json.dumps(memory_for_prompt, ensure_ascii=False, indent=2)}

Resume source: {source_label}
Resume:
{resume}
"""
    prompt = append_agent_progress_guidance(prompt, agent_progress_messages or [])
    response = run_text_task(prompt)
    payload = extract_json_object(response)
    merged_memory = payload.get("memory")
    if not isinstance(merged_memory, dict):
        raise HTTPException(status_code=500, detail="Agent JSON response must include a memory object.")

    if scoped_update:
        merged_memory = merge_scoped_project_memory(
            current_memory,
            merged_memory,
            project_name=project_name,
            project_id=project_id,
        )

    changed = normalized_json(merged_memory) != normalized_json(current_memory)
    if changed:
        source_suffix = f":project:{project_name.strip() or project_id.strip()}" if scoped_update else ""
        agent.replace_profile_memory(merged_memory, source=f"resume-merge:{source_label}{source_suffix}")

    additions = payload.get("additions", [])
    if not isinstance(additions, list):
        additions = []

    return {
        "updated": changed,
        "source": source_label,
        "project_name": project_name.strip(),
        "project_id": project_id.strip(),
        "additions": [str(item) for item in additions if str(item).strip()],
        "memory": merged_memory,
        "path": str(agent.CHROMA_DB_PATH),
        "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
    }


def build_project_analysis_payload(repo_contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for context in repo_contexts:
        commits = []
        for evidence in context.get("contribution_evidence", []):
            for commit in evidence.get("commits", [])[:5]:
                commits.append(
                    {
                        "sha": commit.get("sha"),
                        "message": commit.get("message"),
                        "date": commit.get("date"),
                        "files": commit.get("files", [])[:12],
                        "diff_analysis": commit.get("diff_analysis", {}),
                    }
                )
        payload.append(
            {
                "url": context.get("url"),
                "repository": context.get("repository"),
                "description": context.get("description"),
                "homepage": context.get("homepage"),
                "topics": context.get("topics", []),
                "default_branch": context.get("default_branch"),
                "languages": context.get("languages", []),
                "root_files": context.get("root_files", []),
                "readme": context.get("readme", ""),
                "recent_commit_evidence": commits,
            }
        )
    return payload


def read_current_project_memory() -> dict[str, Any]:
    try:
        current = json.loads(agent.read_project_memory())
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Existing project_memory.json is not valid JSON.") from error
    if not isinstance(current, dict):
        raise HTTPException(status_code=500, detail="Existing project_memory.json must be a JSON object.")
    return current


def update_project_memory_from_repo_analysis(
    repo_contexts: list[dict[str, Any]],
    agent_progress_messages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    if not repo_contexts:
        return {
            "updated": False,
            "source": "repo-analysis",
            "additions": [],
            "project_memory": read_current_project_memory(),
            "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
            "message": "No repository analysis payload is available.",
        }
    if not agent.has_usable_repo_context(repo_contexts):
        return {
            "updated": False,
            "source": "repo-analysis",
            "additions": [],
            "project_memory": read_current_project_memory(),
            "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
            "message": "Repository analysis payload is not usable.",
        }

    usable_contexts = [
        context
        for context in repo_contexts
        if agent.has_usable_repo_context([context])
    ]
    project_memory = read_current_project_memory()
    changed_any = False
    additions: list[str] = []
    processed_repositories: list[str] = []
    skipped_repositories = [
        str(context.get("repository") or context.get("url") or "")
        for context in repo_contexts
        if context not in usable_contexts
    ]

    for context in usable_contexts:
        current_project_memory = project_memory
        repo_analysis = build_project_analysis_payload([context])
        prompt = f"""
{PROJECT_MEMORY_FROM_REPO_ANALYSIS_PROMPT}

Current project_memory.json:
{json.dumps(current_project_memory, ensure_ascii=False, indent=2)}

Repository analysis payload for exactly one repository:
{json.dumps(repo_analysis, ensure_ascii=False, indent=2)}
"""
        prompt = append_agent_progress_guidance(prompt, agent_progress_messages or [])
        response = run_text_task(prompt)
        payload = extract_json_object(response)
        next_project_memory = payload.get("project_memory")
        if not isinstance(next_project_memory, dict):
            raise HTTPException(status_code=500, detail="Agent JSON response must include a project_memory object.")

        changed = normalized_json(next_project_memory) != normalized_json(current_project_memory)
        if changed:
            agent.write_project_memory_file(next_project_memory)
            changed_any = True
        project_memory = next_project_memory

        repo_additions = payload.get("additions", [])
        if isinstance(repo_additions, list):
            additions.extend(str(item) for item in repo_additions if str(item).strip())
        processed_repositories.append(str(context.get("repository") or context.get("url") or ""))

    return {
        "updated": changed_any,
        "source": "repo-analysis",
        "mode": "sequential-per-repository",
        "processed_repositories": processed_repositories,
        "skipped_repositories": skipped_repositories,
        "additions": additions,
        "project_memory": project_memory,
        "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
    }


def build_interview_prep_prompt(
    use_github_context: bool,
    language: str = "zh",
    agent_progress_messages: Optional[list[dict[str, Any]]] = None,
) -> str:
    try:
        job_description = agent.read_job_description()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        resume = agent.read_tailored_resume()
        resume_source = "tailored_resume.txt"
    except (FileNotFoundError, ValueError):
        try:
            resume = agent.read_resume()
            resume_source = "resume.txt"
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    memory = agent.read_memory()
    github_context = read_approved_github_context() if use_github_context else ""

    github_section = (
        f"\nApproved GitHub context:\n{github_context}\n"
        if use_github_context
        else "\nApproved GitHub context: Not requested for this generation.\n"
    )

    section_outline = """  1. Role focus
  2. Technical questions
  3. Project talking points
  4. Behavioral / STAR material
  5. Preparation gaps or facts to verify
  6. Questions to ask the interviewer"""

    prompt = f"""
Create complete interview preparation notes for the job application below.

Rules:
- Use only the job description, resume, memory, and approved GitHub context provided here.
- Do not invent projects, employers, degrees, technologies, metrics, or repository facts.
- If evidence is weak or missing, say what to prepare or verify instead of fabricating.
- Return only the notes content. Do not say that you saved a file. Do not include placeholders.
- Follow the output language requirement below exactly.
- Include the following six concepts, translating every section heading into the job description's language:
{section_outline}

Job description:
{job_description}

Resume source: {resume_source}
Resume:
{resume}

Memory:
{memory}
{github_section}
"""
    return append_agent_progress_guidance(
        prompt + interview_prep_language_instruction(language),
        agent_progress_messages or [],
    )


def looks_like_interview_prep(content: str) -> bool:
    text = content.strip()
    lowered = text.lower()
    if len(text) < 400:
        return False
    rejected_phrases = [
        "saved interview preparation notes",
        "interview preparation notes saved",
        "saved to",
        "placeholder",
        "lorem ipsum",
        "no usable",
    ]
    if any(phrase in lowered[:300] for phrase in rejected_phrases):
        return False
    section_hits = sum(
        1
        for marker in [
            "Role Focus",
            "Technical Questions",
            "Project Talking Points",
            "Behavioral",
            "Preparation Gaps",
            "Questions to Ask",
            "职位重点",
            "技术问题",
            "项目讲述",
            "STAR",
            "行为面试",
            "补强",
            "反问",
        ]
        if marker in text
    )
    structural_hits = len(re.findall(r"(?m)^\s*(?:#{1,6}\s+|\d+[.)]\s+)", text))
    return section_hits >= 3 or structural_hits >= 3


def read_approved_github_context(query: str = "") -> str:
    context = agent.read_stored_github_context(query=query)
    if not context:
        return "No approved GitHub context is available. Ask the user to approve GitHub access in the web UI first."

    if not agent.has_usable_repo_context(context):
        return "No usable approved GitHub context is available."

    return (
        "Supporting GitHub evidence only. Do not write resume bullets directly from this evidence; "
        "use project_memory.json as the primary source.\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


agent.TOOL_FUNCTIONS["read_github_context"] = read_approved_github_context


agent.TOOL_FUNCTIONS["read_project_evidence_map"] = lambda limit_per_project=4: agent.read_project_evidence_map(
    limit_per_project=limit_per_project
)


def truncate_text(value: Any, max_chars: int = MAX_STAGED_TEXT_CHARS) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n... [truncated]"


def compact_value_for_prompt(value: Any, max_string_chars: int = 1200, max_list_items: int = 6, depth: int = 0) -> Any:
    if depth > 5:
        return "[truncated nested value]"
    if isinstance(value, str):
        return truncate_text(value, max_string_chars)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        compacted = [
            compact_value_for_prompt(item, max_string_chars, max_list_items, depth + 1)
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            compacted.append(f"... [{len(value) - max_list_items} more items truncated]")
        return compacted
    if isinstance(value, dict):
        skipped_keys = {
            "patch",
            "readme",
            "source_facts",
            "existing_bullets",
            "bullet_writer_validation",
            "compare_file_changes",
            "file_changes",
            "context",
        }
        compacted = {}
        for key, item in value.items():
            if key in skipped_keys:
                continue
            compacted[key] = compact_value_for_prompt(item, max_string_chars, max_list_items, depth + 1)
        return compacted
    return truncate_text(value, max_string_chars)


def short_signal(value: Any, max_chars: int = MAX_PROMPT_SIGNAL_CHARS) -> str:
    return truncate_text(str(value or "").strip(), max_chars)


def append_unique(items: list[str], value: Any, limit: int) -> None:
    text = short_signal(value)
    if text and text not in items and len(items) < limit:
        items.append(text)


def classify_prompt_signal_priority(signal: str) -> int:
    text = signal.lower()
    high_priority_keywords = [
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "ci/cd",
        "jenkins",
        "github actions",
        "terraform",
        "helm",
        "test",
        "selenium",
        "cypress",
        "playwright",
        "espresso",
        "gradle",
        "maven",
        "linux",
        "unix",
        "shell",
        "powershell",
        "deployment",
        "setup",
        "debug",
        "logging",
        "error handling",
        "backend api",
        "database",
        "persistence",
        "validation",
    ]
    medium_priority_keywords = [
        "frontend",
        "ui",
        "state",
        "documentation",
        "configuration",
        "config",
        "refactor",
        "compression",
    ]
    low_priority_keywords = ["format", "comment", "readme-only", "cosmetic", "lockfile"]
    if any(keyword in text for keyword in high_priority_keywords):
        return 0
    if any(keyword in text for keyword in medium_priority_keywords):
        return 1
    if any(keyword in text for keyword in low_priority_keywords):
        return 3
    return 2


def ranked_prompt_signals(signals: list[str], limit: int = MAX_PROMPT_DIFF_SIGNALS) -> list[str]:
    unique = []
    for signal in signals:
        append_unique(unique, signal, limit * 3)
    unique.sort(key=lambda item: (classify_prompt_signal_priority(item), item.lower()))
    return unique[:limit]


def detect_languages_and_frameworks_from_files(files: list[str], text: str = "") -> list[str]:
    values = []
    lower_text = text.lower()
    checks = [
        ("Python", [".py", "fastapi", "pytest"]),
        ("JavaScript", [".js", ".jsx", "vite", "react"]),
        ("TypeScript", [".ts", ".tsx"]),
        ("React", [".jsx", ".tsx", "react"]),
        ("FastAPI", ["fastapi", "api_server.py"]),
        ("SQLite", ["sqlite", ".sqlite"]),
        ("Docker", ["dockerfile", "docker-compose"]),
        ("GitHub Actions", [".github/workflows"]),
        ("Gradle", ["gradle", "build.gradle"]),
        ("Maven", ["pom.xml", "maven"]),
        ("Android", ["androidmanifest", "espresso"]),
        ("PowerShell", [".ps1", "powershell"]),
        ("Shell", [".sh", "bash"]),
    ]
    file_text = "\n".join(files).lower()
    combined = f"{file_text}\n{lower_text}"
    for label, needles in checks:
        if any(needle in combined for needle in needles):
            append_unique(values, label, 20)
    return values


def extract_added_symbols_from_patch(patch: str) -> list[str]:
    symbols = []
    for line in str(patch or "").splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        content = line[1:].strip()
        patterns = [
            r"def\s+([A-Za-z_][\w]*)\s*\(",
            r"class\s+([A-Za-z_][\w]*)",
            r"function\s+([A-Za-z_$][\w$]*)\s*\(",
            r"const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:\([^)]*\)\s*=>|function)",
            r"export\s+default\s+function\s+([A-Za-z_$][\w$]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                append_unique(symbols, match.group(1), 20)
    return symbols


def extract_diff_signals_for_prompt(file_change: dict[str, Any]) -> dict[str, Any]:
    filename = str(file_change.get("filename") or file_change.get("file") or "").strip()
    patch = str(file_change.get("patch") or file_change.get("diff") or file_change.get("raw_diff") or "")
    status = str(file_change.get("status") or file_change.get("change_type") or "").strip()
    added_symbols = extract_added_symbols_from_patch(patch)
    patch_lower = patch.lower()
    filename_lower = filename.lower()
    signals = []
    allowed_claims = []
    forbidden_claims = []

    checks = [
        ("added backend API handling", ["@app.", "fastapi", "api_server", "route", "endpoint"], "Implemented backend API handling"),
        ("added database persistence logic", ["sqlite", "select ", "insert ", "update ", "delete ", "database"], "Implemented database persistence logic"),
        ("added data validation or schema handling", ["pydantic", "validate", "schema", "required", "invalid"], "Implemented data validation logic"),
        ("added error handling path", ["try:", "except ", "raise ", "httpexception", "error"], "Debugged or implemented error handling paths"),
        ("added logging/debugging signal", ["log", "debug", "traceback"], "Debugged runtime observability or logging behavior"),
        ("added prompt compression helpers", ["for_prompt", "compact_", "truncate", "prompt"], "Implemented prompt compression utilities"),
        ("removed raw evidence from prompt payloads", ["patch", "readme", "validation", "raw_diff", "full_diff"], "Refactored prompt payloads to avoid raw evidence"),
        ("added frontend workflow or state handling", ["usestate", "useeffect", "onchange", "form", "input"], "Implemented frontend workflow or state handling"),
        ("added test automation", ["test_", "unittest", "pytest", "assert", "playwright", "cypress", "espresso"], "Implemented test automation"),
        ("added Gradle or Maven test setup", ["gradle", "maven", "pom.xml"], "Configured Java/Android build or test setup"),
        ("added shell or PowerShell automation", [".sh", ".ps1", "powershell", "bash"], "Automated setup or developer workflows with scripts"),
        ("changed environment variable loading logic", ["env", "environment", "load_dotenv"], "Implemented environment configuration handling"),
        ("added deployment/container configuration", ["docker", "kubernetes", "terraform", "helm", "aws", "gcp", "jenkins"], "Implemented deployment or infrastructure configuration"),
    ]
    combined = f"{filename_lower}\n{patch_lower}"
    for signal, needles, claim in checks:
        if any(needle in combined for needle in needles):
            append_unique(signals, signal, MAX_PROMPT_DIFF_SIGNALS)
            append_unique(allowed_claims, claim, MAX_PROMPT_CLAIMS)

    for symbol in added_symbols[:6]:
        append_unique(signals, f"added or modified function/component `{symbol}`", MAX_PROMPT_DIFF_SIGNALS)

    if filename:
        append_unique(signals, f"changed `{filename}`", MAX_PROMPT_DIFF_SIGNALS)

    infrastructure_keywords = {
        "AWS": ["aws", "ecs", "lambda", "s3"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Terraform": ["terraform"],
        "Jenkins": ["jenkins"],
        "Docker": ["docker"],
    }
    for label, needles in infrastructure_keywords.items():
        if not any(needle in combined for needle in needles):
            append_unique(forbidden_claims, f"Do not claim {label} work unless other evidence supports it", MAX_PROMPT_CLAIMS)

    summary_parts = []
    if status:
        summary_parts.append(status)
    if filename:
        summary_parts.append(filename)
    if signals:
        summary_parts.append("; ".join(ranked_prompt_signals(signals, 4)))
    file_summary = short_signal(" - ".join(summary_parts), MAX_PROMPT_FILE_SUMMARY_CHARS)

    return {
        "file": filename,
        "status": status,
        "change_type": file_change.get("change_type"),
        "summary": file_summary,
        "added_or_modified_symbols": added_symbols[:8],
        "implementation_signals": ranked_prompt_signals(signals),
        "allowed_claims": allowed_claims[:MAX_PROMPT_CLAIMS],
        "forbidden_claims": forbidden_claims[:MAX_PROMPT_CLAIMS],
        "evidence_id": hashlib.sha256(f"{filename}\n{patch[:1000]}".encode("utf-8")).hexdigest()[:12],
        "confidence": "high" if patch and signals else "medium" if signals else "low",
    }


def validation_for_prompt(validation: Any, candidate_id: str = "", project: str = "") -> dict[str, Any]:
    if not isinstance(validation, dict):
        return {
            "candidate_id": candidate_id,
            "project": project,
            "supported_claims": [],
            "unsupported_claims": [],
            "evidence_refs": [],
            "risk_level": "medium",
            "notes": "",
        }
    issues = [str(item) for item in validation.get("issues", []) if str(item).strip()][:6]
    return {
        "candidate_id": candidate_id,
        "project": project,
        "supported_claims": validation.get("supported_claims", [])[:MAX_PROMPT_CLAIMS],
        "unsupported_claims": validation.get("unsupported_claims", issues)[:MAX_PROMPT_CLAIMS],
        "evidence_refs": validation.get("evidence_refs", [])[:8],
        "risk_level": "low" if validation.get("accepted") and not issues else "medium",
        "notes": truncate_text(validation.get("notes") or "; ".join(issues), 400),
    }


def repo_evidence_for_prompt(evidence: Any) -> Any:
    contexts = evidence if isinstance(evidence, list) else [evidence]
    compacted = []
    for context in contexts[:3]:
        if not isinstance(context, dict):
            compacted.append(compact_value_for_prompt(context))
            continue
        file_summaries = []
        repo_allowed_claims = []
        repo_forbidden_claims = []
        repo_signals = []
        contribution_evidence = []
        for contribution in context.get("contribution_evidence", [])[:3]:
            if not isinstance(contribution, dict):
                continue
            commits = []
            for commit in contribution.get("commits", [])[:4]:
                if not isinstance(commit, dict):
                    continue
                commit_file_summaries = []
                for file_change in commit.get("file_changes", [])[:MAX_PROMPT_FILES_PER_REPO]:
                    if isinstance(file_change, dict):
                        extracted = extract_diff_signals_for_prompt(file_change)
                        commit_file_summaries.append(extracted)
                        file_summaries.append(extracted)
                        repo_signals.extend(extracted["implementation_signals"])
                        repo_allowed_claims.extend(extracted["allowed_claims"])
                        repo_forbidden_claims.extend(extracted["forbidden_claims"])
                commits.append(
                    {
                        "sha": commit.get("sha"),
                        "message": commit.get("message"),
                        "date": commit.get("date"),
                        "files": commit.get("files", [])[:8],
                        "diff_analysis": compact_value_for_prompt(commit.get("diff_analysis", {}), 400, 5),
                        "file_summaries": commit_file_summaries[:MAX_PROMPT_FILES_PER_REPO],
                    }
                )
            compare_file_summaries = []
            for file_change in contribution.get("compare_file_changes", [])[:MAX_PROMPT_FILES_PER_REPO]:
                if isinstance(file_change, dict):
                    extracted = extract_diff_signals_for_prompt(file_change)
                    compare_file_summaries.append(extracted)
                    file_summaries.append(extracted)
                    repo_signals.extend(extracted["implementation_signals"])
                    repo_allowed_claims.extend(extracted["allowed_claims"])
                    repo_forbidden_claims.extend(extracted["forbidden_claims"])
            contribution_evidence.append(
                {
                    "method": contribution.get("method"),
                    "github_account": contribution.get("github_account"),
                    "base_sha": contribution.get("base_sha"),
                    "head_sha": contribution.get("head_sha"),
                    "commit_count_checked": contribution.get("commit_count_checked"),
                    "commits": commits,
                    "compare_diff_analysis": compact_value_for_prompt(contribution.get("compare_diff_analysis", {}), 400, 5),
                    "compare_files": contribution.get("compare_files", [])[:8],
                    "compare_file_summaries": compare_file_summaries,
                    "error": contribution.get("error"),
                }
            )
        files = context.get("root_files", [])[:20]
        files.extend(summary.get("file", "") for summary in file_summaries if summary.get("file"))
        languages_frameworks = []
        for language in context.get("languages", [])[:10]:
            append_unique(languages_frameworks, language, 20)
        for detected in detect_languages_and_frameworks_from_files([str(item) for item in files], json.dumps(file_summaries, ensure_ascii=False)):
            append_unique(languages_frameworks, detected, 20)
        ranked_signals = ranked_prompt_signals(repo_signals)
        allowed_claims = ranked_prompt_signals(repo_allowed_claims, MAX_PROMPT_CLAIMS)
        forbidden_claims = ranked_prompt_signals(repo_forbidden_claims, MAX_PROMPT_CLAIMS)
        compacted.append(
            {
                "url": context.get("url"),
                "repository": context.get("repository"),
                "project_name": context.get("project_name") or context.get("project"),
                "description": truncate_text(context.get("description"), 400),
                "topics": context.get("topics", [])[:8],
                "default_branch": context.get("default_branch"),
                "languages_frameworks_detected": languages_frameworks,
                "root_files": context.get("root_files", [])[:20],
                "incremental_update": compact_value_for_prompt(context.get("incremental_update", {}), 400, 5),
                "changed_file_paths": [summary.get("file") for summary in file_summaries if summary.get("file")][:MAX_PROMPT_FILES_PER_REPO],
                "file_level_summaries": file_summaries[:MAX_PROMPT_FILES_PER_REPO],
                "diff_signals": ranked_signals,
                "resume_relevant_keywords": ranked_signals[:10] + languages_frameworks[:10],
                "allowed_claims": allowed_claims,
                "forbidden_claims": forbidden_claims,
                "evidence_confidence": "high" if ranked_signals else "medium" if contribution_evidence else "low",
                "contribution_evidence": contribution_evidence,
            }
        )
    serialized = json.dumps(compacted, ensure_ascii=False)
    if len(serialized) <= MAX_PROMPT_EVIDENCE_CHARS:
        return compacted
    return compact_value_for_prompt(compacted, 700, 4)


def compact_github_evidence_for_prompt(evidence: Any) -> Any:
    return repo_evidence_for_prompt(evidence)


def candidate_for_prompt(candidate: dict[str, Any]) -> dict[str, Any]:
    bullets = candidate.get("recommended_bullets") or candidate.get("final_bullets") or []
    candidate_id = str(candidate.get("project_id") or candidate.get("source_name") or candidate.get("project_name") or "")
    project_name = candidate.get("project_name") or candidate.get("source_name")
    return {
        "section_type": candidate.get("section_type"),
        "project_id": candidate.get("project_id"),
        "project_name": project_name,
        "candidate_id": candidate_id,
        "fit": candidate.get("fit"),
        "keep_or_replace": candidate.get("keep_or_replace"),
        "fit_reason": truncate_text(candidate.get("fit_reason") or candidate.get("job_alignment"), 800),
        "final_bullets": compact_value_for_prompt(bullets, 700, 4),
        "skills_to_emphasize": candidate.get("skills_to_emphasize", [])[:10],
        "risks": candidate.get("risks", [])[:6],
        "role_profile": compact_value_for_prompt(candidate.get("role_profile", {}), 500, 6),
        "jd_requirements": compact_value_for_prompt(candidate.get("jd_requirements", {}), 700, 8),
        "evidence_card": compact_value_for_prompt(candidate.get("evidence_card", {}), 900, 8),
        "role_lens": compact_value_for_prompt(candidate.get("role_lens", {}), 700, 8),
        "allowed_claims": compact_value_for_prompt(candidate.get("allowed_claims", []), 400, MAX_PROMPT_CLAIMS),
        "forbidden_claims": compact_value_for_prompt(candidate.get("forbidden_claims", []), 400, MAX_PROMPT_CLAIMS),
        "validation": validation_for_prompt(candidate.get("bullet_writer_validation"), candidate_id=candidate_id, project=str(project_name or "")),
    }


def compact_bullet_candidate_for_prompt(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate_for_prompt(candidate)


def compact_bullet_candidates_for_prompt(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate_for_prompt(candidate) for candidate in candidates]


ROLE_FAMILY_KEYWORDS = {
    "software_engineering": [
        "software",
        "developer",
        "engineer",
        "api",
        "backend",
        "frontend",
        "full stack",
        "react",
        "java",
        "python",
        "javascript",
        "typescript",
    ],
    "infrastructure_devops": [
        "devops",
        "infrastructure",
        "ci/cd",
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "deployment",
        "linux",
        "scripting",
        "automation",
    ],
    "it_analyst": [
        "it analyst",
        "technical support",
        "troubleshoot",
        "service desk",
        "documentation",
        "configuration",
        "application support",
        "requirements",
    ],
    "data_analyst": [
        "data analyst",
        "sql",
        "reporting",
        "dashboard",
        "metrics",
        "kpi",
        "excel",
        "data validation",
        "analysis",
    ],
    "product_business_analyst": [
        "business analyst",
        "product",
        "requirements",
        "stakeholder",
        "process",
        "workflow",
        "prioritization",
        "user needs",
    ],
    "qa_testing": [
        "qa",
        "quality assurance",
        "testing",
        "test cases",
        "automation testing",
        "selenium",
        "cypress",
        "playwright",
        "bug",
    ],
    "cybersecurity": [
        "security",
        "cybersecurity",
        "vulnerability",
        "incident",
        "access control",
        "risk",
        "compliance",
        "audit",
    ],
    "supply_chain_operations": [
        "supply chain",
        "operations",
        "inventory",
        "logistics",
        "tracking",
        "coordination",
        "deadline",
        "procurement",
    ],
    "healthcare_admin": [
        "healthcare",
        "health",
        "patient",
        "clinical",
        "administrative",
        "records",
        "privacy",
        "scheduling",
    ],
    "design_product": [
        "design",
        "ux",
        "ui",
        "prototype",
        "figma",
        "user research",
        "wireframe",
        "product design",
    ],
}

ROLE_LENS_PRIORITIES = {
    "software_engineering": ["api", "data model", "backend", "frontend", "integration", "validation", "testing", "git"],
    "infrastructure_devops": ["script", "automation", "environment", "configuration", "ci/cd", "docker", "logging", "debugging"],
    "it_analyst": ["troubleshooting", "documentation", "process support", "requirements", "application support", "automation", "configuration"],
    "data_analyst": ["sql", "validation", "reporting", "metrics", "records", "analysis", "kpi"],
    "product_business_analyst": ["requirements", "stakeholder", "documentation", "workflow", "user needs", "prioritization"],
    "qa_testing": ["testing", "test cases", "automation", "bug", "validation", "quality"],
    "cybersecurity": ["security", "risk", "access", "audit", "vulnerability", "incident"],
    "supply_chain_operations": ["accuracy", "tracking", "reporting", "excel", "sql", "coordination", "deadline"],
    "healthcare_admin": ["records", "privacy", "workflow", "documentation", "coordination", "accuracy"],
    "design_product": ["prototype", "user needs", "workflow", "interaction", "research", "design"],
}

PROTECTED_UNSUPPORTED_TOOLS = [
    "AWS",
    "Azure",
    "Kubernetes",
    "Docker",
    "Terraform",
    "Jenkins",
    "SAP",
    "Oracle",
    "Power BI",
    "Microsoft 365",
]


def keyword_hits(text: str, keywords: list[str], limit: int = 20) -> list[str]:
    lower_text = str(text or "").lower()
    hits = []
    for keyword in keywords:
        if keyword.lower() in lower_text:
            append_unique(hits, keyword, limit)
    return hits


def classify_role_family(jd_text: str) -> dict[str, Any]:
    text = str(jd_text or "")
    scores = []
    for family, keywords in ROLE_FAMILY_KEYWORDS.items():
        hits = keyword_hits(text, keywords, 30)
        if hits:
            score = len(hits)
            if family == "software_engineering" and keyword_hits(text, ["backend", "frontend", "api", "developer", "software"], 5):
                score += 1
            if family == "infrastructure_devops" and hits == ["automation"]:
                score -= 1
            scores.append((score, family, hits))
    scores.sort(key=lambda item: (-item[0], item[1]))
    primary = scores[0][1] if scores else "software_engineering"
    secondary = [family for _, family, _ in scores[1:4]]
    role_focus = []
    for _, _, hits in scores[:3]:
        for hit in hits:
            append_unique(role_focus, hit, 12)
    high_priority = role_focus[:10]
    low_priority = []
    for family, keywords in ROLE_FAMILY_KEYWORDS.items():
        if family != primary and family not in secondary:
            for keyword in keywords[:2]:
                append_unique(low_priority, keyword, 10)
    return {
        "role_family": primary,
        "secondary_role_families": secondary,
        "role_focus": role_focus,
        "resume_strategy": (
            f"Emphasize {primary.replace('_', ' ')} evidence first, then support it with "
            "role-relevant methods, artifacts, and business/workflow value."
        ),
        "high_priority_keywords": high_priority,
        "low_priority_keywords": low_priority,
    }


def jd_requirements_for_prompt(jd_text: str) -> dict[str, Any]:
    text = str(jd_text or "")
    core = jd_core_for_prompt(text)
    role_profile = classify_role_family(text)
    all_skill_keywords = sorted({keyword for values in ROLE_FAMILY_KEYWORDS.values() for keyword in values})
    tool_keywords = [
        "Python", "Java", "JavaScript", "TypeScript", "React", "FastAPI", "SQL", "SQLite", "MongoDB",
        "Excel", "Power BI", "Tableau", "Git", "GitHub", "Docker", "Kubernetes", "Terraform", "Jenkins",
        "AWS", "Azure", "Linux", "PowerShell", "SAP", "Oracle", "Microsoft 365",
    ]
    soft_keywords = ["communication", "collaboration", "documentation", "deadline", "team", "stakeholder", "problem solving", "critical thinking"]
    action_verbs = ["analyze", "build", "coordinate", "debug", "document", "implement", "improve", "maintain", "report", "support", "test", "troubleshoot", "validate"]
    repeated = []
    words = re.findall(r"[A-Za-z][A-Za-z+#./-]{2,}", text.lower())
    for word in sorted(set(words)):
        if words.count(word) >= 2 and word not in {"and", "the", "with", "for", "you", "will", "our"}:
            append_unique(repeated, word, 20)
    tools = keyword_hits(text, tool_keywords, 30)
    responsibilities = core.get("responsibilities", [])
    return {
        "job_title": core.get("job_title", ""),
        "company": core.get("company", ""),
        "must_have_skills": keyword_hits(text, all_skill_keywords + tool_keywords, 30),
        "nice_to_have_skills": keyword_hits(text, ["asset", "preferred", "nice to have", "familiarity", "experience with"], 10),
        "responsibilities": responsibilities,
        "tools_platforms": tools,
        "domain_knowledge": keyword_hits(text, ["healthcare", "finance", "education", "supply chain", "operations", "security", "game", "inventory"], 15),
        "soft_skills": keyword_hits(text, soft_keywords, 15),
        "repeated_ats_keywords": repeated,
        "action_verbs": keyword_hits(text, action_verbs, 15),
        "evidence_types_to_emphasize": ROLE_LENS_PRIORITIES.get(role_profile["role_family"], [])[:10],
    }


def list_from_nested(value: Any, keys: list[str], limit: int = 20) -> list[str]:
    values = []
    if isinstance(value, dict):
        for key in keys:
            item = value.get(key)
            if isinstance(item, list):
                for entry in item:
                    append_unique(values, str(entry), limit)
            elif isinstance(item, str):
                append_unique(values, item, limit)
    return values


def infer_structural_results_from_code(evidence_items: list[Any], methods: list[str], features: list[str]) -> list[str]:
    combined = " ".join(
        [
            json.dumps(evidence_items, ensure_ascii=False).lower(),
            " ".join(str(item).lower() for item in methods + features),
        ]
    )
    inferred = []
    patterns = [
        (
            ["cache", "caching", "memo", "reuse", "dedup", "skip unchanged", "unchanged"],
            "Reduced repeated work by reusing cached or unchanged evidence paths; no specific latency/QPS metric is verified.",
        ),
        (
            ["index", "sqlite", "query", "where ", "select ", "retrieval", "ranking", "match"],
            "Improved retrieval or matching path with structured lookup/ranking logic; no exact request-speed metric is verified.",
        ),
        (
            ["batch", "chunk", "queue", "parallel", "concurrent", "async", "pipeline"],
            "Restructured processing into chunked, batched, or pipeline steps to reduce manual/repetitive processing; no exact throughput metric is verified.",
        ),
        (
            ["retry", "fallback", "abort", "cancel", "timeout", "error handling", "validation"],
            "Improved reliability and recovery paths through validation, retry/fallback, cancellation, or error handling; no exact error-rate metric is verified.",
        ),
        (
            ["compress", "truncate", "context", "token", "prompt"],
            "Reduced prompt/context overhead through compression or scoped payload construction; no exact cost/token saving is verified.",
        ),
        (
            ["automation", "automated", "workflow", "orchestration"],
            "Reduced repetitive manual steps by automating the workflow; no exact time saving is verified.",
        ),
    ]
    for needles, claim in patterns:
        if any(needle in combined for needle in needles):
            append_unique(inferred, claim, 6)
    return inferred


def build_project_evidence_card(
    name: str,
    section_type: str,
    source_facts: dict[str, Any],
    prompt_evidence: Any,
) -> dict[str, Any]:
    evidence_items = prompt_evidence if isinstance(prompt_evidence, list) else [prompt_evidence]
    technologies = list_from_nested(source_facts, ["tech_stack", "technologies", "skills", "languages_frameworks_detected"], 20)
    methods = list_from_nested(source_facts, ["workflows", "methods", "confirmed_features"], 20)
    features = list_from_nested(source_facts, ["confirmed_features", "features", "core_features"], 20)
    artifacts = []
    source_refs = []
    allowed_claims = []
    forbidden_claims = []
    testing = []
    debugging = []
    documentation = []
    automation = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        nested_evidence_text = json.dumps(item.get("contribution_evidence", []), ensure_ascii=False)
        for signal in re.findall(r"added [A-Za-z0-9 _/-]{3,80}|implemented [A-Za-z0-9 _/-]{3,80}|debugged [A-Za-z0-9 _/-]{3,80}", nested_evidence_text, re.IGNORECASE):
            append_unique(methods, signal, 25)
        for tech in item.get("languages_frameworks_detected", []):
            append_unique(technologies, str(tech), 20)
        for signal in item.get("diff_signals", []) + item.get("resume_relevant_keywords", []):
            signal_text = str(signal)
            append_unique(methods, signal_text, 25)
            lower = signal_text.lower()
            if "test" in lower:
                append_unique(testing, signal_text, 10)
            if "debug" in lower or "error" in lower:
                append_unique(debugging, signal_text, 10)
            if "document" in lower or "readme" in lower:
                append_unique(documentation, signal_text, 10)
            if "automation" in lower or "script" in lower:
                append_unique(automation, signal_text, 10)
        for file_path in item.get("changed_file_paths", []):
            append_unique(artifacts, str(file_path), 20)
        for claim in item.get("allowed_claims", []):
            append_unique(allowed_claims, str(claim), MAX_PROMPT_CLAIMS)
        for claim in item.get("forbidden_claims", []):
            append_unique(forbidden_claims, str(claim), MAX_PROMPT_CLAIMS)
        if item.get("url"):
            append_unique(source_refs, str(item.get("url")), 8)
        if item.get("repository"):
            append_unique(source_refs, str(item.get("repository")), 8)
    identity = source_facts.get("identity") if isinstance(source_facts.get("identity"), dict) else {}
    business_value = list_from_nested(identity, ["core_value", "core_problem", "positioning"], 10)
    data_or_scale = list_from_nested(source_facts, ["metrics", "data_or_scale"], 10)
    inferred_results = infer_structural_results_from_code(evidence_items, methods, features)
    confidence = "high" if allowed_claims or artifacts else "medium" if technologies or methods else "low"
    return {
        "name": name,
        "type": section_type,
        "technologies": technologies,
        "methods": methods[:20],
        "features": features,
        "artifacts": artifacts,
        "data_or_scale": data_or_scale,
        "collaboration_signals": list_from_nested(source_facts, ["collaboration", "team", "agile"], 10),
        "testing_signals": testing,
        "debugging_signals": debugging,
        "documentation_signals": documentation,
        "automation_signals": automation,
        "business_or_user_value": business_value,
        "inferred_results": inferred_results,
        "source_refs": source_refs,
        "confidence": confidence,
        "allowed_claims": allowed_claims,
        "forbidden_claims": forbidden_claims,
    }


def apply_role_lens(evidence_card: dict[str, Any], role_profile: dict[str, Any]) -> dict[str, Any]:
    families = [role_profile.get("role_family", "software_engineering")] + list(role_profile.get("secondary_role_families", []))
    lens_terms = []
    for family in families:
        for term in ROLE_LENS_PRIORITIES.get(str(family), []):
            append_unique(lens_terms, term, 30)
    searchable_fields = (
        evidence_card.get("technologies", [])
        + evidence_card.get("methods", [])
        + evidence_card.get("features", [])
        + evidence_card.get("testing_signals", [])
        + evidence_card.get("debugging_signals", [])
        + evidence_card.get("documentation_signals", [])
        + evidence_card.get("automation_signals", [])
        + evidence_card.get("business_or_user_value", [])
    )
    ranked = []
    for item in searchable_fields:
        text = str(item)
        score = sum(1 for term in lens_terms if term.lower() in text.lower())
        if score:
            ranked.append((score, text))
    ranked.sort(key=lambda item: (-item[0], item[1].lower()))
    priorities = []
    for _, item in ranked:
        append_unique(priorities, item, 12)
    return {
        "role_family": role_profile.get("role_family"),
        "secondary_role_families": role_profile.get("secondary_role_families", []),
        "lens_terms": lens_terms[:15],
        "ranked_evidence_priorities": priorities,
        "bullet_guidance": (
            "Use the highest-ranked evidence priorities first. Emphasize method + implemented "
            "function/process + role-specific value, while respecting allowed/forbidden claims."
        ),
    }


def validate_bullet_quality(bullet: str, evidence_card: dict[str, Any], role_profile: dict[str, Any]) -> dict[str, Any]:
    text = str(bullet or "")
    lower = text.lower()
    issues = []
    unsupported_claims = []
    evidence_text = json.dumps(evidence_card, ensure_ascii=False).lower()
    role_keywords = [str(item).lower() for item in role_profile.get("high_priority_keywords", []) + role_profile.get("role_focus", [])]
    method_terms = ["cache", "caching", "compare", "comparison", "retrieval", "ranking", "validation", "schema", "parsing", "retry", "fallback", "automation", "orchestration", "sql", "reporting"]
    has_tool_or_method = any(str(item).lower() in lower for item in evidence_card.get("technologies", []) + evidence_card.get("methods", [])) or any(term in lower for term in method_terms)
    has_function = any(str(item).lower() in lower for item in evidence_card.get("features", []) + evidence_card.get("artifacts", [])) or any(word in lower for word in ["workflow", "validation", "reporting", "routing", "selection", "tracking", "analysis", "automation", "scan", "scans", "repository", "records"])
    has_challenge = any(word in lower for word in ["debug", "validate", "prevent", "reduce", "recover", "unsupported", "context", "consistency", "accuracy", "error"])
    has_value = any(word in lower for word in ["reduce", "improve", "preserve", "support", "streamline", "prevent", "clarify", "prioritize", "relevance", "reliability", "accuracy"])
    has_role_keyword = not role_keywords or any(keyword and keyword in lower for keyword in role_keywords)
    element_count = sum([has_tool_or_method, has_function, has_challenge, has_value, has_role_keyword])
    vague_terms = [
        "various",
        "multiple",
        "improved functionality",
        "helped",
        "worked on",
        "responsible for",
        "participated in",
        "familiar with",
        "built a system",
        "used python",
        "used technology",
        "developed system",
        "tasks",
    ]
    if not has_tool_or_method:
        issues.append("lacks a concrete supported tool or method")
    if not has_function:
        issues.append("lacks a specific implemented function, process, or artifact")
    if not has_role_keyword:
        issues.append("lacks target-role relevance")
    if element_count < 3:
        issues.append("does not include at least 3 of method/tool, function, challenge, value, and role keyword")
    if any(term in lower for term in vague_terms):
        issues.append("uses vague or generic wording")
    for tool in PROTECTED_UNSUPPORTED_TOOLS:
        if tool.lower() in lower and tool.lower() not in evidence_text:
            unsupported_claims.append(tool)
    if re.search(r"\b\d+%|\$\d+|\b\d+x\b", lower) and not evidence_card.get("data_or_scale"):
        issues.append("may invent unsupported metrics")
    for forbidden in evidence_card.get("forbidden_claims", []):
        forbidden_text = str(forbidden).lower()
        for tool in PROTECTED_UNSUPPORTED_TOOLS:
            if tool.lower() in lower and tool.lower() in forbidden_text:
                unsupported_claims.append(tool)
    specificity_score = max(0, min(100, element_count * 22 - len([i for i in issues if "vague" in i]) * 20))
    jd_alignment_score = 80 if has_role_keyword else 45
    evidence_support_score = 90 if has_tool_or_method and not unsupported_claims else 35 if unsupported_claims else 65
    return {
        "is_strong": not issues and not unsupported_claims,
        "issues": issues,
        "suggested_fix": "Add a supported method/tool, implemented function, and role-specific value grounded in the evidence card.",
        "unsupported_claims": sorted(set(unsupported_claims)),
        "jd_alignment_score": jd_alignment_score,
        "specificity_score": specificity_score,
        "evidence_support_score": evidence_support_score,
    }


def jd_core_for_prompt(jd: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(jd, dict):
        text = json.dumps(jd, ensure_ascii=False)
    else:
        text = str(jd or "")
    lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    lower_text = text.lower()
    title = ""
    company = ""
    title_patterns = [
        r"(?:job title|position|role)\s*[:\-]\s*([^\n|]{2,100})",
        r"\b([A-Z][A-Za-z0-9 /&+-]{2,80}(?:Engineer|Developer|Intern|Analyst|Specialist|Manager))\b",
    ]
    company_patterns = [
        r"(?:company|employer|organization)\s*[:\-]\s*([^\n|]{2,100})",
        r"\bat\s+([A-Z][A-Za-z0-9 .,&+-]{2,80})",
    ]
    for pattern in title_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            title = truncate_text(match.group(1).strip(), 100)
            break
    for pattern in company_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            company = truncate_text(match.group(1).strip(), 100)
            break

    keyword_groups = {
        "testing_ci_infra_keywords": [
            "aws",
            "gcp",
            "kubernetes",
            "ecs",
            "docker",
            "jenkins",
            "github actions",
            "ci/cd",
            "terraform",
            "helm",
            "elk",
            "selenium",
            "cypress",
            "playwright",
            "espresso",
            "maven",
            "gradle",
            "unix",
            "linux",
            "debug",
        ],
        "must_have_keywords": [
            "python",
            "java",
            "javascript",
            "typescript",
            "react",
            "fastapi",
            "sql",
            "sqlite",
            "mongodb",
            "api",
            "backend",
            "frontend",
            "automation",
            "testing",
            "git",
            "bitbucket",
        ],
    }
    extracted: dict[str, list[str]] = {}
    for key, keywords in keyword_groups.items():
        values = []
        for keyword in keywords:
            if keyword in lower_text:
                append_unique(values, keyword.upper() if keyword in {"aws", "gcp", "sql", "api", "ci/cd", "elk"} else keyword, 20)
        extracted[key] = values

    scored_lines = []
    dense_keywords = keyword_groups["testing_ci_infra_keywords"] + keyword_groups["must_have_keywords"]
    for line in lines:
        lower_line = line.lower()
        score = sum(1 for keyword in dense_keywords if keyword in lower_line)
        if any(word in lower_line for word in ["responsibil", "require", "qualification", "experience", "build", "develop", "debug", "test"]):
            score += 1
        if score:
            scored_lines.append((score, line))
    scored_lines.sort(key=lambda item: (-item[0], len(item[1])))
    responsibilities = [truncate_text(line, 240) for _, line in scored_lines[:8]]

    candidate_positioning = []
    positioning_checks = [
        ("backend/API development", ["api", "backend", "fastapi", "server"]),
        ("test automation", ["test", "selenium", "cypress", "playwright", "espresso"]),
        ("build/debug workflows", ["debug", "gradle", "maven", "ci/cd"]),
        ("scripted setup", ["shell", "powershell", "setup", "automation"]),
        ("Git collaboration", ["git", "github", "bitbucket"]),
        ("frontend workflow", ["react", "frontend", "ui"]),
        ("database persistence", ["sql", "sqlite", "mongodb", "database"]),
    ]
    for label, keywords in positioning_checks:
        if any(keyword in lower_text for keyword in keywords):
            append_unique(candidate_positioning, label, 12)

    ats_keywords = ranked_prompt_signals(
        extracted["testing_ci_infra_keywords"] + extracted["must_have_keywords"],
        limit=20,
    )
    return {
        "job_title": title,
        "company": company,
        "role_focus": responsibilities[:5],
        "must_have_keywords": extracted["must_have_keywords"][:12],
        "preferred_keywords": extracted["testing_ci_infra_keywords"][:12],
        "core_responsibilities": responsibilities,
        "testing_ci_infrastructure_keywords": extracted["testing_ci_infra_keywords"][:12],
        "candidate_positioning": candidate_positioning,
        "important_ats_keywords": ats_keywords,
    }


def latex_section_spans(resume_latex: str) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"\\section\{([^}]+)\}", resume_latex))
    spans = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(resume_latex)
        spans.append(
            {
                "name": match.group(1).strip(),
                "start": match.start(),
                "end": end,
                "text": resume_latex[match.start() : end],
            }
        )
    return spans


def find_latex_section(resume_latex: str, section_name: str) -> dict[str, Any] | None:
    target = section_name.lower()
    aliases = {
        "summary": ["professional summary", "summary", "profile", "summary/profile-section"],
        "summary/profile-section": ["professional summary", "summary", "profile"],
        "skills-section": ["technical skills", "skills"],
        "experience-section": ["experience", "work experience"],
        "project": ["projects"],
        "projects": ["projects"],
    }
    names = aliases.get(target, [target])
    for span in latex_section_spans(resume_latex):
        if span["name"].lower() in names or target in span["name"].lower():
            return span
    return None


def find_project_block(section_text: str, block_hint: str = "") -> str:
    hint = str(block_hint or "").lower().strip()
    starts = [match.start() for match in re.finditer(r"\\resumeProjectHeading", section_text)]
    if not starts:
        return ""
    starts.append(len(section_text))
    blocks = [section_text[starts[index] : starts[index + 1]] for index in range(len(starts) - 1)]
    if hint:
        for block in blocks:
            if hint in block.lower():
                return block
    return blocks[0]


def count_resume_items(latex_block: str) -> int:
    return len(re.findall(r"\\(?:resumeItem|item)\b", latex_block))


def resume_block_for_prompt(resume_latex: str, section_name: str, block_hint: str | None = None) -> dict[str, Any]:
    if section_name.lower().startswith("project"):
        section = find_latex_section(resume_latex, "projects")
        if section:
            block = find_project_block(section["text"], block_hint or "")
            if block:
                return {
                    "scope": "project_block",
                    "section_name": section["name"],
                    "block_hint": block_hint or "",
                    "latex": block,
                    "original_bullet_count": count_resume_items(block),
                    "length_style_constraints": "Keep bullet count close to the original block unless the candidate clearly requires fewer concise bullets.",
                }
            return {
                "scope": "section",
                "section_name": section["name"],
                "block_hint": block_hint or "",
                "latex": section["text"],
                "original_bullet_count": count_resume_items(section["text"]),
                "length_style_constraints": "Preserve complete Projects section structure.",
            }
    section = find_latex_section(resume_latex, section_name)
    if not section:
        section = find_latex_section(resume_latex, section_name.replace("-section", ""))
    if section:
        return {
            "scope": "section",
            "section_name": section["name"],
            "block_hint": block_hint or "",
            "latex": section["text"],
            "original_bullet_count": count_resume_items(section["text"]),
            "length_style_constraints": "Preserve the complete section boundary and current LaTeX style.",
        }
    return {
        "scope": "document",
        "section_name": section_name,
        "block_hint": block_hint or "",
        "latex": agent.extract_latex_document(resume_latex) or resume_latex,
        "original_bullet_count": count_resume_items(resume_latex),
        "length_style_constraints": "No matching section was found; preserve full document validity.",
    }


def replace_resume_block(current_resume: str, block_payload: dict[str, Any], replacement: str) -> str:
    document = agent.extract_latex_document(replacement)
    if document:
        return document
    original = str(block_payload.get("latex") or "")
    replacement = replacement.strip()
    if original and original in current_resume and replacement:
        return current_resume.replace(original, replacement, 1)
    if block_payload.get("scope") == "document" and replacement:
        return replacement
    raise HTTPException(status_code=400, detail="Compact retry did not return a replaceable LaTeX block.")


RAW_PROMPT_MARKERS = ["diff --git", "@@", "+++ ", "--- "]
RAW_PROMPT_KEYS = [
    "patch",
    "raw_diff",
    "full_diff",
    "readme",
    "file_content",
    "validation_blob",
    "bullet_writer_validation",
]


def payload_has_raw_markers(payload: Any) -> bool:
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    return any(marker.lower() in serialized for marker in RAW_PROMPT_MARKERS)


def payload_has_raw_keys(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in RAW_PROMPT_KEYS:
                return True
            if payload_has_raw_keys(value):
                return True
    if isinstance(payload, list):
        return any(payload_has_raw_keys(item) for item in payload)
    return False


def approx_payload_size(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=False))


def merge_retry_payload_for_prompt(original_payload: dict[str, Any]) -> dict[str, Any]:
    section_name = original_payload["section_name"]
    candidate = original_payload["candidate"]
    if section_name.lower().startswith("project"):
        candidate_summary = candidate_for_prompt(candidate)
    else:
        candidate_summary = compact_value_for_prompt(candidate, 900, 8)
    block_hint = str(
        candidate.get("project_name")
        or candidate.get("source_name")
        or candidate.get("project_id")
        or original_payload.get("block_hint")
        or ""
    )
    target_block = resume_block_for_prompt(
        original_payload["current_resume"],
        section_name,
        block_hint=block_hint,
    )
    allowed_claims = []
    forbidden_claims = []
    if isinstance(candidate_summary, dict):
        for claim in candidate_summary.get("allowed_claims", []):
            append_unique(allowed_claims, claim, 10)
        for claim in candidate_summary.get("forbidden_claims", []):
            append_unique(forbidden_claims, claim, 10)
        validation = candidate_summary.get("validation", {})
        if isinstance(validation, dict):
            for claim in validation.get("supported_claims", []):
                append_unique(allowed_claims, claim, 10)
            for claim in validation.get("unsupported_claims", []):
                append_unique(forbidden_claims, claim, 10)
    payload = {
        "retry_reason": "normal merge payload exceeded model context window",
        "compact_jd": jd_core_for_prompt(original_payload["job_description"]),
        "target_resume_block": target_block,
        "candidate": candidate_summary,
        "allowed_claims": allowed_claims[:10],
        "forbidden_claims": forbidden_claims[:10],
        "formatting_rules": [
            "Preserve valid LaTeX and complete begin/end boundaries.",
            "Return only the merged target LaTeX block or section, not the full resume unless target scope is document.",
            "Do not invent technologies, impact metrics, deployment, ownership, or infrastructure claims.",
            "Do not add AWS, Kubernetes, Terraform, Jenkins, Docker, or cloud claims unless allowed_claims explicitly support them.",
        ],
        "length_budget": {
            "original_bullet_count": target_block.get("original_bullet_count", 0),
            "max_candidates": 1,
            "keep_length_close_to_original": True,
        },
        "latex_safety_rules": [
            "Do not cut off LaTeX commands.",
            "Keep itemize/list environments balanced.",
            "Keep section or project heading format consistent with the target block.",
        ],
    }
    if approx_payload_size(payload) > 30000:
        payload["candidate"] = compact_value_for_prompt(payload["candidate"], 500, 4)
    return payload


def build_retry_merge_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are receiving a compact emergency merge payload because the full resume merge payload exceeded the model context window.

Use only the provided compact JD, target LaTeX block or section, candidate, and claim boundaries.
Preserve valid LaTeX. Do not invent technologies, impact metrics, deployment, ownership, or unsupported claims.
Prefer the candidate wording that best matches the JD while remaining evidence-grounded.
Keep the section/project length close to the original budget.
Return only the merged LaTeX block or section. Do not include Markdown fences or explanation.

Compact retry payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def call_model_with_context_retry(
    normal_payload: dict[str, Any],
    build_retry_payload: Callable[[dict[str, Any]], dict[str, Any]],
    call_model: Callable[[dict[str, Any]], str],
) -> str:
    try:
        return call_model(normal_payload)
    except Exception as error:
        if not is_context_window_error(error):
            raise
        retry_payload = build_retry_payload(normal_payload)
        normal_size = approx_payload_size(normal_payload)
        retry_size = approx_payload_size(retry_payload)
        print(
            "Resume merge context retry: "
            f"normal_payload_chars={normal_size}, retry_payload_chars={retry_size}, "
            f"raw_markers_removed={not payload_has_raw_markers(retry_payload)}, "
            f"raw_keys_removed={not payload_has_raw_keys(retry_payload)}"
        )
        if payload_has_raw_markers(retry_payload) or payload_has_raw_keys(retry_payload):
            raise HTTPException(
                status_code=500,
                detail="Compact retry payload still contains raw diff or oversized evidence fields.",
            ) from error
        try:
            return call_model({**normal_payload, "retry_payload": retry_payload, "retry": True})
        except Exception as retry_error:
            raise HTTPException(
                status_code=getattr(retry_error, "status_code", 500),
                detail=f"Compact retry after context-window error failed: {getattr(retry_error, 'detail', retry_error)}",
            ) from retry_error


def project_list_from_memory(project_memory: dict[str, Any]) -> list[dict[str, Any]]:
    projects = project_memory.get("projects", []) if isinstance(project_memory, dict) else []
    if isinstance(projects, dict):
        projects = [projects]
    return [project for project in projects if isinstance(project, dict)]


def compact_project_for_prompt(project: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": project.get("project_id"),
        "project_name": project.get("project_name") or project.get("name"),
        "identity": project.get("identity", {}),
        "tech_stack": project.get("tech_stack", []),
        "workflows": project.get("workflows", []),
        "confirmed_features": project.get("confirmed_features", []),
        "real_metrics": project.get("real_metrics", {}),
        "star_facts": project.get("star_facts", []),
        "recent_changes": project.get("recent_changes", []),
        "rag_refs": project.get("rag_refs", {}),
    }


def project_query(project: dict[str, Any]) -> str:
    queries = agent.project_queries_from_memory({"projects": [project]})
    if queries:
        return queries[0].get("query", "")
    return str(project.get("project_name") or project.get("name") or project.get("project_id") or "")


def retrieve_evidence_for_project(project: dict[str, Any]) -> list[dict[str, Any]]:
    query = project_query(project)
    if not query:
        return []
    return agent.MEMORY_STORE.read_github_contexts(query=query)


STAR_FIELD_LABELS = {
    "situation": "Situation / 业务背景",
    "task": "Task / 个人贡献",
    "action": "Action / 技术动作",
    "result": "Result / 结果数据",
}


def project_display_name(project: dict[str, Any]) -> str:
    return str(project.get("project_name") or project.get("name") or project.get("project_id") or "Project").strip()


def normalize_star_field(field_type: str) -> str:
    field = str(field_type or "").strip().lower()
    aliases = {
        "metric": "result",
        "metrics": "result",
        "business_context": "situation",
        "contribution_scope": "task",
        "technical_detail": "action",
    }
    return aliases.get(field, field if field in STAR_FIELD_LABELS else "result")


def project_star_facts(project: dict[str, Any]) -> list[dict[str, Any]]:
    facts = project.get("star_facts", [])
    if isinstance(facts, list):
        return [fact for fact in facts if isinstance(fact, dict)]
    return []


def fact_matches_field(fact: dict[str, Any], field_type: str) -> bool:
    return normalize_star_field(str(fact.get("field_type") or "")) == field_type


def fact_is_no_data(fact: dict[str, Any]) -> bool:
    text = f"{fact.get('raw_answer', '')} {fact.get('normalized_fact', '')}".lower()
    return any(token in text for token in ["no data", "not sure", "unknown", "不知道", "没有数据", "没有量化", "无数据"])


def star_fact_question_key(project: dict[str, Any], field_type: str, missing_info_type: str = "") -> str:
    project_key = str(project.get("project_id") or project_display_name(project)).strip().lower()
    project_key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", project_key).strip("-")
    return f"{project_key}:{field_type}:{missing_info_type or field_type}"


def star_field_status(project: dict[str, Any], evidence_card: dict[str, Any], field_type: str) -> dict[str, Any]:
    facts = [fact for fact in project_star_facts(project) if fact_matches_field(fact, field_type)]
    if facts:
        latest = facts[-1]
        status = "not_needed" if field_type == "result" and fact_is_no_data(latest) else "user_confirmed"
        return {
            "status": status,
            "source": latest.get("source", "user_confirmed"),
            "summary": str(latest.get("normalized_fact") or latest.get("raw_answer") or "")[:220],
        }

    identity = project.get("identity") if isinstance(project.get("identity"), dict) else {}
    if field_type == "situation":
        values = [
            identity.get("core_problem"),
            identity.get("core_value"),
            identity.get("background"),
            identity.get("positioning"),
            *(project.get("workflows", []) if isinstance(project.get("workflows"), list) else []),
        ]
        summary = "; ".join(str(value).strip() for value in values if str(value or "").strip())[:220]
        if summary:
            return {"status": "found_in_memory", "source": "project_memory", "summary": summary}
    if field_type == "task":
        values = []
        for key in ["ownership", "contribution_scope", "collaboration", "team"]:
            value = project.get(key)
            if isinstance(value, list):
                values.extend(value)
            elif value:
                values.append(value)
        if not values and (evidence_card.get("allowed_claims") or evidence_card.get("artifacts")):
            return {
                "status": "inferred_from_code",
                "source": "project_memory_and_code",
                "summary": "本地项目记忆和代码证据可支持用户参与实现，但不会夸大为独立负责/主导。",
            }
        summary = "; ".join(str(value).strip() for value in values if str(value or "").strip())[:220]
        if summary:
            return {"status": "found_in_memory", "source": "project_memory", "summary": summary}
    if field_type == "action":
        values = (
            evidence_card.get("methods", [])
            + evidence_card.get("technologies", [])
            + evidence_card.get("features", [])
            + evidence_card.get("artifacts", [])
        )
        summary = "; ".join(str(value).strip() for value in values if str(value or "").strip())[:220]
        if summary:
            return {"status": "found_in_code", "source": "project_memory_and_code", "summary": summary}
    if field_type == "result":
        values = []
        for key in ["real_metrics", "metrics", "results", "impact", "data_or_scale"]:
            value = project.get(key)
            if isinstance(value, dict):
                values.extend(f"{k}: {v}" for k, v in value.items() if str(v or "").strip())
            elif isinstance(value, list):
                values.extend(value)
            elif value:
                values.append(value)
        values.extend(evidence_card.get("data_or_scale", []))
        summary = "; ".join(str(value).strip() for value in values if str(value or "").strip())[:220]
        if summary:
            return {"status": "found_in_memory", "source": "project_memory", "summary": summary}
        inferred_results = evidence_card.get("inferred_results", [])
        if inferred_results:
            summary = "; ".join(str(value).strip() for value in inferred_results if str(value or "").strip())[:220]
            return {
                "status": "inferred_from_code",
                "source": "local_diff_and_code_evidence",
                "summary": summary,
            }
    return {"status": "missing", "source": "", "summary": ""}


def build_star_completion(project: dict[str, Any], evidence_card: dict[str, Any]) -> dict[str, Any]:
    fields = {
        field: star_field_status(project, evidence_card, field)
        for field in ["situation", "task", "action", "result"]
    }
    missing = [field for field, payload in fields.items() if payload["status"] == "missing"]
    return {"fields": fields, "missing_fields": missing}


def star_question_for_project(
    project: dict[str, Any],
    completion: dict[str, Any],
    asked_question_keys: set[str],
) -> Optional[dict[str, Any]]:
    priority = ["result", "task", "situation", "action"]
    project_name = project_display_name(project)
    found_summaries = [
        payload["summary"]
        for payload in completion["fields"].values()
        if payload.get("summary")
    ][:3]
    found_text = "；".join(found_summaries) if found_summaries else "我已经先检查了本地项目记忆、已保存 STAR facts 和可用代码证据。"
    for field_type in priority:
        if field_type not in completion["missing_fields"]:
            continue
        missing_info_type = "time_saved_or_verified_impact" if field_type == "result" else field_type
        question_key = star_fact_question_key(project, field_type, missing_info_type)
        if question_key in asked_question_keys:
            continue
        if field_type == "result":
            question = (
                f"我正在完善 {project_name} 这个项目的 Result 部分。"
                f"我已经先检查到：{found_text}\n\n"
                "现在只缺一个关键信息：有没有真实的结果或对比数据？例如生成/处理时间、人工修改时间、准确率、错误率、请求速度或成本。"
                "没有数据也可以直接说“没有数据/不知道”，我会保守写成具体技术产出，不编造百分比。"
            )
            detail = "等待补充：Result / 结果数据"
        elif field_type == "task":
            question = (
                f"我正在确认 {project_name} 这个项目的 Task / 个人贡献。"
                f"本地证据里已经看到：{found_text}\n\n"
                "这里你是独立负责核心模块、主导设计开发，还是和别人协作？我需要确认能否写 Led / Built / Designed，避免夸大贡献。"
            )
            detail = "等待补充：Task / 个人贡献"
        elif field_type == "situation":
            question = (
                f"我正在补齐 {project_name} 的 Situation / 业务背景。"
                f"我已经先读取到：{found_text}\n\n"
                "这个项目主要解决你投递、学习、业务或用户流程里的哪个具体问题？一句话说明场景就可以。"
            )
            detail = "等待补充：Situation / 业务背景"
        else:
            question = (
                f"我正在核对 {project_name} 的 Action / 技术动作。"
                f"我已经先检查到：{found_text}\n\n"
                "这个项目最能代表你的一个关键技术实现是什么？例如检索、缓存、并发、索引、RAG、API 设计、自动化 pipeline 或错误处理。"
            )
            detail = "等待补充：Action / 技术动作"
        return {
            "project_id": str(project.get("project_id") or ""),
            "project_name": project_name,
            "field_type": field_type,
            "missing_info_type": missing_info_type,
            "question_key": question_key,
            "stage_id": f"complete-star-{question_key}",
            "stage_label": f"补全 STAR：{project_name}",
            "stage_detail": detail,
            "context_label": f"Agent · {project_name}",
            "prompt": question,
        }
    return None


def build_resume_star_check(body: ResumeStarCheckBody) -> dict[str, Any]:
    try:
        job_description = agent.read_job_description()
        resume = agent.read_resume()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    project_memory = read_current_project_memory()
    selected_projects = select_staged_projects(job_description, resume, project_memory, body.allow_project_selection)
    asked_keys = {str(key) for key in body.asked_question_keys if str(key).strip()}
    stages = []
    summaries = []
    next_question = None
    for project in selected_projects:
        assert_agent_task_not_cancelled()
        project_name = project_display_name(project)
        evidence = retrieve_evidence_for_project(project)
        evidence_card = build_project_evidence_card(project_name, "project", compact_project_for_prompt(project), compact_github_evidence_for_prompt(evidence))
        completion = build_star_completion(project, evidence_card)
        missing_labels = [STAR_FIELD_LABELS[field] for field in completion["missing_fields"]]
        stage_id = f"complete-star-{str(project.get('project_id') or project_name).lower().replace(' ', '-')}"
        stage = {
            "id": stage_id,
            "label": f"补全 STAR：{project_name}",
            "status": "done" if not completion["missing_fields"] else "pending",
            "detail": "本地信息已足够" if not completion["missing_fields"] else f"待核对：{' / '.join(missing_labels)}",
            "projectName": project_name,
        }
        question = star_question_for_project(project, completion, asked_keys)
        if question and next_question is None:
            stage.update({
                "id": question["stage_id"],
                "status": "waiting_for_user",
                "detail": question["stage_detail"],
                "fieldType": question["field_type"],
            })
            next_question = question
        stages.append(stage)
        found = [
            f"{STAR_FIELD_LABELS[field]}={payload['status']}"
            for field, payload in completion["fields"].items()
            if payload["status"] != "missing"
        ]
        summaries.append(f"{project_name}: " + ("; ".join(found) if found else "暂无可用 STAR 信息"))
    return {
        "stages": stages,
        "next_question": next_question,
        "messages": [
            "我已先读取 job_description.txt、resume.txt、project_memory.json 和可用 GitHub/代码证据。",
            "STAR 扫描结果：" + "；".join(summaries[:4]),
        ],
    }


def save_resume_star_fact(body: ResumeStarFactBody) -> dict[str, Any]:
    field_type = normalize_star_field(body.field_type)
    raw_answer = body.raw_answer.strip()
    if not raw_answer:
        raise HTTPException(status_code=400, detail="raw_answer is required.")
    project_memory = read_current_project_memory()
    projects = project_list_from_memory(project_memory)
    target_project = None
    for project in projects:
        if project_matches(project, project_name=body.project_name, project_id=body.project_id):
            target_project = project
            break
    if target_project is None:
        target_project = {"project_id": body.project_id, "project_name": body.project_name or "Project"}
        projects.append(target_project)
        project_memory["projects"] = projects
    no_data = fact_is_no_data({"raw_answer": raw_answer, "normalized_fact": body.normalized_fact})
    normalized_fact = body.normalized_fact.strip() or (
        "No quantified result is available; use a conservative qualitative result without invented metrics."
        if no_data and field_type == "result"
        else raw_answer
    )
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    question_key = body.question_key or star_fact_question_key(target_project, field_type, body.missing_info_type)
    fact = {
        "question_key": question_key,
        "project_id": str(target_project.get("project_id") or body.project_id or ""),
        "project_name": project_display_name(target_project),
        "field_type": field_type,
        "missing_info_type": body.missing_info_type or field_type,
        "raw_answer": raw_answer,
        "normalized_fact": normalized_fact,
        "source": "user_confirmed",
        "confidence": body.confidence or "high",
        "created_at": now,
        "updated_at": now,
    }
    facts = project_star_facts(target_project)
    replaced = False
    for index, existing in enumerate(facts):
        if existing.get("question_key") == question_key:
            fact["created_at"] = existing.get("created_at") or now
            facts[index] = fact
            replaced = True
            break
    if not replaced:
        facts.append(fact)
    target_project["star_facts"] = facts
    agent.write_project_memory_file(project_memory)
    return {
        "saved": True,
        "project_id": fact["project_id"],
        "project_name": fact["project_name"],
        "field_type": field_type,
        "question_key": question_key,
        "replaced": replaced,
        "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
    }


def select_staged_projects(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    allow_project_selection: bool,
) -> list[dict[str, Any]]:
    projects = project_list_from_memory(project_memory)
    if not projects:
        return []
    if not allow_project_selection:
        return projects[:MAX_STAGED_PROJECTS]

    compact_projects = [{"index": index, **compact_project_for_prompt(project)} for index, project in enumerate(projects)]
    prompt = f"""
Select the strongest Project Memory projects for this job application.

Rules:
- Use only the job description, original resume, and Project Memory project summaries.
- Prefer exactly {PREFERRED_RESUME_PROJECTS} projects for a one-page resume.
- Select {MAX_STAGED_PROJECTS} projects only when the third project is clearly job-critical and can still fit a one-page resume.
- Never select more than {MAX_STAGED_PROJECTS} projects.
- Prefer projects already in the resume unless another Project Memory project is clearly stronger.
- Return selected_indices in strongest-to-weakest order; higher-ranked projects receive more resume bullet space later.
- Return ONLY valid JSON with exactly this shape:
  {{"selected_indices": [0], "reason": "short explanation"}}

Job description:
{truncate_text(job_description, 12000)}

Original resume:
{truncate_text(resume, 18000)}

Project Memory projects:
{json.dumps(compact_projects, ensure_ascii=False, indent=2)}
"""
    try:
        payload = extract_json_object(run_text_task(prompt))
    except HTTPException:
        return projects[:MAX_STAGED_PROJECTS]

    indices = payload.get("selected_indices", [])
    if not isinstance(indices, list):
        return projects[:MAX_STAGED_PROJECTS]

    selected = []
    for value in indices:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(projects) and projects[index] not in selected:
            selected.append(projects[index])
        if len(selected) >= MAX_STAGED_PROJECTS:
            break
    return selected or projects[:MAX_STAGED_PROJECTS]


def run_resume_bullet_writer_tool(
    section_type: str,
    source_name: str,
    job_description: str,
    resume: str,
    source_facts: dict[str, Any],
    evidence: Any,
    existing_bullets: list[str],
    language: str,
    extra_rules: str = "",
) -> dict[str, Any]:
    prompt_source_facts = compact_value_for_prompt(source_facts, 1200, 8)
    prompt_evidence = compact_github_evidence_for_prompt(evidence)
    role_profile = classify_role_family(job_description)
    jd_requirements = jd_requirements_for_prompt(job_description)
    evidence_card = build_project_evidence_card(source_name, section_type, source_facts, prompt_evidence)
    role_lens = apply_role_lens(evidence_card, role_profile)
    allowed_claims = []
    forbidden_claims = []
    for item in prompt_evidence if isinstance(prompt_evidence, list) else [prompt_evidence]:
        if not isinstance(item, dict):
            continue
        for claim in item.get("allowed_claims", []):
            append_unique(allowed_claims, claim, MAX_PROMPT_CLAIMS)
        for claim in item.get("forbidden_claims", []):
            append_unique(forbidden_claims, claim, MAX_PROMPT_CLAIMS)
    for claim in evidence_card.get("allowed_claims", []):
        append_unique(allowed_claims, claim, MAX_PROMPT_CLAIMS)
    for claim in evidence_card.get("forbidden_claims", []):
        append_unique(forbidden_claims, claim, MAX_PROMPT_CLAIMS)
    prompt = f"""
{RESUME_BULLET_WRITER_PROMPT}

Return JSON with exactly these keys:
  "section_type": "project" | "experience",
  "source_name": string,
  "job_alignment": string,
  "star_analysis": array of objects with keys "candidate_fact", "situation", "task", "action", "result", "missing_star_fields", "evidence_source",
  "react_analysis": array of objects with keys "candidate_fact", "why_writable", "why_it_belongs", "business_capability", "technical_capability", "risk_avoided",
  "final_bullets": array of objects with keys "bullet", "evidence", "confidence",
  "skills_to_emphasize": array of strings,
  "risks": array of strings

Output language requirement:
{output_language_instruction(language)}

Section type:
{section_type}

Source name:
{source_name}

Extra rules:
{extra_rules}

Role profile:
{json.dumps(role_profile, ensure_ascii=False, indent=2)}

Structured JD requirements:
{json.dumps(jd_requirements, ensure_ascii=False, indent=2)}

Original resume:
{truncate_text(resume, 22000)}

Source facts:
{json.dumps(prompt_source_facts, ensure_ascii=False, indent=2)}

Project / experience evidence card:
{json.dumps(evidence_card, ensure_ascii=False, indent=2)}

Role lens priorities:
{json.dumps(role_lens, ensure_ascii=False, indent=2)}

Existing bullets:
{json.dumps(existing_bullets, ensure_ascii=False, indent=2)}

Allowed resume claims from compressed evidence:
{json.dumps(allowed_claims, ensure_ascii=False, indent=2)}

Forbidden / unsupported resume claims:
{json.dumps(forbidden_claims, ensure_ascii=False, indent=2)}

STAR enforcement:
- Populate star_analysis before final_bullets.
- If a metric, ownership level, business scale, or result is not supported, list it in missing_star_fields.
- final_bullets must not include unsupported numbers, inflated ownership, or generic stack-only wording.
- evidence_card.inferred_results may be used as conservative qualitative Result evidence from local diff/code
  analysis, but never convert it into verified QPS, P99, latency, cost, accuracy, or percentage claims unless
  data_or_scale or user-confirmed star_facts explicitly supports the number.
- Treat live user guidance from the progress modal as user-provided STAR evidence when present.
"""
    payload = extract_json_object(run_text_task(prompt))
    for key in ["star_analysis", "react_analysis", "final_bullets", "skills_to_emphasize", "risks"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    bullet_quality = []
    for item in payload["final_bullets"]:
        bullet = item.get("bullet") if isinstance(item, dict) else str(item)
        quality = validate_bullet_quality(str(bullet or ""), evidence_card, role_profile)
        bullet_quality.append({"bullet": str(bullet or ""), **quality})

    validation = json.loads(
        agent.write_resume_bullets(
            section_type=section_type,
            source_name=source_name,
            job_alignment=str(payload.get("job_alignment", "")),
            source_facts=source_facts,
            evidence=evidence if isinstance(evidence, list) else [evidence],
            existing_bullets=existing_bullets,
            react_analysis=payload["react_analysis"],
            final_bullets=payload["final_bullets"],
            language=language,
        )
    )
    payload["bullet_writer_validation"] = {
        "accepted": validation.get("accepted", False),
        "issues": validation.get("issues", []),
        "required_mode": validation.get("required_mode", ""),
        "required_pattern": validation.get("required_pattern", ""),
        "role_family": role_profile.get("role_family"),
        "bullet_quality": bullet_quality,
    }
    quality_issues = []
    unsupported_claims = []
    for quality in bullet_quality:
        if not quality.get("is_strong"):
            quality_issues.extend(quality.get("issues", []))
        unsupported_claims.extend(quality.get("unsupported_claims", []))
    if quality_issues or unsupported_claims:
        payload["bullet_writer_validation"]["accepted"] = False
        payload["bullet_writer_validation"]["issues"].extend(sorted(set(quality_issues)))
        payload["bullet_writer_validation"]["unsupported_claims"] = sorted(set(unsupported_claims))
    if validation.get("issues"):
        payload["risks"].extend(validation["issues"])
    if quality_issues:
        payload["risks"].extend(sorted(set(quality_issues)))
    payload["allowed_claims"] = allowed_claims
    payload["forbidden_claims"] = forbidden_claims
    payload["role_profile"] = role_profile
    payload["jd_requirements"] = jd_requirements
    payload["evidence_card"] = evidence_card
    payload["role_lens"] = role_lens
    return payload


def build_project_resume_candidate(
    job_description: str,
    resume: str,
    project: dict[str, Any],
    evidence: list[dict[str, Any]],
    language: str,
    progress_guidance: str = "",
) -> dict[str, Any]:
    source_facts = compact_project_for_prompt(project)
    payload = run_resume_bullet_writer_tool(
        section_type="project",
        source_name=str(project.get("project_name") or project.get("name") or project.get("project_id") or ""),
        job_description=job_description,
        resume=resume,
        source_facts=source_facts,
        evidence=evidence,
        existing_bullets=[],
        language=language,
        extra_rules=(
            "Project Memory is the primary source of truth. Chroma evidence is supporting proof only. "
            "Select only the strongest concise bullets; final layout will allocate more bullets to higher-ranked "
            "projects and fewer bullets to lower-ranked projects. The first bullet must explain what the project is "
            "and what workflow or problem it addresses. Return fit, keep_or_replace, and fit_reason if possible."
            + progress_guidance
        ),
    )
    payload["project_id"] = payload.get("project_id") or project.get("project_id") or ""
    payload["project_name"] = payload.get("project_name") or project.get("project_name") or project.get("name") or ""
    payload["fit"] = payload.get("fit") if payload.get("fit") in {"high", "medium", "low"} else "medium"
    payload["keep_or_replace"] = payload.get("keep_or_replace") or "update"
    payload["fit_reason"] = payload.get("fit_reason") or payload.get("job_alignment", "")
    payload["recommended_bullets"] = payload.get("final_bullets", [])
    return payload


def build_skills_resume_candidate(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    project_candidates: list[dict[str, Any]],
    language: str,
    progress_guidance: str = "",
) -> dict[str, Any]:
    prompt_project_candidates = compact_bullet_candidates_for_prompt(project_candidates)
    prompt_project_memory = compact_value_for_prompt(project_memory, 1200, 8)
    prompt = f"""
Generate structured Skills-section tailoring recommendations.

Rules:
- Do not output a full resume.
- Use only the job description, original resume, Project Memory, and staged project candidates.
- Skills may be reordered, grouped, emphasized, or removed when weakly relevant.
- Add a skill only if it is supported by the original resume, Project Memory, or staged project candidates.
- Do not invent tools, frameworks, platforms, databases, languages, certifications, or proficiency levels.
- Return ONLY valid JSON with exactly these keys:
  "skills_strategy": string,
  "skills_to_emphasize": array of strings,
  "skills_to_deemphasize": array of strings,
  "skills_to_add_if_supported": array of objects with keys "skill", "supporting_source", "confidence",
  "skills_to_remove_or_avoid": array of strings,
  "recommended_skills_section": string,
  "risks": array of strings

Output language requirement:
{output_language_instruction(language)}

Job description:
{truncate_text(job_description, 12000)}

Original resume:
{truncate_text(resume, 22000)}

Project Memory:
{json.dumps(prompt_project_memory, ensure_ascii=False, indent=2)}

Staged project candidates:
{json.dumps(prompt_project_candidates, ensure_ascii=False, indent=2)}
{progress_guidance}
"""
    payload = extract_json_object(run_text_task(prompt))
    for key in ["skills_to_emphasize", "skills_to_deemphasize", "skills_to_add_if_supported", "skills_to_remove_or_avoid", "risks"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    return payload


def build_experience_resume_candidate(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    project_candidates: list[dict[str, Any]],
    skills_candidate: dict[str, Any],
    allow_experience_removal: bool,
    language: str,
    progress_guidance: str = "",
) -> dict[str, Any]:
    prompt_project_candidates = compact_bullet_candidates_for_prompt(project_candidates)
    prompt_skills_candidate = compact_value_for_prompt(skills_candidate, 900, 8)
    source_facts = {
        "project_memory": compact_value_for_prompt(project_memory, 1200, 8),
        "staged_project_candidates": prompt_project_candidates,
        "staged_skills_candidate": prompt_skills_candidate,
        "allow_experience_removal": allow_experience_removal,
    }
    payload = run_resume_bullet_writer_tool(
        section_type="experience",
        source_name="Experience",
        job_description=job_description,
        resume=resume,
        source_facts=source_facts,
        evidence=prompt_project_candidates,
        existing_bullets=[],
        language=language,
        extra_rules=(
            "Generate Experience-section tailoring recommendations. You may reorder factual Experience "
            "bullets, rewrite them for relevance and clarity, and remove weak or redundant bullets. "
            "Preserve every existing Experience entry unless allow_experience_removal is true. Do not "
            "invent employers, roles, dates, responsibilities, technologies, metrics, seniority, or ownership. "
            "Return experience_strategy, entry_recommendations, bullets_to_emphasize, bullets_to_deemphasize, "
            "and unsupported_claims_to_avoid if possible."
            + progress_guidance
        ),
    )
    final_bullets = payload.get("final_bullets", [])
    payload["experience_strategy"] = payload.get("experience_strategy") or payload.get("job_alignment", "")
    if not isinstance(payload.get("entry_recommendations"), list):
        payload["entry_recommendations"] = [
            {
                "entry_name": "Experience",
                "action": "rewrite_bullets",
                "reason": payload.get("job_alignment", ""),
                "recommended_bullets": final_bullets,
                "remove_bullets": [],
                "risks": payload.get("risks", []),
            }
        ]
    for key in ["bullets_to_emphasize", "bullets_to_deemphasize", "unsupported_claims_to_avoid", "risks"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    return payload


def build_summary_resume_candidate(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    project_candidates: list[dict[str, Any]],
    skills_candidate: dict[str, Any],
    experience_candidate: dict[str, Any],
    language: str,
    progress_guidance: str = "",
) -> dict[str, Any]:
    prompt_project_memory = compact_value_for_prompt(project_memory, 1200, 8)
    prompt_project_candidates = compact_bullet_candidates_for_prompt(project_candidates)
    prompt_skills_candidate = compact_value_for_prompt(skills_candidate, 900, 8)
    prompt_experience_candidate = compact_value_for_prompt(experience_candidate, 900, 8)
    prompt = f"""
Generate structured resume Summary/Profile tailoring recommendations.

Rules:
- Do not output a full resume.
- Use only the job description, original resume, Project Memory, staged project candidates,
  staged Skills candidate, and staged Experience candidate.
- The summary must be concise, factual, and aligned to the target job.
- Do not invent years of experience, job titles, domains, achievements, metrics, seniority, or technologies.
- Do not overclaim ownership or production impact unless supported by the staged candidates or original resume.
- If the original resume has no Summary/Profile section, recommend whether to add one only if it improves ATS/relevance.
- Return ONLY valid JSON with exactly these keys:
  "summary_strategy": string,
  "recommended_summary": string,
  "keywords_to_include": array of strings,
  "claims_to_avoid": array of strings,
  "evidence_basis": array of strings,
  "risks": array of strings

Output language requirement:
{output_language_instruction(language)}

Job description:
{truncate_text(job_description, 12000)}

Original resume:
{truncate_text(resume, 26000)}

Project Memory:
{json.dumps(prompt_project_memory, ensure_ascii=False, indent=2)}

Staged project candidates:
{json.dumps(prompt_project_candidates, ensure_ascii=False, indent=2)}

Staged Skills candidate:
{json.dumps(prompt_skills_candidate, ensure_ascii=False, indent=2)}

Staged Experience candidate:
{json.dumps(prompt_experience_candidate, ensure_ascii=False, indent=2)}
{progress_guidance}
"""
    payload = extract_json_object(run_text_task(prompt))
    for key in ["keywords_to_include", "claims_to_avoid", "evidence_basis", "risks"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    return payload


def apply_resume_project_candidate(
    job_description: str,
    current_resume: str,
    project_candidate: dict[str, Any],
    body: TailorBody,
    index: int,
    total: int,
) -> str:
    normal_payload = {
        "section_name": "Project-section",
        "job_description": job_description,
        "current_resume": current_resume,
        "candidate": project_candidate,
        "block_hint": str(project_candidate.get("project_name") or project_candidate.get("project_id") or ""),
        "index": index,
        "total": total,
        "allow_project_selection": body.allow_project_selection,
        "allow_experience_removal": body.allow_experience_removal,
        "language": body.language,
    }

    def call_merge_model(payload: dict[str, Any]) -> str:
        if payload.get("retry"):
            retry_payload = payload["retry_payload"]
            block = run_text_task(build_retry_merge_prompt(retry_payload))
            return replace_resume_block(current_resume, retry_payload["target_resume_block"], block)

        prompt_project_candidate = compact_bullet_candidate_for_prompt(payload["candidate"])
        prompt = (
            output_language_instruction(body.language)
            + original_resume_language_instruction("tailored_resume")
            + f"""
Apply exactly one staged Project-section candidate to the current LaTeX resume.

Loop step:
- Project candidate {payload["index"]} of {payload["total"]}.
- Return the complete updated LaTeX resume.
- Keep all previous edits already present in the current LaTeX resume.
- Use only this one project candidate for this step.
- Do not request or infer raw Chroma evidence.
- Do not create project bullet claims outside this candidate's final_bullets / recommended_bullets.
- Preserve the STAR grounding from the staged candidate. Do not add metrics, ownership level,
  scale, or results that are not present in star_analysis, final_bullets, user guidance, or evidence.
- Reject generic stack-only wording such as "used X to develop Y"; keep action + module + technical method + supported result/value.
- Project selection allowed: {payload["allow_project_selection"]}
- If project selection is not allowed, keep the existing resume project list and only update factual wording.
- Do not invent unsupported metrics, technologies, responsibilities, employers, roles, dates, or repository facts.
- Return only LaTeX code with no Markdown fences and no analysis text.

Job description:
{truncate_text(payload["job_description"], 12000)}

Current LaTeX resume:
{truncate_text(payload["current_resume"], 30000)}

One staged Project candidate:
{json.dumps(prompt_project_candidate, ensure_ascii=False, indent=2)}
"""
        )
        return run_text_task(prompt)

    answer = call_model_with_context_retry(normal_payload, merge_retry_payload_for_prompt, call_merge_model)
    if not agent.looks_like_latex_resume(answer):
        raise HTTPException(status_code=400, detail=f"Agent did not return valid LaTeX resume code after project merge step {index}.")
    return answer


def apply_resume_section_candidate(
    section_name: str,
    job_description: str,
    current_resume: str,
    candidate: dict[str, Any],
    body: TailorBody,
) -> str:
    normal_payload = {
        "section_name": section_name,
        "job_description": job_description,
        "current_resume": current_resume,
        "candidate": candidate,
        "block_hint": section_name,
        "allow_project_selection": body.allow_project_selection,
        "allow_experience_removal": body.allow_experience_removal,
        "language": body.language,
    }

    def call_merge_model(payload: dict[str, Any]) -> str:
        if payload.get("retry"):
            retry_payload = payload["retry_payload"]
            block = run_text_task(build_retry_merge_prompt(retry_payload))
            return replace_resume_block(current_resume, retry_payload["target_resume_block"], block)

        prompt_candidate = compact_value_for_prompt(payload["candidate"], 900, 8)
        prompt = (
            output_language_instruction(body.language)
            + original_resume_language_instruction("tailored_resume")
            + f"""
Apply exactly one staged {section_name} candidate to the current LaTeX resume.

Rules:
- Return the complete updated LaTeX resume.
- Keep all previous edits already present in the current LaTeX resume.
- Use only this staged {section_name} candidate for this step.
- For Project and Experience bullet wording, use only ReAct bullet writer candidates already present in the staged data.
- Preserve STAR grounding. Do not add unsupported business scale, ownership level, before/after metrics,
  users, latency, QPS, cost, accuracy, or production claims during the merge.
- For Experience bullets, keep the user's personal contribution explicit and supported.
- Do not invent unsupported metrics, technologies, responsibilities, employers, roles, dates, or repository facts.
- Entire Experience entry removal allowed: {payload["allow_experience_removal"]}
- Return only LaTeX code with no Markdown fences and no analysis text.

Job description:
{truncate_text(payload["job_description"], 12000)}

Current LaTeX resume:
{truncate_text(payload["current_resume"], 30000)}

One staged {section_name} candidate:
{json.dumps(prompt_candidate, ensure_ascii=False, indent=2)}
"""
        )
        return run_text_task(prompt)

    answer = call_model_with_context_retry(normal_payload, merge_retry_payload_for_prompt, call_merge_model)
    if not agent.looks_like_latex_resume(answer):
        raise HTTPException(status_code=400, detail=f"Agent did not return valid LaTeX resume code after {section_name} merge step.")
    return answer


def project_bullet_budget(index: int, total: int) -> int:
    if total <= 1:
        return 3
    if total == 2:
        return 3 if index == 1 else 2
    if index == 1:
        return 3
    if index == 2:
        return 2
    return 1


def project_layout_candidates_for_prompt(project_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout_candidates = []
    total = len(project_candidates)
    for index, candidate in enumerate(project_candidates, start=1):
        compact_candidate = compact_bullet_candidate_for_prompt(candidate)
        layout_candidates.append(
            {
                **compact_candidate,
                "rank": index,
                "target_bullet_count": project_bullet_budget(index, total),
            }
        )
    return layout_candidates


def apply_projects_section_layout(
    job_description: str,
    current_resume: str,
    project_candidates: list[dict[str, Any]],
    body: TailorBody,
) -> str:
    if not body.allow_project_selection:
        return current_resume

    layout_candidates = project_layout_candidates_for_prompt(project_candidates[:MAX_STAGED_PROJECTS])
    normal_payload = {
        "section_name": "Project-section",
        "job_description": job_description,
        "current_resume": current_resume,
        "candidate": {
            "selected_project_candidates": layout_candidates,
            "preferred_project_count": PREFERRED_RESUME_PROJECTS,
            "maximum_project_count": MAX_STAGED_PROJECTS,
            "ranking_rule": "Higher-ranked selected projects should receive more bullets than lower-ranked projects.",
            "one_page_rule": "Prefer a one-page resume; remove weak or non-selected projects before expanding content.",
        },
        "block_hint": "Projects",
        "allow_project_selection": body.allow_project_selection,
        "allow_experience_removal": body.allow_experience_removal,
        "language": body.language,
    }

    def call_layout_model(payload: dict[str, Any]) -> str:
        if payload.get("retry"):
            retry_payload = payload["retry_payload"]
            block = run_text_task(build_retry_merge_prompt(retry_payload))
            return replace_resume_block(current_resume, retry_payload["target_resume_block"], block)

        prompt_candidate = compact_value_for_prompt(payload["candidate"], 2400, 10)
        prompt = (
            output_language_instruction(body.language)
            + original_resume_language_instruction("tailored_resume")
            + f"""
Apply final Projects-section layout constraints to the current LaTeX resume.

Rules:
- Return the complete updated LaTeX resume.
- Keep the resume optimized for one page.
- Prefer exactly {PREFERRED_RESUME_PROJECTS} Projects-section entries.
- Keep {MAX_STAGED_PROJECTS} Projects-section entries only when the third selected project is clearly job-critical and the resume can still fit one page.
- Never keep more than {MAX_STAGED_PROJECTS} Projects-section entries.
- Remove projects that are not in selected_project_candidates when project selection is allowed.
- Preserve selected_project_candidates ranking order from strongest to weakest.
- Higher-ranked projects must have more bullets than lower-ranked projects when multiple projects are shown.
- Use target_bullet_count for each selected project: rank 1 usually 3 bullets, rank 2 usually 2 bullets, rank 3 usually 1 bullet.
- Lower-ranked project bullets should be compact and only keep the most job-relevant factual claim.
- Do not create new bullet wording outside the selected candidates' final_bullets / recommended_bullets.
- Do not invent unsupported metrics, technologies, responsibilities, employers, roles, dates, or repository facts.
- Preserve LaTeX validity and existing section style.
- Return only LaTeX code with no Markdown fences and no analysis text.

Job description:
{truncate_text(payload["job_description"], 12000)}

Current LaTeX resume:
{truncate_text(payload["current_resume"], 30000)}

Projects-section layout candidate data:
{json.dumps(prompt_candidate, ensure_ascii=False, indent=2)}
"""
        )
        return run_text_task(prompt)

    answer = call_model_with_context_retry(normal_payload, merge_retry_payload_for_prompt, call_layout_model)
    if not agent.looks_like_latex_resume(answer):
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code after Projects-section layout step.")
    return answer


def build_resume_gap_report(
    role_profile: dict[str, Any],
    jd_requirements: dict[str, Any],
    project_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    evidence_text = json.dumps(
        [candidate.get("evidence_card", {}) for candidate in project_candidates],
        ensure_ascii=False,
    ).lower()
    missing_evidence = []
    for keyword in jd_requirements.get("must_have_skills", [])[:20]:
        if str(keyword).lower() not in evidence_text:
            append_unique(missing_evidence, f"No strong project evidence found for `{keyword}`.", 12)
    weak_sections = []
    for candidate in project_candidates:
        validation = candidate.get("bullet_writer_validation", {})
        if not isinstance(validation, dict):
            continue
        weak_bullets = [
            item
            for item in validation.get("bullet_quality", [])
            if isinstance(item, dict) and not item.get("is_strong")
        ]
        if weak_bullets:
            append_unique(
                weak_sections,
                f"{candidate.get('project_name') or candidate.get('source_name')}: {len(weak_bullets)} weak/generic bullet candidates.",
                12,
            )
    recommended_updates = []
    family = role_profile.get("role_family", "")
    if family == "it_analyst" and "powershell" in [str(item).lower() for item in jd_requirements.get("must_have_skills", [])]:
        if "powershell" not in evidence_text:
            recommended_updates.append("Add a real PowerShell/setup/support automation script before claiming stronger PowerShell experience.")
    if family == "data_analyst" and "sql" in [str(item).lower() for item in jd_requirements.get("must_have_skills", [])]:
        if "sql" not in evidence_text:
            recommended_updates.append("Add a concrete SQL query/reporting artifact before emphasizing SQL analysis experience.")
    safe_keywords = []
    for candidate in project_candidates:
        card = candidate.get("evidence_card", {})
        if not isinstance(card, dict):
            continue
        for value in card.get("technologies", []) + card.get("methods", []) + card.get("features", []):
            if str(value).lower() in [str(keyword).lower() for keyword in jd_requirements.get("must_have_skills", [])]:
                append_unique(safe_keywords, str(value), 15)
    unsafe_keywords = []
    for tool in PROTECTED_UNSUPPORTED_TOOLS:
        if tool.lower() in json.dumps(jd_requirements, ensure_ascii=False).lower() and tool.lower() not in evidence_text:
            append_unique(unsafe_keywords, tool, 15)
    return {
        "missing_evidence": missing_evidence,
        "weak_sections": weak_sections,
        "recommended_project_updates": recommended_updates,
        "safe_keywords_to_add": safe_keywords,
        "unsafe_keywords_to_avoid": unsafe_keywords,
    }


def merge_staged_resume(
    job_description: str,
    resume: str,
    project_candidates: list[dict[str, Any]],
    skills_candidate: dict[str, Any],
    experience_candidate: dict[str, Any],
    summary_candidate: dict[str, Any],
    body: TailorBody,
) -> str:
    current_resume = resume
    for index, project_candidate in enumerate(project_candidates, start=1):
        current_resume = apply_resume_project_candidate(
            job_description,
            current_resume,
            project_candidate,
            body,
            index,
            len(project_candidates),
        )

    current_resume = apply_projects_section_layout(
        job_description,
        current_resume,
        project_candidates,
        body,
    )

    current_resume = apply_resume_section_candidate(
        "Skills-section",
        job_description,
        current_resume,
        skills_candidate,
        body,
    )
    current_resume = apply_resume_section_candidate(
        "Experience-section",
        job_description,
        current_resume,
        experience_candidate,
        body,
    )
    current_resume = apply_resume_section_candidate(
        "Summary/Profile-section",
        job_description,
        current_resume,
        summary_candidate,
        body,
    )
    return current_resume


def tailor_resume_staged(body: TailorBody) -> dict[str, Any]:
    try:
        job_description = agent.read_job_description()
        resume = agent.read_resume()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        project_memory = json.loads(agent.read_project_memory())
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="project_memory.json is not valid JSON.") from error

    application_hint = resolve_saved_application_hint(job_description)
    progress_guidance = agent_progress_guidance_text(body.agent_progress_messages)
    selected_projects = select_staged_projects(job_description, resume, project_memory, body.allow_project_selection)
    if not selected_projects:
        raise HTTPException(
            status_code=400,
            detail="Project Memory has no projects. Run GitHub extraction to populate project_memory.json first.",
        )

    candidates = []
    selected_project_memory = {"projects": [compact_project_for_prompt(project) for project in selected_projects]}
    for project in selected_projects:
        evidence = retrieve_evidence_for_project(project)
        candidates.append(build_project_resume_candidate(
            job_description,
            resume,
            project,
            evidence,
            body.language,
            progress_guidance,
        ))

    skills_candidate = build_skills_resume_candidate(
        job_description,
        resume,
        selected_project_memory,
        candidates,
        body.language,
        progress_guidance,
    )
    experience_candidate = build_experience_resume_candidate(
        job_description,
        resume,
        selected_project_memory,
        candidates,
        skills_candidate,
        body.allow_experience_removal,
        body.language,
        progress_guidance,
    )
    summary_candidate = build_summary_resume_candidate(
        job_description,
        resume,
        selected_project_memory,
        candidates,
        skills_candidate,
        experience_candidate,
        body.language,
        progress_guidance,
    )
    role_profile = classify_role_family(job_description)
    jd_requirements = jd_requirements_for_prompt(job_description)
    gap_report = build_resume_gap_report(role_profile, jd_requirements, candidates)

    answer = merge_staged_resume(job_description, resume, candidates, skills_candidate, experience_candidate, summary_candidate, body)
    if not agent.looks_like_latex_resume(answer):
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code.")

    agent.save_tailored_resume(answer, company=application_hint["company"], role=application_hint["role"])
    tailored_resume_outputs = list_output_files(agent.TAILORED_RESUME_OUTPUT_DIR, ".txt", limit=1)
    response: dict[str, Any] = {
        "saved": True,
        "path": tailored_resume_outputs[0]["path"] if tailored_resume_outputs else str(agent.latest_tailored_resume_path()),
        "output_path": tailored_resume_outputs[0]["path"] if tailored_resume_outputs else None,
        "content": agent.read_tailored_resume(),
        "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
        "staged": True,
        "staged_project_count": len(candidates),
        "staged_project_candidates": candidates,
        "staged_skills_candidate": skills_candidate,
        "staged_experience_candidate": experience_candidate,
        "staged_summary_candidate": summary_candidate,
        "role_profile": role_profile,
        "jd_requirements": jd_requirements,
        "gap_report": gap_report,
    }
    if body.include_application_hint:
        response["application_hint"] = application_hint
    return response


def read_github_memory_repo_source() -> str:
    repositories = agent.MEMORY_STORE.list_github_repositories()
    return "\n".join(
        f"https://github.com/{item['repository']}"
        for item in repositories
        if item.get("repository")
    )


def project_scope_to_github_url(project_name: str = "", project_id: str = "") -> str:
    scope = (project_name or project_id or "").strip()
    if not scope:
        return ""

    url_match = re.search(
        r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        scope,
    )
    if url_match:
        owner, repo = url_match.groups()
        return f"https://github.com/{owner}/{repo.removesuffix('.git')}"

    repo_match = re.fullmatch(r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", scope)
    if repo_match:
        owner, repo = repo_match.groups()
        return f"https://github.com/{owner}/{repo.removesuffix('.git')}"

    return ""


def scoped_project_github_source(project_name: str = "", project_id: str = "") -> str:
    direct_url = project_scope_to_github_url(project_name=project_name, project_id=project_id)
    if direct_url:
        return f"Direct GitHub repository from project scope:\n{direct_url}"

    current_memory = load_memory_for_merge()
    profile_scope = (
        scoped_project_memory(current_memory, project_name=project_name, project_id=project_id)
        if isinstance(current_memory, dict)
        else {"projects": []}
    )

    project_memory_scope: dict[str, Any] = {"projects": []}
    try:
        project_memory = read_current_project_memory()
    except HTTPException:
        project_memory = {}
    if isinstance(project_memory, dict):
        project_memory_scope = scoped_project_memory(project_memory, project_name=project_name, project_id=project_id)

    scoped_text = (
        f"Profile project memory:\n{json.dumps(profile_scope, ensure_ascii=False, indent=2)}\n\n"
        f"Project Memory JSON scope:\n{json.dumps(project_memory_scope, ensure_ascii=False, indent=2)}"
    )
    scoped_repositories = []
    scoped_text_lower = scoped_text.lower()
    for item in agent.MEMORY_STORE.list_github_repositories():
        repository = str(item.get("repository", "")).strip()
        if repository and repository.lower() in scoped_text_lower:
            scoped_repositories.append(f"https://github.com/{repository}")

    for owner, repo in re.findall(
        r"(?:Repository|repository|repo)\s*:\s*([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        scoped_text,
    ):
        url = f"https://github.com/{owner}/{repo.removesuffix('.git')}"
        if url not in scoped_repositories:
            scoped_repositories.append(url)

    return f"{scoped_text}\n\nStored GitHub repositories for this project:\n" + "\n".join(scoped_repositories)


def read_github_repo_source(resume_source: str, project_name: str = "", project_id: str = "") -> str:
    if project_name.strip() or project_id.strip():
        return scoped_project_github_source(project_name=project_name, project_id=project_id)

    if resume_source == "resume":
        return agent.read_resume()
    if resume_source == "tailored_resume":
        return agent.read_tailored_resume()
    if resume_source == "memory":
        return f"{agent.read_memory()}\n\n{read_github_memory_repo_source()}"
    if resume_source == "resume_and_memory":
        return (
            f"{agent.read_resume()}\n\n{agent.read_memory()}\n\n"
            f"{read_github_memory_repo_source()}"
        )
    if resume_source == "tailored_resume_and_resume_and_memory":
        try:
            tailored_resume = agent.read_tailored_resume()
        except (FileNotFoundError, ValueError):
            tailored_resume = ""
        return (
            f"{tailored_resume}\n\n{agent.read_resume()}\n\n{agent.read_memory()}\n\n"
            f"{read_github_memory_repo_source()}"
        )
    raise HTTPException(
        status_code=400,
        detail=(
            "resume_source must be 'resume', 'tailored_resume', 'memory', "
            "'resume_and_memory', or 'tailored_resume_and_resume_and_memory'."
        ),
    )


GITHUB_REPO_SCAN_STATE_VERSION = 1


def project_memory_prompt_hash() -> str:
    return hashlib.sha256(PROJECT_MEMORY_FROM_REPO_ANALYSIS_PROMPT.encode("utf-8")).hexdigest()[:16]


def repo_state_key(repo_info: dict[str, Any]) -> str:
    return f"{repo_info['owner']}/{repo_info['repo']}"


def load_github_repo_scan_state() -> dict[str, Any]:
    path = agent.GITHUB_REPO_SCAN_STATE_PATH
    if not path.exists():
        return {"version": GITHUB_REPO_SCAN_STATE_VERSION, "repositories": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": GITHUB_REPO_SCAN_STATE_VERSION, "repositories": {}}
    if not isinstance(state, dict):
        return {"version": GITHUB_REPO_SCAN_STATE_VERSION, "repositories": {}}
    repositories = state.get("repositories")
    if not isinstance(repositories, dict):
        state["repositories"] = {}
    state["version"] = GITHUB_REPO_SCAN_STATE_VERSION
    return state


def save_github_repo_scan_state(state: dict[str, Any]) -> None:
    path = agent.GITHUB_REPO_SCAN_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def response_header(headers: dict[str, Any], name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return str(value)
    return ""


def fetch_github_remote_state(repo_info: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    owner = urllib.parse.quote(repo_info["owner"], safe="")
    repo = urllib.parse.quote(repo_info["repo"], safe="")
    repository = repo_state_key(repo_info)
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    remote_state: dict[str, Any] = {
        "repository": repository,
        "url": repo_info["url"],
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        repo_headers = {}
        if previous.get("repo_etag"):
            repo_headers["If-None-Match"] = previous["repo_etag"]
        repo_response = agent.github_api_request(base_url, extra_headers=repo_headers)
        if repo_response["status"] == 304:
            default_branch = previous.get("default_branch", "")
            remote_state["repo_not_modified"] = True
            remote_state["repo_etag"] = previous.get("repo_etag", "")
        else:
            repo_data = json.loads(repo_response["text"])
            default_branch = str(repo_data.get("default_branch") or "")
            remote_state["repo_not_modified"] = False
            remote_state["repo_etag"] = response_header(repo_response.get("headers", {}), "ETag")
            remote_state["pushed_at"] = repo_data.get("pushed_at")
            remote_state["updated_at"] = repo_data.get("updated_at")

        remote_state["default_branch"] = default_branch
        if not default_branch:
            remote_state["changed"] = True
            remote_state["change_reason"] = "default branch is unavailable"
            return remote_state

        commit_headers = {}
        if previous.get("latest_commit_etag"):
            commit_headers["If-None-Match"] = previous["latest_commit_etag"]
        commit_response = agent.github_api_request(
            f"{base_url}/commits/{urllib.parse.quote(default_branch, safe='')}",
            extra_headers=commit_headers,
        )
        if commit_response["status"] == 304:
            latest_commit_sha = previous.get("latest_commit_sha", "")
            remote_state["commit_not_modified"] = True
            remote_state["latest_commit_etag"] = previous.get("latest_commit_etag", "")
        else:
            commit_data = json.loads(commit_response["text"])
            latest_commit_sha = str(commit_data.get("sha") or "")
            remote_state["commit_not_modified"] = False
            remote_state["latest_commit_etag"] = response_header(commit_response.get("headers", {}), "ETag")
            commit_info = commit_data.get("commit") if isinstance(commit_data, dict) else {}
            committer = commit_info.get("committer") if isinstance(commit_info, dict) else {}
            remote_state["latest_commit_date"] = committer.get("date") if isinstance(committer, dict) else None

        previous_sha = str(previous.get("latest_commit_sha") or "")
        previous_branch = str(previous.get("default_branch") or "")
        remote_state["latest_commit_sha"] = latest_commit_sha
        remote_state["changed"] = (
            not previous.get("context")
            or not latest_commit_sha
            or latest_commit_sha != previous_sha
            or default_branch != previous_branch
        )
        if not previous.get("context"):
            remote_state["change_reason"] = "no cached context"
        elif default_branch != previous_branch:
            remote_state["change_reason"] = "default branch changed"
        elif latest_commit_sha != previous_sha:
            remote_state["change_reason"] = "latest commit changed"
        else:
            remote_state["change_reason"] = "unchanged"
        return remote_state
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        remote_state["changed"] = True
        remote_state["error"] = (
            agent.describe_http_error(error)
            if isinstance(error, urllib.error.HTTPError)
            else str(error)
        )
        remote_state["change_reason"] = "remote state check failed"
        return remote_state


def fetch_github_context_api(
    approved: bool,
    resume_source: str = "resume",
    project_name: str = "",
    project_id: str = "",
    force_refresh: bool = False,
    reanalyze_cached: bool = False,
    agent_progress_messages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    if not approved:
        return {"saved": False, "message": "GitHub context fetch was not approved."}

    try:
        repo_source = read_github_repo_source(resume_source, project_name=project_name, project_id=project_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    repos = agent.extract_github_repos(repo_source)
    if not repos:
        return {"saved": False, "message": "No GitHub repositories found in the selected source."}

    github_identities = agent.read_github_identities()
    if not agent.identity_has_values(github_identities):
        raise HTTPException(
            status_code=400,
            detail=f"Add GitHub identities to {agent.GITHUB_ACCOUNTS_PATH} first.",
        )

    scan_state = load_github_repo_scan_state()
    repositories_state = scan_state.setdefault("repositories", {})
    repo_contexts = []
    fetched_contexts = []
    scan_results = []
    prompt_hash = project_memory_prompt_hash()
    needs_project_memory_reanalysis = bool(reanalyze_cached)
    for repo in repos:
        assert_agent_task_not_cancelled()
        key = repo_state_key(repo)
        previous_state = repositories_state.get(key, {})
        if not isinstance(previous_state, dict):
            previous_state = {}

        remote_state = fetch_github_remote_state(repo, previous_state)
        should_fetch = force_refresh or remote_state.get("changed") or not previous_state.get("context")
        previous_sha = str(previous_state.get("latest_commit_sha") or "")
        latest_sha = str(remote_state.get("latest_commit_sha") or previous_sha)
        can_incremental_fetch = (
            should_fetch
            and not force_refresh
            and previous_state.get("context")
            and previous_sha
            and latest_sha
            and latest_sha != previous_sha
        )
        cache_status = "fetch"
        if remote_state.get("error"):
            repo_context = {
                "url": repo["url"],
                "repository": key,
                "error": f"Could not check repository update state: {remote_state['error']}",
                "contribution_evidence": [],
            }
            cache_status = "remote-state-error"
        elif not should_fetch:
            repo_context = previous_state["context"]
            cache_status = "reused"
        elif can_incremental_fetch:
            repo_context = agent.fetch_incremental_github_repo_context(
                repo,
                previous_state["context"],
                github_identities,
                previous_sha,
                latest_sha,
            )
            repo_context["verified_github_identities"] = github_identities
            incremental_errors = [
                evidence.get("error")
                for evidence in repo_context.get("contribution_evidence", [])
                if isinstance(evidence, dict) and evidence.get("error")
            ]
            if incremental_errors:
                repo_context = agent.fetch_github_repo_context(repo)
                repo_context["verified_github_identities"] = github_identities
                if repo_context.get("error"):
                    repo_context["contribution_evidence"] = []
                else:
                    repo_context["contribution_evidence"] = agent.fetch_user_commits_for_repo(
                        repo, github_identities
                    )
                cache_status = "fetch"
            else:
                cache_status = "incremental"
            fetched_contexts.append(repo_context)
        else:
            repo_context = agent.fetch_github_repo_context(repo)
            repo_context["verified_github_identities"] = github_identities
            if repo_context.get("error"):
                repo_context["contribution_evidence"] = []
            else:
                repo_context["contribution_evidence"] = agent.fetch_user_commits_for_repo(
                    repo, github_identities
                )
            fetched_contexts.append(repo_context)

        if cache_status != "remote-state-error" and agent.has_usable_repo_context([repo_context]):
            repositories_state[key] = {
                **previous_state,
                "repository": key,
                "url": repo["url"],
                "default_branch": remote_state.get("default_branch") or previous_state.get("default_branch", ""),
                "latest_commit_sha": remote_state.get("latest_commit_sha") or previous_state.get("latest_commit_sha", ""),
                "repo_etag": remote_state.get("repo_etag") or previous_state.get("repo_etag", ""),
                "latest_commit_etag": remote_state.get("latest_commit_etag") or previous_state.get("latest_commit_etag", ""),
                "checked_at": remote_state.get("checked_at"),
                "context": repo_context,
            }
            if (
                cache_status == "fetch"
                or cache_status == "incremental"
                or repositories_state[key].get("project_memory_prompt_hash") != prompt_hash
            ):
                needs_project_memory_reanalysis = True

        scan_results.append(
            {
                "repository": key,
                "cache_status": cache_status,
                "changed": bool(remote_state.get("changed")),
                "change_reason": remote_state.get("change_reason", ""),
                "default_branch": remote_state.get("default_branch") or previous_state.get("default_branch", ""),
                "latest_commit_sha": remote_state.get("latest_commit_sha") or previous_state.get("latest_commit_sha", ""),
                "error": remote_state.get("error", ""),
            }
        )
        repo_contexts.append(repo_context)

    assert_agent_task_not_cancelled()
    path = agent.CHROMA_DB_PATH
    if fetched_contexts:
        path = agent.save_github_context_output(fetched_contexts)

    if needs_project_memory_reanalysis:
        assert_agent_task_not_cancelled()
        project_memory_update = update_project_memory_from_repo_analysis(
            repo_contexts,
            agent_progress_messages=agent_progress_messages,
        )
        for result in scan_results:
            key = result["repository"]
            if key in repositories_state and not result.get("error"):
                repositories_state[key]["project_memory_prompt_hash"] = prompt_hash
                repositories_state[key]["project_memory_analyzed_at"] = datetime.now().isoformat(timespec="seconds")
    else:
        project_memory_update = {
            "updated": False,
            "source": "repo-analysis-cache",
            "additions": [],
            "project_memory": read_current_project_memory(),
            "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
            "message": "Repository commit SHAs and Project Memory analysis prompt are unchanged; reused cached GitHub context.",
        }

    assert_agent_task_not_cancelled()
    save_github_repo_scan_state(scan_state)
    return {
        "saved": agent.has_usable_repo_context(repo_contexts),
        "path": str(path),
        "project_name": project_name.strip(),
        "project_id": project_id.strip(),
        "project_memory_update": project_memory_update,
        "scan_results": scan_results,
        "fetched_repository_count": len(fetched_contexts),
        "reused_repository_count": sum(1 for result in scan_results if result["cache_status"] == "reused"),
        "scan_state_path": str(agent.GITHUB_REPO_SCAN_STATE_PATH),
        "context": repo_contexts,
    }


@app.get("/api/status")
def get_status():
    file_metadata = {}
    for name, path in FILE_MAP.items():
        if name == "tailored_resume":
            path = agent.latest_tailored_resume_path()
        if path.exists():
            file_metadata[name] = {
                "mtime": path.stat().st_mtime,
                "mtime_ms": int(path.stat().st_mtime * 1000),
            }
        else:
            file_metadata[name] = {"mtime": None, "mtime_ms": None}

    return {
        "provider": agent.current_provider,
        "model": agent.current_model,
        "supports_images": provider_supports_images(agent.current_provider),
        "provider_configs": build_provider_config_status()["providers"],
        "files": {name: file_ready(name, path) for name, path in FILE_MAP.items()},
        "file_metadata": file_metadata,
        "outputs": {
            "analysis": list_job_analysis_history(),
            "tailored_resumes": list_output_files(agent.TAILORED_RESUME_OUTPUT_DIR, ".txt"),
            "tailored_resume_pdfs": list_output_files(TAILORED_RESUME_PDF_OUTPUT_DIR, ".pdf"),
            "cover_letters": list_output_files(agent.COVER_LETTER_OUTPUT_DIR, ".txt"),
            "interview_prep": list_output_files(agent.INTERVIEW_PREP_OUTPUT_DIR, ".txt"),
            "chat_sessions": list_output_files(CHAT_SESSION_OUTPUT_DIR, ".txt"),
            "github_context": [
                {
                    "name": "Chroma GitHub evidence",
                    "path": str(agent.CHROMA_DB_PATH),
                }
            ]
            if agent.MEMORY_STORE.github_count()
            else [],
        },
    }


@app.post("/api/shutdown")
def shutdown(body: ShutdownBody):
    chat_session = save_chat_session(body.chat_session) if body.chat_session else None
    schedule_shutdown()
    return {
        "shutdown": True,
        "grace_seconds": SHUTDOWN_GRACE_SECONDS,
        "chat_session": chat_session,
    }


@app.post("/api/session/open")
def open_session():
    canceled = cancel_pending_shutdown()
    return {"open": True, "canceled_shutdown": canceled}


@app.post("/api/chat/session")
def persist_chat_session(body: ChatSessionBody):
    return save_chat_session(body)


@app.post("/api/provider")
def set_provider(body: ProviderBody):
    adapter, provider_name = get_adapter(body.provider)
    config = PROVIDER_CONFIGS.get(provider_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unsupported provider '{body.provider}'.")
    agent.current_provider = provider_name
    agent.current_adapter = adapter
    agent.current_model = adapter.default_model()
    values = {
        "MODEL_PROVIDER": provider_name,
        config["model_env"]: agent.current_model,
    }
    write_env_values(values)
    for key, value in values.items():
        os.environ[key] = value
    return {"provider": agent.current_provider, "model": agent.current_model}


@app.get("/api/provider-configs")
def get_provider_configs():
    return build_provider_config_status()


@app.post("/api/provider-configs")
def save_provider_config(body: ProviderConfigBody):
    provider = normalize_provider(body.provider)
    config = PROVIDER_CONFIGS.get(provider)
    if not config:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{body.provider}'.",
        )

    api_key = body.api_key.strip()
    base_url = body.base_url.strip()
    model = body.model.strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    if config["requires_base_url"] and not base_url:
        raise HTTPException(status_code=400, detail="Base URL is required for this provider.")

    values = {
        config["api_key_env"]: api_key,
        config["base_url_env"]: base_url or config["default_base_url"],
        config["model_env"]: model or config["default_model"],
        "MODEL_PROVIDER": provider,
    }
    write_env_values(values)
    for key, value in values.items():
        os.environ[key] = value

    adapter = agent.create_model_adapter(provider)
    agent.current_provider = provider
    agent.current_adapter = adapter
    agent.current_model = values[config["model_env"]]

    return {
        "saved": True,
        "provider": agent.current_provider,
        "model": agent.current_model,
        "provider_configs": build_provider_config_status()["providers"],
    }


@app.post("/api/model")
def set_model(body: ModelBody):
    model = body.model.strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model name cannot be empty.")
    config = PROVIDER_CONFIGS.get(normalize_provider(agent.current_provider))
    if not config:
        raise HTTPException(status_code=400, detail=f"Unsupported provider '{agent.current_provider}'.")
    agent.current_model = model
    values = {
        "MODEL_PROVIDER": agent.current_provider,
        config["model_env"]: agent.current_model,
    }
    write_env_values(values)
    for key, value in values.items():
        os.environ[key] = value
    return {"provider": agent.current_provider, "model": agent.current_model}


@app.get("/api/files/{name}")
def get_file(name: str):
    if name not in FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown file: {name}")
    ready, content = read_file_content(name)
    return {"name": name, "ready": ready, "content": content}


@app.get("/api/output-file")
def get_output_file(path: str = Query(..., min_length=1)):
    output_file = resolve_output_file(path)
    if output_file.suffix.lower() not in {".txt", ".md", ".json", ".tex"}:
        raise HTTPException(status_code=400, detail="This output file cannot be displayed as text.")
    try:
        content = output_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="This output file is not UTF-8 text.") from error
    return {"path": str(output_file), "content": content}


@app.post("/api/output-file/launch")
def launch_output_file(path: str = Query(..., min_length=1)):
    output_file = resolve_output_file(path)
    if output_file.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF output files can be opened here.")
    try:
        if sys.platform == "win32":
            try:
                os.startfile(str(output_file))
            except OSError:
                subprocess.Popen(
                    ["rundll32.exe", "shell32.dll,OpenAs_RunDLL", str(output_file)],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_file)])
        else:
            launch_command = None
            for candidate in (
                ("xdg-open",),
                ("gio", "open"),
                ("gnome-open",),
                ("kde-open5",),
                ("kde-open",),
            ):
                if shutil.which(candidate[0]):
                    launch_command = [*candidate, str(output_file)]
                    break
            if launch_command is None:
                raise OSError("No desktop file opener was found.")
            subprocess.Popen(launch_command)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not launch the system PDF viewer.") from error
    return {"opened": True, "path": str(output_file)}


@app.delete("/api/output-file")
def delete_output_file(path: str = Query(..., min_length=1)):
    output_file = resolve_output_file(path)
    try:
        output_file.unlink()
    except OSError as error:
        raise HTTPException(status_code=500, detail="Could not delete output file.") from error
    if output_file.parent == agent.ANALYSIS_OUTPUT_DIR.resolve():
        remove_analysis_history_path(output_file)
    return {"deleted": True, "path": str(output_file)}


@app.put("/api/files/{name}")
def put_file(name: str, body: FileBody):
    if name not in FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown file: {name}")
    try:
        save_file_content(name, body.content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"saved": True, "name": name}


@app.get("/api/prompt")
def get_prompt():
    return {
        "content": agent.PROMPT_PATH.read_text(encoding="utf-8")
        if agent.PROMPT_PATH.exists()
        else "",
        "example": read_prompt_example(),
    }


@app.put("/api/prompt")
def put_prompt(body: PromptBody):
    content = body.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
    agent.PROMPT_PATH.write_text(content + "\n", encoding="utf-8")
    agent.SYSTEM_PROMPT = content
    return {"saved": True, "content": content}


@app.post("/api/agent/ask")
def agent_ask(body: AgentAskBody):
    with agent_task_context(body.agent_task_id):
        if body.provider:
            agent.current_provider = body.provider.lower().strip()
            agent.current_adapter = get_adapter(body.provider)[0]
        if body.model:
            agent.current_model = body.model.strip()

        images = validate_agent_images(body.images)
        ensure_provider_supports_images(body.provider or agent.current_provider, images)
        message = body.message.strip()
        if not message and not images:
            raise HTTPException(status_code=400, detail="Message or image attachment is required.")
        if not message:
            message = "Please inspect the attached image and describe what you can do with it."

        answer = run_agent_task(
            message
            + output_language_instruction(body.language)
            + original_resume_language_instruction_for_request(message)
            + agent_progress_guidance_text(body.agent_progress_messages),
            body.provider,
            body.model,
            images,
        )
        return {
            "answer": answer,
            "artifacts": {
                "analysis_path": None,
                "tailored_resume_path": str(agent.latest_tailored_resume_path())
                if agent.file_is_ready(agent.latest_tailored_resume_path())
                else None,
                "cover_letter_path": str(agent.COVER_LETTER_PATH)
                if agent.file_is_ready(agent.COVER_LETTER_PATH)
                else None,
            },
        }


@app.post("/api/agent/progress-guidance")
def handle_agent_progress_guidance(body: AgentProgressGuidanceBody):
    with agent_task_context(body.agent_task_id):
        guidance = body.user_message.strip()
        if not guidance:
            raise HTTPException(status_code=400, detail="Guidance message is required.")

        prior = agent_progress_guidance_text(body.prior_messages)
        prompt = f"""
The user sent live guidance while an Agent task was already in its final model stage.

Task title: {body.title or "Agent task"}
Current stage: {body.stage_label or "final stage"}

Latest user guidance:
{guidance}
{prior}

Respond briefly to the user inside the progress modal.
- Acknowledge the guidance.
- Explain that the current in-flight final-stage generation may not be changed unless the user cancels and reruns.
- If useful, summarize exactly how this guidance should be applied on a rerun.
- Do not save files, modify artifacts, or claim the current result has changed.
{output_language_instruction(body.language)}
"""
        answer = run_text_task(prompt)
        return {"answer": answer}


@app.post("/api/agent/cancel")
def cancel_agent_task(body: AgentCancelBody):
    canceled = cancel_agent_task_id(body.agent_task_id)
    return {"cancelled": canceled, "agent_task_id": normalize_agent_task_id(body.agent_task_id)}


@app.post("/api/agent-tasks/start")
def start_agent_task(body: AgentTaskStartBody):
    return create_background_agent_task(body)


@app.get("/api/agent-tasks/{task_id}/status")
def get_agent_task_status(task_id: str):
    task = snapshot_background_task(task_id)
    return {
        "taskId": task["taskId"],
        "status": task["status"],
        "stages": task.get("stages", []),
        "messages": task.get("messages", []),
        "currentStage": task.get("currentStage", ""),
        "resultAvailable": bool(task.get("resultAvailable")),
        "error": task.get("error", ""),
        "created_at": task.get("created_at", ""),
        "updated_at": task.get("updated_at", ""),
    }


@app.post("/api/agent-tasks/{task_id}/message")
def post_agent_task_message(task_id: str, body: AgentTaskMessageBody):
    task = snapshot_background_task(task_id)
    if task.get("status") in {"done", "error", "cancelled"}:
        raise HTTPException(status_code=409, detail="Agent task is not accepting messages.")
    append_background_task_message(task_id, "user", body.content.strip())
    return {"saved": True, "taskId": normalize_agent_task_id(task_id)}


@app.post("/api/agent-tasks/{task_id}/cancel")
def cancel_background_agent_task(task_id: str):
    task_id = normalize_agent_task_id(task_id)
    canceled = cancel_agent_task_id(task_id)
    update_background_task(task_id, status="cancelled", result=None, resultAvailable=False, error="")
    with background_task_lock:
        task = background_agent_tasks.get(task_id)
        if task:
            task["stages"] = [
                {**stage, "status": "cancelled" if stage.get("status") in {"pending", "running", "waiting_for_user"} else stage.get("status")}
                for stage in task.get("stages", [])
            ]
    append_background_task_message(task_id, "system", "任务已取消。")
    return {"cancelled": canceled, "taskId": task_id}


@app.get("/api/agent-tasks/{task_id}/result")
def get_agent_task_result(task_id: str):
    task = snapshot_background_task(task_id)
    if task.get("status") == "cancelled":
        raise HTTPException(status_code=409, detail="Agent task was cancelled.")
    if task.get("status") == "error":
        raise HTTPException(status_code=500, detail=task.get("error") or "Agent task failed.")
    if task.get("status") != "done" or not task.get("resultAvailable"):
        raise HTTPException(status_code=409, detail="Agent task result is not ready.")
    return task.get("result")


@app.post("/api/job-description")
def save_job_description(body: JobDescriptionBody):
    previous_job_description = (
        agent.read_text_file(agent.JOB_DESCRIPTION_PATH).strip()
        if agent.file_is_ready(agent.JOB_DESCRIPTION_PATH)
        else ""
    )
    new_job_description = body.content.strip()
    agent.write_text_file(agent.JOB_DESCRIPTION_PATH, body.content)
    agent.clear_interview_prep()
    tailored_resume_cleared = False
    if new_job_description != previous_job_description:
        agent.clear_tailored_resume()
        tailored_resume_cleared = True
    return {
        "saved": True,
        "path": str(agent.JOB_DESCRIPTION_PATH),
        "interview_prep_cleared": True,
        "tailored_resume_cleared": tailored_resume_cleared,
    }


@app.post("/api/job-description/analyze")
def analyze_job_description(body: AnalyzeBody):
    with agent_task_context(body.agent_task_id):
        return analyze_job_description_task(body)


def analyze_job_description_task(body: AnalyzeBody):
    try:
        job_description = agent.read_job_description()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    message = agent.JOB_AGENT_PROMPT
    if body.use_github_context:
        message += "\nUse GitHub context if available and approved."
    message += (
        job_analysis_language_instruction(body.language)
        + """

Return ONLY valid JSON with exactly these string keys:
- "analysis": the complete job analysis shown to the user.
- "company": hiring employer / company name only.
- "role": job title / position name only.

Rules for this response:
- The company and role are metadata for history only. Do not include a separate company/title section in "analysis".
- If the employer or title is not supported by the job description, use an empty string for that field.
- Do not wrap the JSON in Markdown fences.
"""
    )
    message = append_agent_progress_guidance(message, body.agent_progress_messages)
    answer = run_agent_task(message)
    try:
        payload = extract_json_object(answer)
        analysis = clean_history_text(payload.get("analysis"), "")
        company = clean_history_text(payload.get("company"), "")
        role = clean_history_text(payload.get("role"), "")
    except HTTPException:
        analysis = answer
        hint = resolve_application_hint(job_description)
        company = hint["company"]
        role = hint["role"]

    if not analysis:
        raise HTTPException(status_code=500, detail="Agent did not return job analysis content.")

    analysis_path = agent.save_analysis_output(analysis, company=company, role=role)
    history_entry = update_job_analysis_history(company, role, job_description, analysis_path)
    return {
        "analysis": analysis,
        "analysis_path": str(analysis_path),
        "company": history_entry["company"],
        "role": history_entry["role"],
        "history_entry": {
            "name": job_history_display_name(history_entry),
            "path": str(analysis_path),
            "company": history_entry["company"],
            "role": history_entry["role"],
            "updated_at": history_entry["updated_at"],
        },
    }


@app.post("/api/resume/tailor")
def tailor_resume(body: TailorBody):
    with agent_task_context(body.agent_task_id):
        return tailor_resume_task(body)


@app.post("/api/resume/star-check")
def resume_star_check(body: ResumeStarCheckBody):
    with agent_task_context(body.agent_task_id):
        return build_resume_star_check(body)


@app.post("/api/resume/star-fact")
def resume_star_fact(body: ResumeStarFactBody):
    with agent_task_context(body.agent_task_id):
        return save_resume_star_fact(body)


def tailor_resume_task(body: TailorBody):
    if body.use_github_context:
        return tailor_resume_staged(body)

    project_memory_context = agent.read_project_memory()
    prompt = (
        RESUME_TAILOR_PROMPT
        + output_language_instruction(body.language)
        + original_resume_language_instruction("tailored_resume")
        + f"""

Project Memory, read first and use as the primary project source:
{project_memory_context}
"""
    )
    if not body.allow_project_selection:
        prompt += (
            "\nKeep the existing resume project list. Do not remove projects or add projects from Chroma profile memory. "
            "You may still improve wording when it remains factual."
        )
    if body.allow_experience_removal:
        prompt += (
            "\nThe user explicitly allows removing an entire Experience entry when it is weakly relevant to the saved "
            "job description and removing it improves the tailored resume. Keep stronger relevant Experience entries. "
            "Never remove an entry merely to invent or substitute unsupported experience."
        )
    prompt += "\nGitHub evidence is not requested for this generation; use Project Memory, resume.txt, and job_description.txt."
    prompt += (
        "\nBefore writing or modifying any Project-section or Experience-section bullet, call "
        "write_resume_bullets and use its ReAct analysis plus final_bullets as the source of bullet wording."
    )
    prompt = append_agent_progress_guidance(prompt, body.agent_progress_messages)
    job_description = (
        agent.read_text_file(agent.JOB_DESCRIPTION_PATH)
        if agent.file_is_ready(agent.JOB_DESCRIPTION_PATH)
        else ""
    )
    application_hint = resolve_saved_application_hint(job_description)
    answer = run_agent_task(prompt)
    if not agent.looks_like_latex_resume(answer):
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code.")
    agent.save_tailored_resume(answer, company=application_hint["company"], role=application_hint["role"])
    tailored_resume_outputs = list_output_files(agent.TAILORED_RESUME_OUTPUT_DIR, ".txt", limit=1)
    response: dict[str, Any] = {
        "saved": True,
        "path": tailored_resume_outputs[0]["path"] if tailored_resume_outputs else str(agent.latest_tailored_resume_path()),
        "output_path": tailored_resume_outputs[0]["path"] if tailored_resume_outputs else None,
        "content": agent.read_tailored_resume(),
        "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
    }
    if body.include_application_hint:
        response["application_hint"] = application_hint
    return response


@app.post("/api/resume/update-memory")
def update_resume_memory(body: ResumeMemoryBody):
    with agent_task_context(body.agent_task_id):
        return update_resume_memory_task(body)


def update_resume_memory_task(body: ResumeMemoryBody):
    try:
        return update_memory_from_resume_source(
            body.resume_source,
            project_name=body.project_name,
            project_id=body.project_id,
            agent_progress_messages=body.agent_progress_messages,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/resume/pdf-to-latex")
def resume_pdf_to_latex(body: ResumePdfToLatexBody):
    with agent_task_context(body.agent_task_id):
        return resume_pdf_to_latex_task(body)


def resume_pdf_to_latex_task(body: ResumePdfToLatexBody):
    pdf_bytes = validate_resume_pdf(body)
    extracted = extract_pdf_resume_content(pdf_bytes)
    prompt = build_pdf_to_latex_prompt(
        body.filename,
        extracted,
        body.language,
        body.agent_progress_messages,
    )
    answer = run_text_task(prompt)
    latex = agent.extract_latex_document(answer)
    if not latex or not agent.looks_like_latex_resume(latex):
        raise HTTPException(
            status_code=400,
            detail="Model did not return complete LaTeX resume code. Please try again.",
        )

    agent.write_text_file(agent.RESUME_PATH, latex)
    return {
        "saved": True,
        "path": str(agent.RESUME_PATH),
        "content": latex,
        "pdf": {
            "filename": body.filename,
            "pages": extracted["pages"],
            "links": extracted["links"],
            "truncated": extracted["truncated"],
        },
    }


@app.post("/api/resume/tailored/pdf")
def export_tailored_resume_pdf(body: TailoredResumePdfBody):
    content = body.content.strip()
    application_hint = resolve_saved_application_hint()
    if content:
        document = agent.extract_latex_document(content)
        if not document:
            raise HTTPException(status_code=400, detail="No complete LaTeX document found.")
        content = document
    else:
        try:
            content = agent.read_tailored_resume()
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    output_pdf = compile_tailored_resume_pdf(
        content,
        company=application_hint["company"],
        role=application_hint["role"],
    )
    return {
        "saved": True,
        "path": str(output_pdf),
        "output_path": str(output_pdf),
    }


@app.post("/api/cover-letter/generate")
def generate_cover_letter(body: CoverLetterBody):
    with agent_task_context(body.agent_task_id):
        return generate_cover_letter_task(body)


def generate_cover_letter_task(body: CoverLetterBody):
    style_hint = f"\nPreferred style: {body.style}."
    prompt = (
        agent.COVER_LETTER_AGENT_PROMPT
        + style_hint
        + output_language_instruction(body.language)
        + original_resume_language_instruction("cover_letter")
    )
    if not body.use_tailored_resume:
        prompt += "\nUse resume.txt instead of tailored_resume.txt if the user requested it."
    if body.use_github_context:
        prompt += "\nYou may use GitHub context conservatively when it supports a specific claim."
    prompt = append_agent_progress_guidance(prompt, body.agent_progress_messages)
    cover_letter_mtime = (
        agent.COVER_LETTER_PATH.stat().st_mtime_ns
        if agent.COVER_LETTER_PATH.exists()
        else None
    )
    application_hint = resolve_saved_application_hint()
    answer = run_agent_task(prompt)
    cover_letter_was_saved = (
        agent.COVER_LETTER_PATH.exists()
        and agent.COVER_LETTER_PATH.stat().st_mtime_ns != cover_letter_mtime
    )
    if answer.strip() and not cover_letter_was_saved:
        agent.save_cover_letter(answer, company=application_hint["company"], role=application_hint["role"])
    cover_letter_outputs = list_output_files(agent.COVER_LETTER_OUTPUT_DIR, ".txt", limit=1)
    response: dict[str, Any] = {
        "saved": True,
        "path": str(agent.COVER_LETTER_PATH),
        "output_path": cover_letter_outputs[0]["path"] if cover_letter_outputs else None,
        "content": agent.read_text_file(agent.COVER_LETTER_PATH)
        if agent.file_is_ready(agent.COVER_LETTER_PATH)
        else answer,
    }
    if body.include_application_hint:
        response["application_hint"] = application_hint
    return response


@app.post("/api/interview-prep/generate")
def generate_interview_prep(body: InterviewPrepBody):
    with agent_task_context(body.agent_task_id):
        return generate_interview_prep_task(body)


def generate_interview_prep_task(body: InterviewPrepBody):
    prompt = build_interview_prep_prompt(
        body.use_github_context,
        body.language,
        body.agent_progress_messages,
    )
    application_hint = resolve_saved_application_hint()
    answer = run_text_task(prompt)
    if not looks_like_interview_prep(answer):
        raise HTTPException(
            status_code=400,
            detail="Agent did not return usable interview preparation notes. Please regenerate after checking the job description and resume.",
        )
    agent.save_interview_prep(answer, company=application_hint["company"], role=application_hint["role"])
    interview_prep_outputs = list_output_files(agent.INTERVIEW_PREP_OUTPUT_DIR, ".txt", limit=1)
    return {
        "saved": True,
        "path": str(agent.INTERVIEW_PREP_PATH),
        "output_path": interview_prep_outputs[0]["path"] if interview_prep_outputs else None,
        "content": agent.read_text_file(agent.INTERVIEW_PREP_PATH)
        if agent.file_is_ready(agent.INTERVIEW_PREP_PATH)
        else answer,
    }


@app.post("/api/github/scan")
def github_scan(body: GitHubScanBody):
    with agent_task_context(body.agent_task_id):
        return github_scan_task(body)


def github_scan_task(body: GitHubScanBody):
    try:
        repo_source = read_github_repo_source(
            body.resume_source,
            project_name=body.project_name,
            project_id=body.project_id,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    repos = agent.extract_github_repos(repo_source)
    identities = agent.read_github_identities()
    return {
        "repos": [
            {"owner": repo["owner"], "repo": repo["repo"], "url": repo["url"]}
            for repo in repos
        ],
        "token_configured": agent.github_token_is_configured(),
        "identities": identities,
        "project_name": body.project_name.strip(),
        "project_id": body.project_id.strip(),
    }


@app.get("/api/github/config")
def get_github_config():
    return build_github_config_status()


@app.post("/api/github/config")
def save_github_config(body: GitHubConfigBody):
    write_github_identities(body.usernames, body.author_names, body.author_emails)
    token = body.token.strip()
    if token:
        write_env_values({"GITHUB_TOKEN": token})
        os.environ["GITHUB_TOKEN"] = token

    return {
        "saved": True,
        **build_github_config_status(),
    }


@app.post("/api/github/context")
def github_context(body: GitHubContextBody):
    with agent_task_context(body.agent_task_id):
        return github_context_task(body)


def github_context_task(body: GitHubContextBody):
    try:
        return fetch_github_context_api(
            body.approved,
            body.resume_source,
            project_name=body.project_name,
            project_id=body.project_id,
            force_refresh=body.force_refresh,
            reanalyze_cached=body.reanalyze_cached,
            agent_progress_messages=body.agent_progress_messages,
        )
    except agent.transient_network_errors() as error:
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error


@app.get("/api/applications")
def get_applications(status: str = "", limit: int = Query(default=20, ge=1, le=100)):
    raw = agent.list_application_records(status=status, limit=limit)
    return json.loads(raw)


@app.post("/api/applications")
def create_application(body: ApplicationCreateBody):
    raw = agent.add_application_record(
        company=body.company,
        role=body.role,
        link=body.link,
        status=body.status,
        applied_date=body.applied_date,
        resume_version=body.resume_version,
        cover_letter_version=body.cover_letter_version,
        notes=body.notes,
    )
    return json.loads(raw)


@app.patch("/api/applications/{record_id}")
def patch_application(record_id: int, body: ApplicationUpdateBody):
    payload = body.model_dump(exclude_none=True)
    raw = agent.update_application_record(record_id, **payload)
    result = json.loads(raw)
    if not result.get("updated"):
        raise HTTPException(status_code=404, detail="Application record not found or no changes.")
    return result


@app.delete("/api/applications/{record_id}")
def delete_application(record_id: int):
    raw = agent.delete_application_record(record_id)
    result = json.loads(raw)
    if not result.get("deleted"):
        raise HTTPException(status_code=404, detail="Application record not found.")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8001, reload=True)
