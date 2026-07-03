"""FastAPI HTTP layer for WorkAgent frontend."""

from __future__ import annotations

import base64
import binascii
import hashlib
import inspect
import io
import json
import os
import re
import signal
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import uuid
from contextlib import closing, contextmanager
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
Prefer a one-page resume. The Projects section should look intentionally prioritized. Prefer 2 projects,
use 3 only when the third adds distinct job-relevant evidence, and never include more than 3. Allocate
more bullets to the highest-ranked project. If there are 2 projects, the first project should usually have
about 5 bullets and the second about 3. If there are 3 projects, use about 4, 3, and 1-2 bullets. Do not
distribute bullets evenly unless the projects are genuinely equal in relevance and evidence strength.
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
- The Projects section should look intentionally prioritized. Prefer 2 projects, use 3 only when the third
  adds distinct job-relevant evidence, and never include more than 3.
- Allocate more bullets to the highest-ranked project. If there are 2 projects, the first project should
  usually have about 5 bullets and the second about 3. If there are 3 projects, use about 4, 3, and 1-2
  bullets. Do not distribute bullets evenly unless the projects are genuinely equal in relevance and evidence strength.
- Each bullet should be concise, factual, and ATS-friendly.
- Never invent metrics, technologies, deployment, users, business impact, ownership, or performance claims.

Technical Skills Rules:
- Preserve the base resume's compact Technical Skills LaTeX style: use
  \\begin{itemize}[leftmargin=0.15in, label={}], \\small{...}, and one \\item containing inline
  \\textbf{Category:} skill lists separated by \\\\ line breaks.
- Do not use visible bullets or one \\item per skill category in Technical Skills.
- Keep Technical Skills in the same font/size as the base resume; do not introduce a plain \\begin{itemize}
  skills list.
- Include only concrete, supported skill names. Do not include sentences, Chinese explanatory fragments,
  "...", "[truncated]", "more items", or generic filler such as API, automation, validation,
  requirements, reporting, or documentation by itself.
- Merge duplicates and aliases, such as React/React.js, Git/GitHub, SQLite/better-sqlite3, and HTML/CSS.

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
PROJECT_PRIORITY_INSTRUCTION = (
    "The Projects section should look intentionally prioritized. Prefer 2 projects, use 3 only when the "
    "third adds distinct job-relevant evidence, and never include more than 3. Allocate more bullets to "
    "the highest-ranked project. If there are 2 projects, the first project should usually have about "
    "5 bullets and the second about 3. If there are 3 projects, use about 4, 3, and 1-2 bullets. Do not "
    "distribute bullets evenly unless the projects are genuinely equal in relevance and evidence strength."
)
PROJECT_ONE_PAGE_CUT_ORDER = [
    "Weak or repetitive third project",
    "Lower-ranked project bullets",
    "Less relevant experience bullets",
    "Summary wording",
    "Top-ranked project bullets only as a last resort",
]
MAX_STAGED_TEXT_CHARS = 12000
MAX_PROMPT_FILES_PER_REPO = 12
MAX_PROMPT_DIFF_SIGNALS = 20
MAX_PROMPT_CLAIMS = 12
MAX_PROMPT_SIGNAL_CHARS = 240
MAX_PROMPT_FILE_SUMMARY_CHARS = 500
MAX_PROMPT_EVIDENCE_CHARS = 9000
PROXY_SAFE_MAX_INPUT_CHARS = 25000
PROXY_SAFE_HARD_INPUT_CHARS = 35000
OFFICIAL_DIRECT_MAX_INPUT_CHARS = 80000
OFFICIAL_MAP_REDUCE_THRESHOLD_CHARS = 100000
PROXY_TRANSIENT_STATUS_CODES = {502, 503, 504}
PROXY_TIMEOUT_STATUS_CODES = {524}
PROXY_PROVIDER_CONFIG_STATUS_CODES = {401, 403, 404}
PROXY_RETRY_AFTER_MAX_SECONDS = 30
BULLET_WRITER_RETRY_COMPACT_CHARS = 15000
BULLET_WRITER_EMERGENCY_COMPACT_CHARS = 10000
REPO_CHUNK_TARGET_CHARS = 18000
REDUCE_BATCH_TARGET_CHARS = 18000
MODEL_ROUTING_LOG_PREFIX = "Model routing"
MODEL_CACHE_PATH = agent.INFORMATION_DIR / "model_call_cache.json"
PROJECT_COMPACT_FACTS_PATH = agent.INFORMATION_DIR / "project_compact_facts.json"
CHUNK_CHECKPOINTS_PATH = agent.INFORMATION_DIR / "agent_chunk_checkpoints.json"
RESUME_CANDIDATE_CHECKPOINTS_PATH = agent.INFORMATION_DIR / "resume_candidate_checkpoints.json"
MODEL_FAILURE_LOG_DIR = agent.ROOT_DIR / "logs" / "model_failures"
TECH_STACK_DB_PATH = agent.APPLICATION_DB_PATH


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
    forceChunking: bool = False
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
    forceChunking: bool = False
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


def normalize_model_base_url(raw: str | None) -> str | None:
    value = (raw or "").strip().strip('"').strip("'").rstrip("/")
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value


def infer_provider_kind(base_url: str | None, explicit_kind: str | None = None) -> str:
    if explicit_kind:
        return str(explicit_kind).strip()
    normalized = normalize_model_base_url(base_url)
    if not normalized:
        return "official_openai"
    host = urllib.parse.urlparse(normalized).netloc.lower()
    if host == "api.openai.com":
        return "official_openai"
    return "third_party_proxy"


def provider_env_name(provider_name: str, suffix: str) -> str:
    normalized = normalize_provider(provider_name).upper().replace("-", "_")
    return f"{normalized}_{suffix}"


def provider_base_url(provider_name: str) -> str:
    config = PROVIDER_CONFIGS.get(normalize_provider(provider_name), {})
    env_name = config.get("base_url_env")
    default_base_url = str(config.get("default_base_url") or "")
    return os.getenv(env_name, default_base_url) if env_name else default_base_url


def provider_model_name(provider_name: str, fallback: str = "") -> str:
    config = PROVIDER_CONFIGS.get(normalize_provider(provider_name), {})
    env_name = config.get("model_env")
    default_model = str(config.get("default_model") or fallback)
    return os.getenv(env_name, default_model) if env_name else fallback


def provider_override_bool(provider_name: str, key: str) -> Optional[bool]:
    value = os.getenv(provider_env_name(provider_name, key))
    if value is None:
        value = os.getenv(key)
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def provider_override_int(provider_name: str, key: str, default: int) -> int:
    value = os.getenv(provider_env_name(provider_name, key)) or os.getenv(key)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def provider_override_float(provider_name: str, key: str, default: float) -> float:
    value = os.getenv(provider_env_name(provider_name, key)) or os.getenv(key)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def build_provider_routing_config(provider_name: str) -> dict[str, Any]:
    normalized_provider = normalize_provider(provider_name)
    base_url = provider_base_url(normalized_provider)
    explicit_kind = (
        os.getenv(provider_env_name(normalized_provider, "PROVIDER_KIND"))
        or os.getenv("MODEL_PROVIDER_KIND")
        or None
    )
    provider_kind = infer_provider_kind(base_url, explicit_kind)
    proxy_override = provider_override_bool(normalized_provider, "PROXY_SAFE_MODE")
    proxy_safe_mode = proxy_override if proxy_override is not None else provider_kind == "third_party_proxy"
    shrink_input_override = provider_override_bool(normalized_provider, "SHRINK_INPUT_ON_RETRY")
    reduce_output_override = provider_override_bool(normalized_provider, "REDUCE_OUTPUT_TOKENS_ON_RETRY")
    endpoint_fallback_override = provider_override_bool(normalized_provider, "ENDPOINT_FALLBACK_ENABLED")
    execution_mode = (
        os.getenv(provider_env_name(normalized_provider, "EXECUTION_MODE"))
        or ("proxy_safe" if proxy_safe_mode else "official_direct")
    )
    return {
        "name": normalized_provider,
        "baseUrl": base_url,
        "providerKind": provider_kind,
        "proxySafeMode": proxy_safe_mode,
        "executionMode": execution_mode,
        "directMaxInputChars": provider_override_int(
            normalized_provider,
            "DIRECT_MAX_INPUT_CHARS",
            PROXY_SAFE_MAX_INPUT_CHARS if proxy_safe_mode else OFFICIAL_DIRECT_MAX_INPUT_CHARS,
        ),
        "proxySafeHardInputChars": provider_override_int(
            normalized_provider,
            "PROXY_SAFE_HARD_INPUT_CHARS",
            PROXY_SAFE_HARD_INPUT_CHARS,
        ),
        "mapReduceThresholdChars": provider_override_int(
            normalized_provider,
            "MAP_REDUCE_THRESHOLD_CHARS",
            PROXY_SAFE_MAX_INPUT_CHARS if proxy_safe_mode else OFFICIAL_MAP_REDUCE_THRESHOLD_CHARS,
        ),
        "maxTransientRetries": provider_override_int(
            normalized_provider,
            "MAX_TRANSIENT_RETRIES",
            2 if proxy_safe_mode else 0,
        ),
        "initialRetryDelayMs": provider_override_int(
            normalized_provider,
            "INITIAL_RETRY_DELAY_MS",
            3000,
        ),
        "retryBackoffMultiplier": provider_override_float(
            normalized_provider,
            "RETRY_BACKOFF_MULTIPLIER",
            2.0,
        ),
        "shrinkInputOnRetry": (
            shrink_input_override
            if shrink_input_override is not None else proxy_safe_mode
        ),
        "shrinkRatioOnTransientError": provider_override_float(
            normalized_provider,
            "SHRINK_RATIO_ON_TRANSIENT_ERROR",
            0.75,
        ),
        "reduceOutputTokensOnRetry": (
            reduce_output_override
            if reduce_output_override is not None else proxy_safe_mode
        ),
        "maxOutputTokensPerCall": provider_override_int(
            normalized_provider,
            "MAX_OUTPUT_TOKENS_PER_CALL",
            1200 if proxy_safe_mode else 0,
        ),
        "retryOutputTokens": provider_override_int(
            normalized_provider,
            "RETRY_OUTPUT_TOKENS",
            800,
        ),
        "endpointFallbackEnabled": (
            endpoint_fallback_override
            if endpoint_fallback_override is not None else proxy_safe_mode
        ),
        "supportsStreaming": bool(provider_override_bool(normalized_provider, "SUPPORTS_STREAMING")),
    }


def should_use_proxy_safe_mode(provider: dict[str, Any]) -> bool:
    if provider.get("proxySafeMode") is not None:
        return bool(provider["proxySafeMode"])
    return infer_provider_kind(provider.get("baseUrl"), provider.get("providerKind")) == "third_party_proxy"


def should_use_map_reduce(
    provider: dict[str, Any],
    input_char_count: int,
    user_options: dict[str, Any] | None = None,
) -> bool:
    user_options = user_options or {}
    if user_options.get("forceChunking"):
        return True
    if should_use_proxy_safe_mode(provider):
        return input_char_count > int(provider.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS)
    official_limit = int(provider.get("mapReduceThresholdChars") or OFFICIAL_MAP_REDUCE_THRESHOLD_CHARS)
    return input_char_count > official_limit


def estimate_input_size(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(value))


def routing_decision(
    provider_name: str,
    input_char_count: int,
    user_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = build_provider_routing_config(provider_name)
    normalized_base_url = normalize_model_base_url(provider.get("baseUrl"))
    host = urllib.parse.urlparse(normalized_base_url).netloc.lower() if normalized_base_url else "api.openai.com"
    use_map_reduce = should_use_map_reduce(provider, input_char_count, user_options)
    if user_options and user_options.get("forceChunking"):
        reason = "forceChunking requested"
    elif should_use_proxy_safe_mode(provider):
        limit = int(provider.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS)
        reason = "third-party proxy over safe limit" if input_char_count > limit else "third-party proxy below safe limit"
    else:
        limit = int(provider.get("mapReduceThresholdChars") or OFFICIAL_MAP_REDUCE_THRESHOLD_CHARS)
        reason = "official over map-reduce threshold" if input_char_count > limit else "official below threshold"
    return {
        **provider,
        "host": host,
        "inputCharCount": input_char_count,
        "approxTokenCount": input_char_count // 4,
        "useMapReduce": use_map_reduce,
        "reason": reason,
    }


def log_model_routing(
    decision: dict[str, Any],
    caller: str,
    task_type: str,
    model: str,
    endpoint_type: str,
) -> None:
    print(
        f"{MODEL_ROUTING_LOG_PREFIX}: "
        f"caller={caller}, taskType={task_type}, model={model}, "
        f"provider={decision.get('name')}, host={decision.get('host')}, "
        f"kind={decision.get('providerKind')}, proxySafeMode={str(decision.get('proxySafeMode')).lower()}, "
        f"inputChars={decision.get('inputCharCount')}, approxTokens={decision.get('approxTokenCount')}, "
        f"endpoint={endpoint_type}, executionMode={decision.get('executionMode')}, "
        f"useMapReduce={str(decision.get('useMapReduce')).lower()}, reason={decision.get('reason')}, "
        f"timestamp={datetime.now().isoformat(timespec='seconds')}"
    )


def caller_name_from_stack() -> str:
    for frame in inspect.stack()[2:8]:
        name = frame.function
        if name not in {"run_text_task", "caller_name_from_stack"}:
            return name
    return "unknown"


def enforce_model_input_limits(decision: dict[str, Any], caller: str, allow_oversize_direct: bool = False) -> None:
    if allow_oversize_direct or not should_use_proxy_safe_mode(decision):
        return
    input_chars = int(decision.get("inputCharCount") or 0)
    direct_limit = int(decision.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS)
    hard_limit = int(decision.get("proxySafeHardInputChars") or PROXY_SAFE_HARD_INPUT_CHARS)
    if input_chars <= direct_limit:
        return
    limit = direct_limit
    error_type = "ModelInputTooLargeForProxy"
    message = "Input is too large for third-party proxy. Use project-level chunking before calling the model."
    if input_chars > hard_limit:
        limit = hard_limit
        error_type = "ModelInputHardLimitExceededForProxy"
        message = "Input exceeds the hard limit for third-party proxy. Chunking is required before calling the model."
    raise HTTPException(
        status_code=413,
        detail={
            "ok": False,
            "type": error_type,
            "caller": caller,
            "inputCharCount": input_chars,
            "limit": limit,
            "message": message,
        },
    )


def model_input_limit_exception(
    *,
    caller: str,
    input_char_count: int,
    limit: int,
    hard_limit: int,
    message: str = "",
) -> HTTPException:
    hard_exceeded = input_char_count > hard_limit
    return HTTPException(
        status_code=413,
        detail={
            "ok": False,
            "type": "ModelInputHardLimitExceededForProxy" if hard_exceeded else "ModelInputTooLargeForProxy",
            "caller": caller,
            "inputCharCount": input_char_count,
            "limit": hard_limit if hard_exceeded else limit,
            "message": message
            or (
                "Compact/model input exceeds the hard third-party proxy limit."
                if hard_exceeded
                else "Compact/model input is still too large for third-party proxy."
            ),
        },
    )


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


def extract_model_error_headers(error: Exception) -> dict[str, str]:
    headers: Any = {}
    if isinstance(error, APIStatusError):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) if response is not None else {}
    elif isinstance(error, urllib.error.HTTPError):
        headers = getattr(error, "headers", {}) or {}
    result: dict[str, str] = {}
    try:
        items = headers.items()
    except AttributeError:
        items = []
    for key, value in items:
        result[str(key).lower()] = str(value)
    return result


def retry_after_seconds_from_headers(headers: dict[str, str]) -> Optional[int]:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = int(float(value.strip()))
    except ValueError:
        return None
    return max(0, min(seconds, PROXY_RETRY_AFTER_MAX_SECONDS))


def model_error_type_for_status(status_code: int, detail_text: str = "") -> Optional[str]:
    text = detail_text.lower()
    if status_code in PROXY_TIMEOUT_STATUS_CODES or "timeout" in text or "origin_response_timeout" in text:
        return "ModelProxyTimeout"
    if status_code in PROXY_TRANSIENT_STATUS_CODES or any(
        marker in text
        for marker in [
            "bad gateway",
            "connection reset",
            "upstream unavailable",
            "upstream error",
            "network error",
            "temporarily unavailable",
        ]
    ):
        return "ModelTransientError"
    if status_code in PROXY_PROVIDER_CONFIG_STATUS_CODES:
        return "ModelProviderConfigError"
    return None


def structured_model_error_detail(
    *,
    error_type: str,
    caller: str,
    status_code: int,
    decision: dict[str, Any],
    input_char_count: int,
    message: str,
    retryable: bool,
    attempts: Optional[list[dict[str, Any]]] = None,
    retry_after_seconds: Optional[int] = None,
    cloudflare_ray_id: str = "",
    endpoint: str = "",
    checkpoint_preserved: bool = False,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "ok": False,
        "type": error_type,
        "caller": caller,
        "statusCode": status_code,
        "retryable": retryable,
        "providerKind": decision.get("providerKind"),
        "proxySafeMode": decision.get("proxySafeMode"),
        "inputCharCount": input_char_count,
        "message": message,
    }
    if attempts:
        detail["attempts"] = attempts
    if retry_after_seconds is not None:
        detail["retryAfterSeconds"] = retry_after_seconds
    if cloudflare_ray_id:
        detail["cloudflareRayId"] = cloudflare_ray_id
    if endpoint:
        detail["endpoint"] = endpoint
    if checkpoint_preserved:
        detail["checkpointPreserved"] = True
    return detail


def http_exception_detail_text(error: HTTPException) -> str:
    detail = getattr(error, "detail", "")
    if isinstance(detail, dict):
        try:
            return json.dumps(detail, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(detail)
    return str(detail)


def status_code_from_http_exception(error: HTTPException) -> int:
    detail = getattr(error, "detail", None)
    if isinstance(detail, dict) and detail.get("statusCode"):
        try:
            return int(detail["statusCode"])
        except (TypeError, ValueError):
            pass
    return int(getattr(error, "status_code", 502) or 502)


def retry_after_seconds_from_http_exception(error: HTTPException) -> Optional[int]:
    detail = getattr(error, "detail", None)
    if isinstance(detail, dict):
        value = detail.get("retryAfterSeconds")
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
    return None


def cloudflare_ray_from_http_exception(error: HTTPException) -> str:
    detail = getattr(error, "detail", None)
    if isinstance(detail, dict):
        return str(detail.get("cloudflareRayId") or "")
    match = re.search(r"\bcf-ray[:=]\s*([A-Za-z0-9-]+)", str(detail), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def save_model_failure_log(
    *,
    caller: str,
    decision: dict[str, Any],
    status_code: int,
    input_char_count: int,
    attempt: int,
    endpoint: str,
    stream: bool,
    max_output_tokens: Optional[int],
    retry_action: str,
    error_message: str,
    cloudflare_ray_id: str = "",
) -> None:
    MODEL_FAILURE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "caller": caller,
        "providerKind": decision.get("providerKind"),
        "proxySafeMode": decision.get("proxySafeMode"),
        "statusCode": status_code,
        "inputCharCount": input_char_count,
        "attempt": attempt,
        "endpoint": endpoint,
        "stream": bool(stream),
        "maxOutputTokens": max_output_tokens,
        "retryAction": retry_action,
        "cloudflareRayId": cloudflare_ray_id,
        "errorMessage": short_signal(error_message, 500) if "short_signal" in globals() else str(error_message)[:500],
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = MODEL_FAILURE_LOG_DIR / f"{stamp}_{caller}_{attempt}_{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def raise_model_api_http_exception(error: Exception, caller: str = "") -> None:
    if isinstance(error, APIStatusError):
        upstream_status = error.status_code or 502
        http_status = upstream_status if 400 <= upstream_status < 600 else 502
    elif isinstance(error, urllib.error.HTTPError):
        http_status = error.code if 400 <= error.code < 600 else 502
    else:
        raise error

    message = extract_model_api_error_message(error)
    headers = extract_model_error_headers(error)
    provider_name = normalize_provider(agent.current_provider)
    decision = routing_decision(provider_name, 0)
    error_type = model_error_type_for_status(http_status, message)
    if error_type and should_use_proxy_safe_mode(decision):
        raise HTTPException(
            status_code=http_status,
            detail=structured_model_error_detail(
                error_type=error_type,
                caller=caller,
                status_code=http_status,
                decision=decision,
                input_char_count=0,
                retryable=error_type != "ModelProviderConfigError",
                retry_after_seconds=retry_after_seconds_from_headers(headers),
                cloudflare_ray_id=headers.get("cf-ray", ""),
                message=(
                    "Third-party proxy timed out. The task can be resumed with smaller chunks."
                    if error_type == "ModelProxyTimeout"
                    else "Third-party proxy returned a transient upstream error."
                ),
            ),
        ) from error
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


def chat_completions_adapter_for_provider(provider_name: str) -> Optional[Any]:
    name = normalize_provider(provider_name)
    if name == "openai":
        return agent.OpenAIChatCompletionsAdapter(
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
            model_env="OPENAI_MODEL",
            fallback_model="gpt-5.5",
        )
    if name == "openai-compatible":
        return agent.OpenAIChatCompletionsAdapter(
            api_key_env="OPENAI_COMPATIBLE_API_KEY",
            base_url_env="OPENAI_COMPATIBLE_BASE_URL",
            model_env="OPENAI_COMPATIBLE_MODEL",
            fallback_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        )
    adapter = agent.create_model_adapter(name)
    if "Chat" in adapter.__class__.__name__:
        return adapter
    return None


def provider_default_endpoint(provider_name: str) -> str:
    adapter = agent.create_model_adapter(normalize_provider(provider_name))
    return "chat_completions" if "Chat" in adapter.__class__.__name__ else "responses"


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


def completed_process_output(result: subprocess.CompletedProcess[str]) -> str:
    stdout = result.stdout or ""
    stderr = result.stderr or ""
    return (stdout + "\n" + stderr).strip()


def compile_tailored_resume_pdf(latex: str, company: str = "", role: str = "") -> Path:
    document = agent.extract_latex_document(latex)
    if not document:
        raise HTTPException(status_code=400, detail="No complete LaTeX document found.")
    validation_issues = agent.latex_resume_validation_issues(document)
    if validation_issues:
        raise HTTPException(
            status_code=400,
            detail="Invalid LaTeX resume: " + " ".join(validation_issues),
        )
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
                encoding="utf-8",
                errors="replace",
                timeout=90,
                check=False,
            )
            if result.returncode != 0:
                break

        if result.returncode == 0:
            break

        output = completed_process_output(result)
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
        if isinstance(error.detail, dict):
            error_type = str(error.detail.get("type") or "")
            if error_type == "ModelTransientError":
                return "第三方中转站连续返回 502。已保存当前进度，可以稍后重试或降低输入规模。"
            if error_type == "ModelProxyTimeout":
                return "第三方中转站连续超时。已保存当前进度，可以稍后重试或降低输入规模。"
            detail = str(error.detail.get("message") or error.detail)
        else:
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
        raise_model_api_http_exception(error, caller="run_agent_task")
    except agent.transient_network_errors() as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        unregister_agent_task_adapter(task_id, adapter)


def run_text_task(
    message: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    caller: str = "",
    task_type: str = "text_task",
    user_options: Optional[dict[str, Any]] = None,
    allow_oversize_direct: bool = False,
    endpoint_mode: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    stream: bool = False,
) -> str:
    adapter, provider_name = get_adapter(provider)
    if endpoint_mode == "chat_completions":
        adapter = chat_completions_adapter_for_provider(provider_name) or adapter
    chosen_model = model or adapter.default_model()
    caller = caller or caller_name_from_stack()
    decision = routing_decision(provider_name, len(message), user_options)
    endpoint_type = "chat_completions" if "Chat" in adapter.__class__.__name__ else "responses"
    log_model_routing(decision, caller, task_type, chosen_model, endpoint_type)
    enforce_model_input_limits(decision, caller, allow_oversize_direct=allow_oversize_direct)
    task_id = current_agent_task_id.get("")
    assert_agent_task_not_cancelled()
    register_agent_task_adapter(task_id, adapter)
    try:
        response = adapter.create_response(
            model=chosen_model,
            instructions=agent.SYSTEM_PROMPT,
            tools=[],
            input_items=[{"role": "user", "content": message}],
            max_output_tokens=max_output_tokens,
            stream=stream,
        )
        assert_agent_task_not_cancelled()
        return adapter.output_text(response)
    except AgentTaskCancelled as error:
        raise HTTPException(status_code=499, detail=str(error)) from error
    except (APIStatusError, urllib.error.HTTPError) as error:
        assert_agent_task_not_cancelled()
        raise_model_api_http_exception(error, caller=caller)
    except agent.transient_network_errors() as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        assert_agent_task_not_cancelled()
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        unregister_agent_task_adapter(task_id, adapter)


def invoke_safe_builder(builder: Callable[..., Any], **kwargs: Any) -> Any:
    signature = inspect.signature(builder)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return builder(**kwargs)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return builder(**accepted)


def safe_builder_prompt(result: Any) -> tuple[str, Any]:
    if isinstance(result, str):
        return result, None
    if isinstance(result, dict):
        prompt = result.get("prompt")
        if isinstance(prompt, str):
            return prompt, result.get("payload")
    raise HTTPException(status_code=500, detail="safe_model_call builder must return a prompt string or {'prompt': string}.")


def safe_model_call(
    caller: str,
    prompt: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task_type: str = "text_task",
    compact_builder: Optional[Callable[..., Any]] = None,
    map_reduce_builder: Optional[Callable[..., Any]] = None,
    user_options: Optional[dict[str, Any]] = None,
    allow_oversize_direct: bool = False,
) -> str:
    provider_name = normalize_provider(provider or agent.current_provider)
    original_input_chars = estimate_input_size(prompt)
    decision = routing_decision(provider_name, original_input_chars, user_options)
    soft_limit = int(decision.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS)
    hard_limit = int(decision.get("proxySafeHardInputChars") or PROXY_SAFE_HARD_INPUT_CHARS)
    use_compact_input = False
    use_map_reduce = bool(decision.get("useMapReduce"))
    final_prompt = prompt
    compact_input_chars = 0
    compact_payload = None

    should_compact = should_use_proxy_safe_mode(decision) and original_input_chars > soft_limit
    if should_compact and compact_builder:
        compact_result = invoke_safe_builder(
            compact_builder,
            decision=decision,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            max_chars=soft_limit,
            caller=caller,
        )
        final_prompt, compact_payload = safe_builder_prompt(compact_result)
        compact_input_chars = estimate_input_size(final_prompt)
        use_compact_input = True

    if should_use_proxy_safe_mode(decision) and estimate_input_size(final_prompt) > soft_limit and map_reduce_builder:
        map_reduce_result = invoke_safe_builder(
            map_reduce_builder,
            decision=decision,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            max_chars=soft_limit,
            caller=caller,
            compact_payload=compact_payload,
        )
        final_prompt, compact_payload = safe_builder_prompt(map_reduce_result)
        compact_input_chars = estimate_input_size(final_prompt)
        use_compact_input = True
        use_map_reduce = True

    if not should_use_proxy_safe_mode(decision) and use_map_reduce and map_reduce_builder:
        map_reduce_result = invoke_safe_builder(
            map_reduce_builder,
            decision=decision,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            max_chars=int(decision.get("mapReduceThresholdChars") or OFFICIAL_MAP_REDUCE_THRESHOLD_CHARS),
            caller=caller,
            compact_payload=compact_payload,
        )
        final_prompt, compact_payload = safe_builder_prompt(map_reduce_result)
        compact_input_chars = estimate_input_size(final_prompt)
        use_compact_input = True

    final_input_chars = estimate_input_size(final_prompt)
    print(
        "safe_model_call: "
        f"caller={caller}, taskType={task_type}, providerKind={decision.get('providerKind')}, "
        f"proxySafeMode={str(decision.get('proxySafeMode')).lower()}, "
        f"originalInputCharCount={original_input_chars}, compactInputCharCount={compact_input_chars}, "
        f"finalInputCharCount={final_input_chars}, useCompactInput={str(use_compact_input).lower()}, "
        f"useMapReduce={str(use_map_reduce).lower()}"
    )

    if should_use_proxy_safe_mode(decision) and not allow_oversize_direct and final_input_chars > soft_limit:
        raise model_input_limit_exception(
            caller=caller,
            input_char_count=final_input_chars,
            limit=soft_limit,
            hard_limit=hard_limit,
            message="Input is still too large after compact/map-reduce protection.",
        )

    provider_is_proxy = should_use_proxy_safe_mode(decision)
    max_transient_retries = int(decision.get("maxTransientRetries") or 0) if provider_is_proxy else 0
    original_endpoint = provider_default_endpoint(provider_name)
    fallback_adapter = (
        chat_completions_adapter_for_provider(provider_name)
        if provider_is_proxy and bool(decision.get("endpointFallbackEnabled")) and original_endpoint == "responses"
        else None
    )
    can_fallback_endpoint = fallback_adapter is not None
    attempts: list[dict[str, Any]] = []
    attempt_number = 1
    attempt_prompt = final_prompt
    attempt_input_chars = final_input_chars
    endpoint_mode: Optional[str] = None
    endpoint_name = original_endpoint
    stream = False

    def max_output_tokens_for_attempt(number: int) -> Optional[int]:
        if not provider_is_proxy:
            return None
        if number >= 3 and bool(decision.get("reduceOutputTokensOnRetry")):
            retry_tokens = int(decision.get("retryOutputTokens") or 0)
            return retry_tokens or None
        max_tokens = int(decision.get("maxOutputTokensPerCall") or 0)
        return max_tokens or None

    def retry_delay_seconds(failed_attempt: int, error: HTTPException) -> float:
        retry_after = retry_after_seconds_from_http_exception(error)
        if retry_after is not None:
            return float(retry_after)
        initial_ms = int(decision.get("initialRetryDelayMs") or 0)
        multiplier = float(decision.get("retryBackoffMultiplier") or 1.0)
        delay = (initial_ms / 1000.0) * (multiplier ** max(0, failed_attempt - 1))
        return min(delay, float(PROXY_RETRY_AFTER_MAX_SECONDS))

    def next_retry_action(failed_attempt: int) -> str:
        if failed_attempt == 1:
            return "retry_same_input"
        if can_fallback_endpoint:
            return "endpoint_fallback"
        if bool(decision.get("shrinkInputOnRetry")) and compact_builder:
            return "shrink_input"
        if bool(decision.get("supportsStreaming")):
            return "streaming_retry"
        return "retry_same_input"

    def retry_compact_limit(next_attempt: int, current_chars: int) -> int:
        ratio = float(decision.get("shrinkRatioOnTransientError") or 0.75)
        target = max(1000, int(current_chars * ratio))
        if caller == "run_resume_bullet_writer_tool":
            if next_attempt >= 4:
                return min(target, BULLET_WRITER_EMERGENCY_COMPACT_CHARS)
            return min(target, BULLET_WRITER_RETRY_COMPACT_CHARS)
        return min(target, soft_limit)

    def rebuild_prompt_for_retry(next_attempt: int, current_prompt: str, current_chars: int) -> tuple[str, Any]:
        nonlocal compact_payload
        if not bool(decision.get("shrinkInputOnRetry")) or not compact_builder:
            return current_prompt, compact_payload
        retry_mode = "emergency" if next_attempt >= 4 else "retry"
        target_chars = retry_compact_limit(next_attempt, current_chars)
        compact_result = invoke_safe_builder(
            compact_builder,
            decision=decision,
            soft_limit=soft_limit,
            hard_limit=hard_limit,
            max_chars=target_chars,
            caller=caller,
            compact_payload=compact_payload,
            retry_mode=retry_mode,
            retry_target_chars=target_chars,
        )
        retry_prompt, retry_payload = safe_builder_prompt(compact_result)
        compact_payload = retry_payload
        return retry_prompt, retry_payload

    def record_retry_progress(status_code: int, action: str) -> None:
        task_id = current_agent_task_id.get("")
        if not task_id:
            return
        if action in {"shrink_input", "endpoint_fallback", "streaming_retry"}:
            append_background_task_message(task_id, "agent", f"第三方中转站返回 {status_code}，系统正在缩小输入并重试。")
        else:
            append_background_task_message(task_id, "agent", f"第三方中转站返回 {status_code}，系统正在有限重试。")

    while True:
        max_output_tokens = max_output_tokens_for_attempt(attempt_number)
        try:
            return run_text_task(
                attempt_prompt,
                provider=provider_name,
                model=model,
                caller=caller,
                task_type=task_type,
                user_options=user_options,
                allow_oversize_direct=allow_oversize_direct,
                endpoint_mode=endpoint_mode,
                max_output_tokens=max_output_tokens,
                stream=stream,
            )
        except HTTPException as error:
            status_code = status_code_from_http_exception(error)
            detail_text = http_exception_detail_text(error)
            error_type = model_error_type_for_status(status_code, detail_text)
            retryable = error_type in {"ModelTransientError", "ModelProxyTimeout"}
            if not provider_is_proxy or not error_type or not retryable:
                if provider_is_proxy and error_type == "ModelProviderConfigError":
                    raise HTTPException(
                        status_code=status_code,
                        detail=structured_model_error_detail(
                            error_type=error_type,
                            caller=caller,
                            status_code=status_code,
                            decision=decision,
                            input_char_count=attempt_input_chars,
                            retryable=False,
                            message="Model provider configuration appears invalid; check base_url, API key, protocol, and model name.",
                        ),
                    ) from error
                raise

            retry_action = next_retry_action(attempt_number)
            attempt_record = {
                "attempt": attempt_number,
                "statusCode": status_code,
                "inputCharCount": attempt_input_chars,
                "endpoint": endpoint_name,
                "stream": stream,
                "maxOutputTokens": max_output_tokens,
                "retryAction": retry_action if attempt_number <= max_transient_retries else "give_up",
            }
            attempts.append(attempt_record)
            save_model_failure_log(
                caller=caller,
                decision=decision,
                status_code=status_code,
                input_char_count=attempt_input_chars,
                attempt=attempt_number,
                endpoint=endpoint_name,
                stream=stream,
                max_output_tokens=max_output_tokens,
                retry_action=attempt_record["retryAction"],
                error_message=detail_text,
                cloudflare_ray_id=cloudflare_ray_from_http_exception(error),
            )

            if attempt_number > max_transient_retries:
                message = (
                    "Third-party proxy repeatedly timed out after safe_model_call retry protection."
                    if error_type == "ModelProxyTimeout"
                    else "Third-party proxy returned repeated transient upstream errors after safe_model_call retry protection."
                )
                raise HTTPException(
                    status_code=status_code if status_code in {502, 503, 504, 524} else 502,
                    detail=structured_model_error_detail(
                        error_type=error_type,
                        caller=caller,
                        status_code=status_code,
                        decision=decision,
                        input_char_count=attempt_input_chars,
                        retryable=True,
                        attempts=attempts,
                        cloudflare_ray_id=cloudflare_ray_from_http_exception(error),
                        endpoint=endpoint_name,
                        checkpoint_preserved=True,
                        message=message,
                    ),
                ) from error

            record_retry_progress(status_code, retry_action)
            delay = retry_delay_seconds(attempt_number, error)
            if delay > 0:
                time.sleep(delay)

            attempt_number += 1
            if attempt_number >= 3:
                attempt_prompt, compact_payload = rebuild_prompt_for_retry(
                    attempt_number,
                    attempt_prompt,
                    attempt_input_chars,
                )
                attempt_input_chars = estimate_input_size(attempt_prompt)
                if can_fallback_endpoint:
                    endpoint_mode = "chat_completions"
                    endpoint_name = "chat_completions"
                stream = bool(decision.get("supportsStreaming") and endpoint_name == "chat_completions")


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
        response = safe_model_call(caller="resolve_application_hint", prompt=prompt, task_type="application_hint")
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


def read_json_file(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(normalized_json(value).encode("utf-8")).hexdigest()


def read_model_call_cache() -> dict[str, Any]:
    payload = read_json_file(MODEL_CACHE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def write_model_call_cache(cache: dict[str, Any]) -> None:
    write_json_file(MODEL_CACHE_PATH, cache)


def model_cache_key(task_type: str, model: str, prompt_template_version: str, input_payload: Any) -> str:
    return stable_hash(
        {
            "taskType": task_type,
            "model": model,
            "promptTemplateVersion": prompt_template_version,
            "inputHash": stable_hash(input_payload),
        }
    )


def cached_json_model_call(
    task_type: str,
    prompt_template_version: str,
    input_payload: Any,
    prompt: str,
    caller: str,
) -> dict[str, Any]:
    provider_name = normalize_provider(agent.current_provider)
    model = provider_model_name(provider_name)
    key = model_cache_key(task_type, model, prompt_template_version, input_payload)
    cache = read_model_call_cache()
    cached = cache.get(key)
    if isinstance(cached, dict) and isinstance(cached.get("output_json"), dict):
        return cached["output_json"]
    output = extract_json_object(safe_model_call(caller=caller, prompt=prompt, task_type=task_type))
    cache[key] = {
        "cache_key": key,
        "task_type": task_type,
        "model": model,
        "input_hash": stable_hash(input_payload),
        "output_json": output,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_model_call_cache(cache)
    return output


def read_project_compact_facts_cache() -> dict[str, Any]:
    payload = read_json_file(PROJECT_COMPACT_FACTS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def write_project_compact_facts_cache(cache: dict[str, Any]) -> None:
    write_json_file(PROJECT_COMPACT_FACTS_PATH, cache)


def project_compact_cache_key(project_name: str, repo_name: str, source_hash: str) -> str:
    return stable_hash({"project_name": project_name, "repo_name": repo_name, "source_hash": source_hash})


def get_cached_project_compact_facts(project_name: str, repo_name: str, source_hash: str) -> Optional[dict[str, Any]]:
    cache = read_project_compact_facts_cache()
    record = cache.get(project_compact_cache_key(project_name, repo_name, source_hash))
    if isinstance(record, dict) and isinstance(record.get("compact_facts_json"), dict):
        return record["compact_facts_json"]
    return None


def save_project_compact_facts(project_name: str, repo_name: str, source_hash: str, facts: dict[str, Any]) -> None:
    cache = read_project_compact_facts_cache()
    key = project_compact_cache_key(project_name, repo_name, source_hash)
    now = datetime.now().isoformat(timespec="seconds")
    previous = cache.get(key) if isinstance(cache.get(key), dict) else {}
    cache[key] = {
        "id": key,
        "project_name": project_name,
        "repo_name": repo_name,
        "source_hash": source_hash,
        "compact_facts_json": facts,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
    }
    write_project_compact_facts_cache(cache)


def read_chunk_checkpoints() -> dict[str, Any]:
    payload = read_json_file(CHUNK_CHECKPOINTS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def write_chunk_checkpoints(checkpoints: dict[str, Any]) -> None:
    write_json_file(CHUNK_CHECKPOINTS_PATH, checkpoints)


def checkpoint_key(task_id: str, project_name: str, repo_name: str, chunk_index: int, chunk_hash: str) -> str:
    return stable_hash(
        {
            "task_id": task_id,
            "project_name": project_name,
            "repo_name": repo_name,
            "chunk_index": chunk_index,
            "chunk_hash": chunk_hash,
        }
    )


def get_completed_chunk_checkpoint(
    task_id: str,
    project_name: str,
    repo_name: str,
    chunk_index: int,
    chunk_hash: str,
) -> Optional[dict[str, Any]]:
    checkpoints = read_chunk_checkpoints()
    record = checkpoints.get(checkpoint_key(task_id, project_name, repo_name, chunk_index, chunk_hash))
    if isinstance(record, dict) and record.get("status") == "done" and isinstance(record.get("output_json"), dict):
        return record["output_json"]
    return None


def save_chunk_checkpoint(
    task_id: str,
    project_name: str,
    repo_name: str,
    chunk_index: int,
    chunk_hash: str,
    status: str,
    output_json: Optional[dict[str, Any]] = None,
    error: str = "",
) -> None:
    checkpoints = read_chunk_checkpoints()
    key = checkpoint_key(task_id, project_name, repo_name, chunk_index, chunk_hash)
    now = datetime.now().isoformat(timespec="seconds")
    previous = checkpoints.get(key) if isinstance(checkpoints.get(key), dict) else {}
    checkpoints[key] = {
        "id": key,
        "task_id": task_id,
        "project_name": project_name,
        "repo_name": repo_name,
        "chunk_index": chunk_index,
        "chunk_hash": chunk_hash,
        "status": status,
        "output_json": output_json or {},
        "error": error,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
    }
    write_chunk_checkpoints(checkpoints)


def read_resume_candidate_checkpoints() -> dict[str, Any]:
    payload = read_json_file(RESUME_CANDIDATE_CHECKPOINTS_PATH, {})
    return payload if isinstance(payload, dict) else {}


def write_resume_candidate_checkpoints(checkpoints: dict[str, Any]) -> None:
    write_json_file(RESUME_CANDIDATE_CHECKPOINTS_PATH, checkpoints)


def resume_candidate_checkpoint_key(
    task_id: str,
    project_id: str,
    project_name: str,
    source_hash: str,
) -> str:
    return stable_hash(
        {
            "task_id": task_id,
            "project_id": project_id,
            "project_name": project_name,
            "source_hash": source_hash,
        }
    )


def get_completed_resume_candidate_checkpoint(
    task_id: str,
    project_id: str,
    project_name: str,
    source_hash: str,
) -> Optional[dict[str, Any]]:
    if not task_id:
        return None
    checkpoints = read_resume_candidate_checkpoints()
    key = resume_candidate_checkpoint_key(task_id, project_id, project_name, source_hash)
    record = checkpoints.get(key)
    if isinstance(record, dict) and record.get("status") == "done" and isinstance(record.get("candidate_json"), dict):
        return record["candidate_json"]
    return None


def save_resume_candidate_checkpoint(
    task_id: str,
    project_id: str,
    project_name: str,
    source_hash: str,
    status: str,
    candidate_json: Optional[dict[str, Any]] = None,
    error: str = "",
) -> None:
    if not task_id:
        return
    checkpoints = read_resume_candidate_checkpoints()
    key = resume_candidate_checkpoint_key(task_id, project_id, project_name, source_hash)
    now = datetime.now().isoformat(timespec="seconds")
    previous = checkpoints.get(key) if isinstance(checkpoints.get(key), dict) else {}
    checkpoints[key] = {
        "id": key,
        "task_id": task_id,
        "project_id": project_id,
        "project_name": project_name,
        "source_hash": source_hash,
        "status": status,
        "candidate_json": candidate_json or {},
        "error": error,
        "created_at": previous.get("created_at") or now,
        "updated_at": now,
    }
    write_resume_candidate_checkpoints(checkpoints)


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
    response = safe_model_call(caller="update_memory_from_resume", prompt=prompt, task_type="memory_from_resume")
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


def repo_display_name(context: dict[str, Any]) -> str:
    return str(context.get("repository") or context.get("url") or context.get("project_name") or "unknown-repo")


def repo_project_name(context: dict[str, Any]) -> str:
    return str(context.get("project_name") or context.get("project") or context.get("repository") or context.get("url") or "Unknown Project")


def split_text_into_chunks(text: str, max_chars: int = REPO_CHUNK_TARGET_CHARS) -> list[str]:
    text = str(text or "")
    if len(text) <= max_chars:
        return [text]
    chunks = []
    remaining = text
    boundaries = ["\n\n  {", "\n\n", "\n# ", "\n## ", "\n### ", "\n", ", "]
    while len(remaining) > max_chars:
        cut = -1
        window = remaining[:max_chars]
        for boundary in boundaries:
            candidate = window.rfind(boundary)
            if candidate > max_chars * 0.55:
                cut = candidate + len(boundary)
                break
        if cut <= 0:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [chunk for chunk in chunks if chunk]


def source_files_from_repo_payload(repo_payload: dict[str, Any]) -> list[str]:
    files = [str(item) for item in repo_payload.get("root_files", [])[:20]]
    for commit in repo_payload.get("recent_commit_evidence", [])[:8]:
        if isinstance(commit, dict):
            files.extend(str(item) for item in commit.get("files", [])[:8])
    cleaned = []
    for file_name in files:
        if file_name and file_name not in cleaned:
            cleaned.append(file_name)
        if len(cleaned) >= 30:
            break
    return cleaned


def ensure_compact_prompt(prompt: str, caller: str) -> None:
    decision = routing_decision(normalize_provider(agent.current_provider), len(prompt))
    if should_use_proxy_safe_mode(decision) and len(prompt) > int(decision.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS):
        raise HTTPException(
            status_code=413,
            detail={
                "ok": False,
                "type": "ModelInputTooLargeForProxy",
                "caller": caller,
                "inputCharCount": len(prompt),
                "limit": int(decision.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS),
                "message": "Map-Reduce prompt is still too large for third-party proxy.",
            },
        )


def build_repo_chunk_prompt(
    project_name: str,
    repo_name: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    source_files: list[str],
) -> str:
    return f"""
Analyze one repository chunk and return compact evidence facts only.

Rules:
- Use only the chunk content below.
- Do not write resume bullets; produce evidence-backed facts that can later support Project Memory.
- Do not invent metrics, scale, deployment, users, ownership, or performance.
- Keep every array short.
- Return ONLY valid JSON with exactly this shape:
  {{
    "projectName": "{project_name}",
    "repoName": "{repo_name}",
    "chunkIndex": {chunk_index},
    "technicalFacts": [],
    "resumeRelevantClaims": [],
    "metricCandidates": [],
    "evidenceSources": [],
    "riskFlags": []
  }}

Context header:
projectName: {project_name}
repoName: {repo_name}
chunkIndex: {chunk_index}
totalChunks: {total_chunks}
sourceFiles: {json.dumps(source_files[:30], ensure_ascii=False)}

Repository chunk:
{chunk_text}
"""


def analyze_repo_chunk(
    task_id: str,
    project_name: str,
    repo_name: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    source_files: list[str],
) -> dict[str, Any]:
    chunk_hash = stable_hash(
        {
            "project_name": project_name,
            "repo_name": repo_name,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
        }
    )
    cached = get_completed_chunk_checkpoint(task_id, project_name, repo_name, chunk_index, chunk_hash)
    if cached:
        return cached
    prompt = build_repo_chunk_prompt(project_name, repo_name, chunk_text, chunk_index, total_chunks, source_files)
    ensure_compact_prompt(prompt, "repo_chunk_map")
    try:
        payload = cached_json_model_call(
            "repo_chunk_map",
            "v1",
            {
                "projectName": project_name,
                "repoName": repo_name,
                "chunkIndex": chunk_index,
                "chunkHash": chunk_hash,
            },
            prompt,
            "repo_chunk_map",
        )
    except HTTPException as error:
        if error.status_code != 524:
            save_chunk_checkpoint(task_id, project_name, repo_name, chunk_index, chunk_hash, "failed", error=str(error.detail))
            raise
        retry_text = truncate_text(chunk_text, max(1000, int(len(chunk_text) * 0.7)))
        retry_prompt = build_repo_chunk_prompt(project_name, repo_name, retry_text, chunk_index, total_chunks, source_files)
        ensure_compact_prompt(retry_prompt, "repo_chunk_map_retry")
        try:
            payload = extract_json_object(safe_model_call(caller="repo_chunk_map_retry", prompt=retry_prompt, task_type="repo_chunk_map"))
        except HTTPException as retry_error:
            save_chunk_checkpoint(
                task_id,
                project_name,
                repo_name,
                chunk_index,
                chunk_hash,
                "failed",
                error=str(retry_error.detail),
            )
            raise HTTPException(
                status_code=524,
                detail={
                    "ok": False,
                    "type": "ModelProxyTimeout",
                    "statusCode": 524,
                    "retryable": True,
                    "message": "Third-party proxy timed out. The task has been chunked; reduce chunk size or resume from failed chunk.",
                    "failedChunk": {
                        "projectName": project_name,
                        "repoName": repo_name,
                        "chunkIndex": chunk_index,
                    },
                },
            ) from retry_error
    for key in ["technicalFacts", "resumeRelevantClaims", "metricCandidates", "evidenceSources", "riskFlags"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    payload["projectName"] = payload.get("projectName") or project_name
    payload["repoName"] = payload.get("repoName") or repo_name
    payload["chunkIndex"] = payload.get("chunkIndex") or chunk_index
    save_chunk_checkpoint(task_id, project_name, repo_name, chunk_index, chunk_hash, "done", output_json=payload)
    return payload


def reduce_repo_chunk_facts(project_name: str, repo_name: str, chunk_facts: list[dict[str, Any]]) -> dict[str, Any]:
    batches = split_text_into_chunks(json.dumps(chunk_facts, ensure_ascii=False, indent=2), REDUCE_BATCH_TARGET_CHARS)
    reduced_batches = []
    for batch_index, batch_text in enumerate(batches, start=1):
        prompt = f"""
Reduce compact chunk facts for one repository.

Rules:
- Use only the provided compact chunk facts, not raw repository content.
- Merge duplicates and keep the result short.
- Do not invent metrics or stronger claims.
- Return ONLY valid JSON with exactly these keys:
  "projectName", "repoName", "projectSummary", "technicalStack", "keyModules",
  "resumeRelevantClaims", "metricCandidates", "evidenceSources", "recommendedResumeAngles", "riskFlags".

projectName: {project_name}
repoName: {repo_name}
batchIndex: {batch_index}
totalBatches: {len(batches)}

Chunk facts batch:
{batch_text}
"""
        ensure_compact_prompt(prompt, "repo_level_reduce")
        reduced_batches.append(
            cached_json_model_call(
                "repo_level_reduce",
                "v1",
                {"projectName": project_name, "repoName": repo_name, "batchIndex": batch_index, "batchText": batch_text},
                prompt,
                "repo_level_reduce",
            )
        )
    if len(reduced_batches) == 1:
        payload = reduced_batches[0]
    else:
        prompt = f"""
Final-reduce compact repository facts.

Rules:
- Use only these intermediate compact facts.
- Return ONLY valid JSON with exactly these keys:
  "projectName", "repoName", "projectSummary", "technicalStack", "keyModules",
  "resumeRelevantClaims", "metricCandidates", "evidenceSources", "recommendedResumeAngles", "riskFlags".

projectName: {project_name}
repoName: {repo_name}

Intermediate facts:
{json.dumps(reduced_batches, ensure_ascii=False, indent=2)}
"""
        ensure_compact_prompt(prompt, "repo_level_final_reduce")
        payload = cached_json_model_call(
            "repo_level_final_reduce",
            "v1",
            {"projectName": project_name, "repoName": repo_name, "reducedBatches": reduced_batches},
            prompt,
            "repo_level_final_reduce",
        )
    for key in ["technicalStack", "keyModules", "resumeRelevantClaims", "metricCandidates", "evidenceSources", "recommendedResumeAngles", "riskFlags"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    payload["projectName"] = payload.get("projectName") or project_name
    payload["repoName"] = payload.get("repoName") or repo_name
    return payload


def compact_repo_facts_from_context(context: dict[str, Any]) -> dict[str, Any]:
    repo_payload = build_project_analysis_payload([context])[0]
    project_name = repo_project_name(context)
    repo_name = repo_display_name(context)
    source_hash = stable_hash(repo_payload)
    cached = get_cached_project_compact_facts(project_name, repo_name, source_hash)
    if cached:
        return cached
    serialized = json.dumps(repo_payload, ensure_ascii=False, indent=2)
    chunks = split_text_into_chunks(serialized, REPO_CHUNK_TARGET_CHARS)
    source_files = source_files_from_repo_payload(repo_payload)
    task_id = current_agent_task_id.get("") or "repo-analysis"
    chunk_facts = [
        analyze_repo_chunk(task_id, project_name, repo_name, chunk, index, len(chunks), source_files)
        for index, chunk in enumerate(chunks, start=1)
    ]
    repo_facts = reduce_repo_chunk_facts(project_name, repo_name, chunk_facts)
    repo_facts["chunkCount"] = len(chunks)
    save_project_compact_facts(project_name, repo_name, source_hash, repo_facts)
    return repo_facts


def update_project_memory_from_compact_repo_facts(
    current_project_memory: dict[str, Any],
    repo_facts: dict[str, Any],
    agent_progress_messages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    project_name = str(repo_facts.get("projectName") or repo_facts.get("project_name") or "")
    repo_name = str(repo_facts.get("repoName") or repo_facts.get("repo_name") or "")
    scoped_memory = scoped_project_memory(current_project_memory, project_name=project_name, project_id="")
    prompt = f"""
Update project_memory.json from compact repository facts.

Rules:
- Use only the compact repo facts below. Do not request or infer raw repository content.
- Return a scoped project_memory object containing only the project affected by these repo facts.
- Preserve existing supported facts for that project.
- Keep unsupported metrics empty or omitted. Do not invent product impact, scale, users, performance, or business results.
- Return only valid JSON with exactly these keys:
  "changed": boolean,
  "additions": array of short strings,
  "project_memory": object

Target project: {project_name}
Repository: {repo_name}

Current scoped project_memory.json:
{json.dumps(scoped_memory, ensure_ascii=False, indent=2)}

Compact repository facts:
{json.dumps(repo_facts, ensure_ascii=False, indent=2)}
"""
    prompt = append_agent_progress_guidance(prompt, agent_progress_messages or [])
    ensure_compact_prompt(prompt, "project_level_reduce")
    payload = cached_json_model_call(
        "project_level_reduce",
        "v1",
        {"projectName": project_name, "repoName": repo_name, "scopedMemory": scoped_memory, "repoFacts": repo_facts},
        prompt,
        "project_level_reduce",
    )
    scoped_update = payload.get("project_memory")
    if not isinstance(scoped_update, dict):
        raise HTTPException(status_code=500, detail="Agent JSON response must include a project_memory object.")
    merged = merge_scoped_project_memory(current_project_memory, scoped_update, project_name=project_name, project_id="")
    payload["project_memory"] = merged
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
    force_chunking: bool = False,
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
    map_reduce_repositories: list[dict[str, Any]] = []
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
        decision = routing_decision(
            normalize_provider(agent.current_provider),
            len(prompt),
            {"forceChunking": force_chunking},
        )
        if decision.get("useMapReduce"):
            print(
                "safe_model_call: "
                "caller=update_project_memory_from_repo_analysis, taskType=project_memory_from_repo, "
                f"providerKind={decision.get('providerKind')}, proxySafeMode={str(decision.get('proxySafeMode')).lower()}, "
                f"originalInputCharCount={len(prompt)}, compactInputCharCount=0, "
                f"finalInputCharCount={len(prompt)}, useCompactInput=false, "
                f"useMapReduce={str(decision.get('useMapReduce')).lower()}"
            )
            repo_facts = compact_repo_facts_from_context(context)
            map_reduce_repositories.append(
                {
                    "projectName": repo_facts.get("projectName") or repo_project_name(context),
                    "repoName": repo_facts.get("repoName") or repo_display_name(context),
                    "chunkCount": repo_facts.get("chunkCount") or 1,
                    "reason": decision.get("reason"),
                }
            )
            payload = update_project_memory_from_compact_repo_facts(
                current_project_memory,
                repo_facts,
                agent_progress_messages=agent_progress_messages,
            )
        else:
            response = safe_model_call(
                caller="update_project_memory_from_repo_analysis",
                prompt=prompt,
                task_type="project_memory_from_repo",
                user_options={"forceChunking": force_chunking},
            )
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
        "map_reduce_repositories": map_reduce_repositories,
        "processed_repositories": processed_repositories,
        "skipped_repositories": skipped_repositories,
        "additions": additions,
        "project_memory": project_memory,
        "project_memory_path": str(agent.PROJECT_MEMORY_PATH),
    }


def build_project_memory_status_summary(
    project_memory_update: dict[str, Any],
    *,
    was_reanalyzed: bool,
    scan_results: list[dict[str, Any]],
    before_mtime: Optional[float],
    after_mtime: Optional[float],
) -> dict[str, Any]:
    additions = project_memory_update.get("additions", [])
    additions_count = len(additions) if isinstance(additions, list) else 0
    updated = bool(project_memory_update.get("updated"))
    usable_repo_count = sum(1 for result in scan_results if not result.get("error"))
    changed_repo_count = sum(1 for result in scan_results if result.get("changed"))
    fetched_repo_count = sum(
        1
        for result in scan_results
        if result.get("cache_status") in {"fetch", "incremental"}
    )

    if updated:
        status = "updated"
        label_zh = f"项目记忆已更新：新增 {additions_count} 条事实" if additions_count else "项目记忆已更新"
        label_en = f"Project Memory updated: {additions_count} new fact(s)" if additions_count else "Project Memory updated"
        detail_zh = "GitHub 证据已重新分析，并写入 project_memory.json。"
        detail_en = "GitHub evidence was reanalyzed and written to project_memory.json."
    elif was_reanalyzed and usable_repo_count:
        status = "checked_no_change"
        label_zh = "项目记忆已检查：没有新增可写事实"
        label_en = "Project Memory checked: no new writable facts"
        detail_zh = "仓库证据已交给 agent 分析，但没有发现足够明确、可验证且需要写入的新事实，所以文件时间不会变化。"
        detail_en = "Repository evidence was analyzed, but no clear verified facts needed to be written, so the file timestamp did not change."
    elif was_reanalyzed:
        status = "skipped_no_usable_evidence"
        label_zh = "项目记忆未更新：没有可用仓库证据"
        label_en = "Project Memory not updated: no usable repository evidence"
        detail_zh = "本次没有可用于项目记忆分析的 GitHub 证据。"
        detail_en = "This run did not produce usable GitHub evidence for Project Memory analysis."
    else:
        status = "skipped_cache"
        label_zh = "项目记忆未重分析：仓库与分析提示未变化"
        label_en = "Project Memory not reanalyzed: repository and prompt unchanged"
        detail_zh = "本次复用了缓存证据；如果需要强制重新分析，可勾选重新分析缓存。"
        detail_en = "Cached evidence was reused; enable cached reanalysis if you want the agent to inspect it again."

    return {
        "status": status,
        "updated": updated,
        "reanalyzed": bool(was_reanalyzed),
        "additions_count": additions_count,
        "usable_repository_count": usable_repo_count,
        "changed_repository_count": changed_repo_count,
        "fetched_repository_count": fetched_repo_count,
        "before_mtime": before_mtime,
        "after_mtime": after_mtime,
        "label": label_zh,
        "label_zh": label_zh,
        "label_en": label_en,
        "detail": detail_zh,
        "detail_zh": detail_zh,
        "detail_en": detail_en,
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


def provider_safe_text_limit(default_chars: int, proxy_chars: int) -> int:
    decision = routing_decision(normalize_provider(agent.current_provider), 0)
    if should_use_proxy_safe_mode(decision):
        return min(default_chars, proxy_chars)
    return default_chars


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
        "project_rank": candidate.get("project_rank"),
        "bullet_budget": candidate.get("bullet_budget"),
        "focus_areas": candidate.get("focus_areas", [])[:8],
        "fit_reason": truncate_text(candidate.get("fit_reason") or candidate.get("job_alignment"), 800),
        "final_bullets": compact_value_for_prompt(bullets, 700, 6),
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
        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            document_end = resume_latex.find("\\end{document}", match.end())
            end = document_end if document_end != -1 else len(resume_latex)
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
    def normalized_section(value: str) -> str:
        return re.sub(r"[\s/_-]+", "", str(value or "").lower())

    target = section_name.lower()
    normalized_target = normalized_section(target)
    aliases = {
        "summary": ["professional summary", "summary", "profile", "summary/profile-section", "\u4e13\u4e1a\u6458\u8981", "\u4e2a\u4eba\u7b80\u4ecb", "\u7b80\u4ecb"],
        "summary/profile-section": ["professional summary", "summary", "profile", "\u4e13\u4e1a\u6458\u8981", "\u4e2a\u4eba\u7b80\u4ecb", "\u7b80\u4ecb"],
        "skills-section": ["technical skills", "skills", "\u6280\u80fd", "\u6280\u672f\u6280\u80fd", "\u4e13\u4e1a\u6280\u80fd"],
        "experience-section": ["experience", "work experience", "professional experience", "\u7ecf\u5386", "\u5de5\u4f5c\u7ecf\u5386", "\u5b9e\u4e60\u7ecf\u5386", "\u9879\u76ee\u7ecf\u5386"],
        "project": ["projects", "project-section", "\u9879\u76ee", "\u9879\u76ee\u7ecf\u5386", "\u9879\u76ee\u7ecf\u9a8c"],
        "projects": ["projects", "project-section", "\u9879\u76ee", "\u9879\u76ee\u7ecf\u5386", "\u9879\u76ee\u7ecf\u9a8c"],
        "project-section": ["projects", "\u9879\u76ee", "\u9879\u76ee\u7ecf\u5386", "\u9879\u76ee\u7ecf\u9a8c"],
    }
    names = aliases.get(target, [target])
    normalized_names = {normalized_section(name) for name in names}
    for span in latex_section_spans(resume_latex):
        span_name = span["name"].lower()
        normalized_span = normalized_section(span_name)
        if normalized_span in normalized_names or normalized_target in normalized_span:
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
    replacement = strip_markdown_code_fence(replacement)
    if block_payload.get("scope") == "project_block" and replacement.lstrip().startswith("\\section"):
        section = find_latex_section(current_resume, str(block_payload.get("section_name") or "projects"))
        if section and section["text"] in current_resume and replacement:
            return current_resume.replace(section["text"], replacement, 1)
    if original and original in current_resume and replacement:
        return current_resume.replace(original, replacement, 1)
    if block_payload.get("scope") == "document" and replacement:
        return replacement
    raise HTTPException(status_code=400, detail="Compact retry did not return a replaceable LaTeX block.")


def strip_markdown_code_fence(content: Any) -> str:
    stripped = str(content or "").strip()
    fenced = re.search(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    latex_start = re.search(r"\\(?:documentclass|begin\{document\}|section\{|resume[A-Za-z]+)", stripped)
    if latex_start and latex_start.start() > 0:
        stripped = stripped[latex_start.start() :].strip()
    return stripped


def looks_like_latex_fragment(content: Any) -> bool:
    text = strip_markdown_code_fence(content)
    return bool(re.search(r"\\(?:section|resume[A-Za-z]+|begin\{|end\{|item\b|textbf\{)", text))


def validate_complete_resume_or_raise(resume_latex: str, error_detail: str) -> str:
    if not agent.looks_like_latex_resume(resume_latex):
        raise HTTPException(status_code=400, detail=error_detail)
    validation_issues = agent.latex_resume_validation_issues(resume_latex)
    if validation_issues:
        raise HTTPException(
            status_code=400,
            detail=f"{error_detail} " + " ".join(validation_issues),
        )
    return resume_latex


def complete_resume_from_merge_response(
    answer: str,
    current_resume: str,
    target_block: dict[str, Any],
    error_detail: str,
) -> str:
    normalized_answer = strip_markdown_code_fence(answer)
    document = agent.extract_latex_document(normalized_answer)
    if document:
        return validate_complete_resume_or_raise(document, error_detail)
    if not looks_like_latex_fragment(normalized_answer):
        raise HTTPException(status_code=400, detail=error_detail)
    try:
        merged = replace_resume_block(current_resume, target_block, normalized_answer)
    except HTTPException as error:
        raise HTTPException(status_code=400, detail=error_detail) from error
    return validate_complete_resume_or_raise(merged, error_detail)


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
    if section_name.lower().startswith("project") and "selected_project_candidates" not in candidate:
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
            PROJECT_PRIORITY_INSTRUCTION,
            "Do not re-add omitted projects or expand lower-ranked projects beyond their target bullet budget.",
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
        payload["candidate"] = compact_value_for_prompt(payload["candidate"], 500, 6)
    return payload


def build_retry_merge_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are receiving a compact emergency merge payload because the full resume merge payload exceeded the model context window.

Use only the provided compact JD, target LaTeX block or section, candidate, and claim boundaries.
Preserve valid LaTeX. Do not invent technologies, impact metrics, deployment, ownership, or unsupported claims.
Prefer the candidate wording that best matches the JD while remaining evidence-grounded.
For Projects-section merges: {PROJECT_PRIORITY_INSTRUCTION}
Preserve selected project order, keep omitted projects out, and reduce lower-ranked project bullets first when space is limited.
Keep the section/project length close to the original budget.
Return only the merged LaTeX block or section. Do not include Markdown fences or explanation.

Compact retry payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def compact_merge_candidate(candidate: Any, max_string_chars: int = 420, max_list_items: int = 4) -> Any:
    if not isinstance(candidate, dict):
        return compact_value_for_prompt(candidate, max_string_chars, max_list_items)
    keep_keys = [
        "section_type",
        "project_id",
        "project_name",
        "source_name",
        "candidate_id",
        "fit",
        "keep_or_replace",
        "project_rank",
        "bullet_budget",
        "focus_areas",
        "project_ranking_context",
        "fit_reason",
        "job_alignment",
        "final_bullets",
        "recommended_bullets",
        "selected_project_candidates",
        "project_ranking",
        "preferred_project_count",
        "maximum_project_count",
        "ranking_rule",
        "one_page_rule",
        "one_page_cut_order",
        "recommended_skills_section",
        "skills_to_emphasize",
        "entry_recommendations",
        "recommended_summary",
        "allowed_claims",
        "forbidden_claims",
        "validation",
        "risks",
    ]
    compacted = {
        key: compact_value_for_prompt(candidate.get(key), max_string_chars, max_list_items)
        for key in keep_keys
        if key in candidate
    }
    if not compacted:
        return compact_value_for_prompt(candidate, max_string_chars, max_list_items)
    return compacted


def compact_latex_for_merge_prompt(latex: Any, max_chars: int, block_hint: str = "") -> str:
    text = str(latex or "")
    if len(text) <= max_chars:
        return text
    hint_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9+#.-]{3,}", block_hint or "")]
    important_lines = []
    fallback_lines = []
    for line in text.splitlines():
        stripped = line.rstrip()
        lower = stripped.lower()
        if not stripped:
            continue
        is_important = (
            "\\section" in lower
            or "\\resumeprojectheading" in lower
            or "\\resumesubheading" in lower
            or "\\resumeitem" in lower
            or "\\resumeitemlist" in lower
            or "\\resumesubheadinglist" in lower
            or "\\begin{" in lower
            or "\\end{" in lower
            or any(term in lower for term in hint_terms)
        )
        if is_important:
            important_lines.append(stripped)
        else:
            fallback_lines.append(stripped)
    ordered = important_lines or fallback_lines
    kept = []
    current = 0
    suffix = "\n% ... [target block truncated for compact merge prompt only]"
    budget = max(200, max_chars - len(suffix))
    for line in ordered:
        line_size = len(line) + 1
        if kept and current + line_size > budget:
            break
        kept.append(line)
        current += line_size
    if not kept:
        return truncate_text(text, max_chars)
    return "\n".join(kept) + suffix


def reduce_final_merge_payload_for_limit(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    reduced = json.loads(json.dumps(payload, ensure_ascii=False))

    def prompt_size() -> int:
        return len(build_retry_merge_prompt(reduced))

    if prompt_size() <= max_chars:
        return reduced

    reduced["candidate"] = compact_merge_candidate(reduced.get("candidate"), 420, 6)
    reduced["compact_jd"] = compact_value_for_prompt(reduced.get("compact_jd"), 420, 5)
    reduced["allowed_claims"] = compact_value_for_prompt(reduced.get("allowed_claims", []), 180, 6)
    reduced["forbidden_claims"] = compact_value_for_prompt(reduced.get("forbidden_claims", []), 180, 6)
    reduced["formatting_rules"] = reduced.get("formatting_rules", [])[:3] + [PROJECT_PRIORITY_INSTRUCTION]
    reduced["latex_safety_rules"] = reduced.get("latex_safety_rules", [])[:3]
    if prompt_size() <= max_chars:
        return reduced

    target_block = reduced.get("target_resume_block", {})
    if isinstance(target_block, dict):
        non_latex_size = prompt_size() - len(str(target_block.get("latex") or ""))
        latex_budget = max(1200, max_chars - non_latex_size - 500)
        target_block["latex"] = compact_latex_for_merge_prompt(
            target_block.get("latex"),
            latex_budget,
            str(target_block.get("block_hint") or target_block.get("section_name") or ""),
        )
        target_block["prompt_latex_is_excerpt"] = True
    if prompt_size() <= max_chars:
        return reduced

    reduced["candidate"] = compact_merge_candidate(reduced.get("candidate"), 240, 3)
    reduced["compact_jd"] = compact_value_for_prompt(reduced.get("compact_jd"), 240, 3)
    reduced["allowed_claims"] = compact_value_for_prompt(reduced.get("allowed_claims", []), 120, 3)
    reduced["forbidden_claims"] = compact_value_for_prompt(reduced.get("forbidden_claims", []), 120, 3)
    reduced["formatting_rules"] = [
        "Return only the merged target LaTeX block or section.",
        "Preserve LaTeX list boundaries and do not invent unsupported claims.",
        PROJECT_PRIORITY_INSTRUCTION,
    ]
    reduced["latex_safety_rules"] = ["Keep LaTeX commands and itemize/list environments balanced."]
    if isinstance(target_block, dict):
        target_block["latex"] = compact_latex_for_merge_prompt(
            target_block.get("latex"),
            900,
            str(target_block.get("block_hint") or target_block.get("section_name") or ""),
        )
    if prompt_size() > max_chars:
        reduced["candidate"] = compact_merge_candidate(reduced.get("candidate"), 120, 2)
        reduced["compact_jd"] = compact_value_for_prompt(reduced.get("compact_jd"), 120, 2)
        reduced["allowed_claims"] = []
        reduced["forbidden_claims"] = []
        if isinstance(target_block, dict):
            target_block["latex"] = compact_latex_for_merge_prompt(
                target_block.get("latex"),
                420,
                str(target_block.get("block_hint") or target_block.get("section_name") or ""),
            )
    return reduced


def final_merge_compact_prompt(payload: dict[str, Any], max_chars: int = PROXY_SAFE_MAX_INPUT_CHARS) -> dict[str, Any]:
    retry_payload = payload if "target_resume_block" in payload else merge_retry_payload_for_prompt(payload)
    compact_payload = reduce_final_merge_payload_for_limit(retry_payload, max_chars)
    return {"prompt": build_retry_merge_prompt(compact_payload), "payload": compact_payload}


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
        "tools": project.get("tools", []),
        "coursework": project.get("coursework", []),
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


def default_resume_constraints(resume_constraints: dict[str, Any] | None = None) -> dict[str, Any]:
    constraints = {
        "prefer_one_page": True,
        "preferred_project_count": PREFERRED_RESUME_PROJECTS,
        "maximum_project_count": MAX_STAGED_PROJECTS,
        "project_priority_instruction": PROJECT_PRIORITY_INSTRUCTION,
        "two_project_budgets": [5, 3],
        "three_project_budgets": [4, 3, 2],
        "one_page_cut_order": PROJECT_ONE_PAGE_CUT_ORDER,
    }
    if isinstance(resume_constraints, dict):
        constraints.update({key: value for key, value in resume_constraints.items() if value is not None})
    return constraints


def project_identifier(project: dict[str, Any]) -> str:
    return str(project.get("project_id") or project.get("source_name") or project.get("project_name") or project.get("name") or "").strip()


def normalize_project_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "", text)


def project_labels_match(left: Any, right: Any) -> bool:
    left_norm = normalize_project_label(left)
    right_norm = normalize_project_label(right)
    if not left_norm or not right_norm:
        return False
    return (
        left_norm == right_norm
        or left_norm in right_norm
        or right_norm in left_norm
        or left_norm.endswith(right_norm)
        or right_norm.endswith(left_norm)
    )


def project_card_signal_text(project_card: dict[str, Any]) -> str:
    compact = compact_value_for_prompt(project_card, 550, 10)
    return json.dumps(compact, ensure_ascii=False).lower()


GENERIC_PROJECT_RANKING_TERMS = {
    "all",
    "and",
    "are",
    "based",
    "can",
    "company",
    "experience",
    "job",
    "process",
    "project",
    "projects",
    "role",
    "software",
    "solution",
    "solutions",
    "system",
    "systems",
    "team",
    "technology",
    "the",
    "this",
    "use",
    "user",
    "work",
    "with",
}


def is_meaningful_project_ranking_term(term: Any) -> bool:
    text = str(term or "").strip().lower()
    if len(text) < 3:
        return False
    if text in GENERIC_PROJECT_RANKING_TERMS:
        return False
    if re.fullmatch(r"\d+", text):
        return False
    return True


def ranking_terms_from_profile(jd_profile: dict[str, Any], role_profile: dict[str, Any]) -> list[str]:
    terms = []
    for key in [
        "job_title",
        "must_have_skills",
        "responsibilities",
        "tools_platforms",
        "domain_knowledge",
        "soft_skills",
        "repeated_ats_keywords",
        "action_verbs",
        "evidence_types_to_emphasize",
    ]:
        value = jd_profile.get(key) if isinstance(jd_profile, dict) else None
        for item in listish(value):
            append_unique(terms, item, 60)
    for key in ["role_focus", "high_priority_keywords"]:
        for item in listish(role_profile.get(key) if isinstance(role_profile, dict) else None):
            append_unique(terms, item, 60)
    return [term for term in terms if is_meaningful_project_ranking_term(term)]


def project_card_values(project_card: dict[str, Any], keys: list[str], limit: int = 30) -> list[str]:
    values = []
    sources = [
        project_card,
        project_card.get("identity") if isinstance(project_card.get("identity"), dict) else {},
        project_card.get("evidence_card") if isinstance(project_card.get("evidence_card"), dict) else {},
        project_card.get("role_lens") if isinstance(project_card.get("role_lens"), dict) else {},
        project_card.get("current_project_compact_facts") if isinstance(project_card.get("current_project_compact_facts"), dict) else {},
    ]
    for source in sources:
        for item in list_from_nested(source, keys, limit):
            append_unique(values, item, limit)
    return values


def score_term_hits(text: str, terms: list[str], weight: int = 7, limit: int = 100) -> tuple[int, list[str]]:
    hits = []
    for term in terms:
        term_text = str(term or "").strip().lower()
        if not is_meaningful_project_ranking_term(term_text):
            continue
        if term_text in text:
            append_unique(hits, term, 20)
    return min(limit, len(hits) * weight), hits


def bounded_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def project_confidence_score(project_card: dict[str, Any]) -> int:
    text = project_card_signal_text(project_card)
    confidence_values = []
    for key in ["confidence", "evidence_confidence"]:
        value = project_card.get(key)
        if value:
            confidence_values.append(str(value).lower())
    evidence_card = project_card.get("evidence_card") if isinstance(project_card.get("evidence_card"), dict) else {}
    if evidence_card.get("confidence"):
        confidence_values.append(str(evidence_card.get("confidence")).lower())
    if "high" in confidence_values:
        return 90
    if "medium" in confidence_values:
        return 70
    if "low" in confidence_values:
        return 40
    if "allowed_claims" in text or "resume_relevant_claims" in text or "star_facts" in text:
        return 75
    return 55


def project_evidence_strength_score(project_card: dict[str, Any]) -> int:
    evidence_signals = []
    for key in [
        "allowed_claims",
        "artifacts",
        "source_refs",
        "resumeRelevantClaims",
        "resume_relevant_claims",
        "star_facts",
        "recent_changes",
        "final_bullets",
        "recommended_bullets",
        "metricCandidates",
        "real_metrics",
    ]:
        for item in project_card_values(project_card, [key], 40):
            append_unique(evidence_signals, item, 60)
    confidence = project_confidence_score(project_card)
    base = min(70, len(evidence_signals) * 6)
    if project_card_values(project_card, ["allowed_claims"], 10):
        base += 10
    if project_card_values(project_card, ["star_facts", "metricCandidates", "real_metrics"], 10):
        base += 8
    return bounded_score(base + confidence * 0.18)


def project_technical_depth_score(project_card: dict[str, Any]) -> int:
    depth_signals = []
    for key in [
        "tech_stack",
        "tools",
        "workflows",
        "confirmed_features",
        "methods",
        "features",
        "keyModules",
        "userContributionSignals",
        "testing_signals",
        "debugging_signals",
        "documentation_signals",
        "automation_signals",
        "artifacts",
    ]:
        for item in project_card_values(project_card, [key], 50):
            append_unique(depth_signals, item, 80)
    text = project_card_signal_text(project_card)
    process_bonus = 0
    for keyword in ["workflow", "validation", "ranking", "retrieval", "storage", "database", "testing", "debug", "automation", "documentation"]:
        if keyword in text:
            process_bonus += 4
    return bounded_score(min(72, len(depth_signals) * 5) + process_bonus)


def project_focus_areas(project_card: dict[str, Any], role_profile: dict[str, Any], jd_profile: dict[str, Any]) -> list[str]:
    text = project_card_signal_text(project_card)
    focus = []
    if any(key in text for key in ["workflow", "feature", "module", "implemented", "automation", "pipeline"]):
        append_unique(focus, "core implementation", 8)
    if project_card_values(project_card, ["tech_stack", "tools", "methods"], 12):
        append_unique(focus, "tools/methods used", 8)
    if any(key in text for key in ["data", "database", "sql", "sqlite", "mongodb", "storage", "memory", "retrieval", "workflow"]):
        append_unique(focus, "data/storage/workflow logic", 8)
    if any(key in text for key in ["test", "debug", "troubleshoot", "document", "automation", "validation", "configuration"]):
        append_unique(focus, "testing/debugging/automation/documentation", 8)
    role_terms = ranking_terms_from_profile(jd_profile, role_profile)
    if score_term_hits(text, role_terms, weight=1)[1]:
        append_unique(focus, "target-role relevance", 8)
    if not focus:
        focus = ["core implementation", "target-role relevance"]
    return focus[:5]


def project_term_set(project_card: dict[str, Any]) -> set[str]:
    text = project_card_signal_text(project_card)
    tokens = set(re.findall(r"[a-z][a-z0-9+#.-]{2,}", text))
    generic = {
        "project",
        "projects",
        "source",
        "facts",
        "memory",
        "resume",
        "json",
        "section",
        "candidate",
        "with",
        "from",
        "that",
        "this",
        "using",
        "used",
    }
    return {token for token in tokens if token not in generic}


def project_distinctiveness_scores(project_cards: list[dict[str, Any]]) -> list[int]:
    term_sets = [project_term_set(project) for project in project_cards]
    scores = []
    for index, terms in enumerate(term_sets):
        if not terms or len(term_sets) == 1:
            scores.append(75 if terms else 50)
            continue
        max_overlap = 0.0
        unique_terms = set(terms)
        for other_index, other_terms in enumerate(term_sets):
            if other_index == index or not other_terms:
                continue
            union = terms | other_terms
            if union:
                max_overlap = max(max_overlap, len(terms & other_terms) / len(union))
            unique_terms -= other_terms
        scores.append(bounded_score(62 + min(24, len(unique_terms) * 2) - max_overlap * 50))
    return scores


def project_resume_value_score(
    relevance_score: int,
    evidence_strength: int,
    role_alignment: int,
    technical_depth: int,
    confidence_score: int,
) -> int:
    return bounded_score(
        relevance_score * 0.30
        + evidence_strength * 0.24
        + role_alignment * 0.18
        + technical_depth * 0.16
        + confidence_score * 0.12
    )


def project_omission_reason(scored_project: dict[str, Any]) -> str:
    if scored_project["relevance_score"] < 35:
        return "Lower JD relevance"
    if scored_project["evidence_strength"] < 45:
        return "Weaker supported evidence"
    if scored_project["distinctiveness_score"] < 45:
        return "Repetitive evidence compared with stronger projects"
    return "Insufficient space after stronger, more job-relevant projects"


def project_adds_distinct_third_signal(
    candidate: dict[str, Any],
    selected: list[dict[str, Any]],
    jd_terms: list[str],
) -> bool:
    selected_text = " ".join(item.get("signal_text", "") for item in selected)
    candidate_text = candidate.get("signal_text", "")
    meaningful_terms = [term for term in jd_terms if is_meaningful_project_ranking_term(term)]
    unique_jd_hits = [
        term for term in meaningful_terms
        if str(term or "").lower() in candidate_text and str(term or "").lower() not in selected_text
    ]
    high_value_unique_hits = [
        term for term in unique_jd_hits
        if str(term).lower() in {
            "sql",
            "data",
            "database",
            "testing",
            "test",
            "debug",
            "troubleshoot",
            "documentation",
            "stakeholder",
            "inventory",
            "operations",
            "support",
            "reporting",
            "python",
            "powershell",
            "linux",
            "azure",
            "docker",
            "api",
            "validation",
        }
    ]
    return bool(high_value_unique_hits)


def rank_projects_for_resume(
    project_cards: list[dict],
    jd_profile: dict,
    role_profile: dict,
    resume_constraints: dict | None = None,
) -> dict:
    constraints = default_resume_constraints(resume_constraints)
    cards = [card for card in project_cards if isinstance(card, dict)]
    if not cards:
        return {"selected_projects": [], "omitted_projects": []}

    jd_terms = ranking_terms_from_profile(jd_profile if isinstance(jd_profile, dict) else {}, role_profile if isinstance(role_profile, dict) else {})
    distinctiveness_scores = project_distinctiveness_scores(cards)
    scored_projects = []
    for index, card in enumerate(cards):
        text = project_card_signal_text(card)
        jd_hit_score, jd_hits = score_term_hits(text, jd_terms, weight=8)
        role_terms = []
        for term in ROLE_LENS_PRIORITIES.get(str(role_profile.get("role_family") or "software_engineering"), []):
            append_unique(role_terms, term, 40)
        for term in role_profile.get("role_focus", []) if isinstance(role_profile, dict) else []:
            append_unique(role_terms, term, 40)
        role_hit_score, role_hits = score_term_hits(text, role_terms, weight=9)
        evidence_strength = project_evidence_strength_score(card)
        technical_depth = project_technical_depth_score(card)
        confidence_score = project_confidence_score(card)
        relevance_score = bounded_score(jd_hit_score + min(22, evidence_strength * 0.18) + min(12, technical_depth * 0.10))
        role_alignment = bounded_score(role_hit_score + min(24, relevance_score * 0.24))
        resume_value = project_resume_value_score(
            relevance_score,
            evidence_strength,
            role_alignment,
            technical_depth,
            confidence_score,
        )
        distinctiveness = distinctiveness_scores[index]
        total_score = bounded_score(
            relevance_score * 0.28
            + evidence_strength * 0.22
            + role_alignment * 0.18
            + technical_depth * 0.12
            + distinctiveness * 0.10
            + resume_value * 0.10
        )
        focus_areas = project_focus_areas(card, role_profile if isinstance(role_profile, dict) else {}, jd_profile if isinstance(jd_profile, dict) else {})
        scored_projects.append(
            {
                "project_card": card,
                "project_id": project_identifier(card),
                "project_name": project_display_name(card),
                "original_index": index,
                "relevance_score": relevance_score,
                "evidence_strength": evidence_strength,
                "role_alignment": role_alignment,
                "technical_depth_score": technical_depth,
                "distinctiveness_score": distinctiveness,
                "resume_value_score": resume_value,
                "confidence_score": confidence_score,
                "total_score": total_score,
                "matched_jd_terms": jd_hits[:8],
                "matched_role_terms": role_hits[:8],
                "focus_areas": focus_areas,
                "signal_text": text,
            }
        )

    scored_projects.sort(
        key=lambda item: (
            -item["total_score"],
            -item["relevance_score"],
            -item["evidence_strength"],
            item["original_index"],
        )
    )

    selected = []
    if scored_projects:
        selected.append(scored_projects[0])
    if len(scored_projects) >= 2:
        second = scored_projects[1]
        if second["total_score"] >= 42 or second["relevance_score"] >= 30 or second["evidence_strength"] >= 55:
            selected.append(second)
    if len(scored_projects) >= 3:
        third = scored_projects[2]
        third_is_relevant = third["relevance_score"] >= 45 or third["total_score"] >= 56
        third_has_evidence = third["evidence_strength"] >= 48
        third_is_distinct = third["distinctiveness_score"] >= 52
        third_is_close_to_second = len(selected) < 2 or third["total_score"] >= selected[1]["total_score"] - 4
        third_adds_signal = project_adds_distinct_third_signal(third, selected, jd_terms)
        if (
            len(selected) >= 2
            and third_is_relevant
            and third_has_evidence
            and third_is_distinct
            and third_is_close_to_second
            and third_adds_signal
        ):
            selected.append(third)

    maximum_projects = int(constraints.get("maximum_project_count") or MAX_STAGED_PROJECTS)
    selected = selected[:max(1, min(maximum_projects, MAX_STAGED_PROJECTS))]
    total_selected = len(selected)
    selected_entries = []
    for rank, item in enumerate(selected, start=1):
        focus_areas = list(item["focus_areas"])
        if rank == 1:
            for focus in [
                "core implementation",
                "tools/methods used",
                "data/storage/workflow logic",
                "testing/debugging/automation/documentation",
                "target-role relevance",
            ]:
                append_unique(focus_areas, focus, 5)
        selected_entries.append(
            {
                "project_id": item["project_id"],
                "project_name": item["project_name"],
                "rank": rank,
                "relevance_score": item["relevance_score"],
                "evidence_strength": item["evidence_strength"],
                "role_alignment": item["role_alignment"],
                "technical_depth_score": item["technical_depth_score"],
                "distinctiveness_score": item["distinctiveness_score"],
                "resume_value_score": item["resume_value_score"],
                "confidence_score": item["confidence_score"],
                "total_score": item["total_score"],
                "reason_selected": (
                    "Strong JD fit with supported evidence"
                    if item["matched_jd_terms"]
                    else "Best available supported project evidence for the target role"
                ),
                "bullet_budget": project_bullet_budget(rank, total_selected),
                "focus_areas": focus_areas[:5],
                "matched_jd_terms": item["matched_jd_terms"],
                "matched_role_terms": item["matched_role_terms"],
            }
        )

    selected_names = {entry["project_name"] for entry in selected_entries}
    omitted_entries = []
    for item in scored_projects:
        if item["project_name"] in selected_names:
            continue
        omitted_entries.append(
            {
                "project_id": item["project_id"],
                "project_name": item["project_name"],
                "reason_omitted": project_omission_reason(item),
            }
        )

    return {
        "selected_projects": selected_entries,
        "omitted_projects": omitted_entries,
        "ranking_basis": {
            "criteria": [
                "JD relevance",
                "evidence strength",
                "role-family alignment",
                "technical/process depth",
                "distinctiveness compared with other projects",
                "resume value for the target job",
                "confidence of supported claims",
            ],
            "preferred_project_count": constraints.get("preferred_project_count"),
            "maximum_project_count": constraints.get("maximum_project_count"),
            "one_page_cut_order": constraints.get("one_page_cut_order"),
        },
    }


def project_ranking_entry_for_project(project_ranking: dict[str, Any] | None, project: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(project_ranking, dict):
        return {}
    project_name = project_display_name(project)
    project_id = project_identifier(project)
    for entry in project_ranking.get("selected_projects", []) if isinstance(project_ranking.get("selected_projects"), list) else []:
        if not isinstance(entry, dict):
            continue
        if project_id and str(entry.get("project_id") or "") == project_id:
            return entry
        if project_labels_match(entry.get("project_name"), project_name):
            return entry
    return {}


def projects_from_ranking(projects: list[dict[str, Any]], project_ranking: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for entry in project_ranking.get("selected_projects", []) if isinstance(project_ranking.get("selected_projects"), list) else []:
        if not isinstance(entry, dict):
            continue
        for project in projects:
            if project in selected:
                continue
            if str(entry.get("project_id") or "") and str(entry.get("project_id")) == project_identifier(project):
                selected.append(project)
                break
            if project_labels_match(entry.get("project_name"), project_display_name(project)):
                selected.append(project)
                break
    return selected[:MAX_STAGED_PROJECTS]


def attach_candidate_claims_to_project_ranking(
    project_ranking: dict[str, Any],
    project_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    ranking = json.loads(json.dumps(project_ranking or {}, ensure_ascii=False))
    for entry in ranking.get("selected_projects", []) if isinstance(ranking.get("selected_projects"), list) else []:
        for candidate in project_candidates:
            if (
                str(entry.get("project_id") or "") and str(entry.get("project_id")) == str(candidate.get("project_id") or "")
            ) or project_labels_match(entry.get("project_name"), candidate.get("project_name") or candidate.get("source_name")):
                entry["allowed_claims"] = candidate.get("allowed_claims", [])[:MAX_PROMPT_CLAIMS]
                entry["forbidden_claims"] = candidate.get("forbidden_claims", [])[:MAX_PROMPT_CLAIMS]
                entry["candidate_bullet_count"] = len(candidate.get("recommended_bullets") or candidate.get("final_bullets") or [])
                break
    return ranking


def project_section_text_for_validation(projects_section: dict | str) -> str:
    if isinstance(projects_section, dict):
        for key in ["latex", "text", "content", "projects_section"]:
            if projects_section.get(key):
                return str(projects_section.get(key))
        return json.dumps(projects_section, ensure_ascii=False)
    text = str(projects_section or "")
    section = find_latex_section(text, "projects")
    return section["text"] if section else text


def project_block_heading(block: str) -> str:
    try:
        return agent.project_heading_name(block)
    except Exception:
        match = re.search(r"\\textbf\{([^}]+)\}", block)
        return match.group(1).strip() if match else "Unnamed project"


def project_blocks_for_validation(section_text: str) -> list[str]:
    blocks = agent.project_blocks_from_latex(section_text)
    if blocks:
        return blocks
    starts = [match.start() for match in re.finditer(r"\\resume(?:Project|Sub)Heading\b", section_text)]
    if not starts:
        return []
    starts.append(len(section_text))
    return [section_text[starts[index] : starts[index + 1]] for index in range(len(starts) - 1)]


def bullet_count_for_project_block(block: str) -> int:
    return count_resume_items(block)


def project_block_for_rebuild(block: str) -> str:
    text = str(block or "").strip()
    section_end = text.find(r"\resumeSubHeadingListEnd")
    if section_end != -1:
        text = text[:section_end].rstrip()
    return text


def unsupported_claims_in_project_block(block: str, forbidden_claims: list[Any]) -> list[str]:
    lower = block.lower()
    forbidden_text = json.dumps(forbidden_claims, ensure_ascii=False).lower()
    unsupported = []
    for tool in PROTECTED_UNSUPPORTED_TOOLS:
        if tool.lower() in lower and tool.lower() in forbidden_text:
            append_unique(unsupported, tool, 12)
    for claim in forbidden_claims:
        claim_text = str(claim or "").strip().lower()
        if len(claim_text) >= 18 and claim_text in lower:
            append_unique(unsupported, claim, 12)
    return unsupported


def validate_project_section_allocation(
    projects_section: dict | str,
    project_ranking: dict,
    resume_constraints: dict | None = None,
) -> dict:
    constraints = default_resume_constraints(resume_constraints)
    section_text = project_section_text_for_validation(projects_section)
    blocks = project_blocks_for_validation(section_text)
    headings = [project_block_heading(block) for block in blocks]
    bullet_counts = {
        heading: bullet_count_for_project_block(block)
        for heading, block in zip(headings, blocks)
    }
    selected = [
        entry for entry in project_ranking.get("selected_projects", [])
        if isinstance(entry, dict)
    ] if isinstance(project_ranking, dict) else []
    omitted = [
        entry for entry in project_ranking.get("omitted_projects", [])
        if isinstance(entry, dict)
    ] if isinstance(project_ranking, dict) else []
    issues = []
    suggested_fixes = []
    project_count = len(blocks)
    max_projects = int(constraints.get("maximum_project_count") or MAX_STAGED_PROJECTS)
    preferred_projects = int(constraints.get("preferred_project_count") or PREFERRED_RESUME_PROJECTS)

    if project_count == 0:
        issues.append("Projects section contains no parseable project entries.")
        suggested_fixes.append("Regenerate the Projects section using the selected project ranking.")
    if project_count > max_projects:
        issues.append(f"Projects section contains {project_count} projects, above the maximum of {max_projects}.")
        suggested_fixes.append("Remove projects beyond the selected ranking, starting with omitted or lowest-ranked projects.")
    expected_default_count = min(preferred_projects, len(selected)) if selected else preferred_projects
    if selected and len(selected) >= preferred_projects and project_count < expected_default_count:
        issues.append(f"Projects section contains fewer than the preferred {expected_default_count} selected projects.")
        suggested_fixes.append("Restore the highest-ranked omitted selected project if space allows.")
    if selected and project_count > len(selected):
        issues.append("Projects section includes more projects than the ranking selected.")
        suggested_fixes.append("Remove projects not present in selected_projects.")

    heading_order_matches = True
    selected_present = []
    for heading in headings:
        for entry in selected:
            if project_labels_match(entry.get("project_name"), heading):
                selected_present.append(entry)
                break
    expected_names = [entry.get("project_name") for entry in selected_present]
    actual_names = headings[: len(selected_present)]
    for expected, actual in zip(expected_names, actual_names):
        if not project_labels_match(expected, actual):
            heading_order_matches = False
            break
    if selected_present and not heading_order_matches:
        issues.append("Project order does not follow the ranking.")
        suggested_fixes.append("Reorder Projects entries to match selected_projects rank order.")

    omitted_readded = []
    for omitted_entry in omitted:
        for heading in headings:
            if project_labels_match(omitted_entry.get("project_name"), heading):
                append_unique(omitted_readded, omitted_entry.get("project_name"), 12)
    if omitted_readded:
        issues.append("Omitted projects were re-added: " + ", ".join(omitted_readded))
        suggested_fixes.append("Remove omitted projects from the final Projects section.")

    budget_by_heading = {}
    rank_by_heading = {}
    score_by_heading = {}
    block_by_heading = dict(zip(headings, blocks))
    for entry in selected:
        for heading in headings:
            if project_labels_match(entry.get("project_name"), heading):
                budget_by_heading[heading] = int(entry.get("bullet_budget") or 0)
                rank_by_heading[heading] = int(entry.get("rank") or 0)
                score_by_heading[heading] = int(entry.get("total_score") or entry.get("relevance_score") or 0)
                break

    allocation_matches = True
    for heading, count in bullet_counts.items():
        budget = budget_by_heading.get(heading)
        if not budget:
            continue
        if abs(count - budget) > 1:
            allocation_matches = False
            issues.append(f"{heading} has {count} bullets, outside the intended budget of about {budget}.")
            suggested_fixes.append(f"Adjust {heading} toward {budget} bullets.")
        rank = rank_by_heading.get(heading, 0)
        if rank > 1 and count > budget + 1:
            issues.append(f"Lower-ranked project {heading} is over-expanded relative to its budget.")
            suggested_fixes.append(f"Trim lower-ranked {heading} bullets first.")

    top_has_strongest_detail = True
    if selected:
        top_entry = selected[0]
        top_heading = next((heading for heading in headings if project_labels_match(top_entry.get("project_name"), heading)), "")
        if top_heading:
            top_count = bullet_counts.get(top_heading, 0)
            top_length = len(block_by_heading.get(top_heading, ""))
            for heading, count in bullet_counts.items():
                if heading == top_heading:
                    continue
                score_gap = int(top_entry.get("total_score") or 0) - score_by_heading.get(heading, 0)
                if count > top_count or (count == top_count and score_gap > 8):
                    top_has_strongest_detail = False
                if len(block_by_heading.get(heading, "")) > top_length * 1.15 and score_gap > 8:
                    top_has_strongest_detail = False
        if not top_has_strongest_detail:
            issues.append("The highest-ranked project does not receive the strongest or most detailed treatment.")
            suggested_fixes.append("Move detail back to rank #1 and trim lower-ranked projects first.")

    unsupported_claims = []
    for entry in selected:
        forbidden_claims = entry.get("forbidden_claims", []) if isinstance(entry.get("forbidden_claims"), list) else []
        if not forbidden_claims:
            continue
        for heading, block in block_by_heading.items():
            if project_labels_match(entry.get("project_name"), heading):
                for claim in unsupported_claims_in_project_block(block, forbidden_claims):
                    append_unique(unsupported_claims, f"{heading}: {claim}", 20)
    if unsupported_claims:
        issues.append("Project bullets include unsupported or forbidden claims: " + "; ".join(unsupported_claims))
        suggested_fixes.append("Remove unsupported claim terms or replace them with allowed evidence-backed wording.")

    return {
        "valid": not issues,
        "issues": issues,
        "project_count": project_count,
        "bullet_counts": bullet_counts,
        "allocation_matches_ranking": allocation_matches and heading_order_matches and top_has_strongest_detail,
        "top_project_has_strongest_detail": top_has_strongest_detail,
        "omitted_projects_readded": omitted_readded,
        "unsupported_claims": unsupported_claims,
        "suggested_fixes": suggested_fixes,
    }


def resume_item_spans(text: str) -> list[tuple[int, int, str]]:
    spans = []
    marker = r"\resumeItem{"
    position = 0
    while True:
        start = text.find(marker, position)
        if start == -1:
            break
        brace_index = start + len(marker) - 1
        depth = 0
        end = None
        index = brace_index
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
            index += 1
        if end is None:
            break
        spans.append((start, end, text[start:end]))
        position = end
    return spans


def candidate_bullet_texts(candidate: dict[str, Any] | None) -> list[str]:
    if not isinstance(candidate, dict):
        return []
    raw_bullets = candidate.get("recommended_bullets") or candidate.get("final_bullets") or []
    bullets = []
    for item in raw_bullets if isinstance(raw_bullets, list) else []:
        text = item.get("bullet") if isinstance(item, dict) else item
        if re.search(r"\[\s*truncated\s*\]|\btruncated\b|\bmore items\b|\.\.\.", str(text or ""), flags=re.IGNORECASE):
            continue
        append_unique(bullets, text, 12)
    return bullets


def project_candidate_matches_entry(candidate: dict[str, Any], entry: dict[str, Any]) -> bool:
    if not isinstance(candidate, dict) or not isinstance(entry, dict):
        return False
    entry_id = str(entry.get("project_id") or "")
    if entry_id and entry_id == str(candidate.get("project_id") or ""):
        return True
    return project_labels_match(entry.get("project_name"), candidate.get("project_name") or candidate.get("source_name"))


def find_project_candidate_for_entry(project_candidates: list[dict[str, Any]], entry: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in project_candidates:
        if project_candidate_matches_entry(candidate, entry):
            return candidate
    return None


def latex_resume_items_from_bullets(bullets: list[str]) -> str:
    return "\n".join(f"  \\resumeItem{{{latex_escape_text(bullet)}}}" for bullet in bullets if str(bullet or "").strip())


def replace_project_block_bullets(block: str, bullets: list[str]) -> str:
    if not bullets:
        return block
    item_text = latex_resume_items_from_bullets(bullets)
    start_marker = r"\resumeItemListStart"
    end_marker = r"\resumeItemListEnd"
    start = block.find(start_marker)
    end = block.rfind(end_marker)
    if start != -1 and end != -1 and end > start:
        return (
            block[: start + len(start_marker)].rstrip()
            + "\n"
            + item_text
            + "\n"
            + block[end:].lstrip()
        )
    return block.rstrip() + "\n" + start_marker + "\n" + item_text + "\n" + end_marker


def trim_project_block_to_budget(block: str, budget: int) -> str:
    if budget <= 0:
        return block
    spans = resume_item_spans(block)
    if len(spans) <= budget:
        return block
    keep = spans[:budget]
    pieces = []
    cursor = 0
    kept_indexes = {(start, end) for start, end, _ in keep}
    for start, end, _ in spans:
        if (start, end) in kept_indexes:
            pieces.append(block[cursor:end])
        else:
            pieces.append(block[cursor:start])
        cursor = end
    pieces.append(block[cursor:])
    return "".join(pieces)


def build_project_block_from_candidate(entry: dict[str, Any], candidate: dict[str, Any] | None, budget: int) -> str:
    project_name = str(entry.get("project_name") or (candidate or {}).get("project_name") or "Project").strip()
    evidence_card = (candidate or {}).get("evidence_card") if isinstance((candidate or {}).get("evidence_card"), dict) else {}
    technologies = []
    for item in (candidate or {}).get("skills_to_emphasize", []) if isinstance((candidate or {}).get("skills_to_emphasize"), list) else []:
        append_unique(technologies, item, 6)
    for item in evidence_card.get("technologies", []) if isinstance(evidence_card.get("technologies"), list) else []:
        append_unique(technologies, item, 6)
    tech_suffix = f" $|$ \\emph{{{latex_escape_text(', '.join(technologies[:5]))}}}" if technologies else ""
    bullets = candidate_bullet_texts(candidate)[:budget]
    return (
        "\\resumeProjectHeading\n"
        f"  {{\\textbf{{{latex_escape_text(project_name)}}}{tech_suffix}}}{{}}\n"
        "\\resumeItemListStart\n"
        + latex_resume_items_from_bullets(bullets)
        + "\n\\resumeItemListEnd"
    )


def enforce_project_section_allocation(
    current_resume: str,
    project_ranking: dict[str, Any] | None,
    project_candidates: list[dict[str, Any]] | None = None,
    resume_constraints: dict[str, Any] | None = None,
) -> str:
    if not isinstance(project_ranking, dict):
        return current_resume
    selected = [
        entry for entry in project_ranking.get("selected_projects", [])
        if isinstance(entry, dict)
    ]
    if not selected:
        return current_resume
    constraints = default_resume_constraints(resume_constraints)
    maximum_projects = max(1, min(int(constraints.get("maximum_project_count") or MAX_STAGED_PROJECTS), MAX_STAGED_PROJECTS))
    selected = sorted(selected, key=lambda item: int(item.get("rank") or 99))[:maximum_projects]
    section = find_latex_section(current_resume, "projects")
    if not section:
        return current_resume
    section_text = section["text"]
    blocks = project_blocks_for_validation(section_text)
    first_heading = section_text.find(r"\resumeProjectHeading")
    opening = section_text[:first_heading].strip() if first_heading != -1 else f"\\section{{{section['name']}}}"
    if r"\resumeSubHeadingListStart" not in opening:
        opening += "\n\\resumeSubHeadingListStart"
    block_by_heading = {
        project_block_heading(block): project_block_for_rebuild(block)
        for block in blocks
    }
    project_candidates = project_candidates or []
    rebuilt_blocks = []
    for entry in selected:
        budget = int(entry.get("bullet_budget") or project_bullet_budget(int(entry.get("rank") or len(rebuilt_blocks) + 1), len(selected)))
        existing_block = ""
        for heading, block in block_by_heading.items():
            if project_labels_match(entry.get("project_name"), heading):
                existing_block = block
                break
        candidate = find_project_candidate_for_entry(project_candidates, entry)
        candidate_bullets = candidate_bullet_texts(candidate)
        if existing_block:
            block = existing_block
            if candidate_bullets:
                block = replace_project_block_bullets(block, candidate_bullets[:budget])
            block = trim_project_block_to_budget(block, budget)
        else:
            block = build_project_block_from_candidate(entry, candidate, budget)
        rebuilt_blocks.append(block.strip())
    if not rebuilt_blocks:
        return current_resume
    rebuilt_section = opening.rstrip() + "\n" + "\n\n".join(rebuilt_blocks) + "\n\\resumeSubHeadingListEnd\n"
    return current_resume[: section["start"]] + rebuilt_section + current_resume[section["end"] :]


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
    selected_projects, _ = select_staged_projects_with_ranking(
        job_description,
        resume,
        project_memory,
        allow_project_selection,
    )
    return selected_projects


def select_staged_projects_with_ranking(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    allow_project_selection: bool,
    resume_constraints: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    projects = project_list_from_memory(project_memory)
    if not projects:
        return [], {"selected_projects": [], "omitted_projects": []}
    jd_profile = jd_requirements_for_prompt(job_description)
    role_profile = classify_role_family(job_description)
    constraints = default_resume_constraints(resume_constraints)
    if not allow_project_selection:
        selected = projects[:MAX_STAGED_PROJECTS]
        ranking = rank_projects_for_resume(
            [compact_project_for_prompt(project) for project in selected],
            jd_profile,
            role_profile,
            constraints,
        )
        return selected, ranking

    ranking = rank_projects_for_resume(
        [compact_project_for_prompt(project) for project in projects],
        jd_profile,
        role_profile,
        constraints,
    )
    selected = projects_from_ranking(projects, ranking)
    return (selected or projects[:PREFERRED_RESUME_PROJECTS]), ranking


def compact_resume_snippet_for_bullet_writer(resume: str, section_type: str, source_name: str) -> dict[str, Any]:
    section_name = "Project-section" if section_type == "project" else "Experience-section"
    snippet = resume_block_for_prompt(resume, section_name, source_name)
    if snippet and snippet.get("latex"):
        latex = preserve_resume_snippet_lines(snippet.get("latex"), source_name, max_lines=18)
        return {
            **snippet,
            "latex": latex,
        }
    section = find_latex_section(resume, "projects" if section_type == "project" else "experience")
    latex = preserve_resume_snippet_lines(section.get("text") if section else resume, source_name, max_lines=22)
    return {
        "scope": "resume_excerpt",
        "section_name": section_name,
        "block_hint": source_name,
        "latex": latex,
    }


def compact_bullet_writer_value(value: Any, max_string_chars: int = 450, max_list_items: int = 5) -> Any:
    return compact_value_for_prompt(value, max_string_chars=max_string_chars, max_list_items=max_list_items)


def preserve_resume_snippet_lines(text: Any, source_name: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    source_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9+#.-]{3,}", source_name or "")]
    scored = []
    for index, line in enumerate(lines):
        lower = line.lower()
        score = 0
        if any(term in lower for term in source_terms):
            score += 8
        if "\\resumeprojectheading" in lower or "\\section" in lower:
            score += 6
        if "\\resumeitem" in lower or lower.strip().startswith(r"\item"):
            score += 4
        if any(word in lower for word in ["implemented", "automated", "debugged", "used", "built", "validated"]):
            score += 2
        scored.append((score, index, line))
    kept_indexes = sorted(index for score, index, _ in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_lines])
    return "\n".join(lines[index] for index in kept_indexes)


def compact_jd_terms(job_description: str) -> list[str]:
    requirements = jd_requirements_for_prompt(job_description)
    values = []
    for key in ["must_have_skills", "responsibilities", "tools_platforms", "soft_skills", "repeated_ats_keywords", "evidence_types_to_emphasize"]:
        for item in requirements.get(key, []) if isinstance(requirements.get(key), list) else []:
            append_unique(values, item, 40)
    return values[:20]


def claim_relevance_score(text: Any, jd_terms: list[str], evidence_sources: list[str], confidence: str = "medium") -> int:
    lowered = str(text or "").lower()
    score = 0
    for term in jd_terms:
        term_text = str(term or "").lower()
        if term_text and term_text in lowered:
            score += 4
    if evidence_sources:
        score += 8
    if confidence == "high":
        score += 5
    elif confidence == "medium":
        score += 2
    if any(word in lowered for word in ["metric", "result", "reduced", "improved", "validated", "debugged", "automated"]):
        score += 3
    return score


def shortest_evidence_sources(values: list[Any], limit: int = 4) -> list[str]:
    sources = []
    for value in values:
        text = short_signal(value, 180)
        if not text:
            continue
        append_unique(sources, text, limit * 3)
    sources.sort(key=len)
    return sources[:limit]


def listish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if value is None or value == "":
        return []
    return [value]


def structured_star_facts(source_facts: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped = {field: [] for field in ["situation", "task", "action", "result"]}
    for fact in project_star_facts(source_facts if isinstance(source_facts, dict) else {}):
        field = normalize_star_field(str(fact.get("field_type") or "result"))
        if field not in grouped:
            field = "result"
        text = fact.get("normalized_fact") or fact.get("raw_answer") or fact.get("value") or fact
        grouped[field].append(
            {
                "fact": short_signal(text, 320),
                "evidenceSources": ["user-confirmed STAR facts"],
                "confidence": fact.get("confidence") or "high",
            }
        )
    return {key: values[:5] for key, values in grouped.items()}


def build_resume_relevant_claims(
    source_name: str,
    source_facts: dict[str, Any],
    evidence_card: dict[str, Any],
    allowed_claims: list[str],
    forbidden_claims: list[str],
    jd_terms: list[str],
) -> list[dict[str, Any]]:
    base_sources = shortest_evidence_sources(
        evidence_card.get("source_refs", [])
        + evidence_card.get("artifacts", [])
        + [source_name, "project_memory.json"],
        5,
    )
    forbidden_text = json.dumps(forbidden_claims, ensure_ascii=False).lower()
    candidates = []
    for claim in allowed_claims:
        candidates.append((claim, base_sources, "high"))
    for key in ["workflows", "confirmed_features", "recent_changes", "tech_stack"]:
        values = listish(source_facts.get(key, []) if isinstance(source_facts, dict) else [])
        for value in values:
            candidates.append((value, ["project_memory.json"] + base_sources[:2], "medium"))
    for key in ["methods", "features", "business_or_user_value", "testing_signals", "debugging_signals", "automation_signals"]:
        for value in evidence_card.get(key, []) if isinstance(evidence_card.get(key), list) else []:
            candidates.append((value, base_sources, "medium"))

    claims = []
    seen = set()
    for raw_claim, sources, confidence in candidates:
        claim = short_signal(raw_claim, 280)
        if not claim:
            continue
        normalized = claim.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        safe = not any(token and token in forbidden_text for token in re.findall(r"[A-Za-z0-9+#.-]{3,}", normalized))
        claim_sources = shortest_evidence_sources(sources, 4)
        claims.append(
            {
                "claim": claim,
                "evidenceSources": claim_sources,
                "confidence": confidence,
                "safeForResume": bool(safe and claim_sources),
                "relevanceScore": claim_relevance_score(claim, jd_terms, claim_sources, confidence),
            }
        )
    claims.sort(key=lambda item: (-int(item.get("relevanceScore", 0)), item["claim"].lower()))
    return claims


def build_metric_candidates(
    source_facts: dict[str, Any],
    evidence_card: dict[str, Any],
    jd_terms: list[str],
) -> list[dict[str, Any]]:
    candidates = []
    for value in evidence_card.get("data_or_scale", []) if isinstance(evidence_card.get("data_or_scale"), list) else []:
        sources = shortest_evidence_sources(evidence_card.get("source_refs", []) + ["project_memory.json"], 3)
        candidates.append(
            {
                "metric": short_signal(value, 220),
                "type": "verified",
                "evidenceSources": sources,
                "safeForResume": bool(sources),
                "relevanceScore": claim_relevance_score(value, jd_terms, sources, "high"),
            }
        )
    for fact in project_star_facts(source_facts if isinstance(source_facts, dict) else {}):
        if normalize_star_field(str(fact.get("field_type") or "")) != "result":
            continue
        text = fact.get("normalized_fact") or fact.get("raw_answer") or ""
        if not text:
            continue
        candidates.append(
            {
                "metric": short_signal(text, 220),
                "type": "user-confirmed",
                "evidenceSources": ["user-confirmed STAR facts"],
                "safeForResume": True,
                "relevanceScore": claim_relevance_score(text, jd_terms, ["user-confirmed STAR facts"], "high"),
            }
        )
    candidates.sort(key=lambda item: (-int(item.get("relevanceScore", 0)), item["metric"].lower()))
    return candidates[:8]


def build_current_project_compact_facts(
    source_name: str,
    source_facts: dict[str, Any],
    evidence_card: dict[str, Any],
    allowed_claims: list[str],
    forbidden_claims: list[str],
    job_description: str,
) -> dict[str, Any]:
    jd_terms = compact_jd_terms(job_description)
    identity = source_facts.get("identity") if isinstance(source_facts.get("identity"), dict) else {}
    summary_parts = [
        identity.get("positioning"),
        identity.get("core_problem"),
        identity.get("core_value"),
        source_facts.get("project_name") if isinstance(source_facts, dict) else "",
    ]
    claims = build_resume_relevant_claims(source_name, source_facts, evidence_card, allowed_claims, forbidden_claims, jd_terms)
    metrics = build_metric_candidates(source_facts, evidence_card, jd_terms)
    return {
        "projectName": source_name,
        "projectSummary": " | ".join(short_signal(part, 180) for part in summary_parts if str(part or "").strip())[:700],
        "jdRelevance": jd_terms[:12],
        "technicalStack": shortest_evidence_sources(listish(source_facts.get("tech_stack", []) if isinstance(source_facts, dict) else []) + listish(evidence_card.get("technologies", [])), 12),
        "keyModules": shortest_evidence_sources(listish(evidence_card.get("artifacts", [])) + listish(evidence_card.get("features", [])), 12),
        "userContributionSignals": shortest_evidence_sources(listish(evidence_card.get("methods", [])) + listish(evidence_card.get("automation_signals", [])) + listish(evidence_card.get("debugging_signals", [])), 12),
        "resumeRelevantClaims": claims[:14],
        "metricCandidates": metrics,
        "starFacts": structured_star_facts(source_facts),
        "riskFlags": shortest_evidence_sources(evidence_card.get("forbidden_claims", []) + forbidden_claims, 8),
    }


def build_compact_bullet_writer_input(
    section_type: str,
    source_name: str,
    job_description: str,
    resume: str,
    source_facts: dict[str, Any],
    evidence_card: dict[str, Any],
    role_profile: dict[str, Any],
    role_lens: dict[str, Any],
    existing_bullets: list[str],
    allowed_claims: list[str],
    forbidden_claims: list[str],
    language: str,
    extra_rules: str = "",
) -> dict[str, Any]:
    current_project_facts = build_current_project_compact_facts(
        source_name,
        source_facts,
        evidence_card,
        allowed_claims,
        forbidden_claims,
        job_description,
    )
    return {
        "section_type": section_type,
        "source_name": source_name,
        "compact_jd_requirements": compact_bullet_writer_value(jd_requirements_for_prompt(job_description), 350, 8),
        "role_profile": compact_bullet_writer_value(role_profile, 300, 5),
        "current_project_compact_facts": current_project_facts,
        "current_project_evidence_summary": {
            "evidenceSources": shortest_evidence_sources(listish(evidence_card.get("source_refs", [])) + listish(evidence_card.get("artifacts", [])), 10),
            "technologies": current_project_facts["technicalStack"],
            "keyModules": current_project_facts["keyModules"],
            "safeClaims": [claim for claim in current_project_facts["resumeRelevantClaims"] if claim.get("safeForResume")][:10],
            "metricCandidates": current_project_facts["metricCandidates"],
            "riskFlags": current_project_facts["riskFlags"],
        },
        "role_lens_priorities": compact_bullet_writer_value(role_lens, 300, 5),
        "relevant_star_facts": current_project_facts["starFacts"],
        "existing_resume_snippet": compact_resume_snippet_for_bullet_writer(resume, section_type, source_name),
        "existing_bullets": compact_bullet_writer_value(existing_bullets, 280, 5),
        "bullet_constraints": {
            "language": language,
            "output_language_instruction": output_language_instruction(language),
            "must_return_json": True,
            "required_keys": ["section_type", "source_name", "job_alignment", "star_analysis", "react_analysis", "final_bullets", "skills_to_emphasize", "risks"],
            "star_required": True,
            "do_not_invent": ["metrics", "tools", "files", "commits", "dates", "ownership", "deployment", "users", "business impact"],
            "style": "concise ATS-friendly bullets with concrete method, workflow value, and evidence-grounded result",
            "extra_rules": truncate_text(extra_rules, provider_safe_text_limit(1200, 500)),
        },
    }


def trim_list_field(container: dict[str, Any], key: str, limit: int) -> None:
    if isinstance(container.get(key), list):
        container[key] = container[key][:limit]


def trim_compact_jd_requirements(value: Any, limit: int) -> Any:
    if isinstance(value, list):
        return value[:limit]
    if not isinstance(value, dict):
        return value
    trimmed = json.loads(json.dumps(value, ensure_ascii=False))
    for key, nested in list(trimmed.items()):
        if isinstance(nested, list):
            trimmed[key] = nested[:limit]
    return trimmed


def apply_bullet_writer_retry_mode(payload: dict[str, Any], retry_mode: str) -> dict[str, Any]:
    mode = str(retry_mode or "normal").lower()
    if mode not in {"retry", "emergency"}:
        return payload
    reduced = json.loads(json.dumps(payload, ensure_ascii=False))
    facts = reduced.get("current_project_compact_facts", {})
    evidence_summary = reduced.get("current_project_evidence_summary", {})
    constraints = reduced.get("bullet_constraints", {})
    if isinstance(facts, dict):
        facts["resumeRelevantClaims"] = sorted(
            facts.get("resumeRelevantClaims", []),
            key=lambda item: (
                -int(item.get("relevanceScore", 0)) if isinstance(item, dict) else 0,
                not bool(item.get("safeForResume")) if isinstance(item, dict) else True,
                str(item.get("claim", "")) if isinstance(item, dict) else str(item),
            ),
        )
        facts["metricCandidates"] = sorted(
            facts.get("metricCandidates", []),
            key=lambda item: (
                0 if isinstance(item, dict) and item.get("type") == "user-confirmed" else 1,
                -int(item.get("relevanceScore", 0)) if isinstance(item, dict) else 0,
                str(item.get("metric", "")) if isinstance(item, dict) else str(item),
            ),
        )
        trim_list_field(facts, "resumeRelevantClaims", 5 if mode == "retry" else 3)
        trim_list_field(facts, "metricCandidates", 4 if mode == "retry" else 3)
        trim_list_field(facts, "keyModules", 6 if mode == "retry" else 3)
        trim_list_field(facts, "userContributionSignals", 6 if mode == "retry" else 3)
        trim_list_field(facts, "riskFlags", 5 if mode == "retry" else 4)
        facts["technicalStack"] = facts.get("technicalStack", [])[:8 if mode == "retry" else 5]
        facts["jdRelevance"] = facts.get("jdRelevance", [])[:8 if mode == "retry" else 6]
    if isinstance(evidence_summary, dict):
        trim_list_field(evidence_summary, "evidenceSources", 5 if mode == "retry" else 3)
        trim_list_field(evidence_summary, "safeClaims", 5 if mode == "retry" else 3)
        trim_list_field(evidence_summary, "metricCandidates", 4 if mode == "retry" else 3)
        trim_list_field(evidence_summary, "riskFlags", 5 if mode == "retry" else 4)
    reduced["compact_jd_requirements"] = trim_compact_jd_requirements(
        reduced.get("compact_jd_requirements"),
        8 if mode == "retry" else 6,
    )
    if isinstance(reduced.get("existing_bullets"), list):
        reduced["existing_bullets"] = reduced["existing_bullets"][:3 if mode == "retry" else 2]
    snippet = reduced.get("existing_resume_snippet")
    if isinstance(snippet, dict) and isinstance(snippet.get("latex"), str):
        lines = [line for line in snippet["latex"].splitlines() if line.strip()]
        snippet["latex"] = "\n".join(lines[:8 if mode == "retry" else 4])
    if isinstance(constraints, dict):
        constraints["retry_mode"] = mode
        constraints["max_final_bullets"] = 5 if mode == "retry" else 4
        constraints["extra_rules"] = short_signal(constraints.get("extra_rules"), 300 if mode == "retry" else 160)
        constraints["retry_rules"] = [
            "Use only the current project summary, top JD requirements, high-relevance claims, evidence sources, user-confirmed metrics, and risk flags.",
            "Return enough supported bullets to satisfy the project ranking budget when present; otherwise return 3-5 bullets in retry mode or 2-4 bullets in emergency mode.",
            "Do not invent numbers or drop user-confirmed facts.",
        ]
    return reduced


def build_compact_project_input(**kwargs: Any) -> dict[str, Any]:
    return build_compact_bullet_writer_input(section_type="project", **kwargs)


def build_compact_experience_input(**kwargs: Any) -> dict[str, Any]:
    return build_compact_bullet_writer_input(section_type="experience", **kwargs)


def build_compact_final_merge_input(payload: dict[str, Any], max_chars: int = PROXY_SAFE_MAX_INPUT_CHARS) -> dict[str, Any]:
    retry_payload = payload if "target_resume_block" in payload else merge_retry_payload_for_prompt(payload)
    return reduce_final_merge_payload_for_limit(retry_payload, max_chars)


def build_compact_bullet_writer_prompt(payload: dict[str, Any]) -> str:
    return f"""
You are the compact WorkAgent resume bullet writer.

Use only the compact payload below. Do not request or infer raw repo context, full code, unrelated projects, or full chat history.

Return ONLY valid JSON with exactly these keys:
  "section_type": "project" | "experience",
  "source_name": string,
  "job_alignment": string,
  "star_analysis": array of objects with keys "candidate_fact", "situation", "task", "action", "result", "missing_star_fields", "evidence_source",
  "react_analysis": array of objects with keys "candidate_fact", "why_writable", "why_it_belongs", "business_capability", "technical_capability", "risk_avoided",
  "final_bullets": array of objects with keys "bullet", "evidence", "confidence",
  "skills_to_emphasize": array of strings,
  "risks": array of strings

Rules:
- Write only for payload.source_name and payload.section_type.
- Do not use unrelated projects.
- For project bullets, follow this prioritization rule: {PROJECT_PRIORITY_INSTRUCTION}
- Do not invent metrics, technologies, files, commits, dates, ownership, deployment, users, or business impact.
- Use STAR analysis before final bullets.
- If a metric/result is unsupported, list it in missing_star_fields and write a conservative qualitative result.
- Final bullets must combine a concrete method, a substantive workflow/business capability, and an evidence-grounded result/value.
- Avoid stack-only, CRUD-only, and shallow UI-only bullets.
- When payload.bullet_constraints.max_final_bullets is present, keep final_bullets at or below that count.

Compact bullet writer payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def reduce_compact_bullet_payload_for_limit(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    reduced = json.loads(json.dumps(payload, ensure_ascii=False))

    def prompt_size() -> int:
        return len(build_compact_bullet_writer_prompt(reduced))

    facts = reduced.get("current_project_compact_facts", {})
    evidence_summary = reduced.get("current_project_evidence_summary", {})
    constraints = reduced.get("bullet_constraints", {})

    if isinstance(facts, dict):
        facts["resumeRelevantClaims"] = sorted(
            facts.get("resumeRelevantClaims", []),
            key=lambda item: (-int(item.get("relevanceScore", 0)), not bool(item.get("safeForResume")), item.get("claim", "")),
        )
        facts["metricCandidates"] = sorted(
            facts.get("metricCandidates", []),
            key=lambda item: (-int(item.get("relevanceScore", 0)), not bool(item.get("safeForResume")), item.get("metric", "")),
        )

    reduction_steps = [
        ("existing_bullets", 4),
        ("existing_resume_snippet.latex_lines", 14),
        ("current_project_evidence_summary.safeClaims", 8),
        ("current_project_compact_facts.resumeRelevantClaims", 10),
        ("current_project_compact_facts.metricCandidates", 5),
        ("current_project_compact_facts.keyModules", 8),
        ("current_project_compact_facts.userContributionSignals", 8),
        ("current_project_evidence_summary.evidenceSources", 7),
        ("current_project_compact_facts.resumeRelevantClaims", 7),
        ("current_project_compact_facts.metricCandidates", 3),
        ("current_project_compact_facts.riskFlags", 5),
        ("current_project_compact_facts.resumeRelevantClaims", 5),
    ]

    for path, limit in reduction_steps:
        if prompt_size() <= max_chars:
            break
        if path == "existing_resume_snippet.latex_lines":
            snippet = reduced.get("existing_resume_snippet", {})
            if isinstance(snippet, dict) and isinstance(snippet.get("latex"), str):
                lines = [line for line in snippet["latex"].splitlines() if line.strip()]
                snippet["latex"] = "\n".join(lines[:limit])
            continue
        parent = reduced
        parts = path.split(".")
        for part in parts[:-1]:
            parent = parent.get(part, {}) if isinstance(parent, dict) else {}
        key = parts[-1]
        if isinstance(parent, dict) and isinstance(parent.get(key), list):
            parent[key] = parent[key][:limit]

    if prompt_size() > max_chars and isinstance(constraints, dict):
        constraints["extra_rules"] = short_signal(constraints.get("extra_rules"), 180)

    if prompt_size() > max_chars and isinstance(facts, dict):
        facts["resumeRelevantClaims"] = facts.get("resumeRelevantClaims", [])[:3]
        facts["metricCandidates"] = facts.get("metricCandidates", [])[:2]
        facts["keyModules"] = facts.get("keyModules", [])[:5]
        facts["userContributionSignals"] = facts.get("userContributionSignals", [])[:5]
        if isinstance(evidence_summary, dict):
            evidence_summary["safeClaims"] = evidence_summary.get("safeClaims", [])[:3]
            evidence_summary["evidenceSources"] = evidence_summary.get("evidenceSources", [])[:5]

    return reduced


def bullet_writer_proxy_limit() -> int:
    decision = routing_decision(normalize_provider(agent.current_provider), 0)
    return int(decision.get("directMaxInputChars") or PROXY_SAFE_MAX_INPUT_CHARS)


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
{truncate_text(resume, provider_safe_text_limit(22000, 8000))}

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
- For Projects-section candidates, follow the ranking context and bullet budget from Extra rules; keep lower-ranked
  projects concise and reserve the most detailed treatment for the highest-ranked project.
- evidence_card.inferred_results may be used as conservative qualitative Result evidence from local diff/code
  analysis, but never convert it into verified QPS, P99, latency, cost, accuracy, or percentage claims unless
  data_or_scale or user-confirmed star_facts explicitly supports the number.
- Treat live user guidance from the progress modal as user-provided STAR evidence when present.
"""
    compact_payload = None

    def compact_bullet_prompt(
        max_chars: int = PROXY_SAFE_MAX_INPUT_CHARS,
        retry_mode: str = "normal",
        **_: Any,
    ) -> dict[str, Any]:
        common = dict(
            source_name=source_name,
            job_description=job_description,
            resume=resume,
            source_facts=source_facts,
            evidence_card=evidence_card,
            role_profile=role_profile,
            role_lens=role_lens,
            existing_bullets=existing_bullets,
            allowed_claims=allowed_claims,
            forbidden_claims=forbidden_claims,
            language=language,
            extra_rules=extra_rules,
        )
        if section_type == "project":
            payload = build_compact_project_input(**common)
        elif section_type == "experience":
            payload = build_compact_experience_input(**common)
        else:
            payload = build_compact_bullet_writer_input(section_type=section_type, **common)
        payload = apply_bullet_writer_retry_mode(payload, retry_mode)
        compact_prompt = build_compact_bullet_writer_prompt(payload)
        if len(compact_prompt) > max_chars:
            payload = reduce_compact_bullet_payload_for_limit(payload, max_chars=max_chars)
            compact_prompt = build_compact_bullet_writer_prompt(payload)
        return {"prompt": compact_prompt, "payload": payload}

    response = safe_model_call(
        caller="run_resume_bullet_writer_tool",
        prompt=prompt,
        task_type="resume_bullet_writer",
        compact_builder=compact_bullet_prompt,
    )
    payload = extract_json_object(response)
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
    project_ranking: dict[str, Any] | None = None,
    project_rank_entry: dict[str, Any] | None = None,
    resume_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_facts = compact_project_for_prompt(project)
    project_id = str(project.get("project_id") or "")
    source_name = str(project.get("project_name") or project.get("name") or project.get("project_id") or "")
    constraints = default_resume_constraints(resume_constraints)
    rank_entry = project_rank_entry if isinstance(project_rank_entry, dict) else project_ranking_entry_for_project(project_ranking, project)
    bullet_budget = int(rank_entry.get("bullet_budget") or 0) if isinstance(rank_entry, dict) else 0
    ranking_context = {
        "project_ranking": compact_value_for_prompt(project_ranking or {}, 450, 4),
        "this_project_rank": compact_value_for_prompt(rank_entry or {}, 450, 6),
        "bullet_budget": bullet_budget,
        "one_page_constraints": constraints,
    }
    source_hash = stable_hash(
        {
            "job_description": job_description,
            "project": source_facts,
            "evidence": evidence,
            "language": language,
            "progress_guidance": progress_guidance,
            "ranking_context": ranking_context,
        }
    )
    task_id = current_agent_task_id.get("")
    cached_candidate = get_completed_resume_candidate_checkpoint(task_id, project_id, source_name, source_hash)
    if cached_candidate:
        return cached_candidate
    try:
        payload = run_resume_bullet_writer_tool(
            section_type="project",
            source_name=source_name,
            job_description=job_description,
            resume=resume,
            source_facts=source_facts,
            evidence=evidence,
            existing_bullets=[],
            language=language,
            extra_rules=(
                "Project Memory is the primary source of truth. Chroma evidence is supporting proof only. "
                + PROJECT_PRIORITY_INSTRUCTION
                + " Generate exactly this project's bullet_budget final_bullets when there is enough supported evidence; "
                + "do not stop at 3-4 bullets for the top-ranked project by default. Use fewer bullets only when the "
                + "evidence or one-page constraints truly require it. "
                + "For the highest-ranked project, cover multiple dimensions such as core implementation, "
                + "tools/methods, data/storage/workflow logic, testing/debugging/automation/documentation, "
                + "and target-role relevance. For lower-ranked projects, keep only the strongest job-relevant evidence. "
                + "The first bullet must explain what the project is and what workflow or problem it addresses. "
                + f"Ranking context: {json.dumps(ranking_context, ensure_ascii=False)} "
                + "Return fit, keep_or_replace, and fit_reason if possible."
                + progress_guidance
            ),
        )
    except HTTPException as error:
        save_resume_candidate_checkpoint(
            task_id,
            project_id,
            source_name,
            source_hash,
            "failed",
            error=http_exception_detail_text(error),
        )
        raise
    payload["project_id"] = payload.get("project_id") or project.get("project_id") or ""
    payload["project_name"] = payload.get("project_name") or project.get("project_name") or project.get("name") or ""
    payload["fit"] = payload.get("fit") if payload.get("fit") in {"high", "medium", "low"} else "medium"
    payload["keep_or_replace"] = payload.get("keep_or_replace") or "update"
    payload["fit_reason"] = payload.get("fit_reason") or payload.get("job_alignment", "")
    payload["recommended_bullets"] = payload.get("final_bullets", [])
    if rank_entry:
        payload["project_rank"] = rank_entry.get("rank")
        payload["bullet_budget"] = bullet_budget
        payload["focus_areas"] = rank_entry.get("focus_areas", [])
        payload["project_ranking_context"] = compact_value_for_prompt(rank_entry, 500, 8)
    save_resume_candidate_checkpoint(task_id, project_id, source_name, source_hash, "done", candidate_json=payload)
    return payload


SKILL_CATEGORY_KEYWORDS = {
    "Languages": [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "SQL",
        "C",
        "C++",
        "C#",
        "HTML",
        "CSS",
        "PowerShell",
        "Shell scripting",
        "Shell",
        "Bash",
    ],
    "Backend / API": ["FastAPI", "Flask", "Django", "Node.js", "Express.js", "REST", "API", "OpenAI API", "GitHub API"],
    "Frontend / UI": ["React", "React.js", "Vite", "Electron", "HTML", "CSS", "JavaScript", "TypeScript"],
    "Mobile / Game": ["Android", "Android Studio", "Gradle", "Gradle Kotlin DSL", "Espresso", "Android instrumentation testing", "Unity", "Firebase Auth", "Firebase Messaging"],
    "Database / Storage": [
        "SQLite",
        "better-sqlite3",
        "PostgreSQL",
        "MySQL",
        "MongoDB",
        "Firebase Firestore",
        "Firestore",
        "Chroma",
        "vector store",
        "Redis",
    ],
    "Cloud / DevOps / Infrastructure": [
        "Docker",
        "Kubernetes",
        "Terraform",
        "AWS",
        "Azure",
        "Linux",
        "Unix",
        "CI/CD",
        "GitHub Actions",
        "Jenkins",
        "Microsoft 365",
    ],
    "AI / Automation": [
        "RAG",
        "embedding",
        "embeddings",
        "LLM",
        "OpenAI",
        "prompt",
        "chunking",
        "Map-Reduce",
        "automation",
        "batch",
        "cache",
    ],
    "Testing / Quality": ["pytest", "unittest", "Playwright", "Cypress", "Selenium", "Espresso", "Android instrumentation testing", "Gradle Kotlin DSL", "data validation", "debugging", "validation", "QA", "testing"],
    "Data / Reporting": ["Excel", "Power BI", "Tableau", "reporting", "data analysis", "data validation"],
    "Tools / Workflow": ["Git/GitHub", "Git", "GitHub", "Android Studio", "Gradle", "Maven", "PowerShell", "Shell scripting", "IntelliJ", "technical documentation", "requirements", "documentation"],
    "Collaboration / Documentation": ["documentation", "communication", "collaboration", "stakeholder", "requirements", "reporting"],
}


SKILL_ALIASES = {
    "react.js": "React",
    "reactjs": "React",
    "sqlite3": "SQLite",
    "better-sqlite3": "SQLite",
    "firestore": "Firebase Firestore",
    "firebase firestore": "Firebase Firestore",
    "firebase authentication": "Firebase Auth",
    "firebase cloud messaging": "Firebase Messaging",
    "github integration": "GitHub API",
    "github api": "GitHub API",
    "github": "Git/GitHub",
    "git": "Git/GitHub",
    "git/github": "Git/GitHub",
    "openai api": "OpenAI API",
    "embeddings": "embedding",
    "maps-reduce": "Map-Reduce",
    "map reduce": "Map-Reduce",
    "shell": "Shell scripting",
    "bash": "Shell scripting",
    "powershell fundamentals": "PowerShell",
    "azure fundamentals": "Azure",
    "microsoft 365 familiarity": "Microsoft 365",
    "docker basics": "Docker",
    "kubernetes concepts": "Kubernetes",
    "ci/cd concepts": "CI/CD",
    "html": "HTML/CSS",
    "css": "HTML/CSS",
    "rest": "REST APIs",
    "rest api": "REST APIs",
    "rest apis": "REST APIs",
    "android instrumentation tests": "Android instrumentation testing",
    "gradle kotlin dsl": "Gradle Kotlin DSL",
    "java 11": "Java",
    "sqlite/database files": "SQLite",
    "sqlite database": "SQLite",
    "unity 2022.3.62f3": "Unity",
}


LOW_VALUE_SKILL_SECTION_NAMES = {
    "api",
    "automation",
    "batch",
    "batchfile",
    "cache",
    "camerax",
    "chunking",
    "claude",
    "claude / anthropic",
    "communication",
    "collaboration",
    "database queries",
    "deepseek",
    "documentation",
    "embedding",
    "firebase analytics",
    "gemini",
    "codex",
    "androidx",
    "openai",
    "prompt",
    "qa",
    "reporting",
    "requirements",
    "stakeholder",
    "testing",
    "validation",
}

BAD_SKILL_TEXT_MARKERS = [
    "...",
    "[",
    "]",
    "truncated",
    "more items",
    "求职",
    "申请",
    "使用",
    "开发",
]

JD_MATCH_REQUIRED_SECTION_SKILLS = {
    "android",
    "androidx",
    "camerax",
    "gdscript",
    "godot engine",
    "hlsl",
    "ink",
    "unity",
}


def canonical_skill_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    alias = SKILL_ALIASES.get(normalized.lower())
    return alias or normalized


def clean_resume_skill_name(value: Any) -> str:
    name = canonical_skill_name(value)
    name = re.sub(r"\\[A-Za-z]+\{?|[{}]", "", name)
    name = re.sub(r"\s+", " ", name).strip(" ,.;:")
    if not name:
        return ""
    lower = name.lower()
    if lower in LOW_VALUE_SKILL_SECTION_NAMES:
        return ""
    if any(marker in lower for marker in BAD_SKILL_TEXT_MARKERS):
        return ""
    if re.search(r"[\u4e00-\u9fff]", name):
        return ""
    if re.search(r"\s/\s", name):
        return ""
    if re.search(r"[,;，；。！？]", name):
        return ""
    if len(name) > 44:
        return ""
    word_parts = re.findall(r"[A-Za-z0-9+#./-]+", name)
    if len(word_parts) > 4:
        return ""
    return name


def base_skill_category(skill: str) -> str:
    lower = skill.lower()
    for category, keywords in SKILL_CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower == lower:
                return category
            if len(keyword_lower) > 3 and re.search(
                rf"(?<![A-Za-z0-9+#.]){re.escape(keyword_lower)}(?![A-Za-z0-9+#.])",
                lower,
            ):
                return category
    return "Tools / Methods"


ROLE_TECHNICAL_SKILL_CATEGORIES = {
    "software_engineering": ["Languages", "Backend, Web & Databases", "Testing, Build & Debugging", "Tools & Workflow"],
    "it_analyst": [
        "Languages & Scripting",
        "Application & Automation",
        "Tools & Workflow",
        "Troubleshooting & Documentation",
        "Cloud / Microsoft Fundamentals",
    ],
    "infrastructure_devops": [
        "Languages & Scripting",
        "Cloud & Infrastructure",
        "Containers & Orchestration",
        "CI/CD & Build",
        "Monitoring & Debugging",
        "Tools & Workflow",
    ],
    "data_analyst": ["Languages & Querying", "Databases", "Data Analysis", "Reporting & Visualization", "Tools & Workflow"],
    "product_business_analyst": ["Analysis & Documentation", "Technical Tools", "Data & Reporting", "Collaboration & Workflow"],
}


CAUTIOUS_SKILL_WORDING = {
    "AWS": "AWS fundamentals",
    "Azure": "Azure fundamentals",
    "Microsoft 365": "Microsoft 365 familiarity",
    "Docker": "Docker basics",
    "Kubernetes": "Kubernetes concepts",
    "Terraform": "Terraform concepts",
    "Jenkins": "Jenkins concepts",
    "CI/CD": "CI/CD concepts",
    "PowerShell": "PowerShell fundamentals",
    "Power BI": "Power BI familiarity",
}


WEAK_SKILL_SOURCE_LABELS = {"user_memory", "coursework", "prior_resume_versions", "jd_keywords"}


def skill_category(skill: str, role_family: str = "") -> str:
    base = base_skill_category(skill)
    lower = skill.lower()
    role = role_family or "software_engineering"
    if role == "it_analyst":
        if base == "Languages" or lower in {"powershell", "shell scripting", "bash"}:
            return "Languages & Scripting"
        if base in {"Backend / API", "Frontend / UI", "AI / Automation", "Mobile / Game"}:
            return "Application & Automation"
        if lower in {"azure", "microsoft 365", "aws", "docker", "kubernetes", "ci/cd"}:
            return "Cloud / Microsoft Fundamentals"
        if base in {"Testing / Quality", "Collaboration / Documentation"} or any(term in lower for term in ["troubleshoot", "documentation", "requirements"]):
            return "Troubleshooting & Documentation"
        return "Tools & Workflow"
    if role == "infrastructure_devops":
        if base == "Languages" or lower in {"powershell", "shell scripting", "bash"}:
            return "Languages & Scripting"
        if lower in {"aws", "azure", "linux", "unix", "terraform"}:
            return "Cloud & Infrastructure"
        if lower in {"docker", "kubernetes"}:
            return "Containers & Orchestration"
        if lower in {"ci/cd", "github actions", "jenkins", "gradle", "maven"}:
            return "CI/CD & Build"
        if base == "Testing / Quality" or any(term in lower for term in ["logging", "debug", "validation"]):
            return "Monitoring & Debugging"
        return "Tools & Workflow"
    if role == "data_analyst":
        if base == "Languages" or lower == "sql":
            return "Languages & Querying"
        if base == "Database / Storage":
            return "Databases"
        if lower in {"excel", "power bi", "tableau", "reporting"}:
            return "Reporting & Visualization"
        if base in {"Data / Reporting", "AI / Automation"} or "data" in lower:
            return "Data Analysis"
        return "Tools & Workflow"
    if role == "product_business_analyst":
        if base in {"Collaboration / Documentation"} or lower in {"requirements", "documentation", "communication", "stakeholder"}:
            return "Analysis & Documentation"
        if lower in {"sql", "excel", "power bi", "tableau", "reporting"} or base in {"Database / Storage", "Data / Reporting"}:
            return "Data & Reporting"
        if lower in {"git/github", "git", "github"} or base in {"Tools / Workflow"}:
            return "Collaboration & Workflow"
        return "Technical Tools"
    if base == "Languages":
        if lower in {"powershell", "shell scripting", "bash"}:
            return "Tools & Workflow"
        return "Languages"
    if lower in {"android studio", "intellij", "git/github", "git", "github", "powershell", "shell scripting", "bash"}:
        return "Tools & Workflow"
    if lower in {"espresso", "gradle", "maven", "gradle kotlin dsl", "android instrumentation testing", "debugging", "data validation"}:
        return "Testing, Build & Debugging"
    if base in {"Backend / API", "Frontend / UI", "Mobile / Game", "Database / Storage"}:
        return "Backend, Web & Databases"
    if base in {"Testing / Quality", "Cloud / DevOps / Infrastructure"} and lower not in {"aws", "azure", "kubernetes", "terraform"}:
        return "Testing, Build & Debugging"
    return "Tools & Workflow"


def skill_relevance(skill: str, jd_terms: list[str]) -> str:
    lower = skill.lower()
    if any(str(term).lower() in lower or lower in str(term).lower() for term in jd_terms):
        return "high"
    if any(word in lower for word in ["api", "automation", "sql", "python", "react", "testing", "documentation"]):
        return "medium"
    return "low"


def skill_in_text(skill: str, text: str) -> bool:
    skill = canonical_skill_name(skill)
    if not skill:
        return False
    lowered = str(text or "").lower()
    if not lowered:
        return False
    aliases = [skill]
    aliases.extend(alias for alias, canonical in SKILL_ALIASES.items() if canonical.lower() == skill.lower())
    for alias in sorted(set(aliases), key=len, reverse=True):
        normalized = alias.lower()
        if normalized == "c":
            pattern = r"(?<![A-Za-z0-9+#.])c(?![A-Za-z0-9+#.])"
        elif re.fullmatch(r"[A-Za-z0-9+#./ -]+", normalized):
            pattern = rf"(?<![A-Za-z0-9+#.]){re.escape(normalized)}(?![A-Za-z0-9+#.])"
        else:
            pattern = re.escape(normalized)
        if re.search(pattern, lowered):
            return True
    return False


def all_known_skill_names() -> list[str]:
    names = []
    for keywords in SKILL_CATEGORY_KEYWORDS.values():
        for keyword in keywords:
            append_unique(names, canonical_skill_name(keyword), 200)
    for canonical in SKILL_ALIASES.values():
        append_unique(names, canonical, 200)
    return sorted(names, key=lambda item: (-len(item), item.lower()))


def is_known_resume_skill_name(value: Any) -> bool:
    name = clean_resume_skill_name(value)
    if not name:
        return False
    known = {
        clean_resume_skill_name(skill).lower()
        for skill in all_known_skill_names()
        if clean_resume_skill_name(skill)
    }
    return name.lower() in known


def extract_skill_names_from_text(text: Any, limit: int = 80) -> list[str]:
    value = str(text or "")
    found = []
    for skill in all_known_skill_names():
        if skill_in_text(skill, value):
            append_unique(found, canonical_skill_name(skill), limit)
    return found


def latex_escape_text(value: Any) -> str:
    text = str(value or "")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def cautious_skill_wording(candidate: dict[str, Any]) -> str:
    skill = canonical_skill_name(candidate.get("skill"))
    confidence = str(candidate.get("confidence") or "").lower()
    sources = {str(source) for source in candidate.get("sources", [])}
    weak_only = bool(sources) and sources.issubset(WEAK_SKILL_SOURCE_LABELS)
    if confidence == "medium" and (weak_only or skill in CAUTIOUS_SKILL_WORDING):
        return CAUTIOUS_SKILL_WORDING.get(skill, skill)
    return skill


def clean_skill_display_name(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in set(CAUTIOUS_SKILL_WORDING.values()):
        name = re.sub(r"\\[A-Za-z]+\{?|[{}]", "", raw)
        name = re.sub(r"\s+", " ", name).strip(" ,.;:")
        lower = name.lower()
        if not name:
            return ""
        if any(marker in lower for marker in BAD_SKILL_TEXT_MARKERS):
            return ""
        if re.search(r"[\u4e00-\u9fff]", name):
            return ""
        if any(char in name for char in [",", ";"]):
            return ""
        if len(name) > 44:
            return ""
        return name
    return clean_resume_skill_name(raw)


def add_candidate_skill(
    skills: dict[str, dict[str, Any]],
    skill: Any,
    evidence_project: str,
    evidence_sources: list[Any],
    jd_terms: list[str],
    confidence: str = "medium",
) -> None:
    name = clean_resume_skill_name(skill)
    if not name:
        return
    sources = shortest_evidence_sources(evidence_sources, 5)
    if not sources:
        return
    key = name.lower()
    current = skills.get(key)
    relevance = skill_relevance(name, jd_terms)
    if current is None:
        skills[key] = {
            "skill": name,
            "category": skill_category(name),
            "evidenceProjects": [evidence_project] if evidence_project else [],
            "evidenceSources": sources,
            "jdRelevance": relevance,
            "confidence": confidence,
            "score": claim_relevance_score(name, jd_terms, sources, confidence),
        }
        return
    if evidence_project and evidence_project not in current["evidenceProjects"]:
        current["evidenceProjects"].append(evidence_project)
    for source in sources:
        append_unique(current["evidenceSources"], source, 6)
    if current["jdRelevance"] != "high" and relevance == "high":
        current["jdRelevance"] = "high"
    if current["confidence"] != "high" and confidence == "high":
        current["confidence"] = "high"
    current["score"] = max(int(current.get("score", 0)), claim_relevance_score(name, jd_terms, current["evidenceSources"], current["confidence"]))


def extract_existing_resume_skills(resume: str) -> list[str]:
    section = find_latex_section(resume, "skills")
    text = section.get("text") if section else resume
    values = []
    for skill in extract_skill_names_from_text(text, 80):
        append_unique(values, clean_resume_skill_name(skill), 80)
    for match in re.findall(r"\\textbf\{([^}]+)\}\s*[:：]\s*([^\n]+)", text):
        for item in re.split(r"[,/|;]", match[1]):
            cleaned = clean_resume_skill_name(item)
            append_unique(values, cleaned, 80)
    return values


def init_project_tech_stack_db() -> None:
    TECH_STACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(TECH_STACK_DB_PATH)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS project_tech_stacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL DEFAULT '',
                repository TEXT NOT NULL DEFAULT '',
                skill TEXT NOT NULL,
                normalized_skill TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                confidence TEXT NOT NULL DEFAULT 'medium',
                updated_at TEXT NOT NULL,
                UNIQUE(project_name, repository, normalized_skill, source)
            )
            """
        )
        connection.commit()


def project_tech_stack_rows_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    project_name = str(context.get("project_name") or context.get("name") or context.get("project_id") or "").strip()
    repository = str(context.get("repository") or context.get("url") or "").strip()
    files = [str(item) for item in listish(context.get("root_files", []))]
    text_parts = [
        context.get("readme", ""),
        json.dumps(context.get("topics", []), ensure_ascii=False),
        json.dumps(context.get("languages", []), ensure_ascii=False),
        json.dumps(context.get("languages_frameworks_detected", []), ensure_ascii=False),
        json.dumps(context.get("contribution_evidence", []), ensure_ascii=False),
        json.dumps(context.get("file_level_summaries", []), ensure_ascii=False),
        json.dumps(context.get("diff_signals", []), ensure_ascii=False),
    ]
    for contribution in listish(context.get("contribution_evidence", [])):
        if not isinstance(contribution, dict):
            continue
        for commit in listish(contribution.get("commits", [])):
            if isinstance(commit, dict):
                files.extend(str(item) for item in listish(commit.get("files", [])))
                for change in listish(commit.get("file_changes", [])):
                    if isinstance(change, dict):
                        files.append(str(change.get("filename") or change.get("file") or ""))
        files.extend(str(item) for item in listish(contribution.get("compare_files", [])))
        for change in listish(contribution.get("compare_file_changes", [])):
            if isinstance(change, dict):
                files.append(str(change.get("filename") or change.get("file") or ""))
    detected = []
    for language in listish(context.get("languages", [])):
        append_unique(detected, canonical_skill_name(language), 80)
    for skill in detect_languages_and_frameworks_from_files(files, "\n".join(str(part) for part in text_parts)):
        append_unique(detected, canonical_skill_name(skill), 80)
    for skill in extract_skill_names_from_text("\n".join(files + [str(part) for part in text_parts]), 80):
        append_unique(detected, skill, 80)
    rows = []
    for skill in detected:
        name = canonical_skill_name(skill)
        if not name:
            continue
        rows.append(
            {
                "project_name": project_name,
                "repository": repository,
                "skill": name,
                "normalized_skill": name,
                "category": base_skill_category(name),
                "source": "repository_analysis",
                "evidence": short_signal(repository or project_name or "repository metadata", 220),
                "confidence": "high" if name in [canonical_skill_name(item) for item in listish(context.get("languages", []))] else "medium",
            }
        )
    return rows


def save_project_tech_stack(context: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(context, dict):
        return []
    rows = project_tech_stack_rows_from_context(context)
    if not rows:
        return []
    init_project_tech_stack_db()
    now = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(TECH_STACK_DB_PATH)) as connection:
        for row in rows:
            connection.execute(
                """
                INSERT OR REPLACE INTO project_tech_stacks
                    (project_name, repository, skill, normalized_skill, category, source, evidence, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["project_name"],
                    row["repository"],
                    row["skill"],
                    row["normalized_skill"],
                    row["category"],
                    row["source"],
                    row["evidence"],
                    row["confidence"],
                    now,
                ),
            )
        connection.commit()
    return rows


def query_all_project_tech_stacks(limit: int = 200) -> list[dict[str, Any]]:
    if not TECH_STACK_DB_PATH.exists():
        return []
    init_project_tech_stack_db()
    with closing(sqlite3.connect(TECH_STACK_DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT project_name, repository, skill, normalized_skill, category, source, evidence, confidence, updated_at
            FROM project_tech_stacks
            ORDER BY updated_at DESC, project_name, normalized_skill
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(confidence or "").lower(), 1)


def merged_confidence(current: str, incoming: str) -> str:
    return incoming if confidence_rank(incoming) > confidence_rank(current) else current


def read_user_memory_for_skills() -> dict[str, Any]:
    try:
        raw = agent.read_memory()
    except Exception:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}
    return payload if isinstance(payload, dict) else {"raw_text": text}


def read_prior_generated_resume_skill_names(limit_files: int = 5) -> list[str]:
    directory = agent.TAILORED_RESUME_OUTPUT_DIR
    if not directory.exists():
        return []
    files = sorted(
        [path for path in directory.glob("*.txt") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:limit_files]
    skills = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for skill in extract_existing_resume_skills(text):
            append_unique(skills, skill, 80)
    return skills


def values_for_skill_keys(value: Any, key_terms: set[str], limit: int = 80, depth: int = 0) -> list[str]:
    if depth > 5:
        return []
    values = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            key_matches = any(term in key_text for term in key_terms)
            if key_matches:
                if isinstance(item, (str, int, float)):
                    append_unique(values, str(item), limit)
                elif isinstance(item, list):
                    for entry in item:
                        append_unique(values, entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False), limit)
                elif isinstance(item, dict):
                    append_unique(values, json.dumps(item, ensure_ascii=False), limit)
            for nested in values_for_skill_keys(item, key_terms, limit, depth + 1):
                append_unique(values, nested, limit)
    elif isinstance(value, list):
        for item in value:
            for nested in values_for_skill_keys(item, key_terms, limit, depth + 1):
                append_unique(values, nested, limit)
    return values


def project_name_for_skill_source(project: dict[str, Any]) -> str:
    return str(
        project.get("project_name")
        or project.get("projectName")
        or project.get("source_name")
        or project.get("name")
        or project.get("project_id")
        or "project evidence"
    )


def github_evidence_from_project_cards(selected_project_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence = []
    for card in selected_project_cards:
        if not isinstance(card, dict):
            continue
        for key in ["github_evidence", "githubEvidence", "evidence", "evidence_card", "evidenceCard"]:
            value = card.get(key)
            if isinstance(value, dict):
                evidence.append(value)
            elif isinstance(value, list):
                evidence.extend(item for item in value if isinstance(item, dict))
    return evidence


def collect_skill_candidates_for_prompt(
    jd_profile: dict,
    selected_project_cards: list[dict],
    current_resume: dict | str,
    user_memory: dict | None = None,
    project_database: dict | list[dict] | None = None,
    github_evidence: list[dict] | None = None,
) -> dict:
    role_profile = jd_profile.get("role_profile") if isinstance(jd_profile.get("role_profile"), dict) else {}
    role_family = str(role_profile.get("role_family") or jd_profile.get("role_family") or "software_engineering")
    jd_text = json.dumps(jd_profile, ensure_ascii=False)
    jd_terms = extract_skill_names_from_text(jd_text, 80)
    for value in values_for_skill_keys(jd_profile, {"skill", "tool", "platform", "language", "framework", "database"}, 80):
        for skill in extract_skill_names_from_text(value, 80):
            append_unique(jd_terms, skill, 80)

    skill_map: dict[str, dict[str, Any]] = {}

    def add_skill(
        skill: Any,
        source_label: str,
        evidence_detail: Any,
        confidence: str = "medium",
        evidence_project: str = "",
    ) -> None:
        name = clean_resume_skill_name(skill)
        if not name:
            return
        key = name.lower()
        detail = short_signal(evidence_detail or source_label, 220)
        entry = skill_map.get(key)
        if entry is None:
            entry = {
                "skill": name,
                "category": skill_category(name, role_family),
                "sources": [],
                "evidenceSources": [],
                "evidenceProjects": [],
                "confidence": confidence if confidence in {"high", "medium", "low"} else "medium",
                "jd_relevance": skill_relevance(name, jd_terms),
                "jdRelevance": skill_relevance(name, jd_terms),
                "safe_to_include": False,
                "safeToInclude": False,
                "score": 0,
            }
            skill_map[key] = entry
        append_unique(entry["sources"], source_label, 10)
        append_unique(entry["evidenceSources"], detail, 8)
        if evidence_project:
            append_unique(entry["evidenceProjects"], evidence_project, 6)
        entry["confidence"] = merged_confidence(str(entry.get("confidence") or "medium"), confidence)
        relevance = skill_relevance(name, jd_terms)
        if entry["jd_relevance"] != "high" and relevance == "high":
            entry["jd_relevance"] = "high"
            entry["jdRelevance"] = "high"
        entry["score"] = max(
            int(entry.get("score", 0)),
            claim_relevance_score(name, jd_terms, entry.get("evidenceSources", []), str(entry.get("confidence") or "medium")),
        )

    for skill in jd_terms:
        add_skill(skill, "jd_keywords", "job description keyword", "low")

    resume_text = json.dumps(current_resume, ensure_ascii=False) if isinstance(current_resume, dict) else str(current_resume or "")
    for skill in extract_existing_resume_skills(resume_text):
        add_skill(skill, "current_resume", "current resume Technical Skills/history", "high", "resume")
    for value in values_for_skill_keys(current_resume, {"course", "coursework", "class", "education"}, 80):
        for skill in extract_skill_names_from_text(value, 40):
            add_skill(skill, "coursework", value, "medium")

    for card in selected_project_cards or []:
        if not isinstance(card, dict):
            continue
        project_name = project_name_for_skill_source(card)
        direct_values = []
        for key in [
            "tech_stack",
            "technicalStack",
            "technologies",
            "tools",
            "languages_frameworks_detected",
            "skills",
            "skills_to_emphasize",
        ]:
            for value in listish(card.get(key, [])):
                append_unique(direct_values, value, 80)
        for value in direct_values:
            for skill in extract_skill_names_from_text(value, 40) or [value]:
                add_skill(skill, "project_evidence", f"{project_name}: {value}", "high", project_name)
        project_text_parts = []
        for key in ["workflows", "confirmed_features", "methods", "features", "recommended_bullets", "final_bullets"]:
            if key in card:
                project_text_parts.append(json.dumps(card.get(key), ensure_ascii=False))
        for skill in extract_skill_names_from_text("\n".join(project_text_parts), 60):
            add_skill(skill, "project_evidence", project_name, "medium", project_name)
        for value in values_for_skill_keys(card, {"course", "coursework", "class"}, 40):
            for skill in extract_skill_names_from_text(value, 30):
                add_skill(skill, "coursework", value, "medium", project_name)

    evidence_items = list(github_evidence or []) + github_evidence_from_project_cards(selected_project_cards or [])
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        project_name = project_name_for_skill_source(item)
        detected = []
        for key in ["languages", "languages_frameworks_detected", "technologies", "resume_relevant_keywords"]:
            for value in listish(item.get(key, [])):
                append_unique(detected, value, 80)
        files = [str(value) for value in listish(item.get("root_files", [])) + listish(item.get("changed_file_paths", []))]
        for value in detect_languages_and_frameworks_from_files(files, json.dumps(item, ensure_ascii=False)):
            append_unique(detected, value, 80)
        for value in detected:
            for skill in extract_skill_names_from_text(value, 40) or [value]:
                add_skill(skill, "github_evidence", f"{project_name}: {value}", "high", project_name)
        for skill in extract_skill_names_from_text(json.dumps(item.get("diff_signals", []) + item.get("allowed_claims", []), ensure_ascii=False), 60):
            add_skill(skill, "github_evidence", project_name, "medium", project_name)

    rows = []
    if isinstance(project_database, dict):
        rows = [item for item in project_database.get("skills", []) if isinstance(item, dict)]
    elif isinstance(project_database, list):
        rows = [item for item in project_database if isinstance(item, dict)]
    for row in rows:
        name = row.get("skill") or row.get("normalized_skill") or row.get("name")
        if not name:
            continue
        add_skill(
            name,
            "project_database",
            row.get("evidence") or row.get("repository") or row.get("project_name") or "project tech stack database",
            str(row.get("confidence") or "medium"),
            str(row.get("project_name") or ""),
        )

    memory = user_memory if isinstance(user_memory, dict) else {}
    for value in values_for_skill_keys(memory, {"skill", "technology", "tech_stack", "tool", "language", "framework", "database"}, 120):
        for skill in extract_skill_names_from_text(value, 60):
            add_skill(skill, "user_memory", value, "medium")
    for value in values_for_skill_keys(memory, {"course", "coursework", "class", "education"}, 80):
        for skill in extract_skill_names_from_text(value, 40):
            add_skill(skill, "coursework", value, "medium")

    for skill in read_prior_generated_resume_skill_names():
        add_skill(skill, "prior_resume_versions", "prior generated tailored resume", "medium")

    for entry in skill_map.values():
        sources = set(entry.get("sources", []))
        support_sources = sources - {"jd_keywords"}
        has_support = bool(support_sources)
        confidence = str(entry.get("confidence") or "medium")
        safe = has_support and confidence != "low"
        entry["safe_to_include"] = safe
        entry["safeToInclude"] = safe
        if confidence == "medium" and (sources.issubset(WEAK_SKILL_SOURCE_LABELS) or entry["skill"] in CAUTIOUS_SKILL_WORDING):
            entry["wording_note"] = (
                f"Use as {CAUTIOUS_SKILL_WORDING.get(entry['skill'], entry['skill'])} "
                "when evidence is weaker or only from memory/coursework."
            )
        if entry["skill"] in PROTECTED_UNSUPPORTED_TOOLS and not support_sources:
            entry["safe_to_include"] = False
            entry["safeToInclude"] = False
        if not entry["safe_to_include"]:
            entry["confidence"] = "low" if sources == {"jd_keywords"} else confidence
        entry["score"] = max(
            int(entry.get("score", 0)),
            claim_relevance_score(entry["skill"], jd_terms, entry.get("evidenceSources", []), entry["confidence"]),
        )

    candidates = list(skill_map.values())
    candidates.sort(
        key=lambda item: (
            not bool(item.get("safe_to_include")),
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("jd_relevance", "low")), 2),
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("confidence", "low")), 2),
            -int(item.get("score", 0)),
            item.get("category", ""),
            item.get("skill", "").lower(),
        )
    )
    unsupported_jd = [
        item["skill"]
        for item in candidates
        if "jd_keywords" in item.get("sources", []) and not item.get("safe_to_include")
    ]
    suggested = []
    confirmations = []
    for skill in unsupported_jd:
        if skill in CAUTIOUS_SKILL_WORDING:
            append_unique(suggested, CAUTIOUS_SKILL_WORDING[skill], 20)
        append_unique(confirmations, f"Have you used {skill} directly in coursework, a project, or a work setting?", 20)
    return {
        "skill_candidates": candidates,
        "gap_report": {
            "jd_skills_not_supported": unsupported_jd,
            "suggested_safe_wording": suggested,
            "ask_user_to_confirm": confirmations,
        },
        "role_family": role_family,
        "category_schema": ROLE_TECHNICAL_SKILL_CATEGORIES.get(role_family, ROLE_TECHNICAL_SKILL_CATEGORIES["software_engineering"]),
    }


def skills_section_text(skills_section: dict | str) -> str:
    if isinstance(skills_section, dict):
        return json.dumps(skills_section, ensure_ascii=False)
    return str(skills_section or "")


def validate_technical_skills(
    skills_section: dict | str,
    skill_candidates: dict,
    jd_profile: dict,
) -> dict:
    text = skills_section_text(skills_section)
    skill_text = re.sub(r"\\textbf\{[^}]+\}", "", text)
    included_skills = []
    for skill in extract_skill_names_from_text(skill_text, 120):
        append_unique(included_skills, clean_resume_skill_name(skill), 120)
    candidates = skill_candidates.get("skill_candidates", []) if isinstance(skill_candidates, dict) else []
    candidate_map = {canonical_skill_name(item.get("skill")).lower(): item for item in candidates if isinstance(item, dict)}
    unsupported = []
    wording_adjustments = []
    for skill in included_skills:
        candidate = candidate_map.get(canonical_skill_name(skill).lower())
        if not candidate or not candidate.get("safe_to_include") or set(candidate.get("sources", [])) == {"jd_keywords"}:
            append_unique(unsupported, skill, 40)
            continue
        display = cautious_skill_wording(candidate)
        if display != candidate.get("skill") and display.lower() not in text.lower():
            wording_adjustments.append({"skill": candidate.get("skill"), "suggested_wording": display})

    role_profile = jd_profile.get("role_profile") if isinstance(jd_profile.get("role_profile"), dict) else {}
    role_family = str(role_profile.get("role_family") or jd_profile.get("role_family") or skill_candidates.get("role_family") or "software_engineering")
    allowed_categories = set(ROLE_TECHNICAL_SKILL_CATEGORIES.get(role_family, ROLE_TECHNICAL_SKILL_CATEGORIES["software_engineering"]))
    found_categories = re.findall(r"\\textbf\{([^}]+)\}", text)
    category_notes = []
    for category in found_categories:
        clean = re.sub(r"\\[A-Za-z]+\{?|[{}]", "", category).strip().rstrip(":")
        clean = (
            clean.replace(r"\&", "&")
            .replace(r"\#", "#")
            .replace(r"\_", "_")
            .replace(r"\%", "%")
            .replace(r"\$", "$")
        )
        if clean and clean not in allowed_categories:
            category_notes.append(f"Category `{clean}` does not match the {role_family} skills schema.")

    omitted = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("safe_to_include"):
            continue
        if candidate.get("jd_relevance") != "high" and candidate.get("confidence") != "high":
            continue
        skill = candidate.get("skill", "")
        if skill and canonical_skill_name(skill) not in included_skills:
            append_unique(omitted, skill, 20)

    notes = []
    if len(included_skills) > 32:
        notes.append("Technical Skills may be too broad; keep it below roughly 25-32 skills.")
    if category_notes:
        notes.extend(category_notes)
    jd_only_count = sum(
        1
        for skill in included_skills
        if set(candidate_map.get(canonical_skill_name(skill).lower(), {}).get("sources", [])) == {"jd_keywords"}
    )
    if jd_only_count:
        notes.append("One or more skills appear to be JD-only and should move to the gap report.")

    return {
        "valid": not unsupported and not category_notes and jd_only_count == 0,
        "unsupported_skills": unsupported,
        "omitted_relevant_known_skills": omitted,
        "wording_adjustments": wording_adjustments,
        "notes": notes,
    }


def jd_skill_requirements(job_description: str) -> dict[str, list[str]]:
    text = str(job_description or "")
    result = {
        "languages": [],
        "frameworks": [],
        "cloudInfra": [],
        "databases": [],
        "devops": [],
        "testing": [],
        "automation": [],
        "aiMl": [],
        "softSkills": [],
    }
    mapping = {
        "languages": ["Languages"],
        "frameworks": ["Backend / API", "Frontend / UI"],
        "cloudInfra": ["Cloud / DevOps / Infrastructure"],
        "databases": ["Database / Storage"],
        "devops": ["Cloud / DevOps / Infrastructure"],
        "testing": ["Testing / Quality"],
        "automation": ["AI / Automation"],
        "aiMl": ["AI / Automation"],
        "softSkills": ["Collaboration / Documentation"],
    }
    for output_key, categories in mapping.items():
        for category in categories:
            for skill in SKILL_CATEGORY_KEYWORDS.get(category, []):
                if skill_in_text(skill, text):
                    append_unique(result[output_key], canonical_skill_name(skill), 12)
    for skill in jd_requirements_for_prompt(text).get("soft_skills", []):
        append_unique(result["softSkills"], skill, 12)
    return result


def build_project_skill_evidence(
    project_memory: dict[str, Any],
    project_candidates: list[dict[str, Any]],
    jd_terms: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_by_name = {
        normalize_match_text(candidate.get("project_name") or candidate.get("source_name") or candidate.get("project_id")): candidate
        for candidate in project_candidates
        if isinstance(candidate, dict)
    }
    skill_map: dict[str, dict[str, Any]] = {}
    project_evidence = []
    for project in project_list_from_memory(project_memory):
        project_name = str(project.get("project_name") or project.get("name") or project.get("project_id") or "Project")
        compact_facts = build_current_project_compact_facts(
            project_name,
            project,
            {
                "technologies": listish(project.get("tech_stack", [])) + listish(project.get("tools", [])),
                "methods": listish(project.get("workflows", [])) + listish(project.get("confirmed_features", [])),
                "features": listish(project.get("confirmed_features", [])),
                "artifacts": shortest_evidence_sources(listish(project.get("evidence_notes", [])) + listish(project.get("recent_changes", [])), 8),
                "source_refs": ["project_memory.json"],
                "allowed_claims": [],
                "forbidden_claims": [],
            },
            [],
            [],
            " ".join(jd_terms),
        )
        candidate = candidate_by_name.get(normalize_match_text(project_name), {})
        candidate_bullets = candidate.get("recommended_bullets") or candidate.get("final_bullets") or []
        evidence_sources = ["project_memory.json"] + compact_facts.get("keyModules", [])[:5]
        for skill in compact_facts.get("technicalStack", []) + listish(project.get("tools", [])):
            add_candidate_skill(skill_map, skill, project_name, evidence_sources, jd_terms, "high")
        for signal in compact_facts.get("userContributionSignals", []):
            for category_keywords in SKILL_CATEGORY_KEYWORDS.values():
                for skill in category_keywords:
                    if skill.lower() in str(signal).lower():
                        add_candidate_skill(skill_map, skill, project_name, evidence_sources + [signal], jd_terms, "medium")
        for bullet in candidate_bullets:
            bullet_text = bullet.get("bullet") if isinstance(bullet, dict) else str(bullet)
            for category_keywords in SKILL_CATEGORY_KEYWORDS.values():
                for skill in category_keywords:
                    if skill.lower() in str(bullet_text).lower():
                        add_candidate_skill(skill_map, skill, project_name, evidence_sources + [bullet_text], jd_terms, "high")
        project_evidence.append(
            {
                "projectName": project_name,
                "technicalStack": compact_facts.get("technicalStack", [])[:10],
                "keyModules": compact_facts.get("keyModules", [])[:8],
                "evidenceSources": evidence_sources[:8],
                "relevanceToJD": compact_facts.get("jdRelevance", [])[:8],
            }
        )
    skills = list(skill_map.values())
    skills.sort(key=lambda item: (-int(item.get("score", 0)), item["category"], item["skill"].lower()))
    return project_evidence, skills


def build_compact_skills_input(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    project_candidates: list[dict[str, Any]],
    language: str,
    progress_guidance: str = "",
) -> dict[str, Any]:
    jd_requirements = jd_skill_requirements(job_description)
    role_profile = classify_role_family(job_description)
    jd_profile = {
        "raw_job_description": job_description,
        "requirements": jd_requirements_for_prompt(job_description),
        "skill_requirements": jd_requirements,
        "target_role": jd_core_for_prompt(job_description),
        "role_profile": role_profile,
    }
    jd_terms = []
    for values in jd_requirements.values():
        for value in values:
            append_unique(jd_terms, value, 40)
    existing_skills = extract_existing_resume_skills(resume)
    project_skill_evidence, candidate_skills = build_project_skill_evidence(project_memory, project_candidates, jd_terms)
    selected_project_cards = project_list_from_memory(project_memory) + [candidate for candidate in project_candidates if isinstance(candidate, dict)]
    project_database = query_all_project_tech_stacks()
    skill_candidate_payload = collect_skill_candidates_for_prompt(
        jd_profile=jd_profile,
        selected_project_cards=selected_project_cards,
        current_resume=resume,
        user_memory=read_user_memory_for_skills(),
        project_database=project_database,
        github_evidence=github_evidence_from_project_cards(project_candidates),
    )
    broad_candidate_skills = skill_candidate_payload.get("skill_candidates", [])
    legacy_skill_map = {item["skill"].lower(): item for item in candidate_skills if isinstance(item, dict) and item.get("skill")}
    for item in broad_candidate_skills:
        if not isinstance(item, dict) or not item.get("skill"):
            continue
        key = item["skill"].lower()
        current = legacy_skill_map.get(key)
        if current is None or int(item.get("score", 0)) >= int(current.get("score", 0)):
            legacy_skill_map[key] = item
    candidate_skills = list(legacy_skill_map.values())
    candidate_skills.sort(
        key=lambda item: (
            not bool(item.get("safe_to_include", item.get("safeToInclude", True))),
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("jd_relevance", item.get("jdRelevance", "low"))), 2),
            {"high": 0, "medium": 1, "low": 2}.get(str(item.get("confidence", "low")), 2),
            -int(item.get("score", 0)),
            item.get("category", ""),
            item.get("skill", "").lower(),
        )
    )
    return {
        "compactJdSkillRequirements": jd_requirements,
        "candidateSkills": candidate_skills[:48],
        "skill_candidates": broad_candidate_skills[:80],
        "skillCandidateSources": skill_candidate_payload,
        "gap_report": skill_candidate_payload.get("gap_report", {}),
        "existingResumeSkills": existing_skills[:48],
        "projectSkillEvidence": project_skill_evidence[:8],
        "projectDatabaseSkillEvidence": project_database[:48],
        "targetRole": jd_profile["target_role"],
        "roleProfile": role_profile,
        "categorySchema": skill_candidate_payload.get("category_schema", []),
        "formatConstraints": {
            "language": language,
            "outputLanguageInstruction": output_language_instruction(language),
            "maxCategories": 5,
            "maxSkillsPerCategory": 9,
            "atsFriendly": True,
            "noFabrication": True,
            "requireEvidenceForEverySkill": True,
            "extraGuidance": short_signal(progress_guidance, 500),
        },
        "riskFlags": [
            "Do not add skills without evidenceSources.",
            "Do not include low-relevance skills just to fill space.",
            "Merge aliases such as SQLite/better-sqlite3 and React/React.js.",
            "Move unsupported JD-only skills into gap_report instead of the Technical Skills section.",
        ],
    }


def build_compact_skills_prompt(payload: dict[str, Any]) -> str:
    return f"""
Generate structured Skills-section tailoring recommendations from compact skills evidence only.

Return ONLY valid JSON with exactly these keys:
  "skills_strategy": string,
  "skills_to_emphasize": array of strings,
  "skills_to_deemphasize": array of strings,
  "skills_to_add_if_supported": array of objects with keys "skill", "supporting_source", "confidence",
  "skills_to_remove_or_avoid": array of strings,
  "recommended_skills_section": string,
  "risks": array of strings

Rules:
- Generate the Technical Skills section using the JD, selected resume evidence, current resume skills, user memory, and known project technology stack. Do not limit the skills section only to technologies mentioned in the final project bullets. Include skills that are truthful, relevant, and supported by at least one source. Do not add unsupported JD-only skills. Use cautious wording such as "fundamentals", "familiarity", or "concepts" when evidence is weaker.
- Use only compactJdSkillRequirements, candidateSkills, skill_candidates, existingResumeSkills, projectSkillEvidence, projectDatabaseSkillEvidence, and gap_report.
- Every added or emphasized skill must appear in candidateSkills or skill_candidates with evidenceSources and safe_to_include=true.
- Do not invent tools, frameworks, platforms, databases, languages, certifications, or proficiency levels.
- Merge duplicate or alias skills.
- Prefer high JD relevance and high confidence.
- Keep the section concise and role-targeted: use categorySchema, at most 5 categories, and 8 skills per category.
- Put unsupported JD-only tools in gap_report / risks instead of recommended_skills_section.
- Do not read or infer raw repo context, full project memory, code, or chat history.

Compact skills payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def reduce_compact_skills_payload_for_limit(payload: dict[str, Any], max_chars: int) -> dict[str, Any]:
    reduced = json.loads(json.dumps(payload, ensure_ascii=False))

    def prompt_size() -> int:
        return len(build_compact_skills_prompt(reduced))

    skills = reduced.get("candidateSkills", [])
    if isinstance(skills, list):
        skills.sort(
            key=lambda item: (
                {"high": 0, "medium": 1, "low": 2}.get(str(item.get("jdRelevance", "low")), 2),
                {"high": 0, "medium": 1, "low": 2}.get(str(item.get("confidence", "low")), 2),
                -int(item.get("score", 0)),
                item.get("skill", "").lower(),
            )
        )
    for limit in [36, 28, 22, 16, 12]:
        if prompt_size() <= max_chars:
            break
        if isinstance(reduced.get("candidateSkills"), list):
            reduced["candidateSkills"] = reduced["candidateSkills"][:limit]
        if isinstance(reduced.get("skill_candidates"), list):
            reduced["skill_candidates"] = reduced["skill_candidates"][:limit]
    for limit in [6, 4]:
        if prompt_size() <= max_chars:
            break
        if isinstance(reduced.get("projectSkillEvidence"), list):
            reduced["projectSkillEvidence"] = reduced["projectSkillEvidence"][:limit]
    if prompt_size() > max_chars and isinstance(reduced.get("existingResumeSkills"), list):
        reduced["existingResumeSkills"] = reduced["existingResumeSkills"][:24]
    if prompt_size() > max_chars and isinstance(reduced.get("projectDatabaseSkillEvidence"), list):
        reduced["projectDatabaseSkillEvidence"] = reduced["projectDatabaseSkillEvidence"][:24]
    if prompt_size() > max_chars:
        for skill in list(reduced.get("candidateSkills", [])) + list(reduced.get("skill_candidates", [])):
            if isinstance(skill, dict):
                skill["evidenceSources"] = shortest_evidence_sources(skill.get("evidenceSources", []), 2)
                skill["evidenceProjects"] = skill.get("evidenceProjects", [])[:2]
    return reduced


def skill_source_priority(candidate: dict[str, Any]) -> int:
    sources = set(candidate.get("sources", []))
    if sources & {"current_resume", "project_evidence", "github_evidence", "project_database"}:
        return 0
    if sources & {"user_memory", "coursework"}:
        return 1
    if sources & {"prior_resume_versions"}:
        return 2
    return 3


def skill_candidate_sort_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        {"high": 0, "medium": 1, "low": 2}.get(str(candidate.get("jd_relevance", candidate.get("jdRelevance", "low"))), 2),
        {"high": 0, "medium": 1, "low": 2}.get(str(candidate.get("confidence", "low")), 2),
        skill_source_priority(candidate),
        -int(candidate.get("score", 0)),
        str(candidate.get("category", "")),
        str(candidate.get("skill", "")).lower(),
    )


def select_skill_candidates_for_section(compact_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_candidates = compact_payload.get("skill_candidates") or compact_payload.get("candidateSkills") or []
    candidates = []
    for candidate in raw_candidates:
        if not isinstance(candidate, dict):
            continue
        skill_name = clean_resume_skill_name(candidate.get("skill"))
        if not skill_name:
            continue
        if not bool(candidate.get("safe_to_include", candidate.get("safeToInclude", True))):
            continue
        if str(candidate.get("confidence", "medium")) == "low":
            continue
        jd_relevance = str(candidate.get("jd_relevance", candidate.get("jdRelevance", "low")))
        if not is_known_resume_skill_name(skill_name) and jd_relevance != "high":
            continue
        if jd_relevance == "low" and skill_name.lower() in JD_MATCH_REQUIRED_SECTION_SKILLS:
            continue
        candidate["skill"] = skill_name
        candidates.append(candidate)
    candidates.sort(key=skill_candidate_sort_key)
    category_schema = compact_payload.get("categorySchema") or []
    max_categories = int(compact_payload.get("formatConstraints", {}).get("maxCategories") or 5)
    max_per_category = int(compact_payload.get("formatConstraints", {}).get("maxSkillsPerCategory") or 8)
    selected = []
    category_counts: dict[str, int] = {}
    used_categories: list[str] = []
    for candidate in candidates:
        category = str(candidate.get("category") or "Tools & Workflow")
        if category not in used_categories and len(used_categories) >= max_categories:
            if candidate.get("jd_relevance") != "high":
                continue
        if category_counts.get(category, 0) >= max_per_category:
            continue
        if candidate.get("jd_relevance") == "low" and candidate.get("confidence") != "high" and skill_source_priority(candidate) > 1:
            continue
        selected.append(candidate)
        category_counts[category] = category_counts.get(category, 0) + 1
        append_unique(used_categories, category, max_categories)
        if len(selected) >= max_categories * max_per_category:
            break
    if category_schema:
        selected.sort(
            key=lambda item: (
                category_schema.index(item.get("category")) if item.get("category") in category_schema else len(category_schema),
                skill_candidate_sort_key(item),
            )
        )
    return selected


def render_technical_skills_section(selected_candidates: list[dict[str, Any]], category_schema: list[str]) -> str:
    grouped: dict[str, list[str]] = {}
    for candidate in selected_candidates:
        category = str(candidate.get("category") or "Tools & Workflow")
        grouped.setdefault(category, [])
        skill_name = clean_skill_display_name(cautious_skill_wording(candidate))
        append_unique(grouped[category], skill_name, 12)
    ordered_categories = [category for category in category_schema if category in grouped]
    ordered_categories.extend(category for category in grouped if category not in ordered_categories)
    skill_lines = []
    for category in ordered_categories:
        skills = grouped.get(category, [])
        if not skills:
            continue
        rendered_skills = ", ".join(latex_escape_text(skill) for skill in skills)
        skill_lines.append(f"    \\textbf{{{latex_escape_text(category)}:}} {rendered_skills}")
    lines = [
        "\\section{Technical Skills}",
        "\\begin{itemize}[leftmargin=0.15in, label={}]",
        "\\small{",
        "  \\item{",
    ]
    for index, skill_line in enumerate(skill_lines):
        suffix = r" \\" if index < len(skill_lines) - 1 else ""
        lines.append(f"{skill_line}{suffix}")
    lines.extend(
        [
            "  }",
            "}",
            "\\end{itemize}",
        ]
    )
    return "\n".join(lines)


def build_deterministic_skills_candidate(compact_payload: dict[str, Any]) -> dict[str, Any]:
    selected = select_skill_candidates_for_section(compact_payload)
    category_schema = compact_payload.get("categorySchema") or []
    selected_skill_names = [
        clean_resume_skill_name(candidate.get("skill"))
        for candidate in selected
        if clean_resume_skill_name(candidate.get("skill"))
    ]
    existing = {canonical_skill_name(skill).lower() for skill in compact_payload.get("existingResumeSkills", [])}
    additions = [
        {
            "skill": clean_resume_skill_name(candidate.get("skill")),
            "supporting_source": ", ".join(candidate.get("sources", [])),
            "confidence": candidate.get("confidence", "medium"),
            "safe_to_include": bool(candidate.get("safe_to_include", candidate.get("safeToInclude", True))),
            "wording": clean_skill_display_name(cautious_skill_wording(candidate)),
        }
        for candidate in selected
        if clean_resume_skill_name(candidate.get("skill"))
        and clean_resume_skill_name(candidate.get("skill")).lower() not in existing
    ]
    unsupported = compact_payload.get("gap_report", {}).get("jd_skills_not_supported", [])
    recommended_section = render_technical_skills_section(selected, category_schema)
    jd_profile = {
        "role_profile": compact_payload.get("roleProfile", {}),
        "requirements": compact_payload.get("compactJdSkillRequirements", {}),
    }
    validation = validate_technical_skills(
        recommended_section,
        compact_payload.get("skillCandidateSources", {"skill_candidates": compact_payload.get("skill_candidates", [])}),
        jd_profile,
    )
    return {
        "skills_strategy": (
            "Use a role-targeted skills section sourced from the JD, current resume, selected project evidence, "
            "memory, project tech-stack database rows, GitHub metadata, coursework, and prior resume versions; "
            "exclude unsupported JD-only claims."
        ),
        "skills_to_emphasize": selected_skill_names,
        "skills_to_deemphasize": [
            candidate.get("skill")
            for candidate in compact_payload.get("skill_candidates", [])
            if isinstance(candidate, dict)
            and not candidate.get("safe_to_include")
            and candidate.get("skill")
        ][:20],
        "skills_to_add_if_supported": additions,
        "skills_to_remove_or_avoid": unsupported,
        "recommended_skills_section": recommended_section,
        "skill_candidates": compact_payload.get("skill_candidates", []),
        "gap_report": compact_payload.get("gap_report", {}),
        "validation": validation,
        "risks": [
            f"Unsupported JD-only skill omitted: {skill}" for skill in unsupported[:10]
        ] + [
            f"Use cautious wording for {item['skill']} as {item['suggested_wording']}"
            for item in validation.get("wording_adjustments", [])
        ],
    }


def build_skills_resume_candidate(
    job_description: str,
    resume: str,
    project_memory: dict[str, Any],
    project_candidates: list[dict[str, Any]],
    language: str,
    progress_guidance: str = "",
) -> dict[str, Any]:
    compact_payload = build_compact_skills_input(
        job_description=job_description,
        resume=resume,
        project_memory=project_memory,
        project_candidates=project_candidates,
        language=language,
        progress_guidance=progress_guidance,
    )
    return build_deterministic_skills_candidate(compact_payload)


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


def summary_fit_score(candidate: dict[str, Any]) -> int:
    fit = str(candidate.get("fit") or "").lower()
    score = {"high": 90, "medium": 60, "low": 30}.get(fit, 50)
    validation = candidate.get("bullet_writer_validation") if isinstance(candidate.get("bullet_writer_validation"), dict) else {}
    if validation.get("accepted"):
        score += 8
    if candidate.get("recommended_bullets") or candidate.get("final_bullets"):
        score += 5
    return score


def summary_bullet_texts(candidate: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    bullets = candidate.get("recommended_bullets") or candidate.get("final_bullets") or []
    results = []
    for item in bullets[:limit]:
        if isinstance(item, dict):
            bullet = item.get("bullet") or item.get("text") or ""
            evidence = item.get("evidence") or item.get("supporting_source") or ""
            confidence = item.get("confidence") or "medium"
        else:
            bullet = str(item)
            evidence = ""
            confidence = "medium"
        if not str(bullet).strip():
            continue
        sources = shortest_evidence_sources([evidence, candidate.get("project_name"), candidate.get("source_name"), "staged project candidate"], 3)
        results.append(
            {
                "claim": short_signal(bullet, 260),
                "evidenceSources": sources,
                "confidence": confidence,
                "safeForSummary": bool(sources),
            }
        )
    return results


def compact_skills_for_summary(skills_candidate: dict[str, Any], max_skills: int = 18) -> list[dict[str, Any]]:
    skills = []
    for value in listish(skills_candidate.get("skills_to_emphasize", [])):
        name = canonical_skill_name(value)
        if name:
            skills.append({"skill": name, "evidenceSources": ["skills candidate"], "confidence": "medium"})
    for item in listish(skills_candidate.get("skills_to_add_if_supported", [])):
        if isinstance(item, dict):
            name = canonical_skill_name(item.get("skill"))
            source = item.get("supporting_source") or item.get("evidence") or "skills candidate"
            confidence = item.get("confidence") or "medium"
        else:
            name = canonical_skill_name(item)
            source = "skills candidate"
            confidence = "medium"
        if name:
            skills.append({"skill": name, "evidenceSources": shortest_evidence_sources([source], 2), "confidence": confidence})
    deduped: dict[str, dict[str, Any]] = {}
    for item in skills:
        key = item["skill"].lower()
        if key not in deduped:
            deduped[key] = item
    return list(deduped.values())[:max_skills]


def build_compact_summary_input(
    job_description: str,
    resume: str,
    project_candidates: list[dict[str, Any]],
    skills_candidate: dict[str, Any],
    experience_candidate: dict[str, Any],
    language: str,
    progress_guidance: str = "",
    max_projects: int = 4,
    max_skills: int = 18,
) -> dict[str, Any]:
    role_core = jd_core_for_prompt(job_description)
    role_profile = classify_role_family(job_description)
    sorted_projects = sorted(
        [candidate for candidate in project_candidates if isinstance(candidate, dict)],
        key=lambda item: (-summary_fit_score(item), str(item.get("project_name") or item.get("source_name") or "")),
    )[:max_projects]
    highlights = []
    strongest_evidence = []
    risk_flags = []
    for candidate in sorted_projects:
        project_name = str(candidate.get("project_name") or candidate.get("source_name") or candidate.get("project_id") or "Project")
        evidence_card = candidate.get("evidence_card") if isinstance(candidate.get("evidence_card"), dict) else {}
        claims = summary_bullet_texts(candidate, 3)
        sources = shortest_evidence_sources(
            listish(evidence_card.get("source_refs", []))
            + listish(evidence_card.get("artifacts", []))
            + [project_name, "staged project candidate"],
            5,
        )
        metrics = []
        for item in listish(evidence_card.get("data_or_scale", [])) + listish(candidate.get("metricCandidates", [])):
            metrics.append({"metric": short_signal(item, 180), "evidenceSources": sources[:3], "confidence": "medium"})
        for source in sources:
            append_unique(strongest_evidence, source, 16)
        for risk in listish(candidate.get("risks", [])) + listish(evidence_card.get("forbidden_claims", [])):
            append_unique(risk_flags, risk, 12)
        highlights.append(
            {
                "projectName": project_name,
                "oneLineSummary": short_signal(candidate.get("fit_reason") or candidate.get("job_alignment") or candidate.get("projectSummary") or project_name, 260),
                "keyTechnologies": shortest_evidence_sources(listish(evidence_card.get("technologies", [])) + listish(candidate.get("skills_to_emphasize", [])), 8),
                "strongestClaims": claims,
                "metrics": metrics[:4],
                "evidenceSources": sources,
                "fitScore": summary_fit_score(candidate),
            }
        )
    compact_skills = compact_skills_for_summary(skills_candidate, max_skills)
    for skill in compact_skills:
        for source in skill.get("evidenceSources", []):
            append_unique(strongest_evidence, source, 16)
    return {
        "targetRole": role_core,
        "compactJdRequirements": compact_bullet_writer_value(jd_requirements_for_prompt(job_description), 260, 6),
        "candidatePositioning": {
            "primary": role_profile.get("role_family"),
            "secondary": role_profile.get("secondary_role_families", [])[:3],
            "roleFocus": role_core.get("role_focus", [])[:5],
        },
        "topProjectHighlights": highlights,
        "compactSkills": compact_skills,
        "experienceSignals": compact_bullet_writer_value(
            {
                "strategy": experience_candidate.get("experience_strategy") or experience_candidate.get("job_alignment"),
                "supported_bullets": experience_candidate.get("final_bullets", [])[:4],
                "risks": experience_candidate.get("risks", [])[:5],
            },
            260,
            5,
        ),
        "strongestEvidence": strongest_evidence[:16],
        "riskFlags": risk_flags[:12],
        "summaryConstraints": {
            "language": language,
            "outputLanguageInstruction": output_language_instruction(language),
            "maxLines": 3,
            "atsFriendly": True,
            "noFabrication": True,
            "noUnsupportedMetrics": True,
            "noUnsupportedYearsExperience": True,
            "extraGuidance": short_signal(progress_guidance, 400),
        },
    }


def build_compact_summary_prompt(payload: dict[str, Any]) -> str:
    return f"""
Generate structured resume Summary/Profile tailoring recommendations from compact summary evidence only.

Return ONLY valid JSON with exactly these keys:
  "summary_strategy": string,
  "recommended_summary": string,
  "keywords_to_include": array of strings,
  "claims_to_avoid": array of strings,
  "evidence_basis": array of strings,
  "risks": array of strings

Rules:
- Use only the compact summary payload below.
- Do not invent years of experience, seniority, production scale, users, QPS, P99, cost, accuracy, or unsupported metrics.
- Mention only skills and project strengths with evidenceSources or staged candidate support.
- Prefer concise ATS-friendly language aligned to targetRole and compactJdRequirements.
- If evidence is thin, use conservative positioning rather than inflated claims.
- Do not read or infer raw repo context, full project memory, full achievements, code, or chat history.

Compact summary payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def reduce_compact_summary_payload_for_limit(payload: dict[str, Any], max_chars: int = 12000) -> dict[str, Any]:
    reduced = json.loads(json.dumps(payload, ensure_ascii=False))

    def prompt_size() -> int:
        return len(build_compact_summary_prompt(reduced))

    if isinstance(reduced.get("topProjectHighlights"), list):
        reduced["topProjectHighlights"].sort(key=lambda item: -int(item.get("fitScore", 0)))
    for project_limit, skill_limit, claim_limit in [(3, 16, 3), (3, 14, 2), (2, 12, 2), (2, 10, 1)]:
        if prompt_size() <= max_chars:
            break
        if isinstance(reduced.get("topProjectHighlights"), list):
            reduced["topProjectHighlights"] = reduced["topProjectHighlights"][:project_limit]
            for project in reduced["topProjectHighlights"]:
                if isinstance(project, dict):
                    project["strongestClaims"] = project.get("strongestClaims", [])[:claim_limit]
                    project["metrics"] = project.get("metrics", [])[:2]
                    project["evidenceSources"] = shortest_evidence_sources(project.get("evidenceSources", []), 3)
        if isinstance(reduced.get("compactSkills"), list):
            reduced["compactSkills"] = reduced["compactSkills"][:skill_limit]
        if isinstance(reduced.get("strongestEvidence"), list):
            reduced["strongestEvidence"] = reduced["strongestEvidence"][:10]
    return reduced


class ResumeCompactInputBuilder:
    build_compact_summary_input = staticmethod(build_compact_summary_input)
    build_compact_skills_input = staticmethod(build_compact_skills_input)
    build_compact_project_input = staticmethod(build_compact_project_input)
    build_compact_experience_input = staticmethod(build_compact_experience_input)
    build_compact_final_merge_input = staticmethod(build_compact_final_merge_input)


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
{truncate_text(job_description, provider_safe_text_limit(12000, 7000))}

Original resume:
{truncate_text(resume, provider_safe_text_limit(26000, 9000))}

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
    compact_payload = None

    def compact_summary_prompt(max_chars: int = PROXY_SAFE_MAX_INPUT_CHARS, **_: Any) -> dict[str, Any]:
        nonlocal compact_payload
        compact_payload = build_compact_summary_input(
            job_description=job_description,
            resume=resume,
            project_candidates=project_candidates,
            skills_candidate=skills_candidate,
            experience_candidate=experience_candidate,
            language=language,
            progress_guidance=progress_guidance,
        )
        target = min(12000, max_chars)
        compact_prompt = build_compact_summary_prompt(compact_payload)
        if len(compact_prompt) > target:
            compact_payload = reduce_compact_summary_payload_for_limit(compact_payload, target)
            compact_prompt = build_compact_summary_prompt(compact_payload)
        if len(compact_prompt) > max_chars:
            compact_payload = reduce_compact_summary_payload_for_limit(compact_payload, max_chars)
            compact_prompt = build_compact_summary_prompt(compact_payload)
        return {"prompt": compact_prompt, "payload": compact_payload}

    response = safe_model_call(
        caller="build_summary_resume_candidate",
        prompt=prompt,
        task_type="summary_resume_candidate",
        compact_builder=compact_summary_prompt,
    )
    payload = extract_json_object(response)
    for key in ["keywords_to_include", "claims_to_avoid", "evidence_basis", "risks"]:
        if not isinstance(payload.get(key), list):
            payload[key] = []
    if "compact_payload" in locals() and isinstance(compact_payload, dict):
        supported_skills = {canonical_skill_name(item.get("skill")).lower() for item in compact_payload.get("compactSkills", []) if isinstance(item, dict)}
        if supported_skills:
            payload["keywords_to_include"] = [
                item for item in payload["keywords_to_include"]
                if canonical_skill_name(item).lower() in supported_skills
                or any(canonical_skill_name(item).lower() in str(project).lower() for project in compact_payload.get("topProjectHighlights", []))
            ][:18]
    return payload


def apply_resume_project_candidate(
    job_description: str,
    current_resume: str,
    project_candidate: dict[str, Any],
    body: TailorBody,
    index: int,
    total: int,
    project_ranking: dict[str, Any] | None = None,
    resume_constraints: dict[str, Any] | None = None,
) -> str:
    constraints = default_resume_constraints(resume_constraints)
    normal_payload = {
        "section_name": "Project-section",
        "job_description": job_description,
        "current_resume": current_resume,
        "candidate": project_candidate,
        "project_ranking": compact_value_for_prompt(project_ranking or {}, 700, 6),
        "project_rank_entry": compact_value_for_prompt(project_candidate.get("project_ranking_context") or {}, 700, 6),
        "resume_constraints": constraints,
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
            block = safe_model_call(
                caller="apply_resume_project_candidate_retry",
                prompt=build_retry_merge_prompt(retry_payload),
                task_type="final_resume_merge",
                compact_builder=lambda max_chars=PROXY_SAFE_MAX_INPUT_CHARS, **_: final_merge_compact_prompt(retry_payload, max_chars),
            )
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
- {PROJECT_PRIORITY_INSTRUCTION}
- Respect this candidate's project_rank and bullet_budget. The highest-ranked project should receive the most
  detailed treatment across core implementation, tools/methods, data/storage/workflow logic,
  testing/debugging/automation/documentation, and target-role relevance.
- Lower-ranked projects should stay concise and should not receive equal space unless their ranking scores are genuinely similar.
- Do not re-add projects listed in project_ranking.omitted_projects.
- Preserve the STAR grounding from the staged candidate. Do not add metrics, ownership level,
  scale, or results that are not present in star_analysis, final_bullets, user guidance, or evidence.
- Reject generic stack-only wording such as "used X to develop Y"; keep action + module + technical method + supported result/value.
- Project selection allowed: {payload["allow_project_selection"]}
- If project selection is not allowed, keep the existing resume project list and only update factual wording.
- Do not invent unsupported metrics, technologies, responsibilities, employers, roles, dates, or repository facts.
- Return only LaTeX code with no Markdown fences and no analysis text.

Job description:
{truncate_text(payload["job_description"], provider_safe_text_limit(12000, 6000))}

Current LaTeX resume:
{truncate_text(payload["current_resume"], provider_safe_text_limit(30000, 13000))}

One staged Project candidate:
{json.dumps(prompt_project_candidate, ensure_ascii=False, indent=2)}

Project ranking and one-page constraints:
{json.dumps(compact_value_for_prompt({"project_ranking": payload.get("project_ranking"), "project_rank_entry": payload.get("project_rank_entry"), "resume_constraints": payload.get("resume_constraints")}, 900, 8), ensure_ascii=False, indent=2)}
"""
        )
        return safe_model_call(
            caller="apply_resume_project_candidate",
            prompt=prompt,
            task_type="final_resume_merge",
            compact_builder=lambda max_chars=PROXY_SAFE_MAX_INPUT_CHARS, **_: final_merge_compact_prompt(payload, max_chars),
        )

    answer = call_model_with_context_retry(normal_payload, merge_retry_payload_for_prompt, call_merge_model)
    return complete_resume_from_merge_response(
        answer,
        current_resume,
        merge_retry_payload_for_prompt(normal_payload)["target_resume_block"],
        f"Agent did not return valid LaTeX resume code after project merge step {index}.",
    )


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
            block = safe_model_call(
                caller="apply_resume_section_candidate_retry",
                prompt=build_retry_merge_prompt(retry_payload),
                task_type="final_resume_merge",
                compact_builder=lambda max_chars=PROXY_SAFE_MAX_INPUT_CHARS, **_: final_merge_compact_prompt(retry_payload, max_chars),
            )
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
{truncate_text(payload["job_description"], provider_safe_text_limit(12000, 6000))}

Current LaTeX resume:
{truncate_text(payload["current_resume"], provider_safe_text_limit(30000, 13000))}

One staged {section_name} candidate:
{json.dumps(prompt_candidate, ensure_ascii=False, indent=2)}
"""
        )
        return safe_model_call(
            caller=f"apply_resume_section_candidate:{section_name}",
            prompt=prompt,
            task_type="final_resume_merge",
            compact_builder=lambda max_chars=PROXY_SAFE_MAX_INPUT_CHARS, **_: final_merge_compact_prompt(payload, max_chars),
        )

    answer = call_model_with_context_retry(normal_payload, merge_retry_payload_for_prompt, call_merge_model)
    return complete_resume_from_merge_response(
        answer,
        current_resume,
        merge_retry_payload_for_prompt(normal_payload)["target_resume_block"],
        f"Agent did not return valid LaTeX resume code after {section_name} merge step.",
    )


def apply_skills_section_candidate(
    current_resume: str,
    candidate: dict[str, Any],
) -> str:
    replacement = str(candidate.get("recommended_skills_section") or "").strip()
    if not replacement:
        return current_resume
    replacement = strip_markdown_code_fence(replacement)
    target_block = resume_block_for_prompt(current_resume, "Skills-section")
    if "\\section" not in replacement:
        section_name = str(target_block.get("section_name") or "Technical Skills")
        replacement = f"\\section{{{section_name}}}\n{replacement}"
    if not replacement.endswith("\n"):
        replacement += "\n"
    try:
        merged = replace_resume_block(current_resume, target_block, replacement)
        return merged.replace("\\end{itemize}\\end{document}", "\\end{itemize}\n\\end{document}")
    except HTTPException:
        return current_resume


def project_bullet_budget(index: int, total: int) -> int:
    if total <= 1:
        return 5
    if total == 2:
        return 5 if index == 1 else 3
    if index == 1:
        return 4
    if index == 2:
        return 3
    return 2


def project_layout_candidates_for_prompt(project_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    layout_candidates = []
    total = len(project_candidates)
    for index, candidate in enumerate(project_candidates, start=1):
        compact_candidate = compact_bullet_candidate_for_prompt(candidate)
        rank = int(candidate.get("project_rank") or index)
        budget = int(candidate.get("bullet_budget") or project_bullet_budget(rank, total))
        layout_candidates.append(
            {
                **compact_candidate,
                "rank": rank,
                "target_bullet_count": budget,
                "focus_areas": candidate.get("focus_areas", [])[:8],
            }
        )
    return layout_candidates


def apply_projects_section_layout(
    job_description: str,
    current_resume: str,
    project_candidates: list[dict[str, Any]],
    body: TailorBody,
    project_ranking: dict[str, Any] | None = None,
    resume_constraints: dict[str, Any] | None = None,
) -> str:
    if not body.allow_project_selection:
        return current_resume

    constraints = default_resume_constraints(resume_constraints)
    layout_candidates = project_layout_candidates_for_prompt(project_candidates[:MAX_STAGED_PROJECTS])
    normal_payload = {
        "section_name": "Project-section",
        "job_description": job_description,
        "current_resume": current_resume,
        "candidate": {
            "selected_project_candidates": layout_candidates,
            "project_ranking": compact_value_for_prompt(project_ranking or {}, 900, 8),
            "preferred_project_count": PREFERRED_RESUME_PROJECTS,
            "maximum_project_count": MAX_STAGED_PROJECTS,
            "ranking_rule": PROJECT_PRIORITY_INSTRUCTION,
            "one_page_rule": "Prefer a one-page resume; reduce lower-ranked project bullets before cutting top-ranked project bullets.",
            "one_page_cut_order": constraints.get("one_page_cut_order"),
        },
        "project_ranking": compact_value_for_prompt(project_ranking or {}, 900, 8),
        "resume_constraints": constraints,
        "block_hint": "Projects",
        "allow_project_selection": body.allow_project_selection,
        "allow_experience_removal": body.allow_experience_removal,
        "language": body.language,
    }

    def call_layout_model(payload: dict[str, Any]) -> str:
        if payload.get("retry"):
            retry_payload = payload["retry_payload"]
            block = safe_model_call(
                caller="apply_projects_section_layout_retry",
                prompt=build_retry_merge_prompt(retry_payload),
                task_type="final_resume_merge",
                compact_builder=lambda max_chars=PROXY_SAFE_MAX_INPUT_CHARS, **_: final_merge_compact_prompt(retry_payload, max_chars),
            )
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
- {PROJECT_PRIORITY_INSTRUCTION}
- Preserve selected_project_candidates ranking order from strongest to weakest.
- Preserve the intended target_bullet_count for each project as much as possible.
- If space is limited, cut in this order: weak/repetitive third project, lower-ranked project bullets,
  less relevant Experience bullets, Summary wording, and top-ranked project bullets only as a last resort.
- Never keep more than {MAX_STAGED_PROJECTS} Projects-section entries.
- Remove projects that are not in selected_project_candidates when project selection is allowed.
- Do not re-add omitted projects from project_ranking.omitted_projects.
- Higher-ranked projects must have more bullets than lower-ranked projects when multiple projects are shown, unless
  the ranking scores are genuinely similar.
- Lower-ranked project bullets should be compact and only keep the most job-relevant factual claim.
- Reduce lower-ranked project bullets first when one-page constraints require trimming.
- Do not create new bullet wording outside the selected candidates' final_bullets / recommended_bullets.
- Do not invent unsupported metrics, technologies, responsibilities, employers, roles, dates, or repository facts.
- Preserve LaTeX validity and existing section style.
- Return only LaTeX code with no Markdown fences and no analysis text.

Job description:
{truncate_text(payload["job_description"], provider_safe_text_limit(12000, 6000))}

Current LaTeX resume:
{truncate_text(payload["current_resume"], provider_safe_text_limit(30000, 13000))}

Projects-section layout candidate data:
{json.dumps(prompt_candidate, ensure_ascii=False, indent=2)}

Project ranking and constraints:
{json.dumps(compact_value_for_prompt({"project_ranking": payload.get("project_ranking"), "resume_constraints": payload.get("resume_constraints")}, 1000, 8), ensure_ascii=False, indent=2)}
"""
        )
        return safe_model_call(
            caller="apply_projects_section_layout",
            prompt=prompt,
            task_type="final_resume_merge",
            compact_builder=lambda max_chars=PROXY_SAFE_MAX_INPUT_CHARS, **_: final_merge_compact_prompt(payload, max_chars),
        )

    answer = call_model_with_context_retry(normal_payload, merge_retry_payload_for_prompt, call_layout_model)
    return complete_resume_from_merge_response(
        answer,
        current_resume,
        merge_retry_payload_for_prompt(normal_payload)["target_resume_block"],
        "Agent did not return valid LaTeX resume code after Projects-section layout step.",
    )


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
    project_ranking: dict[str, Any] | None = None,
    resume_constraints: dict[str, Any] | None = None,
) -> str:
    current_resume = resume
    current_resume = enforce_project_section_allocation(
        current_resume,
        project_ranking,
        project_candidates,
        resume_constraints,
    )
    current_resume = validate_complete_resume_or_raise(
        current_resume,
        "Deterministic Projects-section merge produced invalid LaTeX.",
    )

    current_resume = apply_skills_section_candidate(current_resume, skills_candidate)
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
    current_resume = enforce_project_section_allocation(
        current_resume,
        project_ranking,
        project_candidates,
        resume_constraints,
    )
    current_resume = validate_complete_resume_or_raise(
        current_resume,
        "Final deterministic Projects-section enforcement produced invalid LaTeX.",
    )
    return current_resume if current_resume.endswith("\n") else current_resume + "\n"


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
    role_profile = classify_role_family(job_description)
    jd_requirements = jd_requirements_for_prompt(job_description)
    resume_constraints = default_resume_constraints()
    selected_projects, project_ranking = select_staged_projects_with_ranking(
        job_description,
        resume,
        project_memory,
        body.allow_project_selection,
        resume_constraints,
    )
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
            project_ranking,
            project_ranking_entry_for_project(project_ranking, project),
            resume_constraints,
        ))
    project_ranking = attach_candidate_claims_to_project_ranking(project_ranking, candidates)

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
    gap_report = build_resume_gap_report(role_profile, jd_requirements, candidates)
    if isinstance(skills_candidate.get("gap_report"), dict):
        gap_report["technical_skills"] = skills_candidate["gap_report"]
    if isinstance(skills_candidate.get("validation"), dict):
        gap_report["technical_skills_validation"] = skills_candidate["validation"]

    answer = merge_staged_resume(
        job_description,
        resume,
        candidates,
        skills_candidate,
        experience_candidate,
        summary_candidate,
        body,
        project_ranking,
        resume_constraints,
    )
    if not agent.looks_like_latex_resume(answer):
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code.")
    project_section_validation = validate_project_section_allocation(answer, project_ranking, resume_constraints)
    blocking_project_issues = [
        issue for issue in project_section_validation.get("issues", [])
        if any(
            marker in issue
            for marker in [
                "above the maximum",
                "Omitted projects were re-added",
                "Project order does not follow",
                "over-expanded",
            ]
        )
    ]
    if blocking_project_issues:
        raise HTTPException(
            status_code=500,
            detail="Projects section allocation enforcement failed: " + "; ".join(blocking_project_issues),
        )

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
        "project_ranking": project_ranking,
        "project_section_validation": project_section_validation,
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
    force_chunking: bool = False,
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
    project_memory_before_mtime = (
        agent.PROJECT_MEMORY_PATH.stat().st_mtime
        if agent.PROJECT_MEMORY_PATH.exists()
        else None
    )
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
    for repo_context in repo_contexts:
        if isinstance(repo_context, dict) and not repo_context.get("error"):
            try:
                save_project_tech_stack(repo_context)
            except sqlite3.Error:
                pass

    path = agent.CHROMA_DB_PATH
    if fetched_contexts:
        path = agent.save_github_context_output(fetched_contexts)

    if needs_project_memory_reanalysis:
        assert_agent_task_not_cancelled()
        project_memory_update = update_project_memory_from_repo_analysis(
            repo_contexts,
            agent_progress_messages=agent_progress_messages,
            force_chunking=force_chunking,
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

    project_memory_after_mtime = (
        agent.PROJECT_MEMORY_PATH.stat().st_mtime
        if agent.PROJECT_MEMORY_PATH.exists()
        else None
    )
    project_memory_status = build_project_memory_status_summary(
        project_memory_update,
        was_reanalyzed=needs_project_memory_reanalysis,
        scan_results=scan_results,
        before_mtime=project_memory_before_mtime,
        after_mtime=project_memory_after_mtime,
    )
    project_memory_update["status_summary"] = project_memory_status

    assert_agent_task_not_cancelled()
    save_github_repo_scan_state(scan_state)
    return {
        "saved": agent.has_usable_repo_context(repo_contexts),
        "path": str(path),
        "project_name": project_name.strip(),
        "project_id": project_id.strip(),
        "project_memory_update": project_memory_update,
        "project_memory_status": project_memory_status,
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
        answer = safe_model_call(caller="agent_progress_guidance_reply", prompt=prompt, task_type="agent_progress_reply")
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
    answer = safe_model_call(caller="resume_pdf_to_latex", prompt=prompt, task_type="resume_pdf_to_latex")
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
    answer = safe_model_call(caller="generate_interview_prep", prompt=prompt, task_type="interview_prep")
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
            force_chunking=body.forceChunking,
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
