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
import threading
import time
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

FILE_MAP = {
    "resume": agent.RESUME_PATH,
    "tailored_resume": agent.OUTPUT_RESUME_PATH,
    "job_description": agent.JOB_DESCRIPTION_PATH,
    "cover_letter": agent.COVER_LETTER_PATH,
    "interview_prep": agent.INTERVIEW_PREP_PATH,
    "memory": agent.MEMORY_PATH,
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
Based on the saved job_description.txt, Chroma profile memory, resume.txt, and approved GitHub context if useful,
generate the modified complete LaTeX resume code.
Compare the projects currently listed in resume.txt with the factual projects available in Chroma profile memory.
Choose the strongest project mix for the saved job description: you may remove a weaker resume project,
update an existing project's bullets, or add a better-matching memory project that is not currently in the resume.
Tailor the Experience section for the saved job description: you may reorder factual bullets, rewrite bullets
for relevance and clarity, and remove weaker or redundant bullets. Preserve the factual meaning of the source resume.
Keep every existing Experience entry unless the user explicitly allows removing entire Experience entries.
Repository links in Chroma profile memory may be used only as candidates for approved GitHub evidence.
Do not invent claims, technologies, metrics, responsibilities, employers, roles, dates, or repository facts.
Return only LaTeX code with no Markdown fences and no analysis text.
Save with save_tailored_resume when complete.
"""


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


class JobDescriptionBody(BaseModel):
    content: str


class AnalyzeBody(BaseModel):
    use_github_context: bool = False
    language: str = "zh"


class TailorBody(BaseModel):
    use_github_context: bool = True
    allow_project_selection: bool = True
    allow_experience_removal: bool = False
    include_application_hint: bool = False
    language: str = "zh"


class ResumeMemoryBody(BaseModel):
    resume_source: str = "resume"


class ResumePdfToLatexBody(BaseModel):
    filename: str = "resume.pdf"
    data_base64: str
    language: str = "zh"


class TailoredResumePdfBody(BaseModel):
    content: str = ""


class CoverLetterBody(BaseModel):
    use_tailored_resume: bool = True
    use_github_context: bool = False
    style: str = "concise"
    language: str = "zh"


class InterviewPrepBody(BaseModel):
    use_github_context: bool = True
    language: str = "zh"


class GitHubScanBody(BaseModel):
    resume_source: str = "resume"


class GitHubContextBody(BaseModel):
    approved: bool = True
    resume_source: str = "resume"


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
    return {
        "identities": agent.read_github_identities(),
        "token_configured": agent.github_token_is_configured(),
        "memory_repositories": agent.MEMORY_STORE.list_github_repositories(),
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


def list_output_files(directory: Path, suffix: str, limit: int = 5) -> list[dict[str, str]]:
    if not directory.exists():
        return []
    files = sorted(directory.glob(f"*{suffix}"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [{"name": path.name, "path": str(path)} for path in files[:limit]]


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


def list_job_analysis_history(limit: int = 5) -> list[dict[str, str]]:
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

    if name == "tailored_resume" and not agent.file_is_ready(agent.OUTPUT_RESUME_PATH):
        if not agent.LEGACY_OUTPUT_RESUME_PATH.exists():
            return False, ""
        content = agent.LEGACY_OUTPUT_RESUME_PATH.read_text(encoding="utf-8")
        ready = agent.file_is_ready(agent.LEGACY_OUTPUT_RESUME_PATH)
        return ready, content

    path = FILE_MAP[name]
    if not path.exists():
        return False, ""
    content = path.read_text(encoding="utf-8")
    ready = agent.file_is_ready(path)
    return ready, content


def file_ready(name: str, path: Path) -> bool:
    if name == "memory":
        return bool(agent.MEMORY_STORE.profile_count())
    if name == "tailored_resume":
        return agent.file_is_ready(path) or agent.file_is_ready(agent.LEGACY_OUTPUT_RESUME_PATH)
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
    if name == "tailored_resume":
        agent.save_tailored_resume(content)
        return
    if name == "cover_letter":
        agent.save_cover_letter(content)
        return
    if name == "interview_prep":
        agent.save_interview_prep(content)
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
    if normalize_language(language) != "en":
        return ""
    return (
        "\n\nOutput language requirement: respond entirely in English. "
        "All user-facing headings, analysis, recommendations, cover letters, "
        "interview preparation notes, and chat responses must be English."
    )


def original_resume_language_instruction(output_type: str) -> str:
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
    if normalize_language(language) == "en":
        return (
            "\n\nOutput language requirement: respond entirely in English. "
            "Use English section headings and English explanations for the full job analysis."
        )
    return (
        "\n\n输出语言要求：请完全使用中文输出职位分析。"
        "所有标题、匹配分数说明、技能分析、项目建议和简历修改建议都必须使用中文。"
    )


def interview_prep_language_instruction(language: str) -> str:
    if normalize_language(language) == "en":
        return (
            "\n\nOutput language requirement: respond entirely in English. "
            "Use English section headings and English notes only."
        )
    return (
        "\n\nOutput language requirement: produce parallel Chinese-English interview preparation notes. "
        "Every Chinese sentence or bullet must be followed immediately by its matching English translation "
        "on the next line. Keep the pairs one-to-one: do not group all Chinese together and then all English, "
        "and do not put Chinese and English in the same sentence. Use this pattern throughout:\n"
        "中文句子。\n"
        "English sentence.\n"
        "- 中文要点。\n"
        "- English bullet.\n"
        "Section headings must also be paired on adjacent lines."
    )


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


def build_pdf_to_latex_prompt(filename: str, extracted: dict[str, Any], language: str) -> str:
    links_text = "\n".join(
        f"- Page {link['page']}: {link['url']}" for link in extracted["links"]
    ) or "- None detected"
    language_requirement = (
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

    return f"""
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


def latex_commands_for_resume(tex_path: Path, build_dir: Path) -> list[list[str]]:
    commands = []

    xelatex = shutil.which("xelatex")
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

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        commands.append(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={build_dir}",
                str(tex_path),
            ]
        )

    latexmk = shutil.which("latexmk")
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


def compile_tailored_resume_pdf(latex: str) -> Path:
    document = agent.extract_latex_document(latex)
    if not document:
        raise HTTPException(status_code=400, detail="No complete LaTeX document found.")

    TAILORED_RESUME_PDF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"tailored_resume_{agent.timestamp_slug()}"
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

    output_pdf = TAILORED_RESUME_PDF_OUTPUT_DIR / f"{stem}.pdf"
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


def run_agent_task(
    message: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    images: Optional[list[dict[str, str]]] = None,
) -> str:
    adapter, _ = get_adapter(provider)
    chosen_model = model or adapter.default_model()
    try:
        return agent.ask_agent(message, adapter=adapter, model=chosen_model, images=images)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except (APIStatusError, urllib.error.HTTPError) as error:
        raise_model_api_http_exception(error)
    except agent.transient_network_errors() as error:
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def run_text_task(message: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    adapter, _ = get_adapter(provider)
    chosen_model = model or adapter.default_model()
    try:
        response = adapter.create_response(
            model=chosen_model,
            instructions=agent.SYSTEM_PROMPT,
            tools=[],
            input_items=[{"role": "user", "content": message}],
        )
        return adapter.output_text(response)
    except (APIStatusError, urllib.error.HTTPError) as error:
        raise_model_api_http_exception(error)
    except agent.transient_network_errors() as error:
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


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


def update_memory_from_resume_source(resume_source: str) -> dict[str, Any]:
    if resume_source == "tailored_resume":
        resume = agent.read_tailored_resume()
        source_label = "tailored_resume.txt"
    elif resume_source == "resume":
        resume = agent.read_resume()
        source_label = "resume.txt"
    else:
        raise HTTPException(status_code=400, detail="resume_source must be 'resume' or 'tailored_resume'.")

    current_memory = load_memory_for_merge()
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

Current memory JSON:
{json.dumps(current_memory, ensure_ascii=False, indent=2)}

Resume source: {source_label}
Resume:
{resume}
"""
    response = run_text_task(prompt)
    payload = extract_json_object(response)
    merged_memory = payload.get("memory")
    if not isinstance(merged_memory, dict):
        raise HTTPException(status_code=500, detail="Agent JSON response must include a memory object.")

    changed = normalized_json(merged_memory) != normalized_json(current_memory)
    if changed:
        agent.replace_profile_memory(merged_memory, source=f"resume-merge:{source_label}")

    additions = payload.get("additions", [])
    if not isinstance(additions, list):
        additions = []

    return {
        "updated": changed,
        "source": source_label,
        "additions": [str(item) for item in additions if str(item).strip()],
        "memory": merged_memory,
        "path": str(agent.CHROMA_DB_PATH),
    }


def build_interview_prep_prompt(use_github_context: bool, language: str = "zh") -> str:
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

    if normalize_language(language) == "en":
        section_outline = """  1. Role Focus
  2. Technical Questions
  3. Project Talking Points
  4. Behavioral / STAR Material
  5. Preparation Gaps or Facts to Verify
  6. Questions to Ask the Interviewer"""
    else:
        section_outline = """  1. 职位重点
     Role Focus
  2. 技术问题准备
     Technical Questions
  3. 项目讲述要点
     Project Talking Points
  4. 行为面试 / STAR 素材
     Behavioral / STAR Material
  5. 需要补强或确认的内容
     Preparation Gaps or Facts to Verify
  6. 反问面试官的问题
     Questions to Ask the Interviewer"""

    prompt = f"""
Create complete interview preparation notes for the job application below.

Rules:
- Use only the job description, resume, memory, and approved GitHub context provided here.
- Do not invent projects, employers, degrees, technologies, metrics, or repository facts.
- If evidence is weak or missing, say what to prepare or verify instead of fabricating.
- Return only the notes content. Do not say that you saved a file. Do not include placeholders.
- Follow the output language requirement below exactly, regardless of the job description language.
- Include these sections:
  1. 职位重点
  2. 技术问题准备
  3. 项目讲述要点
  4. 行为面试 / STAR 素材
  5. 需要补强或确认的内容
  6. 反问面试官的问题

If any earlier section labels conflict with this outline, use this exact section outline for the final notes:
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
    return prompt + interview_prep_language_instruction(language)


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
    return section_hits >= 3


def read_approved_github_context(query: str = "") -> str:
    context = agent.read_stored_github_context(query=query)
    if not context:
        return "No approved GitHub context is available. Ask the user to approve GitHub access in the web UI first."

    if not agent.has_usable_repo_context(context):
        return "No usable approved GitHub context is available."

    return json.dumps(context, ensure_ascii=False, indent=2)


agent.TOOL_FUNCTIONS["read_github_context"] = read_approved_github_context


def read_github_memory_repo_source() -> str:
    repositories = agent.MEMORY_STORE.list_github_repositories()
    return "\n".join(
        f"https://github.com/{item['repository']}"
        for item in repositories
        if item.get("repository")
    )


def read_github_repo_source(resume_source: str) -> str:
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


def fetch_github_context_api(approved: bool, resume_source: str = "resume") -> dict[str, Any]:
    if not approved:
        return {"saved": False, "message": "GitHub context fetch was not approved."}

    try:
        repo_source = read_github_repo_source(resume_source)
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

    repo_contexts = []
    for repo in repos:
        repo_context = agent.fetch_github_repo_context(repo)
        repo_context["verified_github_identities"] = github_identities
        if repo_context.get("error"):
            repo_context["contribution_evidence"] = []
        else:
            repo_context["contribution_evidence"] = agent.fetch_user_commits_for_repo(
                repo, github_identities
            )
        repo_contexts.append(repo_context)

    path = agent.save_github_context_output(repo_contexts)
    return {
        "saved": agent.has_usable_repo_context(repo_contexts),
        "path": str(path),
        "context": repo_contexts,
    }


@app.get("/api/status")
def get_status():
    return {
        "provider": agent.current_provider,
        "model": agent.current_model,
        "supports_images": provider_supports_images(agent.current_provider),
        "provider_configs": build_provider_config_status()["providers"],
        "files": {name: file_ready(name, path) for name, path in FILE_MAP.items()},
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
    agent.current_provider = provider_name
    agent.current_adapter = adapter
    agent.current_model = adapter.default_model()
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
    agent.current_model = body.model.strip()
    if not agent.current_model:
        raise HTTPException(status_code=400, detail="Model name cannot be empty.")
    return {"provider": agent.current_provider, "model": agent.current_model}


@app.get("/api/files/{name}")
def get_file(name: str):
    if name not in FILE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown file: {name}")
    ready, content = read_file_content(name)
    return {"name": name, "ready": ready, "content": content}


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
        + original_resume_language_instruction_for_request(message),
        body.provider,
        body.model,
        images,
    )
    return {
        "answer": answer,
        "artifacts": {
            "analysis_path": None,
            "tailored_resume_path": str(agent.OUTPUT_RESUME_PATH)
            if agent.file_is_ready(agent.OUTPUT_RESUME_PATH)
            else None,
            "cover_letter_path": str(agent.COVER_LETTER_PATH)
            if agent.file_is_ready(agent.COVER_LETTER_PATH)
            else None,
        },
    }


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

    analysis_path = agent.save_analysis_output(analysis)
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
    prompt = (
        RESUME_TAILOR_PROMPT
        + output_language_instruction(body.language)
        + original_resume_language_instruction("tailored_resume")
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
    if body.use_github_context:
        prompt += "\nUse read_github_context if needed, but only with user approval via the web UI."
    answer = run_agent_task(prompt)
    if agent.looks_like_latex_resume(answer):
        agent.save_tailored_resume(answer)
    else:
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code.")
    tailored_resume_outputs = list_output_files(agent.TAILORED_RESUME_OUTPUT_DIR, ".txt", limit=1)
    response: dict[str, Any] = {
        "saved": True,
        "path": str(agent.OUTPUT_RESUME_PATH),
        "output_path": tailored_resume_outputs[0]["path"] if tailored_resume_outputs else None,
        "content": agent.read_tailored_resume(),
    }
    if body.include_application_hint:
        job_description = (
            agent.read_text_file(agent.JOB_DESCRIPTION_PATH)
            if agent.file_is_ready(agent.JOB_DESCRIPTION_PATH)
            else ""
        )
        response["application_hint"] = resolve_application_hint(job_description)
    return response


@app.post("/api/resume/update-memory")
def update_resume_memory(body: ResumeMemoryBody):
    try:
        return update_memory_from_resume_source(body.resume_source)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/resume/pdf-to-latex")
def resume_pdf_to_latex(body: ResumePdfToLatexBody):
    pdf_bytes = validate_resume_pdf(body)
    extracted = extract_pdf_resume_content(pdf_bytes)
    prompt = build_pdf_to_latex_prompt(body.filename, extracted, body.language)
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
    if content:
        try:
            agent.save_tailored_resume(content)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    else:
        try:
            content = agent.read_tailored_resume()
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    output_pdf = compile_tailored_resume_pdf(content)
    return {
        "saved": True,
        "path": str(output_pdf),
        "output_path": str(output_pdf),
    }


@app.post("/api/cover-letter/generate")
def generate_cover_letter(body: CoverLetterBody):
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
    cover_letter_mtime = (
        agent.COVER_LETTER_PATH.stat().st_mtime_ns
        if agent.COVER_LETTER_PATH.exists()
        else None
    )
    answer = run_agent_task(prompt)
    cover_letter_was_saved = (
        agent.COVER_LETTER_PATH.exists()
        and agent.COVER_LETTER_PATH.stat().st_mtime_ns != cover_letter_mtime
    )
    if answer.strip() and not cover_letter_was_saved:
        agent.save_cover_letter(answer)
    cover_letter_outputs = list_output_files(agent.COVER_LETTER_OUTPUT_DIR, ".txt", limit=1)
    return {
        "saved": True,
        "path": str(agent.COVER_LETTER_PATH),
        "output_path": cover_letter_outputs[0]["path"] if cover_letter_outputs else None,
        "content": agent.read_text_file(agent.COVER_LETTER_PATH)
        if agent.file_is_ready(agent.COVER_LETTER_PATH)
        else answer,
    }


@app.post("/api/interview-prep/generate")
def generate_interview_prep(body: InterviewPrepBody):
    prompt = build_interview_prep_prompt(body.use_github_context, body.language)
    answer = run_text_task(prompt)
    if not looks_like_interview_prep(answer):
        raise HTTPException(
            status_code=400,
            detail="Agent did not return usable interview preparation notes. Please regenerate after checking the job description and resume.",
        )
    agent.save_interview_prep(answer)
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
    try:
        repo_source = read_github_repo_source(body.resume_source)
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
    try:
        return fetch_github_context_api(body.approved, body.resume_source)
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
