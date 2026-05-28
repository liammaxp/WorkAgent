"""FastAPI HTTP layer for WorkAgent frontend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
- Read job_description.txt, tailored_resume.txt (fallback to resume.txt), and memory.json.
- Include likely technical questions, behavioral/STAR prompts, project talking points, and gaps to prepare for.
- Keep claims grounded in the resume and job description.
- Return the complete interview preparation notes directly.
"""

RESUME_TAILOR_PROMPT = """
Based on the saved job_description.txt, memory.json, resume.txt, and approved GitHub context if useful,
generate the modified complete LaTeX resume code.
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


class AgentAskBody(BaseModel):
    message: str
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
    language: str = "zh"


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
        "default_model": "deepseek-chat",
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


def read_file_content(name: str) -> tuple[bool, str]:
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
    if name == "tailored_resume":
        return agent.file_is_ready(path) or agent.file_is_ready(agent.LEGACY_OUTPUT_RESUME_PATH)
    return agent.file_is_ready(path)


def save_file_content(name: str, content: str) -> None:
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


def run_agent_task(message: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    adapter, _ = get_adapter(provider)
    chosen_model = model or adapter.default_model()
    try:
        return agent.ask_agent(message, adapter=adapter, model=chosen_model)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
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
    except agent.transient_network_errors() as error:
        raise HTTPException(status_code=502, detail=f"Network error: {error}") from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


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

    prompt = f"""
Create complete interview preparation notes for the job application below.

Rules:
- Use only the job description, resume, memory, and approved GitHub context provided here.
- Do not invent projects, employers, degrees, technologies, metrics, or repository facts.
- If evidence is weak or missing, say what to prepare or verify instead of fabricating.
- Return only the notes content. Do not say that you saved a file. Do not include placeholders.
- Write in concise Chinese unless the job description strongly implies English-only preparation.
- Include these sections:
  1. 职位重点
  2. 技术问题准备
  3. 项目讲述要点
  4. 行为面试 / STAR 素材
  5. 需要补强或确认的内容
  6. 反问面试官的问题

Job description:
{job_description}

Resume source: {resume_source}
Resume:
{resume}

Memory:
{memory}
{github_section}
"""
    return prompt + output_language_instruction(language)


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


def read_approved_github_context() -> str:
    outputs = list_output_files(agent.GITHUB_CONTEXT_OUTPUT_DIR, ".json", limit=1)
    if not outputs:
        return "No approved GitHub context is available. Ask the user to approve GitHub access in the web UI first."

    path = Path(outputs[0]["path"])
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"Approved GitHub context could not be read: {error}"

    if not agent.has_usable_repo_context(context):
        return "No usable approved GitHub context is available."

    return json.dumps(context, ensure_ascii=False, indent=2)


agent.TOOL_FUNCTIONS["read_github_context"] = read_approved_github_context


def fetch_github_context_api(approved: bool, resume_source: str = "resume") -> dict[str, Any]:
    if not approved:
        return {"saved": False, "message": "GitHub context fetch was not approved."}

    try:
        if resume_source == "tailored_resume":
            resume = agent.read_tailored_resume()
        else:
            resume = agent.read_resume()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    repos = agent.extract_github_repos(resume)
    if not repos:
        return {"saved": False, "message": "No GitHub repositories found in resume."}

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
        "provider_configs": build_provider_config_status()["providers"],
        "files": {name: file_ready(name, path) for name, path in FILE_MAP.items()},
        "outputs": {
            "analysis": list_output_files(agent.ANALYSIS_OUTPUT_DIR, ".txt"),
            "tailored_resumes": list_output_files(agent.TAILORED_RESUME_OUTPUT_DIR, ".txt"),
            "cover_letters": list_output_files(agent.COVER_LETTER_OUTPUT_DIR, ".txt"),
            "interview_prep": list_output_files(agent.INTERVIEW_PREP_OUTPUT_DIR, ".txt"),
            "github_context": list_output_files(agent.GITHUB_CONTEXT_OUTPUT_DIR, ".json"),
        },
    }


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

    answer = run_agent_task(
        body.message + output_language_instruction(body.language),
        body.provider,
        body.model,
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
    agent.write_text_file(agent.JOB_DESCRIPTION_PATH, body.content)
    return {"saved": True, "path": str(agent.JOB_DESCRIPTION_PATH)}


@app.post("/api/job-description/analyze")
def analyze_job_description(body: AnalyzeBody):
    message = agent.JOB_AGENT_PROMPT
    if body.use_github_context:
        message += "\nUse GitHub context if available and approved."
    message += output_language_instruction(body.language)
    answer = run_agent_task(message)
    analysis_path = agent.save_analysis_output(answer)
    return {"analysis": answer, "analysis_path": str(analysis_path)}


@app.post("/api/resume/tailor")
def tailor_resume(body: TailorBody):
    prompt = RESUME_TAILOR_PROMPT + output_language_instruction(body.language)
    if body.use_github_context:
        prompt += "\nUse read_github_context if needed, but only with user approval via the web UI."
    answer = run_agent_task(prompt)
    if agent.looks_like_latex_resume(answer):
        agent.save_tailored_resume(answer)
    else:
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code.")
    tailored_resume_outputs = list_output_files(agent.TAILORED_RESUME_OUTPUT_DIR, ".txt", limit=1)
    return {
        "saved": True,
        "path": str(agent.OUTPUT_RESUME_PATH),
        "output_path": tailored_resume_outputs[0]["path"] if tailored_resume_outputs else None,
        "content": agent.read_tailored_resume(),
    }


@app.post("/api/cover-letter/generate")
def generate_cover_letter(body: CoverLetterBody):
    style_hint = f"\nPreferred style: {body.style}."
    prompt = agent.COVER_LETTER_AGENT_PROMPT + style_hint + output_language_instruction(body.language)
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
        if body.resume_source == "tailored_resume":
            resume = agent.read_tailored_resume()
        else:
            resume = agent.read_resume()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    repos = agent.extract_github_repos(resume)
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
