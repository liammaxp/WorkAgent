"""FastAPI HTTP layer for WorkAgent frontend."""

from __future__ import annotations

import json
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
- Save the final notes with save_interview_prep.
"""

RESUME_TAILOR_PROMPT = """
Based on the saved job_description.txt, memory.json, resume.txt, and approved GitHub context if useful,
generate the modified complete LaTeX resume code.
Return only LaTeX code with no Markdown fences and no analysis text.
Save with save_tailored_resume when complete.
"""


class ProviderBody(BaseModel):
    provider: str


class ModelBody(BaseModel):
    model: str


class FileBody(BaseModel):
    content: str


class AgentAskBody(BaseModel):
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None


class JobDescriptionBody(BaseModel):
    content: str


class AnalyzeBody(BaseModel):
    use_github_context: bool = False


class TailorBody(BaseModel):
    use_github_context: bool = True


class CoverLetterBody(BaseModel):
    use_tailored_resume: bool = True
    use_github_context: bool = False
    style: str = "concise"


class InterviewPrepBody(BaseModel):
    use_github_context: bool = True


class GitHubScanBody(BaseModel):
    resume_source: str = "resume"


class GitHubContextBody(BaseModel):
    approved: bool = True


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


def get_adapter(provider: Optional[str] = None):
    name = (provider or agent.current_provider).lower().strip()
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
    path = FILE_MAP[name]
    if not path.exists():
        return False, ""
    content = path.read_text(encoding="utf-8")
    ready = agent.file_is_ready(path)
    return ready, content


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


def fetch_github_context_api(approved: bool) -> dict[str, Any]:
    if not approved:
        return {"saved": False, "message": "GitHub context fetch was not approved."}

    try:
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
        "files": {name: agent.file_is_ready(path) for name, path in FILE_MAP.items()},
        "outputs": {
            "analysis": list_output_files(agent.ANALYSIS_OUTPUT_DIR, ".txt"),
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


@app.post("/api/agent/ask")
def agent_ask(body: AgentAskBody):
    if body.provider:
        agent.current_provider = body.provider.lower().strip()
        agent.current_adapter = get_adapter(body.provider)[0]
    if body.model:
        agent.current_model = body.model.strip()

    answer = run_agent_task(body.message, body.provider, body.model)
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
    answer = run_agent_task(message)
    analysis_path = agent.save_analysis_output(answer)
    return {"analysis": answer, "analysis_path": str(analysis_path)}


@app.post("/api/resume/tailor")
def tailor_resume(body: TailorBody):
    prompt = RESUME_TAILOR_PROMPT
    if body.use_github_context:
        prompt += "\nUse read_github_context if needed, but only with user approval via the web UI."
    answer = run_agent_task(prompt)
    if agent.looks_like_latex_resume(answer):
        agent.save_tailored_resume(answer)
    else:
        raise HTTPException(status_code=400, detail="Agent did not return valid LaTeX resume code.")
    return {
        "saved": True,
        "path": str(agent.OUTPUT_RESUME_PATH),
        "content": agent.read_tailored_resume(),
    }


@app.post("/api/cover-letter/generate")
def generate_cover_letter(body: CoverLetterBody):
    style_hint = f"\nPreferred style: {body.style}."
    prompt = agent.COVER_LETTER_AGENT_PROMPT + style_hint
    if not body.use_tailored_resume:
        prompt += "\nUse resume.txt instead of tailored_resume.txt if the user requested it."
    if body.use_github_context:
        prompt += "\nYou may use GitHub context conservatively when it supports a specific claim."
    answer = run_agent_task(prompt)
    if answer.strip():
        agent.save_cover_letter(answer)
    return {
        "saved": True,
        "path": str(agent.COVER_LETTER_PATH),
        "content": agent.read_text_file(agent.COVER_LETTER_PATH)
        if agent.file_is_ready(agent.COVER_LETTER_PATH)
        else answer,
    }


@app.post("/api/interview-prep/generate")
def generate_interview_prep(body: InterviewPrepBody):
    prompt = INTERVIEW_PREP_PROMPT
    if body.use_github_context:
        prompt += "\nUse GitHub context conservatively when it helps explain project work."
    answer = run_agent_task(prompt)
    if answer.strip():
        agent.save_interview_prep(answer)
    return {
        "saved": True,
        "path": str(agent.INTERVIEW_PREP_PATH),
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


@app.post("/api/github/context")
def github_context(body: GitHubContextBody):
    try:
        return fetch_github_context_api(body.approved)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
