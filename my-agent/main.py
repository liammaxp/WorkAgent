from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import re
import sqlite3
from datetime import datetime
import urllib.error
import urllib.parse
import urllib.request


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
MEMORY_PATH = BASE_DIR / "memory.json"
RESUME_PATH = BASE_DIR / "resume.txt"
JOB_DESCRIPTION_PATH = BASE_DIR / "job_description.txt"
GITHUB_ACCOUNTS_PATH = BASE_DIR / "github_accounts.txt"
OUTPUT_DIR = BASE_DIR / "outputs"
ANALYSIS_OUTPUT_DIR = OUTPUT_DIR / "analysis"
OUTPUT_RESUME_PATH = BASE_DIR / "tailored_resume.txt"
COVER_LETTER_PATH = BASE_DIR / "cover_letter.txt"
INTERVIEW_PREP_PATH = BASE_DIR / "interview_prep.txt"
APPLICATION_DB_PATH = BASE_DIR / "applications.sqlite3"
PLACEHOLDER_TEXT = "Paste "
DEFAULT_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()
JOB_AGENT_PROMPT = """
Analyze the job description and produce:

1. Job title and company if available
2. Required technical skills
3. Required soft skills
4. Match score from 0 to 100
5. Best matching user projects
6. Resume summary rewrite
7. 3-5 resume bullet points
8. Cover letter draft

Use the available tools to read memory.json, resume.txt, and job_description.txt.
Use GitHub context only if needed and approved by the user.
Be specific and do not exaggerate the user's experience.
"""
GITHUB_REPO_PATTERN = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)
GITHUB_ACCOUNT_PATTERN = re.compile(
    r"(?:https?://(?:www\.)?github\.com/)?@?([A-Za-z0-9-]+)"
)
MAX_README_CHARS = 6000
MAX_COMMITS_PER_ACCOUNT = 20
MAX_COMMIT_DETAILS = 8
MAX_FALLBACK_COMMITS = 100


class ModelAdapter:
    provider_name = "base"

    def default_model(self):
        raise NotImplementedError

    def create_response(self, model, instructions, tools, input_items):
        raise NotImplementedError

    def get_function_calls(self, response):
        raise NotImplementedError

    def append_response_output(self, input_items, response):
        raise NotImplementedError

    def make_tool_output(self, call_id, output):
        raise NotImplementedError

    def output_text(self, response):
        raise NotImplementedError


class SimpleToolCall:
    def __init__(self, name, call_id, arguments="{}"):
        self.name = name
        self.call_id = call_id
        self.arguments = arguments or "{}"


class OpenAIResponsesAdapter(ModelAdapter):
    provider_name = "openai"

    def __init__(
        self,
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        model_env="OPENAI_MODEL",
        fallback_model="gpt-5.5",
    ):
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.model_env = model_env
        self.fallback_model = fallback_model

        client_options = {}
        api_key = os.getenv(api_key_env)
        base_url = os.getenv(base_url_env)
        if api_key:
            client_options["api_key"] = api_key
        if base_url:
            client_options["base_url"] = base_url

        self.client = OpenAI(**client_options)

    def default_model(self):
        return os.getenv(self.model_env, self.fallback_model)

    def create_response(self, model, instructions, tools, input_items):
        return self.client.responses.create(
            model=model,
            instructions=instructions,
            tools=tools,
            input=input_items,
        )

    def get_function_calls(self, response):
        return [
            item
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

    def append_response_output(self, input_items, response):
        input_items += response.output

    def make_tool_output(self, call_id, output):
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }

    def output_text(self, response):
        return response.output_text


class OpenAIChatCompletionsAdapter(ModelAdapter):
    provider_name = "openai-chat"

    def __init__(self, api_key_env, base_url_env, model_env, fallback_model):
        self.model_env = model_env
        self.fallback_model = fallback_model
        client_options = {}
        api_key = os.getenv(api_key_env)
        base_url = os.getenv(base_url_env)
        if api_key:
            client_options["api_key"] = api_key
        if base_url:
            client_options["base_url"] = base_url
        self.client = OpenAI(**client_options)
        self.messages = []

    def default_model(self):
        return os.getenv(self.model_env, self.fallback_model)

    def convert_tools(self, tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            for tool in tools
        ]

    def create_response(self, model, instructions, tools, input_items):
        if not self.messages:
            self.messages = [{"role": "system", "content": instructions}]
            self.messages.extend(input_items)
        return self.client.chat.completions.create(
            model=model,
            messages=self.messages,
            tools=self.convert_tools(tools),
        )

    def get_function_calls(self, response):
        message = response.choices[0].message
        return [
            SimpleToolCall(
                name=tool_call.function.name,
                call_id=tool_call.id,
                arguments=tool_call.function.arguments,
            )
            for tool_call in (message.tool_calls or [])
        ]

    def append_response_output(self, input_items, response):
        self.messages.append(response.choices[0].message)

    def make_tool_output(self, call_id, output):
        message = {"role": "tool", "tool_call_id": call_id, "content": output}
        self.messages.append(message)
        return message

    def output_text(self, response):
        return response.choices[0].message.content or ""


class OpenAICompatibleResponsesAdapter(OpenAIResponsesAdapter):
    provider_name = "openai-compatible"

    def __init__(self):
        super().__init__(
            api_key_env="OPENAI_COMPATIBLE_API_KEY",
            base_url_env="OPENAI_COMPATIBLE_BASE_URL",
            model_env="OPENAI_COMPATIBLE_MODEL",
            fallback_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        )


class DeepSeekAdapter(OpenAIChatCompletionsAdapter):
    provider_name = "deepseek"

    def __init__(self):
        super().__init__(
            api_key_env="DEEPSEEK_API_KEY",
            base_url_env="DEEPSEEK_BASE_URL",
            model_env="DEEPSEEK_MODEL",
            fallback_model="deepseek-chat",
        )
        if not os.getenv("DEEPSEEK_BASE_URL"):
            self.client.base_url = "https://api.deepseek.com"


class ClaudeMessagesAdapter(ModelAdapter):
    provider_name = "claude"

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        self.messages = []

    def default_model(self):
        return os.getenv("ANTHROPIC_MODEL", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5"))

    def convert_tools(self, tools):
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {}),
            }
            for tool in tools
        ]

    def post_json(self, path, body):
        if not self.api_key:
            raise ValueError("Missing ANTHROPIC_API_KEY or CLAUDE_API_KEY in .env.")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_response(self, model, instructions, tools, input_items):
        if not self.messages:
            self.messages.extend(input_items)
        return self.post_json(
            "/v1/messages",
            {
                "model": model,
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
                "system": instructions,
                "messages": self.messages,
                "tools": self.convert_tools(tools),
            },
        )

    def get_function_calls(self, response):
        calls = []
        for block in response.get("content", []):
            if block.get("type") == "tool_use":
                calls.append(
                    SimpleToolCall(
                        name=block.get("name"),
                        call_id=block.get("id"),
                        arguments=json.dumps(block.get("input", {})),
                    )
                )
        return calls

    def append_response_output(self, input_items, response):
        self.messages.append({"role": "assistant", "content": response.get("content", [])})

    def make_tool_output(self, call_id, output):
        message = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}],
        }
        self.messages.append(message)
        return message

    def output_text(self, response):
        return "\n".join(
            block.get("text", "")
            for block in response.get("content", [])
            if block.get("type") == "text"
        ).strip()


class GeminiAdapter(ModelAdapter):
    provider_name = "gemini"

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.base_url = os.getenv(
            "GEMINI_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta",
        )
        self.contents = []

    def default_model(self):
        return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def convert_tools(self, tools):
        return [
            {
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    }
                    for tool in tools
                ]
            }
        ]

    def post_json(self, model, body):
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY in .env.")
        url = (
            f"{self.base_url}/models/{urllib.parse.quote(model, safe='')}:"
            f"generateContent?key={urllib.parse.quote(self.api_key, safe='')}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def create_response(self, model, instructions, tools, input_items):
        if not self.contents:
            for item in input_items:
                self.contents.append(
                    {"role": "user", "parts": [{"text": item.get("content", "")}]}
                )
        return self.post_json(
            model,
            {
                "systemInstruction": {"parts": [{"text": instructions}]},
                "contents": self.contents,
                "tools": self.convert_tools(tools),
            },
        )

    def get_function_calls(self, response):
        calls = []
        candidate = (response.get("candidates") or [{}])[0]
        for part in candidate.get("content", {}).get("parts", []):
            function_call = part.get("functionCall")
            if function_call:
                name = function_call.get("name")
                calls.append(
                    SimpleToolCall(
                        name=name,
                        call_id=name,
                        arguments=json.dumps(function_call.get("args", {})),
                    )
                )
        return calls

    def append_response_output(self, input_items, response):
        candidate = (response.get("candidates") or [{}])[0]
        content = candidate.get("content")
        if content:
            self.contents.append(content)

    def make_tool_output(self, call_id, output):
        message = {
            "role": "user",
            "parts": [{"functionResponse": {"name": call_id, "response": {"result": output}}}],
        }
        self.contents.append(message)
        return message

    def output_text(self, response):
        candidate = (response.get("candidates") or [{}])[0]
        return "\n".join(
            part.get("text", "")
            for part in candidate.get("content", {}).get("parts", [])
            if "text" in part
        ).strip()


def create_model_adapter(provider_name):
    normalized = provider_name.lower().strip()
    if normalized == "openai":
        return OpenAIResponsesAdapter()
    if normalized in ["openai-compatible", "compatible"]:
        return OpenAICompatibleResponsesAdapter()
    if normalized == "deepseek":
        return DeepSeekAdapter()
    if normalized in ["claude", "anthropic"]:
        return ClaudeMessagesAdapter()
    if normalized in ["gemini", "google"]:
        return GeminiAdapter()

    raise ValueError(
        f"Unsupported provider '{provider_name}'. Supported providers: openai, openai-compatible, deepseek, claude, gemini."
    )


def read_text_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content or content.startswith(PLACEHOLDER_TEXT):
        raise ValueError(f"Please add real content to {path}")

    return content


def write_text_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def timestamp_slug():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_analysis_output(content, prefix="job_analysis"):
    path = ANALYSIS_OUTPUT_DIR / f"{prefix}_{timestamp_slug()}.txt"
    write_text_file(path, content)
    return path


def extract_latex_document(content):
    documentclass_index = content.find("\\documentclass")
    begin_index = content.find("\\begin{document}")

    if documentclass_index != -1:
        start_index = documentclass_index
    elif begin_index != -1:
        start_index = begin_index
    else:
        return ""

    latex = content[start_index:].strip()
    end_marker = "\\end{document}"
    end_index = latex.find(end_marker)
    if end_index != -1:
        latex = latex[: end_index + len(end_marker)]

    return latex.strip()


def is_likely_resume_edit_request(text):
    lowered = text.lower()
    resume_keywords = [
        "改简历",
        "修改简历",
        "生成简历",
        "tailor resume",
        "rewrite resume",
        "modify resume",
        "latex",
        "resume code",
    ]
    return any(keyword in lowered for keyword in resume_keywords)


def is_likely_job_description(text):
    lowered = text.lower()
    jd_keywords = [
        "responsibilities",
        "requirements",
        "qualifications",
        "job description",
        "about the role",
        "what you'll do",
        "what you will do",
        "required skills",
        "preferred qualifications",
        "software developer",
        "software engineer",
        "internship",
        "co-op",
        "岗位职责",
        "任职要求",
        "职位描述",
        "岗位要求",
        "资格要求",
        "实习",
        "软件开发",
    ]
    keyword_hits = sum(1 for keyword in jd_keywords if keyword in lowered)
    line_count = len([line for line in text.splitlines() if line.strip()])
    word_count = len(text.split())

    return not is_likely_resume_edit_request(text) and (
        keyword_hits >= 2 or line_count >= 8 or word_count >= 120
    )


def prepare_user_request(user_input):
    if not is_likely_job_description(user_input):
        return user_input, None, False

    write_text_file(JOB_DESCRIPTION_PATH, user_input)
    workflow_request = f"""
The user pasted a new job description. It has already been saved to job_description.txt.

{JOB_AGENT_PROMPT}
"""
    return workflow_request, f"Saved pasted job description to {JOB_DESCRIPTION_PATH}", True


def read_memory():
    if not MEMORY_PATH.exists():
        return json.dumps(
            {
                "note": "memory.json does not exist yet.",
                "expected_fields": [
                    "name",
                    "major",
                    "year",
                    "skills",
                    "projects",
                    "target_roles",
                ],
            },
            ensure_ascii=False,
        )

    content = MEMORY_PATH.read_text(encoding="utf-8").strip()
    if not content or content.startswith(PLACEHOLDER_TEXT):
        return json.dumps({"note": "memory.json is empty or still placeholder content."})

    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return content


def read_resume():
    return read_text_file(RESUME_PATH)


def read_job_description():
    return read_text_file(JOB_DESCRIPTION_PATH)


def save_tailored_resume(content):
    latex = extract_latex_document(content)
    if not latex:
        raise ValueError("No LaTeX resume code found. Refusing to write tailored_resume.txt.")

    write_text_file(OUTPUT_RESUME_PATH, latex)
    return f"Saved tailored resume to {OUTPUT_RESUME_PATH}"


def save_cover_letter(content):
    write_text_file(COVER_LETTER_PATH, content)
    return f"Saved cover letter to {COVER_LETTER_PATH}"


def save_interview_prep(content):
    write_text_file(INTERVIEW_PREP_PATH, content)
    return f"Saved interview preparation notes to {INTERVIEW_PREP_PATH}"


def initialize_application_db():
    with sqlite3.connect(APPLICATION_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                role TEXT NOT NULL,
                link TEXT,
                status TEXT,
                applied_date TEXT,
                resume_version TEXT,
                cover_letter_version TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def add_application_record(
    company,
    role,
    link="",
    status="Interested",
    applied_date="",
    resume_version="",
    cover_letter_version="",
    notes="",
):
    initialize_application_db()
    with sqlite3.connect(APPLICATION_DB_PATH) as connection:
        cursor = connection.execute(
            """
            INSERT INTO applications (
                company,
                role,
                link,
                status,
                applied_date,
                resume_version,
                cover_letter_version,
                notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company,
                role,
                link,
                status,
                applied_date,
                resume_version,
                cover_letter_version,
                notes,
            ),
        )

    return json.dumps(
        {
            "saved": True,
            "id": cursor.lastrowid,
            "database": str(APPLICATION_DB_PATH),
        },
        ensure_ascii=False,
    )


def list_application_records(status="", limit=20):
    initialize_application_db()
    query = """
        SELECT
            id,
            company,
            role,
            link,
            status,
            applied_date,
            resume_version,
            cover_letter_version,
            notes,
            created_at,
            updated_at
        FROM applications
    """
    params = []
    if status:
        query += " WHERE lower(status) = lower(?)"
        params.append(status)

    query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    params.append(limit)

    with sqlite3.connect(APPLICATION_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    return json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2)


def update_application_record(
    record_id,
    company=None,
    role=None,
    link=None,
    status=None,
    applied_date=None,
    resume_version=None,
    cover_letter_version=None,
    notes=None,
):
    initialize_application_db()
    updates = {
        "company": company,
        "role": role,
        "link": link,
        "status": status,
        "applied_date": applied_date,
        "resume_version": resume_version,
        "cover_letter_version": cover_letter_version,
        "notes": notes,
    }
    fields = [(key, value) for key, value in updates.items() if value is not None]
    if not fields:
        return json.dumps({"updated": False, "reason": "No fields provided."})

    assignments = ", ".join([f"{key} = ?" for key, _ in fields])
    values = [value for _, value in fields]
    values.append(record_id)

    with sqlite3.connect(APPLICATION_DB_PATH) as connection:
        cursor = connection.execute(
            f"""
            UPDATE applications
            SET {assignments}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            values,
        )

    return json.dumps(
        {
            "updated": cursor.rowcount > 0,
            "id": record_id,
        },
        ensure_ascii=False,
    )


def read_github_identities():
    if not GITHUB_ACCOUNTS_PATH.exists():
        return {
            "usernames": [],
            "author_names": [],
            "author_emails": [],
        }

    content = GITHUB_ACCOUNTS_PATH.read_text(encoding="utf-8").strip()
    if not content or content.startswith(PLACEHOLDER_TEXT):
        return {
            "usernames": [],
            "author_names": [],
            "author_emails": [],
        }

    identities = {
        "usernames": [],
        "author_names": [],
        "author_emails": [],
    }
    seen = {
        "usernames": set(),
        "author_names": set(),
        "author_emails": set(),
    }
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key = "usernames"
        value = line
        if ":" in line:
            raw_key, raw_value = line.split(":", 1)
            raw_key = raw_key.strip().lower()
            value = raw_value.strip()
            if raw_key in ["username", "user", "github", "github_username"]:
                key = "usernames"
            elif raw_key in ["name", "author", "author_name"]:
                key = "author_names"
            elif raw_key in ["email", "author_email"]:
                key = "author_emails"
            else:
                continue

        if key == "usernames":
            match = GITHUB_ACCOUNT_PATTERN.fullmatch(value.removesuffix("/"))
            if not match:
                continue

            value = match.group(1)

        normalized = value.lower()
        if normalized in seen[key]:
            continue

        seen[key].add(normalized)
        identities[key].append(value)

    return identities


def read_github_accounts():
    return read_github_identities()["usernames"]


def identity_has_values(github_identities):
    return any(github_identities.values())


def commit_matches_identity(commit, github_identities):
    author_login = (commit.get("author") or {}).get("login")
    committer_login = (commit.get("committer") or {}).get("login")
    commit_data = commit.get("commit", {})
    author_data = commit_data.get("author", {})
    committer_data = commit_data.get("committer", {})

    login_values = {
        value.lower()
        for value in [author_login, committer_login]
        if value
    }
    name_values = {
        value.lower()
        for value in [author_data.get("name"), committer_data.get("name")]
        if value
    }
    email_values = {
        value.lower()
        for value in [author_data.get("email"), committer_data.get("email")]
        if value
    }

    for username in github_identities["usernames"]:
        if username.lower() in login_values:
            return True

    for author_name in github_identities["author_names"]:
        if author_name.lower() in name_values:
            return True

    for author_email in github_identities["author_emails"]:
        if author_email.lower() in email_values:
            return True

    return False


def summarize_commit(commit):
    commit_data = commit.get("commit", {})
    author_data = commit_data.get("author", {})
    committer_data = commit_data.get("committer", {})

    return {
        "sha": commit.get("sha"),
        "message": commit_data.get("message", "").splitlines()[0],
        "date": author_data.get("date"),
        "author_name": author_data.get("name"),
        "author_email": author_data.get("email"),
        "committer_name": committer_data.get("name"),
        "committer_email": committer_data.get("email"),
        "github_author_login": (commit.get("author") or {}).get("login"),
        "github_committer_login": (commit.get("committer") or {}).get("login"),
        "files": [],
    }


def github_api_get(url, accept="application/vnd.github+json"):
    headers = {
        "Accept": accept,
        "User-Agent": "liam-job-application-agent",
    }
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    request = urllib.request.Request(
        url,
        headers=headers,
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def github_token_is_configured():
    return bool(os.getenv("GITHUB_TOKEN"))


def print_github_token_status():
    if not github_token_is_configured():
        print("Agent: No GITHUB_TOKEN was loaded from my-agent/.env.")
        return

    try:
        user_data = json.loads(github_api_get("https://api.github.com/user"))
        print(f"Agent: GitHub token loaded. Authenticated as {user_data.get('login')}.")
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        if isinstance(error, urllib.error.HTTPError):
            print(f"Agent: GitHub token check failed: {describe_http_error(error)}")
        else:
            print(f"Agent: GitHub token check failed: {error}")


def describe_http_error(error):
    status = getattr(error, "code", None)
    reason = getattr(error, "reason", "")

    if status == 403:
        return (
            "HTTP Error 403: Forbidden. This usually means the GitHub token is "
            "missing, expired, lacks repository access, was not granted access to "
            "this organization/classroom repository, or the API rate limit was hit."
        )

    if status == 404:
        return (
            "HTTP Error 404: Not Found. The repository may be private, renamed, "
            "deleted, misspelled, or hidden from the current GitHub token. For "
            "private organization/classroom repositories, GitHub often returns 404 "
            "when the token cannot see the repo."
        )

    if status:
        return f"HTTP Error {status}: {reason}"

    return str(error)


def extract_github_repos(text):
    repos = []
    seen = set()

    for owner, repo in GITHUB_REPO_PATTERN.findall(text):
        repo = repo.removesuffix(".git")
        key = (owner, repo)
        if key in seen:
            continue

        seen.add(key)
        repos.append(
            {
                "owner": owner,
                "repo": repo,
                "url": f"https://github.com/{owner}/{repo}",
            }
        )

    return repos


def fetch_commit_files(base_url, commit_context):
    sha = commit_context.get("sha")
    if not sha:
        return commit_context

    try:
        details = json.loads(github_api_get(f"{base_url}/commits/{sha}"))
        commit_context["files"] = [
            file_info.get("filename")
            for file_info in details.get("files", [])
            if file_info.get("filename")
        ][:20]
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        commit_context["files_error"] = (
            describe_http_error(error)
            if isinstance(error, urllib.error.HTTPError)
            else str(error)
        )

    return commit_context


def fetch_fallback_commits_for_repo(base_url, github_identities):
    fallback_context = {
        "method": "recent_commits_identity_match",
        "commit_count_checked": 0,
        "commits": [],
    }

    try:
        commits = json.loads(
            github_api_get(f"{base_url}/commits?per_page={MAX_FALLBACK_COMMITS}")
        )
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        fallback_context["error"] = (
            describe_http_error(error)
            if isinstance(error, urllib.error.HTTPError)
            else str(error)
        )
        return fallback_context

    if not isinstance(commits, list):
        fallback_context["error"] = "Unexpected commits API response."
        return fallback_context

    fallback_context["commit_count_checked"] = len(commits)

    for commit in commits:
        if not commit_matches_identity(commit, github_identities):
            continue

        commit_context = summarize_commit(commit)
        fallback_context["commits"].append(
            fetch_commit_files(base_url, commit_context)
        )

        if len(fallback_context["commits"]) >= MAX_COMMIT_DETAILS:
            break

    return fallback_context


def fetch_user_commits_for_repo(repo_info, github_identities):
    owner = urllib.parse.quote(repo_info["owner"], safe="")
    repo = urllib.parse.quote(repo_info["repo"], safe="")
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    contributions = []

    for account in github_identities["usernames"]:
        author = urllib.parse.quote(account, safe="")
        commits_url = (
            f"{base_url}/commits?author={author}&per_page={MAX_COMMITS_PER_ACCOUNT}"
        )

        account_context = {
            "github_account": account,
            "method": "github_author_login",
            "commit_count_checked": 0,
            "commits": [],
        }

        try:
            commits = json.loads(github_api_get(commits_url))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            account_context["error"] = (
                describe_http_error(error)
                if isinstance(error, urllib.error.HTTPError)
                else str(error)
            )
            contributions.append(account_context)
            continue

        if not isinstance(commits, list):
            account_context["error"] = "Unexpected commits API response."
            contributions.append(account_context)
            continue

        account_context["commit_count_checked"] = len(commits)

        for commit in commits[:MAX_COMMIT_DETAILS]:
            account_context["commits"].append(
                fetch_commit_files(base_url, summarize_commit(commit))
            )

        contributions.append(account_context)

    if (
        github_identities["author_names"]
        or github_identities["author_emails"]
        or not any(context.get("commits") for context in contributions)
    ):
        contributions.append(fetch_fallback_commits_for_repo(base_url, github_identities))

    return contributions


def fetch_github_repo_context(repo_info):
    owner = urllib.parse.quote(repo_info["owner"], safe="")
    repo = urllib.parse.quote(repo_info["repo"], safe="")
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    context = {
        "url": repo_info["url"],
        "repository": f"{repo_info['owner']}/{repo_info['repo']}",
    }

    try:
        repo_data = json.loads(github_api_get(base_url))
        context["description"] = repo_data.get("description")
        context["homepage"] = repo_data.get("homepage")
        context["topics"] = repo_data.get("topics", [])
        context["default_branch"] = repo_data.get("default_branch")
        context["stars"] = repo_data.get("stargazers_count")
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        context["error"] = (
            f"Could not fetch repository metadata: {describe_http_error(error)}"
            if isinstance(error, urllib.error.HTTPError)
            else f"Could not fetch repository metadata: {error}"
        )
        return context

    try:
        languages = json.loads(github_api_get(f"{base_url}/languages"))
        context["languages"] = list(languages.keys())
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        context["languages_error"] = (
            describe_http_error(error)
            if isinstance(error, urllib.error.HTTPError)
            else str(error)
        )

    try:
        root_files = json.loads(github_api_get(f"{base_url}/contents"))
        context["root_files"] = [
            item.get("name")
            for item in root_files
            if isinstance(item, dict) and item.get("name")
        ][:40]
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as error:
        context["root_files_error"] = (
            describe_http_error(error)
            if isinstance(error, urllib.error.HTTPError)
            else str(error)
        )

    try:
        readme = github_api_get(f"{base_url}/readme", accept="application/vnd.github.raw")
        context["readme"] = readme[:MAX_README_CHARS]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        context["readme_error"] = (
            describe_http_error(error)
            if isinstance(error, urllib.error.HTTPError)
            else str(error)
        )

    return context


def print_github_context_summary(repo_contexts):
    print("Agent: GitHub context summary:")
    for index, context in enumerate(repo_contexts, start=1):
        print(f"{index}. {context.get('repository', context.get('url'))}")

        if context.get("error"):
            print(f"   Error: {context['error']}")
            continue

        description = context.get("description") or "No repository description."
        languages = ", ".join(context.get("languages", [])) or "No language data."
        root_files = ", ".join(context.get("root_files", [])[:8]) or "No root files listed."
        readme_status = "README found." if context.get("readme") else "No README fetched."

        print(f"   Description: {description}")
        print(f"   Languages: {languages}")
        print(f"   Root files: {root_files}")
        print(f"   {readme_status}")

        contribution_evidence = context.get("contribution_evidence", [])
        if contribution_evidence:
            for evidence in contribution_evidence:
                commits = evidence.get("commits", [])
                identity = evidence.get("github_account") or evidence.get("method")
                method = evidence.get("method", "unknown")
                checked = evidence.get("commit_count_checked", 0)
                print(
                    f"   {identity}: {len(commits)} matched commit(s), "
                    f"{checked} checked via {method}."
                )
                for commit in commits[:3]:
                    files = ", ".join(commit.get("files", [])[:3]) or "No files listed."
                    print(f"      - {commit.get('message')} | {files}")


def has_usable_repo_context(repo_contexts):
    return any(not context.get("error") for context in repo_contexts)


def build_github_context(resume):
    repos = extract_github_repos(resume)
    if not repos:
        return ""

    github_identities = read_github_identities()
    if not identity_has_values(github_identities):
        print(
            f"\nAgent: I found GitHub repositories, but no GitHub accounts in {GITHUB_ACCOUNTS_PATH}."
        )
        print(
            "Agent: Add your GitHub username, commit author name, or commit email there first so I can verify which commits are yours.\n"
        )
        return ""

    print("\nAgent: I found these GitHub repositories in your resume:")
    for index, repo in enumerate(repos, start=1):
        print(f"{index}. {repo['url']}")
    print("\nAgent: I will verify contribution evidence using these identities:")
    for username in github_identities["usernames"]:
        print(f"- GitHub username: {username}")
    for author_name in github_identities["author_names"]:
        print(f"- Commit author name: {author_name}")
    for author_email in github_identities["author_emails"]:
        print(f"- Commit author email: {author_email}")

    permission = input(
        "\nAllow me to access these public GitHub repositories and commit history before modifying the resume? "
        "Type yes to allow, or anything else to skip: "
    )
    if permission.strip().lower() not in ["y", "yes"]:
        print("\nAgent: Skipping GitHub repository lookup.\n")
        return ""

    print("\nAgent: Fetching public GitHub repository context...\n")
    print_github_token_status()
    repo_contexts = []
    for repo in repos:
        repo_context = fetch_github_repo_context(repo)
        repo_context["verified_github_identities"] = github_identities
        repo_context["contribution_evidence"] = fetch_user_commits_for_repo(
            repo, github_identities
        )
        repo_contexts.append(repo_context)
    print_github_context_summary(repo_contexts)

    if not has_usable_repo_context(repo_contexts):
        print(
            "\nAgent: No GitHub repositories were readable, so GitHub context will be skipped."
        )
        print(
            "Agent: If these are private/classroom repositories, add a valid GITHUB_TOKEN "
            "with access to them in my-agent/.env.\n"
        )
        return ""

    permission = input(
        "\nUse this GitHub context to modify the resume for this request? "
        "Type yes to allow, or anything else to ignore it: "
    )
    if permission.strip().lower() not in ["y", "yes"]:
        print("\nAgent: GitHub context will not be used for this modification.\n")
        return ""

    return json.dumps(repo_contexts, ensure_ascii=False, indent=2)


TOOLS = [
    {
        "type": "function",
        "name": "read_memory",
        "description": "Read the user's long-term profile from memory.json.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_resume",
        "description": "Read the user's current resume LaTeX code from resume.txt.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_job_description",
        "description": "Read the target job description from job_description.txt.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_github_context",
        "description": (
            "Find GitHub repository links in resume.txt, ask the user for permission, "
            "then read public/authorized repository and commit evidence when available."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "save_tailored_resume",
        "description": "Save only complete modified resume LaTeX code to tailored_resume.txt. Do not include analysis or explanation text.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Complete LaTeX resume code to save.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "save_cover_letter",
        "description": "Save a generated cover letter draft to cover_letter.txt.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Cover letter content to save.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "save_interview_prep",
        "description": "Save generated interview preparation notes to interview_prep.txt.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Interview preparation notes to save.",
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "add_application_record",
        "description": "Add a job application record to the local SQLite database.",
        "parameters": {
            "type": "object",
            "properties": {
                "company": {"type": "string"},
                "role": {"type": "string"},
                "link": {"type": "string"},
                "status": {"type": "string"},
                "applied_date": {"type": "string"},
                "resume_version": {"type": "string"},
                "cover_letter_version": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["company", "role"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_application_records",
        "description": "List job application records, optionally filtered by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Optional status filter such as Applied or Interested.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of records to return.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_application_record",
        "description": "Update fields on a job application record by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "record_id": {"type": "integer"},
                "company": {"type": "string"},
                "role": {"type": "string"},
                "link": {"type": "string"},
                "status": {"type": "string"},
                "applied_date": {"type": "string"},
                "resume_version": {"type": "string"},
                "cover_letter_version": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
    },
]


def read_github_context():
    resume = read_resume()
    github_context = build_github_context(resume)
    if not github_context:
        return "No usable GitHub context was provided or approved."

    return github_context


TOOL_FUNCTIONS = {
    "read_memory": read_memory,
    "read_resume": read_resume,
    "read_job_description": read_job_description,
    "read_github_context": read_github_context,
    "save_tailored_resume": save_tailored_resume,
    "save_cover_letter": save_cover_letter,
    "save_interview_prep": save_interview_prep,
    "add_application_record": add_application_record,
    "list_application_records": list_application_records,
    "update_application_record": update_application_record,
}


def execute_tool_call(tool_call, adapter):
    tool_name = getattr(tool_call, "name", None)
    call_id = getattr(tool_call, "call_id", None)
    raw_arguments = getattr(tool_call, "arguments", "{}") or "{}"

    if tool_name not in TOOL_FUNCTIONS:
        output = json.dumps({"error": f"Unknown tool: {tool_name}"})
    else:
        try:
            arguments = json.loads(raw_arguments)
            result = TOOL_FUNCTIONS[tool_name](**arguments)
            output = result if isinstance(result, str) else json.dumps(result)
        except Exception as error:
            output = json.dumps({"error": str(error)})

    return adapter.make_tool_output(call_id, output)


def ask_agent(user_input, adapter, model):
    input_items = [
        {
            "role": "user",
            "content": f"""
User request:
{user_input}

You have tools for reading memory.json, resume.txt, job_description.txt, approved GitHub project context, saving generated files, and managing application records.
For resume tailoring, cover letters, interview prep, job matching, and application tracking, call the tools you need instead of assuming local file contents.
When saving an artifact is useful, call the matching save tool.
If the user asks for a modified resume, generate only complete LaTeX code with no Markdown fences and no analysis text.
Keep job analysis, match scores, recommendations, and explanations separate from resume LaTeX code.
""",
        }
    ]

    for _ in range(6):
        response = adapter.create_response(
            model=model,
            instructions=SYSTEM_PROMPT,
            tools=TOOLS,
            input_items=input_items,
        )

        function_calls = adapter.get_function_calls(response)
        if not function_calls:
            return adapter.output_text(response)

        adapter.append_response_output(input_items, response)
        for tool_call in function_calls:
            input_items.append(execute_tool_call(tool_call, adapter))

    raise RuntimeError("Tool calling loop exceeded the maximum number of steps.")


def looks_like_latex_resume(content):
    return bool(extract_latex_document(content))


current_provider = DEFAULT_PROVIDER
try:
    current_adapter = create_model_adapter(current_provider)
except ValueError as error:
    print(f"Agent: {error}")
    current_provider = "openai"
    current_adapter = create_model_adapter(current_provider)

current_model = current_adapter.default_model()
print(f"Agent: Current provider is {current_provider}")
print(f"Agent: Current model is {current_model}")
print("Agent: Type 'provider PROVIDER_NAME' to switch providers.")
print("Agent: Type 'model MODEL_NAME' to switch models, for example: model gpt-5.4-mini")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    if user_input.lower() == "model":
        print(f"\nAgent: Current model is {current_model}\n")
        continue

    if user_input.lower() == "provider":
        print(f"\nAgent: Current provider is {current_provider}\n")
        print("Agent: Supported providers: openai, openai-compatible, deepseek, claude, gemini\n")
        continue

    if user_input.lower().startswith("model "):
        requested_model = user_input.split(maxsplit=1)[1].strip()
        if not requested_model:
            print(f"\nAgent: Current model is {current_model}\n")
            continue

        current_model = requested_model
        print(f"\nAgent: Model switched to {current_model}\n")
        continue

    if user_input.lower().startswith("provider "):
        requested_provider = user_input.split(maxsplit=1)[1].strip().lower()
        try:
            requested_adapter = create_model_adapter(requested_provider)
        except ValueError as error:
            print(f"\nAgent: {error}\n")
            continue

        current_provider = requested_provider
        current_adapter = requested_adapter
        current_model = current_adapter.default_model()
        print(f"\nAgent: Provider switched to {current_provider}\n")
        print(f"Agent: Current model is {current_model}\n")
        continue

    try:
        prepared_request, preparation_message, is_new_job_description = prepare_user_request(user_input)
        if preparation_message:
            print(f"\nAgent: {preparation_message}\n")

        answer = ask_agent(
            prepared_request,
            adapter=current_adapter,
            model=current_model,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"\nAgent: {error}\n")
        continue

    if looks_like_latex_resume(answer):
        save_tailored_resume(answer)
        print(f"\nAgent: Updated resume LaTeX saved to {OUTPUT_RESUME_PATH}\n")
    else:
        analysis_path = save_analysis_output(answer)
        print(f"\nAgent: Analysis saved to {analysis_path}\n")
        print(f"Agent: {answer}\n")

    if is_new_job_description:
        permission = input(
            "Agent: Do you want me to generate the modified full LaTeX resume code now? "
            "Type yes to generate, or anything else to skip: "
        )
        if permission.strip().lower() in ["y", "yes"]:
            try:
                resume_answer = ask_agent(
                    "Based on the saved job_description.txt, memory.json, resume.txt, and approved GitHub context if useful, generate the modified complete LaTeX resume code. Return only LaTeX code with no Markdown fences.",
                    adapter=current_adapter,
                    model=current_model,
                )
            except (FileNotFoundError, ValueError) as error:
                print(f"\nAgent: {error}\n")
                continue

            save_tailored_resume(resume_answer)
            print(f"\nAgent: Modified resume LaTeX saved to {OUTPUT_RESUME_PATH}\n")
