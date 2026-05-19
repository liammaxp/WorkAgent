from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request


load_dotenv()

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").strip()
RESUME_PATH = BASE_DIR / "resume.txt"
JOB_DESCRIPTION_PATH = BASE_DIR / "job_description.txt"
GITHUB_ACCOUNTS_PATH = BASE_DIR / "github_accounts.txt"
OUTPUT_RESUME_PATH = BASE_DIR / "tailored_resume.txt"
PLACEHOLDER_TEXT = "Paste "
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


def read_text_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content or content.startswith(PLACEHOLDER_TEXT):
        raise ValueError(f"Please add real content to {path}")

    return content


def write_text_file(path, content):
    path.write_text(content.strip() + "\n", encoding="utf-8")


def read_resume():
    return read_text_file(RESUME_PATH)


def read_job_description():
    return read_text_file(JOB_DESCRIPTION_PATH)


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
        commit_context["files_error"] = str(error)

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
        fallback_context["error"] = str(error)
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
            account_context["error"] = str(error)
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
        context["error"] = f"Could not fetch repository metadata: {error}"
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
        context["languages_error"] = str(error)

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
        context["root_files_error"] = str(error)

    try:
        readme = github_api_get(f"{base_url}/readme", accept="application/vnd.github.raw")
        context["readme"] = readme[:MAX_README_CHARS]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        context["readme_error"] = str(error)

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
    repo_contexts = []
    for repo in repos:
        repo_context = fetch_github_repo_context(repo)
        repo_context["verified_github_identities"] = github_identities
        repo_context["contribution_evidence"] = fetch_user_commits_for_repo(
            repo, github_identities
        )
        repo_contexts.append(repo_context)
    print_github_context_summary(repo_contexts)

    permission = input(
        "\nUse this GitHub context to modify the resume for this request? "
        "Type yes to allow, or anything else to ignore it: "
    )
    if permission.strip().lower() not in ["y", "yes"]:
        print("\nAgent: GitHub context will not be used for this modification.\n")
        return ""

    return json.dumps(repo_contexts, ensure_ascii=False, indent=2)


def ask_agent(user_input, resume, job_description, github_context=""):
    github_section = ""
    if github_context:
        github_section = f"""
Here is additional public GitHub repository context.
The user explicitly approved fetching this information:

{github_context}
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"""
Here is my current resume LaTeX code from resume.txt:

{resume}

Here is the job description from job_description.txt:

{job_description}

{github_section}

User request:
{user_input}

Please generate the modified resume as complete LaTeX code.
Return only the final LaTeX code, with no Markdown fences and no extra explanation.
""",
            },
        ],
    )
    return response.output_text


while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        resume = read_resume()
        job_description = read_job_description()
        github_context = build_github_context(resume)
        answer = ask_agent(user_input, resume, job_description, github_context)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nAgent: {error}\n")
        continue

    write_text_file(OUTPUT_RESUME_PATH, answer)
    print(f"\nAgent: Updated resume LaTeX saved to {OUTPUT_RESUME_PATH}\n")
