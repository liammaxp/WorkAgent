import { useEffect, useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  ConfirmDialog,
  LoadingBar,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

function listToText(values = []) {
  return values.join("\n");
}

function textToList(value) {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatMemoryUpdatedAt(value, language) {
  const match = value?.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!match) return value || "-";
  const [, year, month, day, hour, minute, second] = match;
  return new Intl.DateTimeFormat(language === "en" ? "en" : "zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`));
}

function formatUnixUpdatedAt(value, language) {
  if (!value) return "-";
  return new Intl.DateTimeFormat(language === "en" ? "en" : "zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(value * 1000));
}

function resolveProjectMemoryUpdatedAt(githubConfig, status) {
  return (
    githubConfig?.project_memory_updated_at ||
    status?.file_metadata?.project_memory?.mtime ||
    null
  );
}

function normalizeProjectAlias(value) {
  let textValue = String(value || "").trim().toLowerCase();
  const githubMatch = textValue.match(/https?:\/\/(?:www\.)?github\.com\/([a-z0-9_.-]+)\/([a-z0-9_.-]+)/);
  if (githubMatch) textValue = `${githubMatch[1]}/${githubMatch[2]}`;
  textValue = textValue.replace(/^https?:\/\/(?:www\.)?github\.com\//, "");
  const repoMatch = textValue.match(/^([a-z0-9_.-]+)\/([a-z0-9_.-]+)$/);
  if (repoMatch) textValue = repoMatch[2];
  textValue = textValue.replace(/\.git$/, "");
  return textValue.replace(/[^a-z0-9]+/g, "");
}

function appendProjectOption(options, usedKeys, aliases, preferredValue) {
  const aliasValues = aliases.map((alias) => String(alias || "").trim()).filter(Boolean);
  if (!aliasValues.length) return;
  const keys = Array.from(new Set(aliasValues.map(normalizeProjectAlias).filter(Boolean)));
  if (!keys.length) return;
  const existing = options.find((option) => option.keys.some((key) => keys.includes(key)));
  if (existing) {
    keys.forEach((key) => {
      if (!usedKeys.has(key)) existing.keys.push(key);
      usedKeys.add(key);
    });
    return;
  }
  const label = String(preferredValue || aliasValues[0]).trim();
  options.push({ label, keys });
  keys.forEach((key) => usedKeys.add(key));
}

function collectProjectOptions(projectMemory) {
  const options = [];
  const usedKeys = new Set();
  const rawProjects = projectMemory?.projects;
  const projects = Array.isArray(rawProjects)
    ? rawProjects
    : rawProjects && typeof rawProjects === "object"
      ? [rawProjects]
      : [];

  projects.forEach((project) => {
    if (!project || typeof project !== "object") return;
    const aliases = [
      project.project_name,
      project.project_id,
      project.name,
      project.title,
      project.repository,
    ];

    const identity = project.identity;
    if (identity && typeof identity === "object") {
      aliases.push(identity.project_name, identity.project_id, identity.name);
    }

    const evidenceNotes = String(project.evidence_notes || "");
    evidenceNotes.replace(
      /(?:Repository|repository|repo)\s*:\s*([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)/g,
      (_, repository) => {
        aliases.push(repository.replace(/\.git$/, ""));
        return "";
      },
    );

    appendProjectOption(
      options,
      usedKeys,
      aliases,
      project.project_name || project.name || project.title || project.project_id || project.repository,
    );
  });

  return options.sort((left, right) => left.label.localeCompare(right.label));
}

export default function GitHubContext() {
  const { language } = useLanguage();
  const copy = text[language].github;
  const [scan, setScan] = useState(null);
  const [context, setContext] = useState(null);
  const [source, setSource] = useState("tailored_resume_and_resume_and_memory");
  const [projectScope, setProjectScope] = useState("");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [githubForm, setGithubForm] = useState({
    usernames: "",
    author_names: "",
    author_emails: "",
    token: "",
  });
  const [tokenConfigured, setTokenConfigured] = useState(false);
  const [memoryRepositories, setMemoryRepositories] = useState([]);
  const [projectOptions, setProjectOptions] = useState([]);
  const [projectMemoryUpdatedAt, setProjectMemoryUpdatedAt] = useState(null);
  const { loading, error, success, run } = useAsyncAction();

  const loadGithubConfig = () =>
    run(async () => {
      const [data, status, projectMemoryFile] = await Promise.all([
        api.getGithubConfig(),
        api.getStatus(),
        api.getFile("project_memory").catch(() => ({ content: "" })),
      ]);
      setGithubForm((current) => ({
        usernames: listToText(data.identities?.usernames),
        author_names: listToText(data.identities?.author_names),
        author_emails: listToText(data.identities?.author_emails),
        token: current.token,
      }));
      setTokenConfigured(data.token_configured);
      setMemoryRepositories(data.memory_repositories || []);
      try {
        setProjectOptions(collectProjectOptions(JSON.parse(projectMemoryFile.content || "{}")));
      } catch {
        setProjectOptions([]);
      }
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(data, status));
      return data;
    });

  useEffect(() => {
    loadGithubConfig();
  }, []);

  const saveGithubConfig = () =>
    run(async () => {
      const data = await api.saveGithubConfig({
        usernames: textToList(githubForm.usernames),
        author_names: textToList(githubForm.author_names),
        author_emails: textToList(githubForm.author_emails),
        token: githubForm.token,
      });
      setGithubForm((current) => ({ ...current, token: "" }));
      setTokenConfigured(data.token_configured);
      setMemoryRepositories(data.memory_repositories || []);
      const status = await api.getStatus();
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(data, status));
      setScan((current) =>
        current
          ? { ...current, identities: data.identities, token_configured: data.token_configured }
          : current
      );
      return data;
    }, copy.saved);

  const scanRepos = () =>
    run(async () => {
      const data = await api.scanGithub(source, {
        project_name: projectScope.trim(),
      });
      setScan(data);
      setContext(null);
      setTokenConfigured(data.token_configured);
      return data;
    }, copy.scanned);

  const approveFetchContext = () =>
    run(async () => {
      const data = await api.fetchGithubContext(true, source, {
        project_name: projectScope.trim(),
      });
      const [githubConfig, status] = await Promise.all([api.getGithubConfig(), api.getStatus()]);
      setContext(data);
      setMemoryRepositories(githubConfig.memory_repositories || []);
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(githubConfig, status));
      setConfirmOpen(false);
      return data;
    }, copy.fetched);

  const identities = scan?.identities || {
    usernames: textToList(githubForm.usernames),
    author_names: textToList(githubForm.author_names),
    author_emails: textToList(githubForm.author_emails),
  };
  const identityItems = [
    ...(identities.usernames || []).map((value) => `GitHub username: ${value}`),
    ...(identities.author_names || []).map((value) => `Commit author name: ${value}`),
    ...(identities.author_emails || []).map((value) => `Commit author email: ${value}`),
  ];
  const projectScopeLabels = [];
  const projectScopeKeys = new Set();
  projectOptions.forEach((option) => {
    if (option.label && !projectScopeLabels.includes(option.label)) {
      projectScopeLabels.push(option.label);
    }
    (option.keys || []).forEach((key) => projectScopeKeys.add(key));
  });
  [...(scan?.repos || []).map((repo) => `${repo.owner}/${repo.repo}`), ...memoryRepositories.map((repo) => repo.repository)]
    .filter(Boolean)
    .forEach((option) => {
      const key = normalizeProjectAlias(option);
      if (key && !projectScopeKeys.has(key) && !projectScopeLabels.includes(option)) {
        projectScopeKeys.add(key);
        projectScopeLabels.push(option);
      }
    });
  const projectScopeOptions = projectScopeLabels
    .sort((left, right) => left.localeCompare(right));
  const repositoryEvidenceTitle =
    language === "en" ? (copy.memoryRepositories || "Repositories in Chroma Evidence DB") : "Chroma 证据库中的仓库";
  const repositoryEvidenceHint =
    language === "en"
      ? (copy.memoryRepositoriesHint || "These repositories already have local Chroma evidence records. Reading this list does not access GitHub.")
      : "这些仓库已经有本地 Chroma 证据库记录。读取列表不会访问 GitHub 云端。";
  const chromaEvidenceUpdatedAt =
    language === "en" ? (copy.chromaEvidenceUpdatedAt || "Chroma evidence DB updated: ") : "Chroma 证据库更新：";
  const projectMemoryUpdatedAtLabel =
    language === "en" ? (copy.projectMemoryUpdatedAt || "Project Memory JSON updated: ") : "Project Memory JSON 更新：";
  const noRepositoryEvidence =
    language === "en" ? (copy.noMemoryRepositories || "No Chroma repository evidence yet") : "暂无 Chroma 仓库证据";

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">{copy.config}</h2>
        <div className="grid-2">
          <div className="field">
            <label>{copy.username}</label>
            <textarea
              className="short"
              value={githubForm.usernames}
              onChange={(event) => setGithubForm((current) => ({ ...current, usernames: event.target.value }))}
              placeholder="e.g. liammaxp"
            />
          </div>
          <div className="field">
            <label>{copy.token}</label>
            <input
              type="password"
              autoComplete="off"
              value={githubForm.token}
              onChange={(event) => setGithubForm((current) => ({ ...current, token: event.target.value }))}
              placeholder={tokenConfigured ? copy.tokenConfigured : copy.pasteToken}
            />
            <div className="status-line">
              {copy.tokenStatus}<StatusBadge ready={tokenConfigured} />
            </div>
          </div>
        </div>
        <div className="grid-2">
          <div className="field">
            <label>{copy.authorName}</label>
            <textarea
              className="short"
              value={githubForm.author_names}
              onChange={(event) => setGithubForm((current) => ({ ...current, author_names: event.target.value }))}
              placeholder="e.g. liammmmax"
            />
          </div>
          <div className="field">
            <label>{copy.authorEmail}</label>
            <textarea
              className="short"
              value={githubForm.author_emails}
              onChange={(event) => setGithubForm((current) => ({ ...current, author_emails: event.target.value }))}
              placeholder="e.g. name@example.com"
            />
          </div>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={saveGithubConfig} disabled={loading}>
            {copy.saveConfig}
          </button>
          <span className="helper-text">{copy.configHint}</span>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">{copy.scanSettings}</h2>
        <div className="field">
          <label>{copy.resumeSource}</label>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="tailored_resume_and_resume_and_memory">
              {copy.allResumeSources || "定制简历、基础简历与记忆项目"}
            </option>
            <option value="resume_and_memory">{copy.resumeAndMemory || "简历与记忆中的项目"}</option>
            <option value="resume">{copy.baseResume}</option>
            <option value="tailored_resume">{copy.tailoredResume}</option>
            <option value="memory">{copy.memoryProjects || "仅记忆中的项目"}</option>
          </select>
        </div>
        <div className="field compact-field">
          <label>{copy.projectScopeLabel || "Project scope"}</label>
          <input
            list="github-project-scope-options"
            value={projectScope}
            onChange={(event) => setProjectScope(event.target.value)}
            disabled={loading}
            placeholder={copy.projectScopePlaceholder || "Choose a project, or type a project name / ID"}
          />
          <datalist id="github-project-scope-options">
            {projectScopeOptions.map((option) => (
              <option key={option} value={option} />
            ))}
          </datalist>
          <p className="helper-text">
            {copy.projectScopeHint || "Optional. Suggestions come from Project Memory, scanned repositories, and local evidence; leave blank to scan every repository in the selected sources."}
          </p>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={scanRepos} disabled={loading}>
            {copy.scanRepos}
          </button>
          <button type="button" className="btn btn-primary" onClick={() => setConfirmOpen(true)} disabled={loading || !scan?.repos?.length}>
            {copy.confirmFetch}
          </button>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">{repositoryEvidenceTitle}</h2>
        <p className="helper-text">{repositoryEvidenceHint}</p>
        <p className="status-line">
          {projectMemoryUpdatedAtLabel}{formatUnixUpdatedAt(projectMemoryUpdatedAt, language)}
        </p>
        {memoryRepositories.length ? (
          <div className="repo-list">
            {memoryRepositories.map((repo) => (
              <div key={repo.repository} className="repo-item">
                <span>{repo.repository}</span>
                <span className="status-line">
                  {chromaEvidenceUpdatedAt}{formatMemoryUpdatedAt(repo.updated_at, language)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-state">{noRepositoryEvidence}</p>
        )}
      </section>

      {scan && (
        <section className="card">
          <h2 className="card-title">{copy.scanResult}</h2>
          <p className="status-line">
            {copy.tokenStatus}<StatusBadge ready={scan.token_configured} />
          </p>
          {identityItems.length > 0 && (
            <ul className="output-list" style={{ marginBottom: 16 }}>
              {identityItems.map((item) => <li key={item}>{item}</li>)}
            </ul>
          )}
          {scan.repos?.length ? (
            <div className="repo-list">
              {scan.repos.map((repo) => (
                <div key={repo.url} className="repo-item">
                  <span>{repo.owner}/{repo.repo}</span>
                  <a href={repo.url} target="_blank" rel="noreferrer">{copy.open}</a>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-state">{copy.noRepos}</p>
          )}
        </section>
      )}

      {context?.context && (
        <section className="card">
          <h2 className="card-title">{copy.contextSummary}</h2>
          <pre className="json-preview">{JSON.stringify(context.context, null, 2)}</pre>
        </section>
      )}

      <ConfirmDialog
        open={confirmOpen}
        title={copy.allowTitle}
        confirmLabel={copy.allowConfirm}
        loading={loading}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={approveFetchContext}
      >
        <p>{copy.allowBody}</p>
        {scan?.repos?.length ? (
          <ul>
            {scan.repos.map((repo) => <li key={repo.url}>{repo.owner}/{repo.repo}</li>)}
          </ul>
        ) : (
          <p>{copy.noReadableRepos}</p>
        )}
        <p className="status-line">
          {copy.tokenStatus}<StatusBadge ready={scan?.token_configured} />
        </p>
        {identityItems.length > 0 && (
          <>
            <p>{copy.identityIntro}</p>
            <ul>
              {identityItems.map((item) => <li key={item}>{item}</li>)}
            </ul>
          </>
        )}
      </ConfirmDialog>
    </>
  );
}
