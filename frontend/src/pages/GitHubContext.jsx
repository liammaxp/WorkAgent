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

export default function GitHubContext() {
  const { language } = useLanguage();
  const copy = text[language].github;
  const [scan, setScan] = useState(null);
  const [context, setContext] = useState(null);
  const [source, setSource] = useState("tailored_resume_and_resume_and_memory");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [githubForm, setGithubForm] = useState({
    usernames: "",
    author_names: "",
    author_emails: "",
    token: "",
  });
  const [tokenConfigured, setTokenConfigured] = useState(false);
  const { loading, error, success, run } = useAsyncAction();

  const loadGithubConfig = () =>
    run(async () => {
      const data = await api.getGithubConfig();
      setGithubForm((current) => ({
        usernames: listToText(data.identities?.usernames),
        author_names: listToText(data.identities?.author_names),
        author_emails: listToText(data.identities?.author_emails),
        token: current.token,
      }));
      setTokenConfigured(data.token_configured);
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
      setScan((current) =>
        current
          ? { ...current, identities: data.identities, token_configured: data.token_configured }
          : current
      );
      return data;
    }, copy.saved);

  const scanRepos = () =>
    run(async () => {
      const data = await api.scanGithub(source);
      setScan(data);
      setContext(null);
      setTokenConfigured(data.token_configured);
      return data;
    }, copy.scanned);

  const approveFetchContext = () =>
    run(async () => {
      const data = await api.fetchGithubContext(true, source);
      setContext(data);
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
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={scanRepos} disabled={loading}>
            {copy.scanRepos}
          </button>
          <button type="button" className="btn btn-primary" onClick={() => setConfirmOpen(true)} disabled={loading || !scan?.repos?.length}>
            {copy.confirmFetch}
          </button>
        </div>
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
