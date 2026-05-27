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
  const [scan, setScan] = useState(null);
  const [context, setContext] = useState(null);
  const [source, setSource] = useState("resume");
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
          ? {
              ...current,
              identities: data.identities,
              token_configured: data.token_configured,
            }
          : current
      );
      return data;
    }, "GitHub 配置已保存");

  const scanRepos = () =>
    run(async () => {
      const data = await api.scanGithub(source);
      setScan(data);
      setContext(null);
      setTokenConfigured(data.token_configured);
      return data;
    }, "仓库扫描完成");

  const fetchContext = () => setConfirmOpen(true);

  const approveFetchContext = () =>
    run(async () => {
      const data = await api.fetchGithubContext(true, source);
      setContext(data);
      setConfirmOpen(false);
      return data;
    }, "GitHub 上下文已获取");

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
      <PageHeader
        title="GitHub 证据"
        description="配置 GitHub 身份和 Token，扫描简历中的仓库，并读取 README、提交记录与 diff 信号。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">GitHub 配置</h2>
        <div className="grid-2">
          <div className="field">
            <label>GitHub 用户名</label>
            <textarea
              className="short"
              value={githubForm.usernames}
              onChange={(event) =>
                setGithubForm((current) => ({
                  ...current,
                  usernames: event.target.value,
                }))
              }
              placeholder="例如 liammaxp"
            />
          </div>
          <div className="field">
            <label>GitHub Token（可选）</label>
            <input
              type="password"
              autoComplete="off"
              value={githubForm.token}
              onChange={(event) =>
                setGithubForm((current) => ({
                  ...current,
                  token: event.target.value,
                }))
              }
              placeholder={tokenConfigured ? "已配置；留空表示不修改" : "粘贴 GitHub Token"}
            />
            <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
              Token 状态：<StatusBadge ready={tokenConfigured} />
            </div>
          </div>
        </div>
        <div className="grid-2">
          <div className="field">
            <label>提交作者名称</label>
            <textarea
              className="short"
              value={githubForm.author_names}
              onChange={(event) =>
                setGithubForm((current) => ({
                  ...current,
                  author_names: event.target.value,
                }))
              }
              placeholder="例如 liammmmax"
            />
          </div>
          <div className="field">
            <label>提交邮箱</label>
            <textarea
              className="short"
              value={githubForm.author_emails}
              onChange={(event) =>
                setGithubForm((current) => ({
                  ...current,
                  author_emails: event.target.value,
                }))
              }
              placeholder="例如 name@example.com"
            />
          </div>
        </div>
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={saveGithubConfig}
            disabled={loading}
          >
            保存 GitHub 配置
          </button>
          <span className="helper-text">
            每行或逗号分隔多个值；会自动写入 github_accounts.txt 和 .env。
          </span>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">扫描设置</h2>
        <div className="field">
          <label>简历来源</label>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="resume">原始简历</option>
            <option value="tailored_resume">定制简历</option>
          </select>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={scanRepos} disabled={loading}>
            扫描仓库
          </button>
          <button type="button" className="btn btn-primary" onClick={fetchContext} disabled={loading || !scan?.repos?.length}>
            确认并获取上下文
          </button>
        </div>
      </section>

      {scan && (
        <section className="card">
          <h2 className="card-title">扫描结果</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
            Token 状态：<StatusBadge ready={scan.token_configured} />
          </p>
          {identityItems.length > 0 && (
            <ul className="output-list" style={{ marginBottom: 16 }}>
              {identityItems.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          )}
          {scan.repos?.length ? (
            <div className="repo-list">
              {scan.repos.map((repo) => (
                <div key={repo.url} className="repo-item">
                  <span>{repo.owner}/{repo.repo}</span>
                  <a href={repo.url} target="_blank" rel="noreferrer">
                    打开
                  </a>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-state">未在简历中发现 GitHub 仓库链接</p>
          )}
        </section>
      )}

      {context?.context && (
        <section className="card">
          <h2 className="card-title">上下文摘要</h2>
          <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, lineHeight: 1.6, maxHeight: 480, overflow: "auto", margin: 0 }}>
            {JSON.stringify(context.context, null, 2)}
          </pre>
        </section>
      )}

      <ConfirmDialog
        open={confirmOpen}
        title="允许读取 GitHub 仓库信息？"
        confirmLabel="允许并获取"
        loading={loading}
        onCancel={() => setConfirmOpen(false)}
        onConfirm={approveFetchContext}
      >
        <p>WorkAgent 将访问下面这些仓库的公开或已授权信息，包括 README、语言、目录、提交记录和 diff 信号，用来辅助简历、求职信或面试准备。</p>
        {scan?.repos?.length ? (
          <ul>
            {scan.repos.map((repo) => (
              <li key={repo.url}>{repo.owner}/{repo.repo}</li>
            ))}
          </ul>
        ) : (
          <p>当前没有可读取的仓库。</p>
        )}
        <p style={{ marginTop: 12 }}>
          Token 状态：<StatusBadge ready={scan?.token_configured} />
        </p>
        {identityItems.length > 0 && (
          <>
            <p>将用这些身份匹配你的提交：</p>
            <ul>
              {identityItems.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </>
        )}
      </ConfirmDialog>
    </>
  );
}
