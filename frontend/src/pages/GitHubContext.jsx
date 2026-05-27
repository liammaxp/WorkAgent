import { useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  ConfirmDialog,
  LoadingBar,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";

export default function GitHubContext() {
  const [scan, setScan] = useState(null);
  const [context, setContext] = useState(null);
  const [source, setSource] = useState("resume");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const { loading, error, success, run } = useAsyncAction();

  const scanRepos = () =>
    run(async () => {
      const data = await api.scanGithub(source);
      setScan(data);
      setContext(null);
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

  const identities = scan?.identities || {};
  const identityItems = [
    ...(identities.usernames || []).map((value) => `GitHub username: ${value}`),
    ...(identities.author_names || []).map((value) => `Commit author name: ${value}`),
    ...(identities.author_emails || []).map((value) => `Commit author email: ${value}`),
  ];

  return (
    <>
      <PageHeader
        title="GitHub 证据"
        description="从简历识别仓库，读取 README、提交与 diff 信号，辅助真实项目描述。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

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
            Token 状态：
            <StatusBadge ready={scan.token_configured} />
          </p>
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
