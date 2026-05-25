import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client.js";
import {
  Alert,
  LoadingBar,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";

const FILE_LABELS = {
  resume: "原始简历",
  tailored_resume: "定制简历",
  job_description: "职位描述",
  cover_letter: "求职信",
  interview_prep: "面试准备",
  memory: "用户背景",
  github_accounts: "GitHub 账号",
};

export default function Dashboard() {
  const [status, setStatus] = useState(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const { loading, error, success, run } = useAsyncAction();

  const loadStatus = () =>
    run(async () => {
      const data = await api.getStatus();
      setStatus(data);
      setProvider(data.provider);
      setModel(data.model);
      return data;
    });

  useEffect(() => {
    loadStatus();
  }, []);

  const saveProvider = () =>
    run(async () => {
      const data = await api.setProvider(provider);
      setProvider(data.provider);
      setModel(data.model);
      await loadStatus();
      return data;
    }, "模型供应商已更新");

  const saveModel = () =>
    run(async () => {
      const data = await api.setModel(model);
      setModel(data.model);
      return data;
    }, "模型已更新");

  return (
    <>
      <PageHeader
        title="工作台概览"
        description="查看当前 Agent 状态、文件就绪情况与最近输出。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      {status && (
        <>
          <div className="grid-2">
            <section className="card stat-card">
              <div className="stat-label">当前供应商</div>
              <div className="stat-value">{status.provider}</div>
              <div className="field" style={{ marginTop: 16 }}>
                <label>切换供应商</label>
                <select
                  value={provider}
                  onChange={(event) => setProvider(event.target.value)}
                >
                  <option value="openai">openai</option>
                  <option value="openai-compatible">openai-compatible</option>
                  <option value="deepseek">deepseek</option>
                  <option value="claude">claude</option>
                  <option value="gemini">gemini</option>
                </select>
              </div>
              <div className="btn-row">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={saveProvider}
                  disabled={loading}
                >
                  应用供应商
                </button>
              </div>
            </section>

            <section className="card stat-card">
              <div className="stat-label">当前模型</div>
              <div className="stat-value">{status.model}</div>
              <div className="field" style={{ marginTop: 16 }}>
                <label>切换模型</label>
                <input
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  placeholder="例如 gpt-4o"
                />
              </div>
              <div className="btn-row">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={saveModel}
                  disabled={loading}
                >
                  应用模型
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={loadStatus}
                  disabled={loading}
                >
                  刷新
                </button>
              </div>
            </section>
          </div>

          <section className="card">
            <h2 className="card-title">文件状态</h2>
            <div className="grid-3">
              {Object.entries(status.files).map(([key, ready]) => (
                <div key={key} className="stat-card">
                  <div className="stat-label">{FILE_LABELS[key] || key}</div>
                  <StatusBadge ready={ready} />
                </div>
              ))}
            </div>
          </section>

          <div className="grid-2">
            <section className="card">
              <h2 className="card-title">最近职位分析</h2>
              {status.outputs.analysis.length ? (
                <ul className="output-list">
                  {status.outputs.analysis.map((item) => (
                    <li key={item.path}>{item.name}</li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">暂无分析输出</p>
              )}
              <div className="btn-row">
                <Link to="/job" className="btn btn-primary">
                  去分析职位
                </Link>
              </div>
            </section>

            <section className="card">
              <h2 className="card-title">最近 GitHub 上下文</h2>
              {status.outputs.github_context.length ? (
                <ul className="output-list">
                  {status.outputs.github_context.map((item) => (
                    <li key={item.path}>{item.name}</li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">暂无 GitHub 上下文</p>
              )}
              <div className="btn-row">
                <Link to="/github" className="btn btn-primary">
                  获取 GitHub 证据
                </Link>
              </div>
            </section>
          </div>

          <section className="card">
            <h2 className="card-title">快速开始</h2>
            <p style={{ color: "var(--text-muted)", margin: "0 0 16px" }}>
              推荐流程：粘贴职位描述 → 分析匹配度 → 生成定制简历 → 撰写求职信 → 记录申请。
            </p>
            <div className="btn-row">
              <Link to="/job" className="btn btn-secondary">
                1. 职位描述
              </Link>
              <Link to="/resume" className="btn btn-secondary">
                2. 定制简历
              </Link>
              <Link to="/cover-letter" className="btn btn-secondary">
                3. 求职信
              </Link>
              <Link to="/applications" className="btn btn-secondary">
                4. 申请记录
              </Link>
            </div>
          </section>
        </>
      )}
    </>
  );
}
