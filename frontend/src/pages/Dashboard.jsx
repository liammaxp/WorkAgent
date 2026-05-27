import { useEffect, useMemo, useState } from "react";
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

const DEFAULT_PROVIDERS = [
  {
    provider: "openai",
    label: "OpenAI",
    configured: false,
    base_url: "",
    model: "gpt-5.5",
    default_base_url: "",
    default_model: "gpt-5.5",
    requires_base_url: false,
  },
  {
    provider: "openai-compatible",
    label: "OpenAI Compatible",
    configured: false,
    base_url: "",
    model: "gpt-5.5",
    default_base_url: "",
    default_model: "gpt-5.5",
    requires_base_url: true,
  },
  {
    provider: "deepseek",
    label: "DeepSeek",
    configured: false,
    base_url: "https://api.deepseek.com",
    model: "deepseek-chat",
    default_base_url: "https://api.deepseek.com",
    default_model: "deepseek-chat",
    requires_base_url: false,
  },
  {
    provider: "claude",
    label: "Claude / Anthropic",
    configured: false,
    base_url: "https://api.anthropic.com",
    model: "claude-sonnet-4-5",
    default_base_url: "https://api.anthropic.com",
    default_model: "claude-sonnet-4-5",
    requires_base_url: false,
  },
  {
    provider: "gemini",
    label: "Gemini",
    configured: false,
    base_url: "https://generativelanguage.googleapis.com/v1beta",
    model: "gemini-2.5-flash",
    default_base_url: "https://generativelanguage.googleapis.com/v1beta",
    default_model: "gemini-2.5-flash",
    requires_base_url: false,
  },
];

function emptyProviderForm(provider = DEFAULT_PROVIDERS[0]) {
  return {
    provider: provider.provider,
    api_key: "",
    base_url: provider.base_url || provider.default_base_url || "",
    model: provider.model || provider.default_model || "",
  };
}

export default function Dashboard() {
  const [status, setStatus] = useState(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [providerConfigs, setProviderConfigs] = useState(DEFAULT_PROVIDERS);
  const [providerForm, setProviderForm] = useState(emptyProviderForm());
  const { loading, error, success, run } = useAsyncAction();

  const selectedProviderConfig = useMemo(
    () =>
      providerConfigs.find((item) => item.provider === providerForm.provider) ||
      providerConfigs[0],
    [providerConfigs, providerForm.provider]
  );

  const loadStatus = () =>
    run(async () => {
      const data = await api.getStatus();
      const configs = data.provider_configs?.length
        ? data.provider_configs
        : DEFAULT_PROVIDERS;
      setStatus(data);
      setProvider(data.provider);
      setModel(data.model);
      setProviderConfigs(configs);
      setProviderForm((current) => {
        const currentConfig =
          configs.find((item) => item.provider === current.provider) ||
          configs.find((item) => item.provider === data.provider) ||
          configs[0];
        return {
          ...emptyProviderForm(currentConfig),
          api_key: current.api_key,
        };
      });
      return data;
    });

  useEffect(() => {
    loadStatus();
  }, []);

  const chooseConfigProvider = (providerName) => {
    const config =
      providerConfigs.find((item) => item.provider === providerName) ||
      DEFAULT_PROVIDERS[0];
    setProviderForm(emptyProviderForm(config));
    setProvider(providerName);
    setModel(config.model || config.default_model || "");
  };

  const saveProvider = () =>
    run(async () => {
      const data = await api.setProvider(provider);
      setProvider(data.provider);
      setModel(data.model);
      setStatus((current) =>
        current ? { ...current, provider: data.provider, model: data.model } : current
      );
      await loadStatus();
      return data;
    }, "模型供应商已更新");

  const saveProviderConfig = () =>
    run(async () => {
      const payload = {
        ...providerForm,
        model: model || providerForm.model,
      };
      const data = await api.saveProviderConfig(payload);
      setProvider(data.provider);
      setModel(data.model);
      setStatus((current) =>
        current ? { ...current, provider: data.provider, model: data.model } : current
      );
      if (data.provider_configs) {
        setProviderConfigs(data.provider_configs);
      }
      setProviderForm((current) => ({ ...current, api_key: "" }));
      await loadStatus();
      return data;
    }, "API Key 已保存并启用");

  const saveModel = () =>
    run(async () => {
      const data = await api.setModel(model);
      setModel(data.model);
      setStatus((current) =>
        current ? { ...current, model: data.model } : current
      );
      return data;
    }, "模型已更新");

  return (
    <>
      <PageHeader
        title="工作台概览"
        description="配置模型供应商、添加 API Key，并查看文件就绪情况与最近输出。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      {status && (
        <>
          <div className="dashboard-config-grid">
            <section className="card stat-card">
              <h2 className="card-title">模型设置</h2>
              <div className="summary-row">
                <div>
                  <div className="stat-label">当前供应商</div>
                  <div className="stat-value">{status.provider}</div>
                </div>
                <div>
                  <div className="stat-label">当前模型</div>
                  <div className="stat-value">{status.model}</div>
                </div>
              </div>
              <div className="field">
                <label>切换供应商</label>
                <select
                  value={provider}
                  onChange={(event) => setProvider(event.target.value)}
                >
                  {providerConfigs.map((item) => (
                    <option key={item.provider} value={item.provider}>
                      {item.label}
                      {item.configured ? " 已配置" : ""}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>切换模型</label>
                <input
                  value={model}
                  onChange={(event) => setModel(event.target.value)}
                  placeholder="例如 gpt-5.5"
                />
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

            <section className="card">
              <h2 className="card-title">添加 API Key</h2>
              <div className="api-key-grid">
                <div className="field">
                  <label>API 厂商</label>
                  <select
                    value={providerForm.provider}
                    onChange={(event) => chooseConfigProvider(event.target.value)}
                  >
                    {providerConfigs.map((item) => (
                      <option key={item.provider} value={item.provider}>
                        {item.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label>
                    Base URL
                    {selectedProviderConfig.requires_base_url ? "（必填）" : "（可选）"}
                  </label>
                  <input
                    value={providerForm.base_url}
                    onChange={(event) =>
                      setProviderForm((current) => ({
                        ...current,
                        base_url: event.target.value,
                      }))
                    }
                    placeholder={selectedProviderConfig.default_base_url || "默认官方地址"}
                  />
                </div>
              </div>
              <div className="field">
                <label>API Key</label>
                <input
                  type="password"
                  autoComplete="off"
                  value={providerForm.api_key}
                  onChange={(event) =>
                    setProviderForm((current) => ({
                      ...current,
                      api_key: event.target.value,
                    }))
                  }
                  placeholder={`粘贴 ${selectedProviderConfig.label} API Key`}
                />
              </div>
              <div className="btn-row">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={saveProviderConfig}
                  disabled={loading || !providerForm.api_key.trim()}
                >
                  保存并启用
                </button>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => chooseConfigProvider(providerForm.provider)}
                  disabled={loading}
                >
                  重置格式
                </button>
                <span className="helper-text">
                  会自动写入对应变量名，不需要手动编辑 .env。
                </span>
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
