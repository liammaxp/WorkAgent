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
import { text, useLanguage } from "../i18n.jsx";

const DEFAULT_PROVIDERS = [
  { provider: "openai", label: "OpenAI", configured: false, base_url: "", model: "gpt-5.5", default_base_url: "", default_model: "gpt-5.5", requires_base_url: false },
  { provider: "openai-compatible", label: "OpenAI Compatible", configured: false, base_url: "", model: "gpt-5.5", default_base_url: "", default_model: "gpt-5.5", requires_base_url: true },
  { provider: "deepseek", label: "DeepSeek", configured: false, base_url: "https://api.deepseek.com", model: "deepseek-chat", default_base_url: "https://api.deepseek.com", default_model: "deepseek-chat", requires_base_url: false },
  { provider: "claude", label: "Claude / Anthropic", configured: false, base_url: "https://api.anthropic.com", model: "claude-sonnet-4-5", default_base_url: "https://api.anthropic.com", default_model: "claude-sonnet-4-5", requires_base_url: false },
  { provider: "gemini", label: "Gemini", configured: false, base_url: "https://generativelanguage.googleapis.com/v1beta", model: "gemini-2.5-flash", default_base_url: "https://generativelanguage.googleapis.com/v1beta", default_model: "gemini-2.5-flash", requires_base_url: false },
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
  const { language } = useLanguage();
  const copy = text[language].dashboard;
  const common = text[language].common;
  const [status, setStatus] = useState(null);
  const [provider, setProvider] = useState("");
  const [model, setModel] = useState("");
  const [providerConfigs, setProviderConfigs] = useState(DEFAULT_PROVIDERS);
  const [providerForm, setProviderForm] = useState(emptyProviderForm());
  const [applications, setApplications] = useState([]);
  const { loading, error, success, run } = useAsyncAction();

  const selectedProviderConfig = useMemo(
    () =>
      providerConfigs.find((item) => item.provider === providerForm.provider) ||
      providerConfigs[0],
    [providerConfigs, providerForm.provider]
  );

  const loadStatus = () =>
    run(async () => {
      const [data, applicationRecords] = await Promise.all([
        api.getStatus(),
        api.getApplications("", 100),
      ]);
      const configs = data.provider_configs?.length
        ? data.provider_configs
        : DEFAULT_PROVIDERS;
      setStatus(data);
      setApplications(applicationRecords || []);
      setProvider(data.provider);
      setModel(data.model);
      setProviderConfigs(configs);
      setProviderForm((current) => {
        const currentConfig =
          configs.find((item) => item.provider === current.provider) ||
          configs.find((item) => item.provider === data.provider) ||
          configs[0];
        return { ...emptyProviderForm(currentConfig), api_key: current.api_key };
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
    }, copy.providerUpdated);

  const saveProviderConfig = () =>
    run(async () => {
      const payload = { ...providerForm, model: model || providerForm.model };
      const data = await api.saveProviderConfig(payload);
      setProvider(data.provider);
      setModel(data.model);
      setStatus((current) =>
        current ? { ...current, provider: data.provider, model: data.model } : current
      );
      if (data.provider_configs) setProviderConfigs(data.provider_configs);
      setProviderForm((current) => ({ ...current, api_key: "" }));
      await loadStatus();
      return data;
    }, copy.keySaved);

  const saveModel = () =>
    run(async () => {
      const data = await api.setModel(model);
      setModel(data.model);
      setStatus((current) => (current ? { ...current, model: data.model } : current));
      return data;
    }, copy.modelUpdated);

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      {status && (
        <>
          <div className="card dashboard-config-grid">
            <section className="dashboard-config-section">
              <h2 className="card-title">{copy.modelSettings}</h2>
              <div className="summary-row">
                <div>
                  <div className="stat-label">{copy.currentProvider}</div>
                  <div className="stat-value">{status.provider}</div>
                </div>
                <div>
                  <div className="stat-label">{copy.currentModel}</div>
                  <div className="stat-value">{status.model}</div>
                </div>
              </div>
              <div className="field">
                <label>{copy.switchProvider}</label>
                <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                  {providerConfigs.map((item) => (
                    <option key={item.provider} value={item.provider}>
                      {item.label}{item.configured ? ` ${common.configured}` : ""}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label>{copy.switchModel}</label>
                <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="e.g. gpt-5.5" />
              </div>
              <div className="btn-row">
                <button type="button" className="btn btn-secondary" onClick={saveProvider} disabled={loading}>
                  {copy.applyProvider}
                </button>
                <button type="button" className="btn btn-secondary" onClick={saveModel} disabled={loading}>
                  {copy.applyModel}
                </button>
                <button type="button" className="btn btn-secondary" onClick={loadStatus} disabled={loading}>
                  {common.refresh}
                </button>
              </div>
            </section>

            <section className="dashboard-config-section">
              <h2 className="card-title">{copy.addApiKey}</h2>
              <div className="api-key-grid">
                <div className="field">
                  <label>{copy.apiProvider}</label>
                  <select value={providerForm.provider} onChange={(event) => chooseConfigProvider(event.target.value)}>
                    {providerConfigs.map((item) => (
                      <option key={item.provider} value={item.provider}>{item.label}</option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label>Base URL{selectedProviderConfig.requires_base_url ? copy.required : copy.optional}</label>
                  <input
                    value={providerForm.base_url}
                    onChange={(event) => setProviderForm((current) => ({ ...current, base_url: event.target.value }))}
                    placeholder={selectedProviderConfig.default_base_url || copy.defaultOfficialUrl}
                  />
                </div>
              </div>
              <div className="field">
                <label>API Key</label>
                <input
                  type="password"
                  autoComplete="off"
                  value={providerForm.api_key}
                  onChange={(event) => setProviderForm((current) => ({ ...current, api_key: event.target.value }))}
                  placeholder={`${copy.pasteApiKey} (${selectedProviderConfig.label})`}
                />
              </div>
              <div className="btn-row">
                <button type="button" className="btn btn-primary" onClick={saveProviderConfig} disabled={loading || !providerForm.api_key.trim()}>
                  {copy.saveAndEnable}
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => chooseConfigProvider(providerForm.provider)} disabled={loading}>
                  {copy.resetForm}
                </button>
                <span className="helper-text">{copy.keyHint}</span>
              </div>
            </section>
          </div>

          <section className="card">
            <h2 className="card-title">{copy.fileStatus}</h2>
            <div className="grid-3">
              {Object.entries(status.files).map(([key, ready]) => (
                <div key={key} className="stat-card">
                  <div className="stat-label">{copy.fileLabels[key] || key}</div>
                  <StatusBadge ready={ready} />
                </div>
              ))}
            </div>
          </section>

          <div className="card recent-panel">
            <section className="recent-section">
              <h2 className="card-title">{copy.recentAnalysis}</h2>
              {status.outputs.analysis.length ? (
                <ul className="output-list">
                  {status.outputs.analysis.map((item) => (
                    <li key={item.path}>{new Date(item.updated_at).toLocaleString(language === "zh" ? "zh-CN" : "en-US", { hour12: false })}</li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">{copy.noAnalysis}</p>
              )}
              <div className="btn-row">
                <Link to="/job" className="btn btn-primary">{copy.analyzeJob}</Link>
              </div>
            </section>

            <section className="recent-section">
              <h2 className="card-title">{copy.recentApplications}</h2>
              {applications.length ? (
                <ul className="output-list scroll-list">
                  {applications.map((item, index) => (
                    <li key={item.id}>{index + 1}. {item.company}：{item.role}</li>
                  ))}
                </ul>
              ) : (
                <p className="empty-state">{copy.noApplications}</p>
              )}
            </section>
          </div>
        </>
      )}
    </>
  );
}
