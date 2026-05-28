import { useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

export default function Chat() {
  const { language } = useLanguage();
  const copy = text[language].chat;
  const [message, setMessage] = useState("");
  const [history, setHistory] = useState([]);
  const { loading, error, success, run } = useAsyncAction();

  const send = () =>
    run(async () => {
      const trimmed = message.trim();
      if (!trimmed) return null;

      const userEntry = { role: "user", text: trimmed };
      setHistory((prev) => [...prev, userEntry]);
      setMessage("");

      const data = await api.askAgent(trimmed);
      const agentEntry = { role: "agent", text: data.answer || "" };
      setHistory((prev) => [...prev, agentEntry]);
      return data;
    });

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card" style={{ minHeight: 360 }}>
        {history.length === 0 ? (
          <p className="empty-state">{copy.empty}</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {history.map((entry, index) => (
              <div
                key={index}
                style={{
                  alignSelf: entry.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "85%",
                  padding: "12px 16px",
                  borderRadius: 12,
                  background: entry.role === "user" ? "var(--accent-soft)" : "var(--surface-muted)",
                  fontSize: 14,
                  lineHeight: 1.7,
                  whiteSpace: "pre-wrap",
                }}
              >
                <strong style={{ display: "block", marginBottom: 4, fontSize: 12, color: "var(--text-muted)" }}>
                  {entry.role === "user" ? copy.you : "Agent"}
                </strong>
                {entry.text}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <div className="field">
          <label>{copy.message}</label>
          <textarea
            className="short"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={copy.placeholder}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
            }}
          />
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={send} disabled={loading || !message.trim()}>
            {loading ? copy.thinking : copy.send}
          </button>
        </div>
      </section>
    </>
  );
}
