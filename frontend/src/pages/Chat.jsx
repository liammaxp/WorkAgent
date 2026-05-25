import { useState } from "react";
import { api } from "../api/client.js";
import {
  Alert,
  LoadingBar,
  PageHeader,
  useAsyncAction,
} from "../components/ui.jsx";

export default function Chat() {
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
      <PageHeader
        title="Agent 对话"
        description="直接与 WorkAgent 对话，可请求职位分析、简历修改、求职信等任务。"
      />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card" style={{ minHeight: 360 }}>
        {history.length === 0 ? (
          <p className="empty-state">
            输入消息开始对话。你也可以粘贴完整职位描述，Agent 会自动识别并保存。
          </p>
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
                  {entry.role === "user" ? "你" : "Agent"}
                </strong>
                {entry.text}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <div className="field">
          <label>消息</label>
          <textarea
            className="short"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="例如：分析当前职位描述，或帮我写 cover letter…"
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
            }}
          />
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={send} disabled={loading || !message.trim()}>
            {loading ? "思考中…" : "发送 (Ctrl+Enter)"}
          </button>
        </div>
      </section>
    </>
  );
}
