import { useEffect, useMemo, useRef, useState } from "react";

function stageIcon(status) {
  if (status === "running") return <span className="agent-progress-spinner" aria-hidden="true" />;
  if (status === "waiting_for_user") return <span className="agent-progress-icon waiting" aria-hidden="true" />;
  if (status === "done") return <span className="agent-progress-icon done">✓</span>;
  if (status === "error") return <span className="agent-progress-icon error">×</span>;
  if (status === "cancelled") return <span className="agent-progress-icon cancelled">−</span>;
  return <span className="agent-progress-icon pending" />;
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export default function AgentProgressModal({ task, onCancel, onClose, onSend }) {
  const [draft, setDraft] = useState("");
  const messagesEndRef = useRef(null);
  const isOpen = Boolean(task);
  const running = task?.status === "running";
  const currentStage = useMemo(
    () => task?.stages.find((stage) => stage.id === task.currentStageId) || task?.stages.find((stage) => stage.status === "running"),
    [task],
  );

  useEffect(() => {
    if (!isOpen) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [task?.messages?.length]);

  useEffect(() => {
    if (!isOpen) setDraft("");
  }, [isOpen]);

  if (!task) return null;

  const submit = () => {
    const content = draft.trim();
    if (!content || !running) return;
    onSend(content);
    setDraft("");
  };

  const closeOrCancel = () => {
    if (running) {
      onCancel();
      return;
    }
    onClose();
  };

  return (
    <div className="agent-progress-backdrop" role="presentation">
      <section className="agent-progress-modal" role="dialog" aria-modal="true" aria-labelledby="agent-progress-title">
        <header className="agent-progress-header">
          <div>
            <h2 id="agent-progress-title">{task.title}</h2>
            <p>
              当前阶段：
              {currentStage?.detail || currentStage?.label || (task.status === "success" ? "已完成" : "正在调用 Agent")}
            </p>
          </div>
          <button type="button" className="agent-progress-close" aria-label="关闭" onClick={closeOrCancel}>
            ×
          </button>
        </header>

        <div className="agent-progress-body">
          <aside className="agent-progress-stages" aria-label="阶段进度">
            <div className="agent-progress-section-label">阶段进度</div>
            {task.stages.map((stage) => (
              <div
                key={stage.id}
                className={`agent-progress-stage ${stage.status}${stage.id === currentStage?.id ? " active" : ""}`}
              >
                {stageIcon(stage.status)}
                <div>
                  <span>{stage.label}</span>
                  {stage.detail && <small>{stage.detail}</small>}
                </div>
              </div>
            ))}
          </aside>

          <main className="agent-progress-chat" aria-label="Agent 对话">
            <div className="agent-progress-messages">
              {task.messages.map((message) => (
                <div key={message.id} className={`agent-progress-message ${message.role}`}>
                  <div className="agent-progress-message-meta">
                    <span>{message.contextLabel || (message.role === "user" ? "User" : message.role === "agent" ? "Agent" : "System")}</span>
                    <time>{formatTime(message.timestamp)}</time>
                  </div>
                  <div className="agent-progress-message-content">{message.content}</div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </main>
        </div>

        {task.error && <div className="agent-progress-error" role="alert">{task.error}</div>}

        <footer className="agent-progress-footer">
          <textarea
            value={draft}
            disabled={!running}
            placeholder={running ? "输入补充要求 / 回答 agent 问题..." : "任务已结束"}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
          />
          <div className="agent-progress-actions">
            <button type="button" className="btn btn-danger" onClick={running ? onCancel : onClose}>
              {running ? "取消任务" : "关闭"}
            </button>
            <button type="button" className="btn btn-primary" onClick={submit} disabled={!running || !draft.trim()}>
              发送
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
