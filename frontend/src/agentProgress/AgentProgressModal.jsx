import { useEffect, useMemo, useRef, useState } from "react";

function stageIcon(status) {
  if (status === "running") return <span className="agent-progress-spinner" aria-hidden="true" />;
  if (status === "waiting_for_user") return <span className="agent-progress-icon waiting" aria-hidden="true" />;
  if (status === "done") return <span className="agent-progress-icon done">✓</span>;
  if (status === "error") return <span className="agent-progress-icon error">×</span>;
  if (status === "cancelled") return <span className="agent-progress-icon cancelled">-</span>;
  return <span className="agent-progress-icon pending" />;
}

function formatTime(value) {
  if (!value) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function statusLabel(status) {
  if (status === "success") return "任务完成";
  if (status === "error") return "任务失败";
  if (status === "cancelled") return "已取消";
  return "正在执行";
}

function ErrorDetail({ detail }) {
  if (!detail || typeof detail !== "object") return null;
  const failedChunk = detail.failedChunk || {};
  const rows = [
    ["Type", detail.type],
    ["Caller", detail.caller],
    ["Status", detail.statusCode || detail.status],
    ["Retryable", detail.retryable === true ? "yes" : detail.retryable === false ? "no" : ""],
    ["Project", failedChunk.projectName],
    ["Repository", failedChunk.repoName],
    ["Chunk", failedChunk.chunkIndex],
    ["Input chars", detail.inputCharCount],
    ["Limit", detail.limit],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");

  if (!rows.length) return null;

  return (
    <div className="agent-progress-error-detail">
      <div className="agent-progress-error-detail-title">Structured error detail</div>
      <dl>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{String(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function AgentProgressModal({
  task,
  onCancel,
  onClose,
  onMinimize,
  onRestore,
  onSend,
}) {
  const [draft, setDraft] = useState("");
  const messagesEndRef = useRef(null);
  const isOpen = Boolean(task) && !task?.minimized;
  const running = task?.status === "running";
  const currentStage = useMemo(
    () =>
      task?.stages.find((stage) => stage.id === task.currentStageId) ||
      task?.stages.find((stage) => stage.status === "running") ||
      task?.stages.find((stage) => stage.status === "waiting_for_user"),
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
    if (!isOpen) return;
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [isOpen, task?.messages?.length]);

  useEffect(() => {
    if (!task) setDraft("");
  }, [task]);

  if (!task) return null;

  const submit = () => {
    const content = draft.trim();
    if (!content || !running) return;
    onSend(content);
    setDraft("");
  };

  const closeOrMinimize = () => {
    if (running) {
      onMinimize();
      return;
    }
    onClose();
  };

  const currentStageLabel =
    currentStage?.detail ||
    currentStage?.label ||
    (task.status === "success" ? "已完成，5 秒后自动关闭" : "正在调用 Agent");

  if (task.minimized) {
    return (
      <button type="button" className="agent-progress-fab" onClick={onRestore} aria-label="重新打开 Agent 进度弹窗">
        <span className={`agent-progress-fab-dot ${task.status}`} />
        <span>
          <strong>{task.title}</strong>
          <small>{statusLabel(task.status)} · {currentStageLabel}</small>
        </span>
      </button>
    );
  }

  return (
    <div className="agent-progress-backdrop" role="presentation">
      <section className="agent-progress-modal" role="dialog" aria-modal="true" aria-labelledby="agent-progress-title">
        <header className="agent-progress-header">
          <div>
            <h2 id="agent-progress-title">{task.title}</h2>
            <p>当前阶段：{currentStageLabel}</p>
          </div>
          <button
            type="button"
            className="agent-progress-close"
            aria-label={running ? "最小化" : "关闭"}
            title={running ? "最小化到右下角" : "关闭"}
            onClick={closeOrMinimize}
          >
            {running ? "_" : "×"}
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

        {task.error && (
          <div className="agent-progress-error" role="alert">
            <div>{task.error}</div>
            <ErrorDetail detail={task.errorDetail} />
          </div>
        )}

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
