import { useCallback, useState } from "react";

export function PageHeader({ title, description }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      {description && <p>{description}</p>}
    </header>
  );
}

export function Alert({ type = "error", message }) {
  if (!message) return null;
  return <div className={`alert ${type}`}>{message}</div>;
}

export function LoadingBar({ loading }) {
  if (!loading) return null;
  return <div className="loading-bar" />;
}

export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel = "确认",
  cancelLabel = "取消",
  loading = false,
  onConfirm,
  onCancel,
}) {
  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title">
        <h2 id="confirm-dialog-title" className="modal-title">{title}</h2>
        <div className="modal-body">{children}</div>
        <div className="modal-actions">
          <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={loading}>
            {cancelLabel}
          </button>
          <button type="button" className="btn btn-primary" onClick={onConfirm} disabled={loading}>
            {loading ? "处理中…" : confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}

export function useAsyncAction() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const run = useCallback(async (action, successMessage = "") => {
    setLoading(true);
    setError("");
    setSuccess("");
    try {
      const result = await action();
      if (successMessage) setSuccess(successMessage);
      return result;
    } catch (err) {
      setError(err.message || "操作失败");
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { loading, error, success, run, setError, setSuccess };
}

export function StatusBadge({ ready }) {
  return (
    <span className={`badge ${ready ? "ok" : "missing"}`}>
      {ready ? "已就绪" : "未配置"}
    </span>
  );
}

export function EditorCard({
  title,
  value,
  onChange,
  onSave,
  saving,
  placeholder,
  short = false,
}) {
  return (
    <section className="card">
      <h2 className="card-title">{title}</h2>
      <div className="field">
        <textarea
          className={short ? "short" : ""}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
        />
      </div>
      {onSave && (
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={onSave}
            disabled={saving}
          >
            {saving ? "保存中…" : "保存"}
          </button>
        </div>
      )}
    </section>
  );
}
