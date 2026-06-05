import { useCallback, useRef, useState } from "react";
import { text, useLanguage } from "../i18n.jsx";

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
  return (
    <div className={`alert ${type}`} role={type === "error" ? "alert" : "status"}>
      {message}
    </div>
  );
}

export function LoadingBar({ loading }) {
  if (!loading) return null;
  return <div className="loading-bar" />;
}

export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel,
  cancelLabel,
  confirmDisabled = false,
  loading = false,
  onConfirm,
  onCancel,
}) {
  const { language } = useLanguage();
  const common = text[language].common;

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title">
        <h2 id="confirm-dialog-title" className="modal-title">{title}</h2>
        <div className="modal-body">{children}</div>
        <div className="modal-actions">
          <button type="button" className="btn btn-secondary" onClick={onCancel} disabled={loading}>
            {cancelLabel || common.cancel}
          </button>
          <button type="button" className="btn btn-primary" onClick={onConfirm} disabled={loading || confirmDisabled}>
            {loading ? common.loading : confirmLabel || common.confirm}
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
  const pendingCountRef = useRef(0);

  const run = useCallback(async (action, successMessage = "") => {
    pendingCountRef.current += 1;
    setLoading(true);
    setError("");
    setSuccess("");
    try {
      const result = await action();
      if (successMessage) setSuccess(successMessage);
      return result;
    } catch (err) {
      setError(err.message || "Operation failed");
      return null;
    } finally {
      pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);
      if (pendingCountRef.current === 0) setLoading(false);
    }
  }, []);

  return { loading, error, success, run, setError, setSuccess };
}

export function StatusBadge({ ready }) {
  const { language } = useLanguage();
  const common = text[language].common;

  return (
    <span className={`badge ${ready ? "ok" : "missing"}`}>
      {ready ? common.ready : common.missing}
    </span>
  );
}

export function EditorCard({
  title,
  value,
  onChange,
  onSave,
  saving,
  disabled = false,
  placeholder,
  short = false,
  extraActions = null,
}) {
  const { language } = useLanguage();
  const common = text[language].common;

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
            disabled={disabled || saving}
          >
            {saving ? common.saving : common.save}
          </button>
          {extraActions}
        </div>
      )}
    </section>
  );
}
