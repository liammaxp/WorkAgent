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
      if (err?.name === "AgentCancelledError" || err?.name === "AbortError") {
        return null;
      }
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

function outputBaseName(value = "") {
  return String(value || "").split(/[\\/]/).pop() || "";
}

function outputSequenceSuffix(rawName = "") {
  const withoutExtension = String(rawName || "").replace(/\.[^.\\/]+$/, "");
  const match = withoutExtension.match(/_(\d+)$/);
  return match ? ` (${match[1]})` : "";
}

function cleanOutputFallbackName(rawName = "") {
  return String(rawName || "")
    .replace(/\.[^.\\/]+$/, "")
    .replace(/_(\d+)$/, " ($1)")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseCompanyRoleFromOutputName(rawName = "") {
  const withoutExtension = String(rawName || "").replace(/\.[^.\\/]+$/, "");
  const stem = withoutExtension.replace(/_(\d+)$/, "").trim();
  if (/^(tailored_resume|cover_letter|interview_prep|chat_session)(?:_|$)/i.test(stem)) return "";
  const parts = stem.split("_").map((part) => part.trim()).filter(Boolean);
  if (parts.length < 2) return "";
  return `${parts[0]} / ${parts.slice(1).join(" ")}`;
}

function outputDisplayName(file) {
  const rawName = outputBaseName(file.name || file.path || "");
  const company = String(file.company || file.employer || "").trim();
  const role = String(file.role || file.position || file.job_title || "").trim();
  const sequence = outputSequenceSuffix(rawName);

  if (company || role) {
    return `${[company, role].filter(Boolean).join(" / ")}${sequence}`;
  }

  const parsedName = parseCompanyRoleFromOutputName(rawName);
  if (parsedName) return `${parsedName}${sequence}`;

  return cleanOutputFallbackName(rawName) || file.generated_at_display || file.updated_at_display || "-";
}

export function OutputFileSelect({
  files = [],
  value = "",
  onSelect,
  onOpen = null,
  onDelete = null,
  disabled = false,
  showWhenEmpty = false,
  label: customLabel = "",
  placeholder: customPlaceholder = "",
  inline = false,
}) {
  const { language } = useLanguage();
  const label = customLabel || (language === "zh" ? "历史输出" : "Output history");
  const placeholder = customPlaceholder || (language === "zh" ? "选择公司 / 职位以查看内容" : "Choose a company / role to view");
  const openLabel = language === "zh" ? "打开" : "Open";
  const deleteLabel = language === "zh" ? "删除" : "Delete";

  if (!files.length && !showWhenEmpty) return null;

  return (
    <div className={`field output-file-select${inline ? " output-file-select-inline" : ""}`}>
      {!inline && <label>{label}</label>}
      <div className="output-file-control">
        {inline && <span className="output-file-label">{label}</span>}
        <select
          value={files.some((file) => file.path === value) ? value : ""}
          disabled={disabled || !files.length}
          aria-label={label}
          onChange={(event) => {
            const path = event.target.value;
            if (path) onSelect(path);
          }}
        >
          <option value="">{placeholder}</option>
          {files.map((file) => {
            const displayName = outputDisplayName(file);
            return <option key={file.path} value={file.path}>{displayName}</option>;
          })}
        </select>
      </div>
      {(onOpen || onDelete) && (
        <div className="output-file-actions">
          {onOpen && (
            <button type="button" className="btn btn-secondary" disabled={disabled || !value} onClick={() => onOpen(value)}>
              {openLabel}
            </button>
          )}
          {onDelete && (
            <button type="button" className="btn btn-danger" disabled={disabled || !value} onClick={() => onDelete(value)}>
              {deleteLabel}
            </button>
          )}
        </div>
      )}
    </div>
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
