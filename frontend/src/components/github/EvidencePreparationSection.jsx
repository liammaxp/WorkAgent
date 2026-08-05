import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import { useLanguage } from "../../i18n.jsx";
import { canStartPreparation, outcomeMessage, safeRemainingRepositoryText, shouldReconcileAfterOutcome, statusMessage } from "./evidencePreparation.js";

const COPY = {
  en: {
    title: "Prepare project evidence",
    description: "Prepare saved GitHub information so WorkAgent can understand the projects connected above.",
    clarification: "Preparing evidence does not generate or update a resume.",
    loading: "Loading project evidence status...", refresh: "Refresh status", prepare: "Prepare evidence",
    preparing: "Preparing project evidence...", confirmQuestion: "Prepare saved GitHub information for the projects connected above?",
    confirmSupport: "This prepares project evidence locally. It does not generate or update a resume.", cancel: "Cancel",
    unknownReconciled: "The request result could not be confirmed. The latest evidence status has been loaded.",
    unknownUnresolved: "The request result could not be confirmed. Refresh the status before trying again.",
  },
  zh: {
    title: "准备项目证据", description: "准备已保存的 GitHub 信息，帮助 WorkAgent 理解上方已关联的项目。",
    clarification: "准备证据不会生成或更新简历。", loading: "正在加载项目证据状态...", refresh: "刷新状态",
    prepare: "准备证据", preparing: "正在准备项目证据...", confirmQuestion: "是否为上方已关联的项目准备已保存的 GitHub 信息？",
    confirmSupport: "此操作会在本地准备项目证据，不会生成或更新简历。", cancel: "取消",
    unknownReconciled: "无法确认请求结果，已加载最新的证据状态。", unknownUnresolved: "无法确认请求结果，请先刷新状态再重试。",
  },
};

export default function EvidencePreparationSection({ refreshSignal = 0 }) {
  const { language } = useLanguage();
  const copy = COPY[language] || COPY.en;
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [pending, setPending] = useState(false);
  const [feedback, setFeedback] = useState({ type: "", text: "" });
  const pendingRef = useRef(false);

  const loadStatus = useCallback(async ({ signal, preserveFeedback = false } = {}) => {
    setLoading(true); setLoadError(false);
    if (!preserveFeedback) setFeedback({ type: "", text: "" });
    try {
      const result = await api.getGitHubEvidencePreparationStatus({ signal });
      if (signal?.aborted) return null;
      setStatus(result || null);
      return result || null;
    } catch (error) {
      if (signal?.aborted || error?.name === "AbortError") return null;
      setLoadError(true);
      return null;
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadStatus({ signal: controller.signal });
    return () => controller.abort();
  }, [loadStatus, refreshSignal]);

  const startPreparation = async () => {
    if (pendingRef.current || !canStartPreparation(status)) return;
    pendingRef.current = true; setPending(true); setFeedback({ type: "", text: "" });
    try {
      const result = await api.runGitHubEvidencePreparation();
      const resultStatus = result?.status || "error";
      const success = ["created", "updated", "unchanged"].includes(resultStatus);
      const warning = ["partial", "empty", "busy", "degraded"].includes(resultStatus);
      setFeedback({ type: success ? "success" : warning ? "warning" : "error", text: outcomeMessage(resultStatus, language) });
      setConfirming(false);
      if (shouldReconcileAfterOutcome(resultStatus)) await loadStatus({ preserveFeedback: true });
    } catch {
      const reconciled = await loadStatus({ preserveFeedback: true });
      setFeedback({ type: "warning", text: reconciled ? copy.unknownReconciled : copy.unknownUnresolved });
    } finally {
      pendingRef.current = false; setPending(false);
    }
  };

  const currentStatus = loadError ? "error" : status?.status || "error";
  const remainingText = safeRemainingRepositoryText(status, language);
  const actionAvailable = !loading && !loadError && canStartPreparation(status);
  return (
    <section className="card evidence-preparation-section" aria-labelledby="evidence-preparation-title">
      <div className="evidence-preparation-heading"><div>
        <h2 id="evidence-preparation-title" className="card-title">{copy.title}</h2><p className="helper-text">{copy.description}</p>
      </div><button type="button" className="btn btn-secondary" onClick={() => loadStatus()} disabled={loading || pending}>{copy.refresh}</button></div>
      {feedback.text && <div className={`evidence-preparation-feedback ${feedback.type}`} role="status" aria-live="polite">{feedback.text}</div>}
      {loading ? <p className="evidence-preparation-loading" role="status">{copy.loading}</p> :
        <div className={`evidence-preparation-state ${currentStatus}`} role={loadError ? "alert" : "status"}>
          <p>{statusMessage(currentStatus, language)}</p>{remainingText && <p className="helper-text">{remainingText}</p>}
          <p className="helper-text" id="evidence-preparation-clarification">{copy.clarification}</p>
        </div>}
      {actionAvailable && !confirming && <div className="evidence-preparation-actions"><button type="button" className="btn btn-primary" onClick={() => setConfirming(true)} disabled={pending} aria-describedby="evidence-preparation-clarification">{copy.prepare}</button></div>}
      {actionAvailable && confirming && <div className="evidence-preparation-confirmation" role="group" aria-labelledby="evidence-preparation-confirmation-title">
        <h3 id="evidence-preparation-confirmation-title">{copy.confirmQuestion}</h3><p className="helper-text">{copy.confirmSupport}</p>
        <div className="evidence-preparation-actions"><button type="button" className="btn btn-secondary" onClick={() => setConfirming(false)} disabled={pending}>{copy.cancel}</button>
          <button type="button" className="btn btn-primary" onClick={startPreparation} disabled={pending}>{pending ? copy.preparing : copy.prepare}</button></div>
      </div>}
    </section>
  );
}
