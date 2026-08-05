import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import { useLanguage } from "../../i18n.jsx";
import {
  associationExists,
  buildConfirmationPayload,
  isCanonicalRepositoryInput,
  repositoryDisplayName,
  repositoryItemKey,
} from "./repositoryAssociation.js";

const COPY = {
  en: {
    title: "Connect GitHub repositories",
    description: "Link each GitHub repository to the project it belongs to so WorkAgent can use the correct project evidence.",
    loading: "Loading repository associations…",
    loadError: "Repository associations could not be loaded.",
    projectsError: "Projects could not be loaded. Retry before connecting a repository.",
    retry: "Retry",
    complete: "All detected GitHub repositories are linked to projects.",
    incomplete: "The complete GitHub repository is required.",
    enterFull: "Enter the full GitHub repository in owner/repository format.",
    detected: "Detected repository",
    conflict: "This repository has conflicting associations and cannot be connected here.",
    connect: "Connect repository",
    cancel: "Cancel",
    repository: "GitHub repository",
    repositoryPlaceholder: "owner/repository",
    project: "Project",
    selectProject: "Select project",
    linked: "Already linked",
    invalidRepository: "Enter a valid repository in owner/repository format.",
    projectRequired: "Select a project.",
    connecting: "Connecting…",
    success: "Repository connected to project.",
    successRemaining: "Repository connected. Additional repositories still need to be linked.",
    unchanged: "This repository is already connected to the selected project.",
    blocked: "This repository is already associated with another project.",
    degraded: "Repository connected, but its status could not be refreshed. Reload the page to check again.",
    ambiguousConnected: "The connection was interrupted, but the repository association is saved.",
    ambiguous: "The connection result is uncertain. The list was refreshed; check before trying again.",
  },
  zh: {
    title: "关联 GitHub 仓库",
    description: "将每个 GitHub 仓库关联到对应项目，帮助 WorkAgent 使用正确的项目证据。",
    loading: "正在加载仓库关联…",
    loadError: "暂时无法加载仓库关联。",
    projectsError: "暂时无法加载项目。请重试后再关联仓库。",
    retry: "重试",
    complete: "检测到的 GitHub 仓库都已关联到项目。",
    incomplete: "需要完整的 GitHub 仓库标识。",
    enterFull: "请输入 owner/repository 格式的完整 GitHub 仓库。",
    detected: "检测到的仓库",
    conflict: "此仓库存在关联冲突，无法在这里继续关联。",
    connect: "关联仓库",
    cancel: "取消",
    repository: "GitHub 仓库",
    repositoryPlaceholder: "owner/repository",
    project: "项目",
    selectProject: "选择项目",
    linked: "已关联",
    invalidRepository: "请输入有效的 owner/repository 格式仓库。",
    projectRequired: "请选择项目。",
    connecting: "正在关联…",
    success: "仓库已关联到项目。",
    successRemaining: "仓库已关联，仍有其他仓库需要处理。",
    unchanged: "此仓库已经关联到所选项目。",
    blocked: "此仓库已经关联到另一个项目。",
    degraded: "仓库已关联，但暂时无法刷新状态。请重新加载页面后查看。",
    ambiguousConnected: "连接曾中断，但仓库关联已经保存。",
    ambiguous: "连接结果暂不确定。列表已刷新，请确认后再重试。",
  },
};

export default function RepositoryAssociationSection({ onAssociationChanged }) {
  const { language } = useLanguage();
  const copy = COPY[language] || COPY.en;
  const [repositories, setRepositories] = useState(null);
  const [projects, setProjects] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [projectError, setProjectError] = useState("");
  const [selected, setSelected] = useState(null);
  const [repository, setRepository] = useState("");
  const [projectId, setProjectId] = useState("");
  const [fieldError, setFieldError] = useState({ repository: "", project: "" });
  const [message, setMessage] = useState({ type: "", text: "" });
  const [pending, setPending] = useState(false);
  const pendingRef = useRef(false);
  const repositoryInputRef = useRef(null);

  const loadAssociations = useCallback(async (signal) => {
    setLoading(true);
    setLoadError("");
    setProjectError("");
    const [repositoryResult, projectResult] = await Promise.allSettled([
      api.getUnresolvedRepositoryMappings({ signal }),
      api.getRepositoryMappingProjects({ signal }),
    ]);
    if (signal?.aborted) return null;
    const nextRepositories = repositoryResult.status === "fulfilled"
      ? (Array.isArray(repositoryResult.value?.repositories) ? repositoryResult.value.repositories : [])
      : null;
    const nextProjects = projectResult.status === "fulfilled"
      ? (Array.isArray(projectResult.value?.projects) ? projectResult.value.projects : [])
      : null;
    setRepositories(nextRepositories);
    setProjects(nextProjects);
    if (nextRepositories === null) setLoadError(copy.loadError);
    if (nextProjects === null) setProjectError(copy.projectsError);
    setLoading(false);
    return { unresolved: nextRepositories || [], projects: nextProjects || [] };
  }, [copy.loadError, copy.projectsError]);

  useEffect(() => {
    const controller = new AbortController();
    loadAssociations(controller.signal);
    return () => controller.abort();
  }, [loadAssociations]);

  useEffect(() => {
    if (selected) repositoryInputRef.current?.focus();
  }, [selected]);

  const beginAssociation = (item) => {
    if (pending || item.conflicting) return;
    setSelected(item);
    setRepository(item.canonical ? item.repository || "" : "");
    setProjectId("");
    setFieldError({ repository: "", project: "" });
    setMessage({ type: "", text: "" });
  };

  const cancelAssociation = () => {
    if (pending) return;
    setSelected(null);
    setRepository("");
    setProjectId("");
    setFieldError({ repository: "", project: "" });
  };

  const confirmAssociation = async (event) => {
    event.preventDefault();
    if (pendingRef.current) return;
    const errors = {
      repository: isCanonicalRepositoryInput(repository) ? "" : copy.invalidRepository,
      project: projectId ? "" : copy.projectRequired,
    };
    setFieldError(errors);
    if (errors.repository || errors.project) return;
    const payload = buildConfirmationPayload({
      projectId,
      repository,
      repositoryAlias: selected?.canonical ? "" : selected?.repository_alias,
    });
    pendingRef.current = true;
    setPending(true);
    setMessage({ type: "", text: "" });
    try {
      const result = await api.confirmRepositoryMapping(payload);
      if (["created", "updated", "unchanged"].includes(result?.status)) {
        const refreshed = await loadAssociations();
        const remaining = refreshed?.unresolved?.length > 0;
        setMessage({
          type: "success",
          text: result.status === "unchanged" ? copy.unchanged : remaining ? copy.successRemaining : copy.success,
        });
        setSelected(null);
        onAssociationChanged?.();
      } else if (result?.status === "degraded") {
        await loadAssociations();
        setMessage({ type: "warning", text: copy.degraded });
        setSelected(null);
        onAssociationChanged?.();
      } else {
        setMessage({ type: "error", text: copy.blocked });
      }
    } catch {
      const refreshed = await loadAssociations();
      const saved = associationExists({
        unresolved: refreshed?.unresolved,
        projects: refreshed?.projects,
        repository: payload.repository,
        projectId: payload.project_id,
      });
      setMessage({ type: saved ? "success" : "warning", text: saved ? copy.ambiguousConnected : copy.ambiguous });
      if (saved) {
        setSelected(null);
        onAssociationChanged?.();
      }
    } finally {
      pendingRef.current = false;
      setPending(false);
    }
  };

  return (
    <section className="card repository-association-section" aria-labelledby="repository-association-title">
      <div className="repository-association-heading">
        <div>
          <h2 id="repository-association-title" className="card-title">{copy.title}</h2>
          <p className="helper-text">{copy.description}</p>
        </div>
        {(loadError || projectError) && (
          <button type="button" className="btn btn-secondary" onClick={() => loadAssociations()} disabled={loading || pending}>
            {copy.retry}
          </button>
        )}
      </div>

      <div className={`repository-association-message ${message.type}`} role="status" aria-live="polite">
        {message.text}
      </div>
      {loading && <p className="repository-association-loading" role="status">{copy.loading}</p>}
      {!loading && loadError && <p className="repository-association-error" role="alert">{loadError}</p>}
      {!loading && !loadError && projectError && <p className="repository-association-error" role="alert">{projectError}</p>}
      {!loading && repositories?.length === 0 && !loadError && (
        <p className="empty-state">{copy.complete}</p>
      )}

      {!loading && repositories?.length > 0 && (
        <div className="repository-association-list">
          {repositories.map((item, index) => {
            const key = repositoryItemKey(item, index);
            const active = selected && repositoryItemKey(selected, index) === key;
            return (
              <article className={`repository-association-item${item.conflicting ? " conflict" : ""}`} key={key}>
                <div className="repository-association-summary">
                  <div className="repository-association-name">{repositoryDisplayName(item)}</div>
                  {!item.canonical && <p className="helper-text">{copy.incomplete} {copy.enterFull}</p>}
                  {item.conflicting && <p className="repository-association-conflict">{copy.conflict}</p>}
                </div>
                {!active && (
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => beginAssociation(item)}
                    disabled={pending || item.conflicting || !projects}
                    aria-label={`${copy.connect}: ${repositoryDisplayName(item)}`}
                  >
                    {copy.connect}
                  </button>
                )}
                {active && (
                  <form className="repository-association-form" onSubmit={confirmAssociation} noValidate>
                    {!item.canonical && <p className="helper-text">{copy.detected}: <strong>{item.repository_alias}</strong></p>}
                    <div className="field">
                      <label htmlFor={`${key}-repository`}>{copy.repository}</label>
                      <input
                        ref={repositoryInputRef}
                        id={`${key}-repository`}
                        value={repository}
                        onChange={(event) => setRepository(event.target.value)}
                        placeholder={copy.repositoryPlaceholder}
                        maxLength={500}
                        autoComplete="off"
                        disabled={pending}
                        aria-invalid={Boolean(fieldError.repository)}
                        aria-describedby={fieldError.repository ? `${key}-repository-error` : undefined}
                      />
                      {fieldError.repository && <p id={`${key}-repository-error`} className="field-error">{fieldError.repository}</p>}
                    </div>
                    <div className="field">
                      <label htmlFor={`${key}-project`}>{copy.project}</label>
                      <select
                        id={`${key}-project`}
                        value={projectId}
                        onChange={(event) => setProjectId(event.target.value)}
                        disabled={pending || !projects}
                        aria-invalid={Boolean(fieldError.project)}
                        aria-describedby={fieldError.project ? `${key}-project-error` : undefined}
                      >
                        <option value="">{copy.selectProject}</option>
                        {(projects || []).map((project) => (
                          <option key={project.project_id} value={project.project_id}>
                            {project.project_name}
                            {project.already_linked_repositories?.length
                              ? ` — ${copy.linked}: ${project.already_linked_repositories.join(", ")}`
                              : ""}
                          </option>
                        ))}
                      </select>
                      {fieldError.project && <p id={`${key}-project-error`} className="field-error">{fieldError.project}</p>}
                    </div>
                    <div className="repository-association-actions">
                      <button type="button" className="btn btn-secondary" onClick={cancelAssociation} disabled={pending}>{copy.cancel}</button>
                      <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={pending || !projects || !projectId || !isCanonicalRepositoryInput(repository)}
                      >
                        {pending ? copy.connecting : copy.connect}
                      </button>
                    </div>
                  </form>
                )}
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
