import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../../api/client.js";
import { useLanguage } from "../../i18n.jsx";
import {
  REVIEW_STATUS,
  createReviewRequest,
  draftFromReview,
  normalizeReviewResponse,
  reviewEntryVisible,
  shouldCloseReview,
} from "./hiringContextRankingReview.js";
import {
  EMPHASIS_ACTION,
  EMPHASIS_ACTIONS,
  createPreferenceOverlay,
  preferenceControlsVisible,
  projectAction,
  reconcilePreferenceOverlay,
  resetAllPreferences,
  setProjectPreference,
  setStoryPreference,
  storyAction,
  summarizePreferences,
} from "./resumeEmphasisPreferences.js";

const COPY = {
  en: {
    open: "Review tailoring context",
    close: "Close review",
    title: "Tailoring context review",
    description: "See what the role appears to value and which factual project stories may be worth emphasizing.",
    loading: "Reviewing what to emphasize…",
    error: "Could not prepare the tailoring review.",
    retry: "Try again",
    unavailable: "The tailoring review is not available yet.",
    empty: "No engineering stories are available to rank yet.",
    contextTitle: "What this role appears to value",
    contextBoundary: "These are employer and role signals, not claims about your experience.",
    company: "Company",
    team: "Team",
    role: "Role",
    roleFocus: "Primary role focus",
    secondaryFocus: "Other role focus",
    confidence: "Context confidence",
    unknown: "Not identified",
    projectsTitle: "Projects worth emphasizing",
    strongest: "Strongest stories",
    additional: "Other relevant stories",
    why: "Why this is relevant",
    correctionTitle: "Review the job identity",
    previewOnly: "Preview only. These edits change this review, not your saved job description or candidate experience.",
    apply: "Update preview",
    reset: "Reset preview",
    preferenceSession: "These emphasis choices apply only to this review session.",
    preferenceLabel: "Resume preference for",
    defaultAction: "Default",
    emphasizeMore: "Emphasize more",
    emphasizeLess: "Emphasize less",
    keep: "Keep",
    exclude: "Exclude from this resume",
    keepBoundary: "Keep retains an item in consideration. It does not guarantee final inclusion.",
    projectExcluded: "Excluded for this resume. Restore the project to change Story preferences.",
    summaryTitle: "Your emphasis choices",
    emphasized: "emphasized",
    lowered: "lowered",
    kept: "kept",
    excluded: "excluded",
    resetAll: "Reset all emphasis choices",
    contextReset: "Emphasis choices were reset because the role context changed.",
    removedReset: "Choices for projects or Stories no longer in this review were removed.",
  },
  zh: {
    open: "查看定制重点",
    close: "关闭评审",
    title: "简历定制重点评审",
    description: "查看这个职位重视什么，以及哪些真实项目故事更值得突出。",
    loading: "正在评审值得突出的内容…",
    error: "暂时无法准备定制重点评审。",
    retry: "重试",
    unavailable: "定制重点评审暂时不可用。",
    empty: "目前还没有可排序的工程故事。",
    contextTitle: "这个职位看起来重视什么",
    contextBoundary: "这些是雇主与职位信号，不代表你的候选人经历。",
    company: "公司",
    team: "团队",
    role: "职位",
    roleFocus: "主要职位方向",
    secondaryFocus: "其他职位方向",
    confidence: "上下文理解程度",
    unknown: "尚未识别",
    projectsTitle: "值得重点展示的项目",
    strongest: "最强故事",
    additional: "其他相关故事",
    why: "相关原因",
    correctionTitle: "核对职位信息",
    previewOnly: "仅用于预览。这些修改只会更新本次评审，不会更改已保存的职位描述或候选人经历。",
    apply: "更新预览",
    reset: "重置预览",
    preferenceSession: "这些重点选择仅适用于当前评审会话。",
    preferenceLabel: "本份简历对此项的偏好：",
    defaultAction: "默认",
    emphasizeMore: "更多突出",
    emphasizeLess: "减少突出",
    keep: "保留考虑",
    exclude: "从本份简历中排除",
    keepBoundary: "“保留考虑”表示明确留在候选集合中，并不保证最终一定进入简历。",
    projectExcluded: "已从本份简历中排除。恢复项目后才能修改其故事偏好。",
    summaryTitle: "你的重点选择",
    emphasized: "项更多突出",
    lowered: "项减少突出",
    kept: "项保留考虑",
    excluded: "项已排除",
    resetAll: "重置全部重点选择",
    contextReset: "职位上下文已改变，原有重点选择已重置。",
    removedReset: "已移除不再出现在本次评审中的项目或故事选择。",
  },
};

function actionLabel(action, copy) {
  return {
    [EMPHASIS_ACTION.DEFAULT]: copy.defaultAction,
    [EMPHASIS_ACTION.EMPHASIZE_MORE]: copy.emphasizeMore,
    [EMPHASIS_ACTION.EMPHASIZE_LESS]: copy.emphasizeLess,
    [EMPHASIS_ACTION.KEEP]: copy.keep,
    [EMPHASIS_ACTION.EXCLUDE]: copy.exclude,
  }[action];
}

function PreferenceControl({ id, itemLabel, value, disabled = false, onChange, copy }) {
  const label = `${copy.preferenceLabel} ${itemLabel}`;
  return (
    <div className="ranking-review-preference-control">
      <label htmlFor={id}>{label}</label>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        aria-label={label}
      >
        {EMPHASIS_ACTIONS.map((action) => (
          <option value={action} key={action}>{actionLabel(action, copy)}</option>
        ))}
      </select>
    </div>
  );
}

function StoryList({
  stories,
  label,
  copy,
  project,
  groupKey,
  preferenceEnabled,
  overlay,
  onStoryPreference,
}) {
  if (!stories.length) return null;
  const projectExcluded = projectAction(overlay, project.projectId) === EMPHASIS_ACTION.EXCLUDE;
  return (
    <div className="ranking-review-story-group">
      <h5>{label}</h5>
      <ol className="ranking-review-story-list">
        {stories.map((story, index) => {
          const action = storyAction(overlay, project.projectId, story.storyId);
          return (
            <li
              key={story.storyId}
              className={preferenceEnabled && action === EMPHASIS_ACTION.EXCLUDE ? "is-resume-excluded" : undefined}
            >
              <div className="ranking-review-story-heading">{story.label}</div>
              {preferenceEnabled && (
                <PreferenceControl
                  id={`ranking-review-story-preference-${project.position}-${groupKey}-${index + 1}`}
                  itemLabel={story.label}
                  value={action}
                  disabled={projectExcluded}
                  onChange={(nextAction) => onStoryPreference(project.projectId, story.storyId, nextAction)}
                  copy={copy}
                />
              )}
              {story.relevanceReasons.length > 0 && (
                <div className="ranking-review-traits" aria-label={copy.why}>
                  {story.relevanceReasons.map((reason) => (
                    <span className="ranking-review-trait" key={reason}>{reason}</span>
                  ))}
                </div>
              )}
              {story.notices.map((notice) => (
                <p className="ranking-review-notice" key={notice}>{notice}</p>
              ))}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function HiringContextSummary({ context, copy }) {
  return (
    <section className="ranking-review-context" aria-labelledby="ranking-review-context-title">
      <h3 id="ranking-review-context-title">{copy.contextTitle}</h3>
      <p className="helper-text">{copy.contextBoundary}</p>
      <dl className="ranking-review-identity">
        <div><dt>{copy.company}</dt><dd>{context.company || copy.unknown}</dd></div>
        <div><dt>{copy.role}</dt><dd>{context.roleTitle || copy.unknown}</dd></div>
        {context.team && <div><dt>{copy.team}</dt><dd>{context.team}</dd></div>}
        <div><dt>{copy.roleFocus}</dt><dd>{context.primaryRoleFamily}</dd></div>
        {context.secondaryRoleFamilies.length > 0 && (
          <div><dt>{copy.secondaryFocus}</dt><dd>{context.secondaryRoleFamilies.join(", ")}</dd></div>
        )}
        <div><dt>{copy.confidence}</dt><dd>{context.confidence}</dd></div>
      </dl>
      {context.contextSignals.length > 0 && (
        <div className="ranking-review-traits">
          {context.contextSignals.map((signal) => (
            <span className="ranking-review-trait" key={signal}>{signal}</span>
          ))}
        </div>
      )}
    </section>
  );
}

function PreferenceSummary({ overlay, copy, onResetAll }) {
  const summary = summarizePreferences(overlay);
  if (!summary.total) return null;
  return (
    <aside className="ranking-review-preference-summary" aria-labelledby="ranking-review-preference-summary-title">
      <div>
        <h3 id="ranking-review-preference-summary-title">{copy.summaryTitle}</h3>
        <p aria-live="polite">
          {summary.emphasized > 0 && <span>{summary.emphasized} {copy.emphasized}</span>}
          {summary.lowered > 0 && <span>{summary.lowered} {copy.lowered}</span>}
          {summary.kept > 0 && <span>{summary.kept} {copy.kept}</span>}
          {summary.excluded > 0 && <span>{summary.excluded} {copy.excluded}</span>}
        </p>
      </div>
      <button type="button" className="btn btn-secondary" onClick={onResetAll}>{copy.resetAll}</button>
    </aside>
  );
}

function ProjectReview({
  projects,
  copy,
  preferenceEnabled,
  overlay,
  onProjectPreference,
  onStoryPreference,
}) {
  if (!projects.length) return null;
  return (
    <section className="ranking-review-projects" aria-labelledby="ranking-review-projects-title">
      <h3 id="ranking-review-projects-title">{copy.projectsTitle}</h3>
      <ol className="ranking-review-project-list">
        {projects.map((project) => {
          const action = projectAction(overlay, project.projectId);
          const excluded = preferenceEnabled && action === EMPHASIS_ACTION.EXCLUDE;
          return (
            <li key={project.projectId}>
              <article className={`ranking-review-project${excluded ? " is-resume-excluded" : ""}`}>
              {preferenceEnabled ? (
                <div className="ranking-review-project-heading">
                  <h4>{project.displayName}</h4>
                  <PreferenceControl
                    id={`ranking-review-project-preference-${project.position}`}
                    itemLabel={project.displayName}
                    value={action}
                    onChange={(nextAction) => onProjectPreference(project.projectId, nextAction)}
                    copy={copy}
                  />
                </div>
              ) : <h4>{project.displayName}</h4>}
              {excluded && <p className="ranking-review-excluded-note">{copy.projectExcluded}</p>}
              {project.relevanceReasons.map((reason) => (
                <p className="ranking-review-project-reason" key={reason}>{reason}</p>
              ))}
              <StoryList
                stories={project.strongestStories}
                label={copy.strongest}
                copy={copy}
                project={project}
                groupKey="strongest"
                preferenceEnabled={preferenceEnabled}
                overlay={overlay}
                onStoryPreference={onStoryPreference}
              />
              {project.additionalStories.length > 0 && (
                <details className="ranking-review-additional">
                  <summary>{copy.additional}</summary>
                  <StoryList
                    stories={project.additionalStories}
                    label={copy.additional}
                    copy={copy}
                    project={project}
                    groupKey="additional"
                    preferenceEnabled={preferenceEnabled}
                    overlay={overlay}
                    onStoryPreference={onStoryPreference}
                  />
                </details>
              )}
              </article>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

export default function HiringContextRankingReview() {
  const { language } = useLanguage();
  const copy = COPY[language] || COPY.en;
  const [availability, setAvailability] = useState(null);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [review, setReview] = useState(null);
  const [draft, setDraft] = useState({ company: "", team: "", roleTitle: "" });
  const [preferenceSession, setPreferenceSession] = useState({
    overlay: null,
    contextStale: false,
    removed: false,
  });
  const triggerRef = useRef(null);
  const headingRef = useRef(null);
  const requestRef = useRef({ sequence: 0, controller: null });

  useEffect(() => {
    const controller = new AbortController();
    api.getHiringContextRankingReviewAvailability({ signal: controller.signal })
      .then((value) => {
        if (!controller.signal.aborted) setAvailability(value);
      })
      .catch(() => {
        if (!controller.signal.aborted) setAvailability({ available: false });
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (open) headingRef.current?.focus();
  }, [open]);

  useEffect(() => () => requestRef.current.controller?.abort(), []);

  const preferenceEnabled = preferenceControlsVisible(availability);

  useEffect(() => {
    if (!preferenceEnabled) {
      setPreferenceSession({ overlay: null, contextStale: false, removed: false });
      return;
    }
    if (review?.status !== REVIEW_STATUS.READY) return;
    setPreferenceSession((current) => reconcilePreferenceOverlay(current.overlay, review));
  }, [preferenceEnabled, review]);

  const loadReview = useCallback(async (nextDraft = null) => {
    requestRef.current.controller?.abort();
    const controller = new AbortController();
    const sequence = requestRef.current.sequence + 1;
    requestRef.current = { sequence, controller };
    setLoading(true);
    setError(false);
    setReview(null);
    try {
      const value = await api.reviewHiringContext(
        createReviewRequest(nextDraft, language, nextDraft !== null),
        { signal: controller.signal },
      );
      if (controller.signal.aborted || requestRef.current.sequence !== sequence) return;
      const normalized = normalizeReviewResponse(value);
      if (!normalized) throw new Error("invalid review response");
      setReview(normalized);
      setDraft(draftFromReview(normalized));
    } catch (requestError) {
      if (requestError?.name !== "AbortError" && requestRef.current.sequence === sequence) {
        setError(true);
      }
    } finally {
      if (requestRef.current.sequence === sequence) setLoading(false);
    }
  }, [language]);

  const openReview = () => {
    setOpen(true);
    loadReview();
  };

  const closeReview = () => {
    requestRef.current.controller?.abort();
    setOpen(false);
    window.setTimeout(() => triggerRef.current?.focus(), 0);
  };

  const handleKeyDown = (event) => {
    if (!shouldCloseReview(event.key)) return;
    event.preventDefault();
    closeReview();
  };

  const handleProjectPreference = (projectId, action) => {
    setPreferenceSession((current) => {
      const overlay = current.overlay || createPreferenceOverlay(review);
      return {
        overlay: setProjectPreference(overlay, review, projectId, action),
        contextStale: false,
        removed: false,
      };
    });
  };

  const handleStoryPreference = (projectId, storyId, action) => {
    setPreferenceSession((current) => {
      const overlay = current.overlay || createPreferenceOverlay(review);
      return {
        overlay: setStoryPreference(overlay, review, projectId, storyId, action),
        contextStale: false,
        removed: false,
      };
    });
  };

  if (!reviewEntryVisible(availability)) return null;

  return (
    <div className="ranking-review-shell">
      <button
        ref={triggerRef}
        type="button"
        className="btn btn-secondary"
        onClick={open ? closeReview : openReview}
        aria-expanded={open}
        aria-controls="hiring-context-ranking-review"
      >
        {open ? copy.close : copy.open}
      </button>

      {open && (
        <section
          id="hiring-context-ranking-review"
          className="ranking-review-panel"
          aria-labelledby="hiring-context-ranking-review-title"
          onKeyDown={handleKeyDown}
        >
          <div className="ranking-review-header">
            <div>
              <h2 id="hiring-context-ranking-review-title" ref={headingRef} tabIndex="-1">
                {copy.title}
              </h2>
              <p className="helper-text">{copy.description}</p>
            </div>
            <button type="button" className="btn btn-secondary" onClick={closeReview}>
              {copy.close}
            </button>
          </div>

          {loading && <p className="ranking-review-loading" role="status" aria-live="polite">{copy.loading}</p>}
          {!loading && error && (
            <div className="ranking-review-error" role="alert">
              <p>{copy.error}</p>
              <button type="button" className="btn btn-secondary" onClick={() => loadReview()}>{copy.retry}</button>
            </div>
          )}

          {!loading && !error && review && review.status !== REVIEW_STATUS.ERROR && (
            <>
              <HiringContextSummary context={review.hiringContext} copy={copy} />

              {preferenceEnabled && review.status === REVIEW_STATUS.READY && (
                <>
                  <section className="ranking-review-preference-intro" aria-label={copy.summaryTitle}>
                    <p className="helper-text">{copy.preferenceSession}</p>
                    <p className="helper-text">{copy.keepBoundary}</p>
                    {preferenceSession.contextStale && (
                      <p className="ranking-review-preference-notice" role="status">{copy.contextReset}</p>
                    )}
                    {preferenceSession.removed && (
                      <p className="ranking-review-preference-notice" role="status">{copy.removedReset}</p>
                    )}
                  </section>
                  <PreferenceSummary
                    overlay={preferenceSession.overlay}
                    copy={copy}
                    onResetAll={() => setPreferenceSession({
                      overlay: resetAllPreferences(preferenceSession.overlay),
                      contextStale: false,
                      removed: false,
                    })}
                  />
                </>
              )}

              <form
                className="ranking-review-correction"
                onSubmit={(event) => {
                  event.preventDefault();
                  loadReview(draft);
                }}
              >
                <h3>{copy.correctionTitle}</h3>
                <p className="helper-text">{copy.previewOnly}</p>
                <div className="grid-2">
                  <div className="field">
                    <label htmlFor="ranking-review-company">{copy.company}</label>
                    <input
                      id="ranking-review-company"
                      value={draft.company}
                      maxLength={200}
                      onChange={(event) => setDraft((current) => ({ ...current, company: event.target.value }))}
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="ranking-review-role">{copy.role}</label>
                    <input
                      id="ranking-review-role"
                      value={draft.roleTitle}
                      maxLength={240}
                      onChange={(event) => setDraft((current) => ({ ...current, roleTitle: event.target.value }))}
                    />
                  </div>
                </div>
                <div className="field">
                  <label htmlFor="ranking-review-team">{copy.team}</label>
                  <input
                    id="ranking-review-team"
                    value={draft.team}
                    maxLength={200}
                    onChange={(event) => setDraft((current) => ({ ...current, team: event.target.value }))}
                  />
                </div>
                <div className="btn-row">
                  <button type="submit" className="btn btn-primary">{copy.apply}</button>
                  <button type="button" className="btn btn-secondary" onClick={() => loadReview()}>{copy.reset}</button>
                </div>
              </form>

              {review.status === REVIEW_STATUS.EMPTY && <p className="empty-state" role="status">{copy.empty}</p>}
              {review.status === REVIEW_STATUS.UNAVAILABLE && <p className="empty-state" role="status">{copy.unavailable}</p>}
              {review.status === REVIEW_STATUS.READY && (
                <ProjectReview
                  projects={review.projects}
                  copy={copy}
                  preferenceEnabled={preferenceEnabled}
                  overlay={preferenceSession.overlay}
                  onProjectPreference={handleProjectPreference}
                  onStoryPreference={handleStoryPreference}
                />
              )}
            </>
          )}
          {!loading && !error && review?.status === REVIEW_STATUS.ERROR && (
            <div className="ranking-review-error" role="alert">
              <p>{copy.error}</p>
              <button type="button" className="btn btn-secondary" onClick={() => loadReview()}>{copy.retry}</button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
