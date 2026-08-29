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
  },
};

function StoryList({ stories, label, copy }) {
  if (!stories.length) return null;
  return (
    <div className="ranking-review-story-group">
      <h5>{label}</h5>
      <ol className="ranking-review-story-list">
        {stories.map((story) => (
          <li key={story.storyId}>
            <div className="ranking-review-story-heading">{story.label}</div>
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
        ))}
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

function ProjectReview({ projects, copy }) {
  if (!projects.length) return null;
  return (
    <section className="ranking-review-projects" aria-labelledby="ranking-review-projects-title">
      <h3 id="ranking-review-projects-title">{copy.projectsTitle}</h3>
      <ol className="ranking-review-project-list">
        {projects.map((project) => (
          <li key={project.projectId}>
            <article className="ranking-review-project">
              <h4>{project.displayName}</h4>
              {project.relevanceReasons.map((reason) => (
                <p className="ranking-review-project-reason" key={reason}>{reason}</p>
              ))}
              <StoryList stories={project.strongestStories} label={copy.strongest} copy={copy} />
              {project.additionalStories.length > 0 && (
                <details className="ranking-review-additional">
                  <summary>{copy.additional}</summary>
                  <StoryList stories={project.additionalStories} label={copy.additional} copy={copy} />
                </details>
              )}
            </article>
          </li>
        ))}
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
              {review.status === REVIEW_STATUS.READY && <ProjectReview projects={review.projects} copy={copy} />}
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
