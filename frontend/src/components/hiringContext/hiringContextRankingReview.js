export const REVIEW_STATUS = Object.freeze({
  READY: "ready",
  EMPTY: "empty",
  UNAVAILABLE: "unavailable",
  ERROR: "error",
});

const REVIEW_STATUSES = new Set(Object.values(REVIEW_STATUS));

function cleanText(value, maximum = 300) {
  if (value === null || value === undefined) return null;
  if (typeof value !== "string") return null;
  const cleaned = value.replace(/\s+/g, " ").trim();
  if (!cleaned || cleaned.length > maximum) return null;
  return cleaned;
}

function cleanList(value, maximum) {
  if (!Array.isArray(value) || value.length > maximum) return null;
  const result = [];
  for (const item of value) {
    const cleaned = cleanText(item, 300);
    if (!cleaned) return null;
    result.push(cleaned);
  }
  return result;
}

function normalizeStory(value) {
  if (!value || typeof value !== "object") return null;
  const storyId = cleanText(value.story_id, 300);
  const label = cleanText(value.label, 300);
  const relevanceReasons = cleanList(value.relevance_reasons, 3);
  const notices = cleanList(value.notices, 2);
  if (!storyId || !label || !relevanceReasons || !notices) return null;
  return {
    storyId,
    label,
    relevanceReasons,
    notices,
  };
}

function normalizeStories(value) {
  if (!Array.isArray(value) || value.length > 512) return null;
  const stories = value.map(normalizeStory);
  return stories.every(Boolean) ? stories : null;
}

function normalizeProject(value, expectedPosition) {
  if (!value || typeof value !== "object") return null;
  const projectId = cleanText(value.project_id, 300);
  const displayName = cleanText(value.display_name, 300);
  const relevanceReasons = cleanList(value.relevance_reasons, 3);
  const strongestStories = normalizeStories(value.strongest_stories);
  const additionalStories = normalizeStories(value.additional_stories);
  if (
    !projectId
    || !displayName
    || value.position !== expectedPosition
    || !relevanceReasons
    || !strongestStories
    || !additionalStories
  ) return null;
  return {
    projectId,
    displayName,
    position: value.position,
    relevanceReasons,
    strongestStories,
    additionalStories,
  };
}

function normalizeContext(value) {
  if (!value || typeof value !== "object") return null;
  const primaryRoleFamily = cleanText(value.primary_role_family, 160);
  const confidence = cleanText(value.confidence, 160);
  const secondaryRoleFamilies = cleanList(value.secondary_role_families, 6);
  const contextSignals = cleanList(value.context_signals, 8);
  if (!primaryRoleFamily || !confidence || !secondaryRoleFamilies || !contextSignals) return null;
  return {
    company: cleanText(value.company, 200),
    team: cleanText(value.team, 200),
    roleTitle: cleanText(value.role_title, 240),
    primaryRoleFamily,
    secondaryRoleFamilies,
    contextSignals,
    confidence,
  };
}

export function normalizeReviewResponse(value) {
  if (!value || typeof value !== "object" || !REVIEW_STATUSES.has(value.status)) return null;
  const hiringContext = normalizeContext(value.hiring_context);
  if (!hiringContext || value.corrections_persisted !== false || !Array.isArray(value.projects)) return null;
  if (value.status !== REVIEW_STATUS.READY) {
    return {
      status: value.status,
      hiringContext,
      projects: [],
      correctionsPersisted: false,
    };
  }
  if (value.projects.length === 0 || value.projects.length > 64) return null;
  const projects = value.projects.map((project, index) => normalizeProject(project, index + 1));
  if (!projects.every(Boolean)) return null;
  return {
    status: value.status,
    hiringContext,
    projects,
    correctionsPersisted: false,
  };
}

export function createReviewRequest(draft, language, includeDraft = false) {
  const request = { language: language === "zh" ? "zh" : "en" };
  if (!includeDraft) return request;
  return {
    ...request,
    company: String(draft?.company || "").trim().slice(0, 200),
    team: String(draft?.team || "").trim().slice(0, 200),
    role_title: String(draft?.roleTitle || "").trim().slice(0, 240),
  };
}

export function draftFromReview(review) {
  return {
    company: review?.hiringContext?.company || "",
    team: review?.hiringContext?.team || "",
    roleTitle: review?.hiringContext?.roleTitle || "",
  };
}

export function reviewEntryVisible(availability) {
  return availability?.available === true;
}

export function shouldCloseReview(key) {
  return key === "Escape";
}
