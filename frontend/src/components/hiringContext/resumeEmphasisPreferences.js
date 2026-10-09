export const EMPHASIS_ACTION = Object.freeze({
  DEFAULT: "default",
  EMPHASIZE_MORE: "emphasize_more",
  EMPHASIZE_LESS: "emphasize_less",
  KEEP: "keep",
  EXCLUDE: "exclude",
});

export const EMPHASIS_ACTIONS = Object.freeze(Object.values(EMPHASIS_ACTION));
const EXPLICIT_ACTIONS = new Set(EMPHASIS_ACTIONS.slice(1));

function bindingFromReview(review) {
  return review?.preferenceBinding || null;
}

function storyIndex(review) {
  const stories = new Map();
  for (const project of review?.projects || []) {
    for (const story of [...project.strongestStories, ...project.additionalStories]) {
      stories.set(`${project.projectId}\u0000${story.storyId}`, story.currentRevisionId);
    }
  }
  return stories;
}

function projectIds(review) {
  return new Set((review?.projects || []).map((project) => project.projectId));
}

function sameContext(left, right) {
  return Boolean(
    left
    && right
    && left.hiringContextProfileId === right.hiringContextProfileId
    && left.hiringContextFingerprint === right.hiringContextFingerprint
  );
}

function compareExact(left, right) {
  if (left < right) return -1;
  if (left > right) return 1;
  return 0;
}

function sortOverlay(overlay) {
  return {
    binding: overlay.binding,
    projects: [...overlay.projects].sort((left, right) => compareExact(left.projectId, right.projectId)),
    stories: [...overlay.stories].sort((left, right) => (
      compareExact(left.projectId, right.projectId)
      || compareExact(left.storyId, right.storyId)
    )),
  };
}

function requireAction(action) {
  if (!EMPHASIS_ACTIONS.includes(action)) throw new TypeError("Unknown emphasis action");
  return action;
}

export function preferenceControlsVisible(availability) {
  return availability?.available === true
    && availability?.resume_emphasis_preferences_available === true;
}

export function createPreferenceOverlay(review) {
  const binding = bindingFromReview(review);
  if (!binding || !Array.isArray(review?.projects)) return null;
  return { binding, projects: [], stories: [] };
}

export function hasPreferences(overlay) {
  return Boolean(overlay && (overlay.projects.length || overlay.stories.length));
}

export function projectAction(overlay, projectId) {
  return overlay?.projects.find((item) => item.projectId === projectId)?.action
    || EMPHASIS_ACTION.DEFAULT;
}

export function storyAction(overlay, projectId, storyId) {
  if (projectAction(overlay, projectId) === EMPHASIS_ACTION.EXCLUDE) {
    return EMPHASIS_ACTION.EXCLUDE;
  }
  return overlay?.stories.find(
    (item) => item.projectId === projectId && item.storyId === storyId,
  )?.action || EMPHASIS_ACTION.DEFAULT;
}

export function setProjectPreference(overlay, review, projectId, action) {
  if (!overlay || !projectIds(review).has(projectId)) throw new TypeError("Unknown project");
  requireAction(action);
  const projects = overlay.projects.filter((item) => item.projectId !== projectId);
  if (action !== EMPHASIS_ACTION.DEFAULT) projects.push({ projectId, action });
  const stories = action === EMPHASIS_ACTION.EXCLUDE
    ? overlay.stories.filter((item) => item.projectId !== projectId)
    : overlay.stories;
  return sortOverlay({ ...overlay, projects, stories });
}

export function setStoryPreference(overlay, review, projectId, storyId, action) {
  if (!overlay) throw new TypeError("Missing preference overlay");
  requireAction(action);
  const revisionId = storyIndex(review).get(`${projectId}\u0000${storyId}`);
  if (!revisionId) throw new TypeError("Unknown Story");
  if (projectAction(overlay, projectId) === EMPHASIS_ACTION.EXCLUDE) {
    throw new TypeError("Excluded projects cannot retain child Story preferences");
  }
  const stories = overlay.stories.filter(
    (item) => item.projectId !== projectId || item.storyId !== storyId,
  );
  if (action !== EMPHASIS_ACTION.DEFAULT) {
    stories.push({ projectId, storyId, currentRevisionId: revisionId, action });
  }
  return sortOverlay({ ...overlay, stories });
}

export function resetAllPreferences(overlay) {
  if (!overlay) return null;
  return { binding: overlay.binding, projects: [], stories: [] };
}

export function summarizePreferences(overlay) {
  const summary = {
    emphasized: 0,
    lowered: 0,
    kept: 0,
    excluded: 0,
    total: 0,
  };
  for (const item of [...(overlay?.projects || []), ...(overlay?.stories || [])]) {
    if (!EXPLICIT_ACTIONS.has(item.action)) continue;
    summary.total += 1;
    if (item.action === EMPHASIS_ACTION.EMPHASIZE_MORE) summary.emphasized += 1;
    if (item.action === EMPHASIS_ACTION.EMPHASIZE_LESS) summary.lowered += 1;
    if (item.action === EMPHASIS_ACTION.KEEP) summary.kept += 1;
    if (item.action === EMPHASIS_ACTION.EXCLUDE) summary.excluded += 1;
  }
  return summary;
}

export function reconcilePreferenceOverlay(overlay, review) {
  const fresh = createPreferenceOverlay(review);
  if (!fresh) return { overlay: null, contextStale: false, removed: false };
  if (!overlay) return { overlay: fresh, contextStale: false, removed: false };
  if (!sameContext(overlay.binding, fresh.binding)) {
    return {
      overlay: fresh,
      contextStale: hasPreferences(overlay),
      removed: false,
    };
  }
  const knownProjects = projectIds(review);
  const knownStories = storyIndex(review);
  const projects = overlay.projects.filter((item) => knownProjects.has(item.projectId));
  const stories = overlay.stories.flatMap((item) => {
    const revisionId = knownStories.get(`${item.projectId}\u0000${item.storyId}`);
    return revisionId ? [{ ...item, currentRevisionId: revisionId }] : [];
  });
  return {
    overlay: sortOverlay({ binding: fresh.binding, projects, stories }),
    contextStale: false,
    removed: projects.length !== overlay.projects.length || stories.length !== overlay.stories.length,
  };
}
