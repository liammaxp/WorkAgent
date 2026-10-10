import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  EMPHASIS_ACTION,
  createPreferenceOverlay,
  hasPreferences,
  preferenceControlsVisible,
  projectAction,
  reconcilePreferenceOverlay,
  resetAllPreferences,
  setProjectPreference,
  setStoryPreference,
  storyAction,
  summarizePreferences,
} from "./resumeEmphasisPreferences.js";

const componentSource = readFileSync(
  new URL("./HiringContextRankingReview.jsx", import.meta.url),
  "utf8",
);
const helperSource = readFileSync(
  new URL("./resumeEmphasisPreferences.js", import.meta.url),
  "utf8",
);
const cssSource = readFileSync(new URL("../../index.css", import.meta.url), "utf8");

function review(overrides = {}) {
  return {
    status: "ready",
    preferenceBinding: {
      rankingStateId: "state_1",
      rankingStateFingerprint: "a".repeat(64),
      hiringContextProfileId: "context_1",
      hiringContextFingerprint: "b".repeat(64),
      ...overrides.binding,
    },
    hiringContext: {
      company: "Example",
      team: "Systems",
      roleTitle: "Backend Engineer",
    },
    projects: overrides.projects || [
      {
        projectId: "project_beta",
        displayName: "Beta Project",
        position: 1,
        relevanceReasons: ["Reliability"],
        strongestStories: [
          {
            storyId: "story_beta",
            currentRevisionId: "story_beta_r1",
            label: "Reliable API",
            relevanceReasons: ["Reliability"],
            notices: ["More context could strengthen this story"],
          },
        ],
        additionalStories: [],
      },
      {
        projectId: "project_alpha",
        displayName: "Alpha Project",
        position: 2,
        relevanceReasons: ["Data flow"],
        strongestStories: [
          {
            storyId: "story_alpha",
            currentRevisionId: "story_alpha_r1",
            label: "Data flow",
            relevanceReasons: ["Data flow"],
            notices: [],
          },
        ],
        additionalStories: [],
      },
    ],
  };
}

test("preference feature is hidden when the review flag is off", () => {
  assert.equal(preferenceControlsVisible({ available: false, resume_emphasis_preferences_available: true }), false);
});

test("preference feature is hidden when its independent flag is off", () => {
  assert.equal(preferenceControlsVisible({ available: true, resume_emphasis_preferences_available: false }), false);
});

test("preference controls appear only for two explicit enabled guards", () => {
  assert.equal(preferenceControlsVisible({ available: true, resume_emphasis_preferences_available: true }), true);
});

test("missing availability fails closed", () => {
  assert.equal(preferenceControlsVisible(null), false);
});

test("new overlay starts with all preferences at Default", () => {
  const overlay = createPreferenceOverlay(review());
  assert.equal(hasPreferences(overlay), false);
  assert.equal(projectAction(overlay, "project_beta"), EMPHASIS_ACTION.DEFAULT);
  assert.equal(storyAction(overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.DEFAULT);
});

test("project can be emphasized more", () => {
  const current = review();
  const overlay = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  assert.equal(projectAction(overlay, "project_beta"), EMPHASIS_ACTION.EMPHASIZE_MORE);
});

test("project can be emphasized less", () => {
  const current = review();
  const overlay = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EMPHASIZE_LESS);
  assert.equal(projectAction(overlay, "project_beta"), EMPHASIS_ACTION.EMPHASIZE_LESS);
});

test("project can be explicitly kept", () => {
  const current = review();
  const overlay = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  assert.equal(projectAction(overlay, "project_beta"), EMPHASIS_ACTION.KEEP);
});

test("project can be excluded for this resume", () => {
  const current = review();
  const overlay = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.equal(projectAction(overlay, "project_beta"), EMPHASIS_ACTION.EXCLUDE);
});

test("Story can be emphasized more", () => {
  const current = review();
  const overlay = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  assert.equal(storyAction(overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.EMPHASIZE_MORE);
});

test("Story can be emphasized less", () => {
  const current = review();
  const overlay = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EMPHASIZE_LESS);
  assert.equal(storyAction(overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.EMPHASIZE_LESS);
});

test("Story can be explicitly kept", () => {
  const current = review();
  const overlay = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.KEEP);
  assert.equal(storyAction(overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.KEEP);
});

test("Story can be excluded for this resume", () => {
  const current = review();
  const overlay = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.equal(storyAction(overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.EXCLUDE);
});

test("excluded project remains reversible by choosing Default", () => {
  const current = review();
  const excluded = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EXCLUDE);
  const restored = setProjectPreference(excluded, current, "project_beta", EMPHASIS_ACTION.DEFAULT);
  assert.equal(projectAction(restored, "project_beta"), EMPHASIS_ACTION.DEFAULT);
});

test("excluded Story remains reversible by choosing Default", () => {
  const current = review();
  const excluded = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EXCLUDE);
  const restored = setStoryPreference(excluded, current, "project_beta", "story_beta", EMPHASIS_ACTION.DEFAULT);
  assert.equal(storyAction(restored, "project_beta", "story_beta"), EMPHASIS_ACTION.DEFAULT);
});

test("project exclusion clears incompatible child Story choices", () => {
  const current = review();
  const storyPreferred = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  const excluded = setProjectPreference(storyPreferred, current, "project_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.deepEqual(excluded.stories, []);
});

test("project exclusion dominates effective child action", () => {
  const current = review();
  const excluded = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.equal(storyAction(excluded, "project_beta", "story_beta"), EMPHASIS_ACTION.EXCLUDE);
});

test("project lower and child emphasize-more remain compatible", () => {
  const current = review();
  const lowered = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EMPHASIZE_LESS);
  const overlay = setStoryPreference(lowered, current, "project_beta", "story_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  assert.equal(projectAction(overlay, "project_beta"), EMPHASIS_ACTION.EMPHASIZE_LESS);
  assert.equal(storyAction(overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.EMPHASIZE_MORE);
});

test("child preference cannot be added while project is excluded", () => {
  const current = review();
  const excluded = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.throws(
    () => setStoryPreference(excluded, current, "project_beta", "story_beta", EMPHASIS_ACTION.KEEP),
    /Excluded projects/,
  );
});

test("reset one project preference does not reset another project", () => {
  const current = review();
  let overlay = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  overlay = setProjectPreference(overlay, current, "project_alpha", EMPHASIS_ACTION.EMPHASIZE_MORE);
  overlay = setProjectPreference(overlay, current, "project_beta", EMPHASIS_ACTION.DEFAULT);
  assert.equal(projectAction(overlay, "project_alpha"), EMPHASIS_ACTION.EMPHASIZE_MORE);
});

test("reset one Story preference does not reset another Story", () => {
  const current = review();
  let overlay = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.KEEP);
  overlay = setStoryPreference(overlay, current, "project_alpha", "story_alpha", EMPHASIS_ACTION.EMPHASIZE_MORE);
  overlay = setStoryPreference(overlay, current, "project_beta", "story_beta", EMPHASIS_ACTION.DEFAULT);
  assert.equal(storyAction(overlay, "project_alpha", "story_alpha"), EMPHASIS_ACTION.EMPHASIZE_MORE);
});

test("reset all returns every item to Default", () => {
  const current = review();
  const preferred = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  assert.equal(hasPreferences(resetAllPreferences(preferred)), false);
});

test("preference summary counts every explicit action category", () => {
  const current = review();
  let overlay = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  overlay = setProjectPreference(overlay, current, "project_alpha", EMPHASIS_ACTION.EMPHASIZE_LESS);
  overlay = setStoryPreference(overlay, current, "project_beta", "story_beta", EMPHASIS_ACTION.KEEP);
  overlay = setStoryPreference(overlay, current, "project_alpha", "story_alpha", EMPHASIS_ACTION.EXCLUDE);
  assert.deepEqual(summarizePreferences(overlay), {
    emphasized: 1,
    lowered: 1,
    kept: 1,
    excluded: 1,
    total: 4,
  });
});

test("setting a project preference does not mutate incoming review data", () => {
  const current = review();
  const before = JSON.stringify(current);
  setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.equal(JSON.stringify(current), before);
});

test("setting a Story preference does not mutate incoming review data", () => {
  const current = review();
  const before = JSON.stringify(current);
  setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EXCLUDE);
  assert.equal(JSON.stringify(current), before);
});

test("base ranking order remains unchanged", () => {
  const current = review();
  const before = current.projects.map((item) => item.projectId);
  setProjectPreference(createPreferenceOverlay(current), current, "project_alpha", EMPHASIS_ACTION.EMPHASIZE_MORE);
  assert.deepEqual(current.projects.map((item) => item.projectId), before);
});

test("same ranking state preserves local preference", () => {
  const current = review();
  const preferred = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  const result = reconcilePreferenceOverlay(preferred, current);
  assert.equal(projectAction(result.overlay, "project_beta"), EMPHASIS_ACTION.KEEP);
  assert.equal(result.contextStale, false);
});

test("same context new ranking generation preserves exact project identity", () => {
  const current = review();
  const preferred = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  const next = review({ binding: { rankingStateId: "state_2", rankingStateFingerprint: "c".repeat(64) } });
  const result = reconcilePreferenceOverlay(preferred, next);
  assert.equal(projectAction(result.overlay, "project_beta"), EMPHASIS_ACTION.KEEP);
  assert.equal(result.overlay.binding.rankingStateId, "state_2");
});

test("Story revision refresh preserves canonical preference and updates revision", () => {
  const current = review();
  const preferred = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  const projects = structuredClone(current.projects);
  projects[0].strongestStories[0].currentRevisionId = "story_beta_r2";
  const next = review({
    binding: { rankingStateId: "state_2", rankingStateFingerprint: "c".repeat(64) },
    projects,
  });
  const result = reconcilePreferenceOverlay(preferred, next);
  assert.equal(storyAction(result.overlay, "project_beta", "story_beta"), EMPHASIS_ACTION.EMPHASIZE_MORE);
  assert.equal(result.overlay.stories[0].currentRevisionId, "story_beta_r2");
});

test("removed Story drops stale control state", () => {
  const current = review();
  const preferred = setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "story_beta", EMPHASIS_ACTION.EXCLUDE);
  const projects = structuredClone(current.projects);
  projects[0].strongestStories = [];
  const result = reconcilePreferenceOverlay(preferred, review({ projects }));
  assert.deepEqual(result.overlay.stories, []);
  assert.equal(result.removed, true);
});

test("removed project drops stale project and child state", () => {
  const current = review();
  let preferred = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  preferred = setStoryPreference(preferred, current, "project_beta", "story_beta", EMPHASIS_ACTION.EMPHASIZE_MORE);
  const result = reconcilePreferenceOverlay(preferred, review({ projects: current.projects.slice(1) }));
  assert.deepEqual(result.overlay.projects, []);
  assert.deepEqual(result.overlay.stories, []);
  assert.equal(result.removed, true);
});

test("material Hiring Context change clears preferences and marks stale", () => {
  const current = review();
  const preferred = setProjectPreference(createPreferenceOverlay(current), current, "project_beta", EMPHASIS_ACTION.KEEP);
  const next = review({ binding: {
    hiringContextProfileId: "context_game",
    hiringContextFingerprint: "d".repeat(64),
    rankingStateId: "state_game",
    rankingStateFingerprint: "e".repeat(64),
  } });
  const result = reconcilePreferenceOverlay(preferred, next);
  assert.equal(hasPreferences(result.overlay), false);
  assert.equal(result.contextStale, true);
});

test("context change without preferences does not show a false reset notice", () => {
  const current = review();
  const result = reconcilePreferenceOverlay(createPreferenceOverlay(current), review({ binding: {
    hiringContextProfileId: "context_game",
    hiringContextFingerprint: "d".repeat(64),
  } }));
  assert.equal(result.contextStale, false);
});

test("unknown project action fails closed", () => {
  const current = review();
  assert.throws(
    () => setProjectPreference(createPreferenceOverlay(current), current, "unknown", EMPHASIS_ACTION.KEEP),
    /Unknown project/,
  );
});

test("unknown Story action target fails closed", () => {
  const current = review();
  assert.throws(
    () => setStoryPreference(createPreferenceOverlay(current), current, "project_beta", "unknown", EMPHASIS_ACTION.KEEP),
    /Unknown Story/,
  );
});

test("unknown action vocabulary fails closed", () => {
  const current = review();
  assert.throws(
    () => setProjectPreference(createPreferenceOverlay(current), current, "project_beta", "boost"),
    /Unknown emphasis action/,
  );
});

test("company role and team corrections are absent from preference overlay", () => {
  const serialized = JSON.stringify(createPreferenceOverlay(review()));
  for (const field of ["company", "roleTitle", "team"]) assert.equal(serialized.includes(field), false);
});

test("Story sufficiency notices do not create an automatic preference", () => {
  const overlay = createPreferenceOverlay(review());
  assert.equal(overlay.stories.length, 0);
});

test("preference state has no clarification or question decision", () => {
  const serialized = JSON.stringify(createPreferenceOverlay(review()));
  assert.equal(serialized.includes("shouldAsk"), false);
  assert.equal(serialized.includes("clarification"), false);
});

test("native selects provide keyboard-accessible project and Story controls", () => {
  assert.match(componentSource, /<select/);
  assert.match(componentSource, /ranking-review-project-preference-/);
  assert.match(componentSource, /ranking-review-story-preference-/);
});

test("every compact preference select receives a screen-reader label", () => {
  assert.match(componentSource, /<label htmlFor=\{id\}>/);
  assert.match(componentSource, /aria-label=\{label\}/);
});

test("select controls have visible focus treatment", () => {
  assert.match(cssSource, /button, input, select, summary/);
  assert.match(cssSource, /focus-visible/);
});

test("preference controls adapt to the existing mobile breakpoint", () => {
  assert.match(cssSource, /@media \(max-width: 700px\)/);
  assert.match(cssSource, /ranking-review-preference-control/);
});

test("excluded project and Story remain rendered for reversal", () => {
  assert.match(componentSource, /is-resume-excluded/);
  assert.equal(componentSource.includes(".filter((project)"), false);
  assert.equal(componentSource.includes(".filter((story)"), false);
});

test("preference summary and reset-all interaction are product visible", () => {
  assert.match(componentSource, /PreferenceSummary/);
  assert.match(componentSource, /resetAllPreferences/);
  assert.match(componentSource, /copy\.resetAll/);
});

test("KEEP copy explicitly avoids guaranteed inclusion", () => {
  assert.match(componentSource, /does not guarantee final inclusion/);
  assert.equal(componentSource.includes("guaranteed bullet"), false);
});

test("no adjusted score or fake changed ranking is displayed", () => {
  for (const forbidden of ["preference_score", "adjusted rank", "Rank increased", "final_user_adjusted_relevance"]) {
    assert.equal(componentSource.includes(forbidden), false, forbidden);
    assert.equal(helperSource.includes(forbidden), false, forbidden);
  }
});

test("no resume budget behavior or copy is introduced", () => {
  for (const forbidden of ["bullet count", "line count", "Story budget", "project budget", "page-space"]) {
    assert.equal(componentSource.includes(forbidden), false, forbidden);
    assert.equal(helperSource.includes(forbidden), false, forbidden);
  }
});

test("opaque project and Story IDs are never rendered as visible text", () => {
  const visibleSource = componentSource
    .replace(/key=\{project\.projectId\}/g, "")
    .replace(/key=\{story\.storyId\}/g, "")
    .replace(/onProjectPreference\(project\.projectId, nextAction\)/g, "")
    .replace(/onStoryPreference\(project\.projectId, story\.storyId, nextAction\)/g, "")
    .replace(/projectAction\(overlay, project\.projectId\)/g, "")
    .replace(/storyAction\(overlay, project\.projectId, story\.storyId\)/g, "");
  assert.equal(visibleSource.includes("{project.projectId}"), false);
  assert.equal(visibleSource.includes("{story.storyId}"), false);
});

test("no debug overlay viewer is introduced", () => {
  for (const forbidden of ["<pre", "JSON.stringify", "debug", "overlay_fingerprint", "currentRevisionId"] ) {
    assert.equal(componentSource.includes(forbidden), false, forbidden);
  }
});

test("session-only implementation performs no browser or backend persistence", () => {
  for (const forbidden of ["localStorage", "sessionStorage", "api.save", "api.update", "fetch("]) {
    assert.equal(helperSource.includes(forbidden), false, forbidden);
    assert.equal(componentSource.includes(forbidden), false, forbidden);
  }
});

test("preference controls remain independent from preview correction request", () => {
  assert.match(componentSource, /createReviewRequest\(nextDraft, language, nextDraft !== null\)/);
  assert.equal(componentSource.includes("createReviewRequest(preference"), false);
});
