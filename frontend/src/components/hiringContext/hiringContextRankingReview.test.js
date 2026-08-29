import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  REVIEW_STATUS,
  createReviewRequest,
  draftFromReview,
  normalizeReviewResponse,
  reviewEntryVisible,
  shouldCloseReview,
} from "./hiringContextRankingReview.js";

const componentSource = readFileSync(
  new URL("./HiringContextRankingReview.jsx", import.meta.url),
  "utf8",
);

function readyPayload() {
  return {
    status: "ready",
    hiring_context: {
      company: "The Coalition",
      team: "Online systems",
      role_title: "Software Engineering Intern",
      primary_role_family: "Software engineering",
      secondary_role_families: ["Game development"],
      context_signals: ["Real-time interactive software"],
      confidence: "High confidence",
      fingerprint: "hidden-context-fingerprint",
    },
    projects: [
      {
        project_id: "project_beta",
        display_name: "Beta Project",
        position: 1,
        relevance_reasons: ["Strong evidence in Reliability"],
        strongest_stories: [
          {
            story_id: "story_beta_2",
            label: "Reliability and validation",
            relevance_reasons: ["Reliability", "Validation and repair"],
            notices: ["Some claims may need confirmation", "More context could strengthen this story"],
            total_relevance_score: 0.999,
          },
          {
            story_id: "story_beta_1",
            label: "API and system architecture",
            relevance_reasons: ["API and system design"],
            notices: [],
          },
        ],
        additional_stories: [
          {
            story_id: "story_beta_3",
            label: "Operational hardening",
            relevance_reasons: ["Operational hardening"],
            notices: [],
          },
        ],
        redundancy_score: 0.2,
      },
      {
        project_id: "project_alpha",
        display_name: "Alpha Project",
        position: 2,
        relevance_reasons: ["Strong evidence in Data flow"],
        strongest_stories: [],
        additional_stories: [],
      },
    ],
    corrections_persisted: false,
    policy_id: "hidden-policy",
  };
}

test("feature entry is hidden when availability is absent", () => {
  assert.equal(reviewEntryVisible(null), false);
});

test("feature entry is hidden when the rollout guard is off", () => {
  assert.equal(reviewEntryVisible({ available: false }), false);
});

test("feature entry appears only for explicit availability", () => {
  assert.equal(reviewEntryVisible({ available: true }), true);
});

test("Escape requests review close", () => {
  assert.equal(shouldCloseReview("Escape"), true);
});

test("other keys do not close review", () => {
  assert.equal(shouldCloseReview("Enter"), false);
});

test("initial request contains only language", () => {
  assert.deepEqual(createReviewRequest({}, "en"), { language: "en" });
});

test("Chinese language is preserved", () => {
  assert.deepEqual(createReviewRequest({}, "zh"), { language: "zh" });
});

test("unknown language fails to English product copy", () => {
  assert.deepEqual(createReviewRequest({}, "fr"), { language: "en" });
});

test("preview request contains only bounded identity fields", () => {
  assert.deepEqual(
    createReviewRequest({ company: " Example ", team: " Team ", roleTitle: " Role " }, "en", true),
    { language: "en", company: "Example", team: "Team", role_title: "Role" },
  );
});

test("preview request cannot carry candidate evidence", () => {
  const request = createReviewRequest({ company: "Example", candidateFact: "invented" }, "en", true);
  assert.deepEqual(Object.keys(request), ["language", "company", "team", "role_title"]);
});

test("preview request cannot carry project controls", () => {
  const request = createReviewRequest({ company: "Example", excludeProject: "alpha" }, "en", true);
  assert.equal("excludeProject" in request, false);
});

test("normalized review exposes populated Hiring Context", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).hiringContext.company, "The Coalition");
});

test("company is preserved as product identity", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).hiringContext.company, "The Coalition");
});

test("role title is preserved as product identity", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).hiringContext.roleTitle, "Software Engineering Intern");
});

test("team is preserved only when available", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).hiringContext.team, "Online systems");
});

test("primary role family is already product-safe copy", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).hiringContext.primaryRoleFamily, "Software engineering");
});

test("secondary role families stay bounded", () => {
  assert.deepEqual(normalizeReviewResponse(readyPayload()).hiringContext.secondaryRoleFamilies, ["Game development"]);
});

test("organization domain is rendered only as context signal", () => {
  assert.deepEqual(normalizeReviewResponse(readyPayload()).hiringContext.contextSignals, ["Real-time interactive software"]);
});

test("Coalition context does not create a candidate persona", () => {
  const review = normalizeReviewResponse(readyPayload());
  const projectCopy = JSON.stringify(review.projects).toLowerCase();
  assert.equal(projectCopy.includes("candidate is a game developer"), false);
});

test("ranked project API order is preserved", () => {
  assert.deepEqual(
    normalizeReviewResponse(readyPayload()).projects.map((item) => item.displayName),
    ["Beta Project", "Alpha Project"],
  );
});

test("project display names remain exact", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).projects[0].displayName, "Beta Project");
});

test("Story API order is preserved", () => {
  assert.deepEqual(
    normalizeReviewResponse(readyPayload()).projects[0].strongestStories.map((item) => item.label),
    ["Reliability and validation", "API and system architecture"],
  );
});

test("additional Story hierarchy remains separate", () => {
  assert.deepEqual(
    normalizeReviewResponse(readyPayload()).projects[0].additionalStories.map((item) => item.label),
    ["Operational hardening"],
  );
});

test("numeric relevance scores are discarded", () => {
  const story = normalizeReviewResponse(readyPayload()).projects[0].strongestStories[0];
  assert.equal("total_relevance_score" in story, false);
});

test("redundancy values are discarded", () => {
  assert.equal("redundancy_score" in normalizeReviewResponse(readyPayload()).projects[0], false);
});

test("policy identity is discarded", () => {
  assert.equal("policy_id" in normalizeReviewResponse(readyPayload()), false);
});

test("fingerprints are discarded", () => {
  assert.equal("fingerprint" in normalizeReviewResponse(readyPayload()).hiringContext, false);
});

test("claim gap product message is preserved", () => {
  assert.equal(
    normalizeReviewResponse(readyPayload()).projects[0].strongestStories[0].notices[0],
    "Some claims may need confirmation",
  );
});

test("Story-completion product message is preserved", () => {
  assert.equal(
    normalizeReviewResponse(readyPayload()).projects[0].strongestStories[0].notices[1],
    "More context could strengthen this story",
  );
});

test("both sufficiency messages remain independent", () => {
  assert.equal(normalizeReviewResponse(readyPayload()).projects[0].strongestStories[0].notices.length, 2);
});

test("empty state strips any projects", () => {
  const payload = readyPayload();
  payload.status = REVIEW_STATUS.EMPTY;
  assert.deepEqual(normalizeReviewResponse(payload).projects, []);
});

test("unavailable state strips any projects", () => {
  const payload = readyPayload();
  payload.status = REVIEW_STATUS.UNAVAILABLE;
  assert.deepEqual(normalizeReviewResponse(payload).projects, []);
});

test("integrity error state strips partial projects", () => {
  const payload = readyPayload();
  payload.status = REVIEW_STATUS.ERROR;
  assert.deepEqual(normalizeReviewResponse(payload).projects, []);
});

test("malformed response fails closed", () => {
  assert.equal(normalizeReviewResponse({ status: "ready" }), null);
});

test("nonconsecutive project positions fail closed", () => {
  const payload = readyPayload();
  payload.projects[1].position = 4;
  assert.equal(normalizeReviewResponse(payload), null);
});

test("invalid Story array fails closed", () => {
  const payload = readyPayload();
  payload.projects[0].strongest_stories = "raw json";
  assert.equal(normalizeReviewResponse(payload), null);
});

test("correction draft is derived only from Hiring Context", () => {
  assert.deepEqual(draftFromReview(normalizeReviewResponse(readyPayload())), {
    company: "The Coalition",
    team: "Online systems",
    roleTitle: "Software Engineering Intern",
  });
});

test("review component announces loading and errors", () => {
  assert.match(componentSource, /role="status"/);
  assert.match(componentSource, /role="alert"/);
  assert.match(componentSource, /aria-live="polite"/);
});

test("review component exposes accessible disclosure semantics", () => {
  assert.match(componentSource, /aria-expanded=/);
  assert.match(componentSource, /aria-controls=/);
  assert.match(componentSource, /aria-labelledby=/);
});

test("review component supports focus return and Escape", () => {
  assert.match(componentSource, /triggerRef\.current\?\.focus/);
  assert.match(componentSource, /shouldCloseReview\(event\.key\)/);
});

test("review component uses ordered project and Story lists", () => {
  assert.ok((componentSource.match(/<ol/g) || []).length >= 2);
});

test("review component uses native disclosure for additional Stories", () => {
  assert.match(componentSource, /<details/);
  assert.match(componentSource, /<summary>/);
});

test("preview fields have explicit accessible labels", () => {
  for (const id of ["ranking-review-company", "ranking-review-role", "ranking-review-team"]) {
    assert.match(componentSource, new RegExp(`htmlFor="${id}"`));
    assert.match(componentSource, new RegExp(`id="${id}"`));
  }
});

test("component renders fixed loading empty unavailable and error copy", () => {
  for (const token of ["copy.loading", "copy.empty", "copy.unavailable", "copy.error"]) {
    assert.ok(componentSource.includes(token));
  }
});

test("component does not rerank API arrays", () => {
  assert.equal(componentSource.includes(".sort("), false);
});

test("component never renders opaque identities as text", () => {
  const visibleSource = componentSource
    .replace(/key=\{project\.projectId\}/g, "")
    .replace(/key=\{story\.storyId\}/g, "");
  assert.equal(visibleSource.includes("{project.projectId}"), false);
  assert.equal(visibleSource.includes("{story.storyId}"), false);
});

test("component contains no raw or developer presentation surface", () => {
  for (const token of ["dangerouslySetInnerHTML", "<pre", "JSON.stringify", "raw Story", "debug", "Chroma", "fingerprint", "revision"] ) {
    assert.equal(componentSource.includes(token), false, token);
  }
});

test("component contains no question or selection controls", () => {
  for (const token of ["Emphasize more", "Emphasize less", ">Keep<", ">Exclude<", "question composer", "answer box"] ) {
    assert.equal(componentSource.includes(token), false, token);
  }
});

test("component contains no resume-space controls", () => {
  for (const token of ["bullet count", "line budget", "Story budget", "project budget"] ) {
    assert.equal(componentSource.includes(token), false, token);
  }
});
