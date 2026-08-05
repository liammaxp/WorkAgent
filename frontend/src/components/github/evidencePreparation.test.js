import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { canStartPreparation, outcomeMessage, safeRemainingRepositoryText, shouldReconcileAfterOutcome, statusMessage } from "./evidencePreparation.js";

test("all backend status and outcome values have safe product copy", () => {
  for (const status of ["disabled", "mapping_required", "ready_to_prepare", "prepared", "partial", "blocked", "error"]) {
    assert.ok(statusMessage(status, "en")); assert.ok(statusMessage(status, "zh"));
  }
  for (const outcome of ["created", "updated", "unchanged", "empty", "busy", "degraded", "error"]) {
    assert.ok(outcomeMessage(outcome, "en")); assert.ok(outcomeMessage(outcome, "zh"));
  }
});

test("preparation is offered only when the backend explicitly allows a preparable state", () => {
  assert.equal(canStartPreparation({ status: "ready_to_prepare", can_prepare: true }), true);
  assert.equal(canStartPreparation({ status: "partial", can_prepare: true }), true);
  assert.equal(canStartPreparation({ status: "prepared", can_prepare: true }), false);
  assert.equal(canStartPreparation({ status: "ready_to_prepare", can_prepare: false }), false);
});

test("remaining repository count and reconciliation policy stay bounded", () => {
  assert.match(safeRemainingRepositoryText({ status: "mapping_required", remaining_repository_count: 2 }), /^2 /);
  assert.equal(safeRemainingRepositoryText({ status: "ready_to_prepare", remaining_repository_count: 2 }), "");
  assert.equal(shouldReconcileAfterOutcome("created"), true);
  assert.equal(shouldReconcileAfterOutcome("busy"), false);
  assert.equal(shouldReconcileAfterOutcome("disabled"), false);
});

test("component requires confirmation and preserves product safety boundaries", () => {
  const source = readFileSync(new URL("./EvidencePreparationSection.jsx", import.meta.url), "utf8");
  for (const required of ["aria-live=\"polite\"", "loadError ? \"alert\" : \"status\"", "setConfirming(true)", "runGitHubEvidencePreparation", "refreshSignal"]) assert.equal(source.includes(required), true, required);
  for (const forbidden of ["setInterval", "setTimeout", "localStorage", "sessionStorage", "document", "raw_content", "capability", "vector", "Chroma"]) assert.equal(source.includes(forbidden), false, forbidden);
  assert.equal((source.match(/runGitHubEvidencePreparation/g) || []).length, 1);
});
