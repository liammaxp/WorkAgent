import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  associationExists,
  buildConfirmationPayload,
  isCanonicalRepositoryInput,
  repositoryDisplayName,
  repositoryItemKey,
} from "./repositoryAssociation.js";

test("canonical repository validation is strict and bounded", () => {
  assert.equal(isCanonicalRepositoryInput("owner/repository"), true);
  assert.equal(isCanonicalRepositoryInput(" Owner/Repository "), true);
  for (const invalid of [
    "WorkAgent", "", "   ", "owner/", "/repository", "owner/../repository",
    "https://user:secret@github.com/owner/repository", "owner/repository?token=x",
    "owner/repository#fragment", `owner/${"r".repeat(500)}`,
  ]) {
    assert.equal(isCanonicalRepositoryInput(invalid), false, invalid);
  }
});

test("confirmation payload contains only the strict backend fields", () => {
  assert.deepEqual(buildConfirmationPayload({
    projectId: " project-a ", repository: " owner/repository ", metadata: "ignored",
  }), {
    project_id: "project-a", repository: "owner/repository", confirmed: true,
  });
  assert.deepEqual(buildConfirmationPayload({
    projectId: "project-a", repository: "owner/repository", repositoryAlias: "WorkAgent",
  }), {
    project_id: "project-a", repository: "owner/repository", confirmed: true,
    aliases: ["WorkAgent"],
  });
});

test("canonical and bare entries have deterministic display identities without suggestions", () => {
  const canonical = { repository: "owner/repository", canonical: true };
  const bare = { repository: null, repository_alias: "WorkAgent", canonical: false };
  assert.equal(repositoryDisplayName(canonical), "owner/repository");
  assert.equal(repositoryDisplayName(bare), "WorkAgent");
  assert.equal(repositoryItemKey(canonical), "repository:owner/repository");
  assert.equal(repositoryItemKey(bare, 2), "alias:workagent:2");
  assert.equal("project_id" in bare, false);
});

test("network reconciliation trusts refreshed server state", () => {
  const values = {
    unresolved: [{ repository: "owner/other" }],
    projects: [{
      project_id: "project-a", project_name: "A",
      already_linked_repositories: ["owner/repository"],
    }],
    repository: "owner/repository", projectId: "project-a",
  };
  assert.equal(associationExists(values), true);
  assert.equal(associationExists({
    ...values, unresolved: [{ repository: "owner/repository" }],
  }), false);
  assert.equal(associationExists({ ...values, projectId: "project-b" }), false);
});

test("product component keeps accessibility and operation boundaries explicit", () => {
  const source = readFileSync(new URL("./RepositoryAssociationSection.jsx", import.meta.url), "utf8");
  for (const required of [
    "aria-live=\"polite\"", "role=\"alert\"", "aria-invalid", "aria-describedby",
    "getUnresolvedRepositoryMappings", "getRepositoryMappingProjects", "confirmRepositoryMapping", "onAssociationChanged",
  ]) {
    assert.equal(source.includes(required), true, required);
  }
  for (const forbidden of [
    "materialize", "retrieval-v2", "resume/generate", "vector/search", "capability/build",
    "localStorage", "sessionStorage", "console.log", "console.debug",
  ]) {
    assert.equal(source.includes(forbidden), false, forbidden);
  }
});
