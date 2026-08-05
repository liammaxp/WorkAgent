import assert from "node:assert/strict";
import test from "node:test";
import { api, ApiError } from "./client.js";

function response(payload, { ok = true, status = 200 } = {}) {
  return { ok, status, text: async () => JSON.stringify(payload) };
}

test("mapping GET helpers use semantic endpoints and pass cancellation signals", async (context) => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return response({ repositories: [], projects: [] });
  };
  context.after(() => { delete globalThis.fetch; });
  const controller = new AbortController();
  await api.getUnresolvedRepositoryMappings({ signal: controller.signal });
  await api.getRepositoryMappingProjects({ signal: controller.signal });
  assert.deepEqual(calls.map((call) => call.url), [
    "/api/github/repository-mappings/unresolved",
    "/api/github/repository-mappings/projects",
  ]);
  assert.equal(calls.every((call) => call.options.signal === controller.signal), true);
  assert.equal(calls.every((call) => !call.options.method || call.options.method === "GET"), true);
});

test("confirmation helper sends one strict POST body", async (context) => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return response({ status: "created" });
  };
  context.after(() => { delete globalThis.fetch; });
  await api.confirmRepositoryMapping({
    project_id: "project-a", repository: "owner/repository", confirmed: true,
    aliases: ["WorkAgent"], metadata: { internal: true },
  });
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/api/github/repository-mappings/confirm");
  assert.equal(calls[0].options.method, "POST");
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    project_id: "project-a", repository: "owner/repository", confirmed: true,
    aliases: ["WorkAgent"],
  });
});

test("API failures use the existing typed error boundary", async (context) => {
  globalThis.fetch = async () => response({ detail: "Unavailable" }, { ok: false, status: 503 });
  context.after(() => { delete globalThis.fetch; });
  await assert.rejects(
    api.getUnresolvedRepositoryMappings(),
    (error) => error instanceof ApiError && error.status === 503,
  );
});
