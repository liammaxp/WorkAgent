import assert from "node:assert/strict";
import test from "node:test";
import { ApiError, api } from "./client.js";

function response(body, ok = true, status = 200) {
  return { ok, status, text: async () => JSON.stringify(body) };
}

test("status GET uses the exact endpoint and forwards only the signal", async () => {
  const calls = []; const signal = new AbortController().signal;
  globalThis.fetch = async (...args) => { calls.push(args); return response({ status: "disabled" }); };
  await api.getGitHubEvidencePreparationStatus({ signal, ignored: true });
  assert.equal(calls.length, 1); assert.equal(calls[0][0], "/api/github/evidence-preparation");
  assert.equal(calls[0][1].signal, signal); assert.equal(calls[0][1].method, undefined); assert.equal(calls[0][1].body, undefined);
});

test("run POST sends exact confirmation body once", async () => {
  const calls = [];
  globalThis.fetch = async (...args) => { calls.push(args); return response({ status: "created" }); };
  await api.runGitHubEvidencePreparation({ ignored: true });
  assert.equal(calls.length, 1); assert.equal(calls[0][0], "/api/github/evidence-preparation/run");
  assert.equal(calls[0][1].method, "POST"); assert.deepEqual(JSON.parse(calls[0][1].body), { confirmed: true });
});

test("API errors remain typed and are not retried", async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls += 1; return response({ detail: "blocked" }, false, 409); };
  await assert.rejects(() => api.runGitHubEvidencePreparation(), (error) => error instanceof ApiError && error.status === 409);
  assert.equal(calls, 1);
});
