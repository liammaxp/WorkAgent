import assert from "node:assert/strict";
import test from "node:test";
import { api } from "./client.js";

function response(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(body),
  };
}

test("availability uses the semantic read-only route", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const calls = [];
  globalThis.fetch = async (...args) => {
    calls.push(args);
    return response({ available: true });
  };
  assert.deepEqual(await api.getHiringContextRankingReviewAvailability(), { available: true });
  assert.equal(calls[0][0], "/api/hiring-context/review/availability");
  assert.equal(calls[0][1].method, undefined);
});

test("review posts only the supplied product request", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const calls = [];
  globalThis.fetch = async (...args) => {
    calls.push(args);
    return response({ status: "empty" });
  };
  const payload = {
    language: "en",
    company: "Preview Employer",
    team: "Preview Team",
    role_title: "Preview Role",
  };
  await api.reviewHiringContext(payload);
  assert.equal(calls[0][0], "/api/hiring-context/review");
  assert.equal(calls[0][1].method, "POST");
  assert.deepEqual(JSON.parse(calls[0][1].body), payload);
});

test("availability supports cancellation", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const controller = new AbortController();
  let options;
  globalThis.fetch = async (_path, value) => {
    options = value;
    return response({ available: true });
  };
  await api.getHiringContextRankingReviewAvailability({ signal: controller.signal });
  assert.equal(options.signal, controller.signal);
});

test("review supports cancellation without adding internal fields", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const controller = new AbortController();
  let options;
  globalThis.fetch = async (_path, value) => {
    options = value;
    return response({ status: "empty" });
  };
  await api.reviewHiringContext({ language: "en" }, { signal: controller.signal });
  assert.equal(options.signal, controller.signal);
  assert.deepEqual(JSON.parse(options.body), { language: "en" });
});
