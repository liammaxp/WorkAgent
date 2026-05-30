const API_BASE = "/api";

let shutdownSent = false;

function currentLanguage() {
  return localStorage.getItem("workagent-language") === "en" ? "en" : "zh";
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const detail =
      typeof data === "object" && data?.detail
        ? typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail)
        : `Request failed (${response.status})`;
    throw new Error(detail);
  }

  return data;
}

export const api = {
  getStatus: () => request("/status"),
  openSession: () => request("/session/open", { method: "POST" }),
  shutdown: () => request("/shutdown", { method: "POST", keepalive: true }),
  sendShutdownBeacon: () => {
    if (shutdownSent) return;
    shutdownSent = true;
    if (navigator.sendBeacon) {
      const blob = new Blob(["{}"], { type: "application/json" });
      navigator.sendBeacon(`${API_BASE}/shutdown`, blob);
      return;
    }
    fetch(`${API_BASE}/shutdown`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      keepalive: true,
    }).catch(() => {});
  },
  setProvider: (provider) =>
    request("/provider", { method: "POST", body: JSON.stringify({ provider }) }),
  getProviderConfigs: () => request("/provider-configs"),
  saveProviderConfig: (payload) =>
    request("/provider-configs", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  setModel: (model) =>
    request("/model", { method: "POST", body: JSON.stringify({ model }) }),
  getFile: (name) => request(`/files/${name}`),
  saveFile: (name, content) =>
    request(`/files/${name}`, {
      method: "PUT",
      body: JSON.stringify({ content }),
    }),
  getPrompt: () => request("/prompt"),
  savePrompt: (content) =>
    request("/prompt", {
      method: "PUT",
      body: JSON.stringify({ content }),
    }),
  askAgent: (message) =>
    request("/agent/ask", {
      method: "POST",
      body: JSON.stringify({ message, language: currentLanguage() }),
    }),
  saveJobDescription: (content) =>
    request("/job-description", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  analyzeJob: (use_github_context = false) =>
    request("/job-description/analyze", {
      method: "POST",
      body: JSON.stringify({ use_github_context, language: currentLanguage() }),
    }),
  tailorResume: (use_github_context = true) =>
    request("/resume/tailor", {
      method: "POST",
      body: JSON.stringify({ use_github_context, language: currentLanguage() }),
    }),
  updateMemoryFromResume: (resume_source = "resume") =>
    request("/resume/update-memory", {
      method: "POST",
      body: JSON.stringify({ resume_source }),
    }),
  generateCoverLetter: (options = {}) =>
    request("/cover-letter/generate", {
      method: "POST",
      body: JSON.stringify({
        use_tailored_resume: true,
        use_github_context: false,
        style: "concise",
        language: currentLanguage(),
        ...options,
      }),
    }),
  generateInterviewPrep: (use_github_context = true) =>
    request("/interview-prep/generate", {
      method: "POST",
      body: JSON.stringify({ use_github_context, language: currentLanguage() }),
    }),
  scanGithub: (resume_source = "resume") =>
    request("/github/scan", {
      method: "POST",
      body: JSON.stringify({ resume_source }),
    }),
  getGithubConfig: () => request("/github/config"),
  saveGithubConfig: (payload) =>
    request("/github/config", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  fetchGithubContext: (approved = true, resume_source = "resume") =>
    request("/github/context", {
      method: "POST",
      body: JSON.stringify({ approved, resume_source }),
    }),
  getApplications: (status = "", limit = 50) => {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    params.set("limit", String(limit));
    return request(`/applications?${params}`);
  },
  createApplication: (payload) =>
    request("/applications", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateApplication: (id, payload) =>
    request(`/applications/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteApplication: (id) =>
    request(`/applications/${id}`, {
      method: "DELETE",
    }),
};
