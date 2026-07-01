const API_BASE = "/api";

let shutdownSent = false;

export class ApiError extends Error {
  constructor(message, detail = null, status = 0) {
    super(message);
    this.name = "ApiError";
    this.detail = detail;
    this.status = status;
  }
}

function currentLanguage() {
  return localStorage.getItem("workagent-language") === "en" ? "en" : "zh";
}

function formatApiDetail(detail, status) {
  if (!detail) return `Request failed (${status})`;
  if (typeof detail === "string") return detail;
  if (typeof detail !== "object") return String(detail);

  const type = detail.type || "";
  if (type === "ModelProxyTimeout") {
    const zh = currentLanguage() === "zh";
    const chunk = detail.failedChunk || {};
    const where = [
      chunk.projectName ? `project=${chunk.projectName}` : "",
      chunk.repoName ? `repo=${chunk.repoName}` : "",
      chunk.chunkIndex ? `chunk=${chunk.chunkIndex}` : "",
    ].filter(Boolean).join(", ");
    if (zh) {
      return [
        detail.message || "第三方中转站连续超时。已保存当前进度，可以稍后重试或降低输入规模。",
        where ? `失败位置：${where}。` : "",
        detail.retryable ? "已完成的 chunk / project 会通过 checkpoint 复用。" : "",
      ].filter(Boolean).join(" ");
    }
    return [
      detail.message || "Third-party proxy timed out while processing a model chunk.",
      where ? `Failed chunk: ${where}.` : "",
      detail.retryable ? "You can retry; completed chunks are checkpointed." : "",
    ].filter(Boolean).join(" ");
  }

  if (type === "ModelTransientError") {
    if (currentLanguage() === "zh") {
      return [
        detail.message || "第三方中转站连续返回 502。已保存当前进度，可以稍后重试或降低输入规模。",
        detail.caller ? `Caller: ${detail.caller}.` : "",
        detail.inputCharCount ? `最终输入 ${detail.inputCharCount} chars。` : "",
      ].filter(Boolean).join(" ");
    }
    return [
      detail.message || "Third-party proxy returned repeated transient errors after retry protection.",
      detail.caller ? `Caller: ${detail.caller}.` : "",
      detail.inputCharCount ? `Final input ${detail.inputCharCount} chars.` : "",
    ].filter(Boolean).join(" ");
  }

  if (type === "ModelInputTooLargeForProxy" || type === "ModelInputHardLimitExceededForProxy") {
    return [
      detail.message || "Model input is too large for the configured proxy.",
      detail.caller ? `Caller: ${detail.caller}.` : "",
      detail.inputCharCount && detail.limit ? `Input ${detail.inputCharCount} chars, limit ${detail.limit}.` : "",
    ].filter(Boolean).join(" ");
  }

  if (detail.message) return String(detail.message);
  return JSON.stringify(detail);
}

function chatSessionPayload(session) {
  return {
    session_id: session.sessionId,
    created_at: session.createdAt,
    language: currentLanguage(),
    message: session.message,
    images: session.images,
    attachment_error: session.attachmentError,
    history: session.history,
  };
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
    const detail = typeof data === "object" && data?.detail ? data.detail : null;
    throw new ApiError(formatApiDetail(detail, response.status), detail, response.status);
  }

  return data;
}

export const api = {
  getStatus: () => request("/status"),
  getOutputFile: (path) => request(`/output-file?${new URLSearchParams({ path })}`),
  launchOutputFile: (path) =>
    request(`/output-file/launch?${new URLSearchParams({ path })}`, { method: "POST" }),
  deleteOutputFile: (path) =>
    request(`/output-file?${new URLSearchParams({ path })}`, { method: "DELETE" }),
  openSession: () => request("/session/open", { method: "POST" }),
  shutdown: () => request("/shutdown", { method: "POST", keepalive: true }),
  saveChatSession: (session) =>
    request("/chat/session", {
      method: "POST",
      body: JSON.stringify(chatSessionPayload(session)),
    }),
  sendShutdownBeacon: (session) => {
    if (shutdownSent) return;
    shutdownSent = true;
    const body = JSON.stringify({
      chat_session: session ? chatSessionPayload(session) : null,
    });
    if (navigator.sendBeacon) {
      const blob = new Blob([body], { type: "application/json" });
      if (navigator.sendBeacon(`${API_BASE}/shutdown`, blob)) return;
    }
    fetch(`${API_BASE}/shutdown`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
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
  askAgent: (message, images = [], options = {}) =>
    request("/agent/ask", {
      method: "POST",
      body: JSON.stringify({
        message,
        images,
        language: currentLanguage(),
        agent_progress_messages: options.agentProgressMessages || [],
        agent_task_id: options.agentTaskId || "",
      }),
      signal: options.signal,
    }),
  cancelAgentTask: (agentTaskId) =>
    request("/agent/cancel", {
      method: "POST",
      keepalive: true,
      body: JSON.stringify({ agent_task_id: agentTaskId }),
    }).catch(() => null),
  sendAgentProgressGuidance: (payload, options = {}) =>
    request("/agent/progress-guidance", {
      method: "POST",
      body: JSON.stringify({
        language: currentLanguage(),
        agent_task_id: options.agentTaskId || "",
        ...payload,
      }),
      signal: options.signal,
    }),
  startAgentTask: (payload, options = {}) =>
    request("/agent-tasks/start", {
      method: "POST",
      body: JSON.stringify(payload),
      signal: options.signal,
    }),
  getAgentTaskStatus: (taskId, options = {}) =>
    request(`/agent-tasks/${encodeURIComponent(taskId)}/status`, {
      signal: options.signal,
    }),
  sendAgentTaskMessage: (taskId, content, options = {}) =>
    request(`/agent-tasks/${encodeURIComponent(taskId)}/message`, {
      method: "POST",
      body: JSON.stringify({ content }),
      signal: options.signal,
    }),
  cancelAgentTaskRun: (taskId) =>
    request(`/agent-tasks/${encodeURIComponent(taskId)}/cancel`, {
      method: "POST",
      keepalive: true,
    }).catch(() => null),
  getAgentTaskResult: (taskId, options = {}) =>
    request(`/agent-tasks/${encodeURIComponent(taskId)}/result`, {
      signal: options.signal,
    }),
  saveJobDescription: (content) =>
    request("/job-description", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  analyzeJob: (use_github_context = false, options = {}) =>
    request("/job-description/analyze", {
      method: "POST",
      body: JSON.stringify({
        use_github_context,
        language: currentLanguage(),
        agent_progress_messages: options.agentProgressMessages || [],
        agent_task_id: options.agentTaskId || "",
      }),
      signal: options.signal,
    }),
  tailorResume: (
    use_github_context = true,
    allow_project_selection = true,
    allow_experience_removal = false,
    include_application_hint = false,
    options = {},
  ) =>
    request("/resume/tailor", {
      method: "POST",
      body: JSON.stringify({
        use_github_context,
        allow_project_selection,
        allow_experience_removal,
        include_application_hint,
        language: currentLanguage(),
        agent_progress_messages: options.agentProgressMessages || [],
        agent_task_id: options.agentTaskId || "",
      }),
      signal: options.signal,
    }),
  startTailorResumeTask: (
    use_github_context = true,
    allow_project_selection = true,
    allow_experience_removal = false,
    include_application_hint = false,
    options = {},
  ) =>
    request("/agent-tasks/start", {
      method: "POST",
      body: JSON.stringify({
        task_id: options.agentTaskId || "",
        taskType: "resume_tailor",
        language: currentLanguage(),
        body: {
          use_github_context,
          allow_project_selection,
          allow_experience_removal,
          include_application_hint,
          language: currentLanguage(),
          agent_progress_messages: options.agentProgressMessages || [],
          agent_task_id: options.agentTaskId || "",
        },
      }),
      signal: options.signal,
    }),
  checkResumeStarFacts: (options = {}, requestOptions = {}) =>
    request("/resume/star-check", {
      method: "POST",
      body: JSON.stringify({
        allow_project_selection: true,
        language: currentLanguage(),
        ...options,
        agent_task_id: requestOptions.agentTaskId || "",
      }),
      signal: requestOptions.signal,
    }),
  saveResumeStarFact: (payload, options = {}) =>
    request("/resume/star-fact", {
      method: "POST",
      body: JSON.stringify({
        ...payload,
        language: currentLanguage(),
        agent_task_id: options.agentTaskId || "",
      }),
      signal: options.signal,
    }),
  updateMemoryFromResume: (resume_source = "resume", options = {}, requestOptions = {}) =>
    request("/resume/update-memory", {
      method: "POST",
      body: JSON.stringify({
        resume_source,
        ...options,
        agent_progress_messages: requestOptions.agentProgressMessages || [],
        agent_task_id: requestOptions.agentTaskId || "",
      }),
      signal: requestOptions.signal,
    }),
  convertResumePdfToLatex: (payload, options = {}) =>
    request("/resume/pdf-to-latex", {
      method: "POST",
      body: JSON.stringify({
        ...payload,
        language: currentLanguage(),
        agent_progress_messages: options.agentProgressMessages || [],
        agent_task_id: options.agentTaskId || "",
      }),
      signal: options.signal,
    }),
  exportTailoredResumePdf: (content) =>
    request("/resume/tailored/pdf", {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  generateCoverLetter: (options = {}, requestOptions = {}) =>
    request("/cover-letter/generate", {
      method: "POST",
      body: JSON.stringify({
        use_tailored_resume: true,
        use_github_context: false,
        style: "concise",
        language: currentLanguage(),
        ...options,
        agent_progress_messages: requestOptions.agentProgressMessages || [],
        agent_task_id: requestOptions.agentTaskId || "",
      }),
      signal: requestOptions.signal,
    }),
  generateInterviewPrep: (use_github_context = true, options = {}) =>
    request("/interview-prep/generate", {
      method: "POST",
      body: JSON.stringify({
        use_github_context,
        language: currentLanguage(),
        agent_progress_messages: options.agentProgressMessages || [],
        agent_task_id: options.agentTaskId || "",
      }),
      signal: options.signal,
    }),
  scanGithub: (resume_source = "resume", options = {}, requestOptions = {}) =>
    request("/github/scan", {
      method: "POST",
      body: JSON.stringify({
        resume_source,
        ...options,
        agent_progress_messages: requestOptions.agentProgressMessages || [],
        agent_task_id: requestOptions.agentTaskId || "",
      }),
      signal: requestOptions.signal,
    }),
  getGithubConfig: () => request("/github/config"),
  saveGithubConfig: (payload) =>
    request("/github/config", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  fetchGithubContext: (approved = true, resume_source = "resume", options = {}, requestOptions = {}) =>
    request("/github/context", {
      method: "POST",
      body: JSON.stringify({
        approved,
        resume_source,
        ...options,
        agent_progress_messages: requestOptions.agentProgressMessages || [],
        agent_task_id: requestOptions.agentTaskId || "",
      }),
      signal: requestOptions.signal,
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
