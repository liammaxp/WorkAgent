import { Fragment, useEffect, useState } from "react";
import { api } from "../api/client.js";
import { useAgentProgress } from "../agentProgress/AgentProgressContext.jsx";
import {
  Alert,
  LoadingBar,
  PageHeader,
  StatusBadge,
  useAsyncAction,
} from "../components/ui.jsx";
import { text, useLanguage } from "../i18n.jsx";

function listToText(values = []) {
  return values.join("\n");
}

function textToList(value) {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function formatMemoryUpdatedAt(value, language) {
  const match = value?.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
  if (!match) return value || "-";
  const [, year, month, day, hour, minute, second] = match;
  return new Intl.DateTimeFormat(language === "en" ? "en" : "zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(`${year}-${month}-${day}T${hour}:${minute}:${second}`));
}

function formatUnixUpdatedAt(value, language) {
  if (!value) return "-";
  return new Intl.DateTimeFormat(language === "en" ? "en" : "zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(value * 1000));
}

function formatStatusTimestamp(value, language) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(language === "en" ? "en" : "zh-CN", {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(date);
}

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function githubDiagnosticsText(language) {
  if (language === "en") {
    return {
      yes: "Yes",
      no: "No",
      statusTitle: "Saved GitHub Context Status",
      refresh: "Refresh",
      statusLoading: "Loading GitHub context status...",
      statusError: "Unable to load GitHub context status.",
      statusDisabled: "GitHub context diagnostics are disabled. Enable USE_GITHUB_CONTEXT_PHASE2=1 to view saved/indexed context diagnostics.",
      endpointEnabled: "Endpoint enabled",
      saved: "Saved",
      lastSync: "Last sync",
      repoCount: "Repo count",
      recordCount: "Record count",
      rawChars: "Raw chars",
      indexedCount: "Indexed count",
      sources: "Sources",
      projects: "Projects",
      projectId: "Project ID",
      projectName: "Name",
      repository: "Repo",
      records: "Records",
      previewAvailable: "Preview",
      action: "Action",
      preview: "Preview",
      loading: "Loading",
      projectCount: "Project count",
      emptyProjects: "No saved GitHub context projects found.",
      firstProjects: "Showing first 20 projects.",
      moreProjects: "more project summaries are available from the status endpoint.",
      diagnosticsIssuesPrefix: "Status diagnostics reported",
      diagnosticsIssuesSuffix: "issue(s). Raw GitHub content is not shown.",
      previewLoading: "Loading preview...",
      previewError: "Unable to load GitHub context preview.",
      previewDisabled: "GitHub context preview is disabled. Enable USE_GITHUB_CONTEXT_PHASE2=1 to view bounded previews.",
      previewEmpty: "No bounded preview items are available for this project.",
      inspectRaw: "Inspect Raw",
      loadingRaw: "Loading Raw",
      rawLoading: "Loading bounded raw content...",
      rawError: "Unable to inspect raw GitHub context.",
      rawDisabled: "GitHub raw inspect is disabled. Enable USE_GITHUB_CONTEXT_PHASE2=1 to inspect bounded raw context.",
      totalRawChars: "Total raw chars",
      returnedChars: "Returned chars",
      truncated: "Raw content truncated",
      completeWithinLimit: "Complete within limit",
      hideRaw: "Hide Raw",
      chars: "chars",
      safetyNote: "Note: this page shows status and summaries by default. It does not automatically load full GitHub raw content.",
      rawSafetyNote: "Raw inspection is only loaded after clicking Inspect Raw, and the backend returns only a bounded segment using max_chars.",
      rawNotRendered: "Raw GitHub evidence is saved locally and summarized in the status panel above. Full raw context is not rendered on this page.",
    };
  }
  return {
    yes: "是",
    no: "否",
    statusTitle: "已保存的 GitHub 上下文状态",
    refresh: "刷新",
    statusLoading: "正在加载 GitHub 上下文状态...",
    statusError: "无法加载 GitHub 上下文状态。",
    statusDisabled: "GitHub 上下文诊断功能未启用。如需查看保存和索引状态，请在启动后端前设置：USE_GITHUB_CONTEXT_PHASE2=1",
    endpointEnabled: "诊断功能",
    saved: "保存状态",
    lastSync: "最后同步时间",
    repoCount: "仓库数量",
    recordCount: "记录数量",
    rawChars: "原始内容字符数",
    indexedCount: "已索引数量",
    sources: "数据来源",
    projects: "项目列表",
    projectId: "项目 ID",
    projectName: "项目名称",
    repository: "仓库",
    records: "记录",
    previewAvailable: "可预览",
    action: "操作",
    preview: "预览",
    loading: "正在加载",
    projectCount: "项目数量",
    emptyProjects: "暂无已保存的 GitHub 上下文项目。",
    firstProjects: "当前显示前 20 个项目。",
    moreProjects: "个更多项目摘要可通过状态接口获取。",
    diagnosticsIssuesPrefix: "状态诊断报告了",
    diagnosticsIssuesSuffix: "个问题。页面不会显示完整 GitHub 原始内容。",
    previewLoading: "正在加载预览...",
    previewError: "无法加载 GitHub 上下文预览。",
    previewDisabled: "GitHub 上下文预览功能未启用。请设置 USE_GITHUB_CONTEXT_PHASE2=1 后查看有界预览。",
    previewEmpty: "该项目暂无可预览内容。",
    inspectRaw: "查看原始内容",
    loadingRaw: "正在加载原文",
    rawLoading: "正在加载有界原始内容...",
    rawError: "无法加载原始内容。",
    rawDisabled: "GitHub 原始内容查看功能未启用。请设置 USE_GITHUB_CONTEXT_PHASE2=1 后查看有界原文。",
    totalRawChars: "原始字符总数",
    returnedChars: "已返回字符数",
    truncated: "原始内容已截断",
    completeWithinLimit: "未超过返回上限",
    hideRaw: "隐藏原始内容",
    chars: "字符",
    safetyNote: "注意：页面默认只显示状态和摘要，不会自动加载完整 GitHub 原始内容。",
    rawSafetyNote: "点击“查看原始内容”后，也只会按 max_chars 返回一小段用于调试的原文。",
    rawNotRendered: "GitHub 原始证据已保存在本地，并在上方状态面板中以摘要显示；本页面不会渲染完整原始上下文。",
  };
}

function githubEvidenceMemoryText(language) {
  if (language === "en") {
    return {
      yes: "Yes",
      no: "No",
      trueLabel: "true",
      falseLabel: "false",
      panelTitle: "GitHub Evidence Memory",
      refresh: "Refresh",
      loadingStatus: "Loading evidence memory status, health, and safe samples...",
      loadingMessage: "Loading evidence memory diagnostics...",
      disabledMessage: "GitHub evidence memory is disabled. Enable USE_GITHUB_CONTEXT_PHASE2=1 to view and build evidence memory.",
      noRawSourcesMessage: "Evidence memory is enabled, but no raw sources are available yet. Sync GitHub context first, then run the evidence build.",
      completeMessage: "Evidence memory is ready: raw sources, chunks, change summaries, evidence cards, and capability facts are available.",
      availableMessage: "Evidence memory is available for inspection.",
      status: "Status",
      health: "Health",
      enabled: "Enabled",
      available: "Available",
      rawSources: "Raw sources",
      chunks: "Chunks",
      changeSummaries: "Change summaries",
      evidenceCards: "Evidence cards",
      capabilityFacts: "Capability facts",
      rawChars: "Raw chars",
      repos: "Repos",
      pipelineComplete: "Pipeline complete",
      nextAction: "Next action",
      hasRawSources: "Has raw sources",
      hasChunks: "Has chunks",
      hasChangeSummaries: "Has change summaries",
      hasEvidenceCards: "Has evidence cards",
      hasCapabilityFacts: "Has capability facts",
      missingStages: "Missing stages",
      recommended: "Recommended",
      projectSummaries: "Project summaries",
      emptyProjects: "No evidence memory project summaries yet.",
      projectId: "Project ID",
      repo: "Repo",
      manualBuild: "Manual build",
      projectPlaceholder: "Leave empty for all projects",
      stage: "Stage",
      limit: "Limit",
      optional: "Optional",
      noRawSourcesWarning: "No raw sources are saved yet. Sync GitHub context first; this button will not trigger sync.",
      runBuild: "Run evidence build",
      runningBuild: "Running evidence build",
      enableBeforeBuild: "Enable USE_GITHUB_CONTEXT_PHASE2=1 before building.",
      buildResult: "Build result",
      buildCompleted: "Evidence build completed.",
      ranStages: "Ran stages",
      errors: "Errors",
      warnings: "Warnings",
      countsBefore: "Counts before",
      countsAfter: "Counts after",
      deltas: "Deltas",
      ok: "OK",
      processed: "Processed",
      createdUpdated: "Created/updated",
      skipped: "Skipped",
      message: "Message",
      stageSucceeded: "Completed",
      stageFailed: "Failed",
      safeSamples: "Safe inspect samples",
      inspectDisabled: "Evidence memory inspection is disabled.",
      noSafeSamples: "No safe samples available.",
      loadError: "Unable to load evidence memory diagnostics.",
      buildError: "Unable to run evidence memory build.",
      stageLabels: {
        all: "All stages",
        chunk: "Chunk raw sources",
        summarize_changes: "Summarize changes",
        build_evidence_cards: "Build evidence cards",
        build_capability_facts: "Build capability facts",
      },
      sampleLabels: {
        raw_sources: "Raw sources",
        chunks: "Chunks",
        raw_change_summaries: "Raw change summaries",
        evidence_cards: "Evidence cards",
        capability_facts: "Capability facts",
      },
      actionLabels: {
        enable_phase2: "Enable evidence memory",
        wait_for_raw_sources: "Sync GitHub context first",
        run_chunk: "Build chunks",
        summarize_changes: "Summarize changes",
        build_evidence_cards: "Build evidence cards",
        build_capability_facts: "Build capability facts",
        complete: "Complete",
      },
      fieldLabels: {
        raw_sources_count: "Raw sources",
        chunks_count: "Chunks",
        raw_change_summaries_count: "Change summaries",
        evidence_cards_count: "Evidence cards",
        capability_facts_count: "Capability facts",
        raw_chars: "Raw chars",
        repos_count: "Repos",
        source_id: "Source ID",
        source_type: "Source type",
        project_id: "Project ID",
        project_name: "Project name",
        repo: "Repo",
        path: "Path",
        chunk_id: "Chunk ID",
        chunk_type: "Chunk type",
        symbol: "Symbol",
        change_id: "Change ID",
        evidence_id: "Evidence ID",
        capability_id: "Capability ID",
        capability_type: "Capability type",
        resume_angle: "Resume angle",
        summary: "Summary",
        problem: "Problem",
        mechanism: "Mechanism",
        safe_impact: "Safe impact",
        allowed_claims: "Allowed claims",
        direct_code_evidence: "Direct code evidence",
        technical_tags: "Technical tags",
        created_at: "Created at",
        updated_at: "Updated at",
      },
    };
  }
  return {
    yes: "是",
    no: "否",
    trueLabel: "是",
    falseLabel: "否",
    panelTitle: "GitHub 证据记忆",
    refresh: "刷新",
    loadingStatus: "正在加载证据记忆状态、健康检查和安全样例...",
    loadingMessage: "正在加载证据记忆诊断...",
    disabledMessage: "GitHub 证据记忆未启用。如需查看和构建证据记忆，请设置 USE_GITHUB_CONTEXT_PHASE2=1。",
    noRawSourcesMessage: "证据记忆已启用，但还没有原始来源。请先同步 GitHub 上下文，再运行证据构建。",
    completeMessage: "证据记忆已准备好：原始来源、分块、变更摘要、证据卡片和能力事实均可用。",
    availableMessage: "证据记忆可供查看。",
    status: "状态",
    health: "健康检查",
    enabled: "已启用",
    available: "可用",
    rawSources: "原始来源",
    chunks: "分块",
    changeSummaries: "变更摘要",
    evidenceCards: "证据卡片",
    capabilityFacts: "能力事实",
    rawChars: "原始字符数",
    repos: "仓库数",
    pipelineComplete: "构建完成",
    nextAction: "下一步",
    hasRawSources: "已有原始来源",
    hasChunks: "已有分块",
    hasChangeSummaries: "已有变更摘要",
    hasEvidenceCards: "已有证据卡片",
    hasCapabilityFacts: "已有能力事实",
    missingStages: "缺少步骤",
    recommended: "建议操作",
    projectSummaries: "项目摘要",
    emptyProjects: "暂无证据记忆项目摘要。",
    projectId: "项目 ID",
    repo: "仓库",
    manualBuild: "手动构建",
    projectPlaceholder: "留空表示全部项目",
    stage: "构建步骤",
    limit: "数量上限",
    optional: "可选",
    noRawSourcesWarning: "当前还没有保存原始来源。请先同步 GitHub 上下文；此按钮不会触发同步。",
    runBuild: "运行证据构建",
    runningBuild: "正在构建证据",
    enableBeforeBuild: "构建前请先启用 USE_GITHUB_CONTEXT_PHASE2=1。",
    buildResult: "构建结果",
    buildCompleted: "证据构建已完成。",
    ranStages: "已运行步骤",
    errors: "错误",
    warnings: "警告",
    countsBefore: "构建前数量",
    countsAfter: "构建后数量",
    deltas: "变化量",
    ok: "成功",
    processed: "已处理",
    createdUpdated: "新增或更新",
    skipped: "已跳过",
    message: "消息",
    stageSucceeded: "已完成",
    stageFailed: "失败",
    safeSamples: "安全检查样例",
    inspectDisabled: "证据记忆检查未启用。",
    noSafeSamples: "暂无安全样例。",
    loadError: "无法加载证据记忆诊断。",
    buildError: "无法运行证据记忆构建。",
    stageLabels: {
      all: "全部步骤",
      chunk: "生成原始来源分块",
      summarize_changes: "生成变更摘要",
      build_evidence_cards: "生成证据卡片",
      build_capability_facts: "生成能力事实",
    },
    sampleLabels: {
      raw_sources: "原始来源",
      chunks: "分块",
      raw_change_summaries: "变更摘要",
      evidence_cards: "证据卡片",
      capability_facts: "能力事实",
    },
    actionLabels: {
      enable_phase2: "启用证据记忆",
      wait_for_raw_sources: "先同步 GitHub 上下文",
      run_chunk: "生成分块",
      summarize_changes: "生成变更摘要",
      build_evidence_cards: "生成证据卡片",
      build_capability_facts: "生成能力事实",
      complete: "已完成",
    },
    fieldLabels: {
      raw_sources_count: "原始来源",
      chunks_count: "分块",
      raw_change_summaries_count: "变更摘要",
      evidence_cards_count: "证据卡片",
      capability_facts_count: "能力事实",
      raw_chars: "原始字符数",
      repos_count: "仓库数",
      source_id: "来源 ID",
      source_type: "来源类型",
      project_id: "项目 ID",
      project_name: "项目名称",
      repo: "仓库",
      path: "路径",
      chunk_id: "分块 ID",
      chunk_type: "分块类型",
      symbol: "符号",
      change_id: "变更 ID",
      evidence_id: "证据 ID",
      capability_id: "能力事实 ID",
      capability_type: "能力类型",
      resume_angle: "简历角度",
      summary: "摘要",
      problem: "问题",
      mechanism: "机制",
      safe_impact: "安全影响",
      allowed_claims: "可使用表述",
      direct_code_evidence: "直接代码证据",
      technical_tags: "技术标签",
      created_at: "创建时间",
      updated_at: "更新时间",
    },
  };
}

function yesNo(value, ui) {
  return value ? ui.yes : ui.no;
}

function statusSourceLine(label, source, ui, countLabel = ui.projectCount) {
  if (!source) return null;
  const exists = source.exists ?? source.available;
  const count = source.project_count ?? source.count ?? 0;
  return (
    <div className="github-status-source" key={label}>
      <span>{label}</span>
      <strong>{yesNo(Boolean(exists), ui)}</strong>
      <small>{countLabel}: {formatCount(count)}</small>
    </div>
  );
}

function previewProjectKey(project) {
  return String(project?.project_id || project?.repo || project?.project_name || "").trim();
}

const PHASE2_STAGE_OPTIONS = [
  { value: "all", label: "All stages" },
  { value: "chunk", label: "chunk" },
  { value: "summarize_changes", label: "summarize_changes" },
  { value: "build_evidence_cards", label: "build_evidence_cards" },
  { value: "build_capability_facts", label: "build_capability_facts" },
];

const PHASE2_SAMPLE_LABELS = {
  raw_sources: "Raw sources",
  chunks: "Chunks",
  raw_change_summaries: "Raw change summaries",
  evidence_cards: "Evidence cards",
  capability_facts: "Capability facts",
};

const PHASE2_SENSITIVE_SAMPLE_KEYS = new Set([
  "raw_text",
  "text",
  "chunk_text",
  "full_text",
  "raw",
  "patch",
  "content",
]);

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function truncateDisplayValue(value, limit = 400) {
  const textValue = String(value ?? "");
  if (textValue.length <= limit) return textValue;
  return `${textValue.slice(0, limit - 3).trimEnd()}...`;
}

function sanitizePhase2SampleForDisplay(sample) {
  if (Array.isArray(sample)) {
    return sample.slice(0, 12).map((item) => sanitizePhase2SampleForDisplay(item));
  }
  if (sample && typeof sample === "object") {
    return Object.fromEntries(
      Object.entries(sample)
        .filter(([key]) => !PHASE2_SENSITIVE_SAMPLE_KEYS.has(String(key).toLowerCase()))
        .map(([key, value]) => [key, sanitizePhase2SampleForDisplay(value)]),
    );
  }
  if (typeof sample === "string") {
    return truncateDisplayValue(sample, 400);
  }
  return sample;
}

function phase2DisplayValue(value, ui) {
  if (Array.isArray(value)) {
    if (!value.length) return "-";
    return value.map((item) => phase2DisplayValue(item, ui)).join(", ");
  }
  if (value && typeof value === "object") {
    return truncateDisplayValue(JSON.stringify(sanitizePhase2SampleForDisplay(value)), 400);
  }
  if (typeof value === "boolean") return value ? ui.trueLabel : ui.falseLabel;
  if (value === null || value === undefined || value === "") return "-";
  return truncateDisplayValue(value, 400);
}

function phase2ProjectOptions(status, inspect) {
  const projects = [
    ...safeArray(status?.projects),
    ...safeArray(inspect?.projects),
  ];
  return Array.from(
    new Set(
      projects
        .map((project) => String(project?.project_id || "").trim())
        .filter(Boolean),
    ),
  ).sort((left, right) => left.localeCompare(right));
}

function phase2StageLabel(stage, ui) {
  return ui.stageLabels?.[stage] || stage || "-";
}

function phase2StageList(stages, ui) {
  const items = safeArray(stages).map((stage) => phase2StageLabel(stage, ui));
  return items.join(", ") || "-";
}

function phase2ActionLabel(action, ui) {
  return ui.actionLabels?.[action] || action || "-";
}

function phase2FieldLabel(field, ui) {
  return ui.fieldLabels?.[field] || field;
}

function phase2StageMessage(stage, ui) {
  return stage?.ok ? ui.stageSucceeded : ui.stageFailed;
}

function GitHubRawPanel({ rawState, onClose, ui }) {
  if (!rawState) return null;
  if (rawState.loading) {
    return <div className="github-raw-panel helper-text">{ui.rawLoading}</div>;
  }
  if (rawState.error) {
    return (
      <div className="github-raw-panel github-status-message github-status-error">
        {ui.rawError}
      </div>
    );
  }
  const raw = rawState.data;
  if (!raw) return null;
  if (raw.enabled === false) {
    return (
      <div className="github-raw-panel github-status-message">
        {ui.rawDisabled}
      </div>
    );
  }
  return (
    <div className="github-raw-panel">
      <div className="github-raw-toolbar">
        <span>{ui.totalRawChars}: {formatCount(raw.raw_chars_total)}</span>
        <span>{ui.returnedChars}: {formatCount(raw.returned_chars)} / {formatCount(raw.max_chars)}</span>
        <span>{raw.truncated ? ui.truncated : ui.completeWithinLimit}</span>
        <button type="button" className="btn btn-secondary btn-small" onClick={onClose}>
          {ui.hideRaw}
        </button>
      </div>
      <pre className="github-raw-text">{raw.raw_text || ""}</pre>
    </div>
  );
}

function GitHubPreviewPanel({ preview, rawInspections, onInspectRaw, onCloseRaw, ui }) {
  if (!preview) return null;
  if (preview.loading) {
    return <div className="github-preview-panel helper-text">{ui.previewLoading}</div>;
  }
  if (preview.error) {
    return (
      <div className="github-preview-panel github-status-message github-status-error">
        {ui.previewError}
      </div>
    );
  }
  if (preview.data?.enabled === false) {
    return (
      <div className="github-preview-panel github-status-message">
        {ui.previewDisabled}
      </div>
    );
  }
  const items = Array.isArray(preview.data?.items) ? preview.data.items : [];
  if (!items.length) {
    return <div className="github-preview-panel empty-state">{ui.previewEmpty}</div>;
  }
  return (
    <div className="github-preview-panel">
      {items.map((item) => (
        <div className="github-preview-item" key={item.source_id}>
          <div className="github-preview-meta">
            <strong>{item.source_id}</strong>
            <span>{item.source_type || "-"}</span>
            <span>{item.repo || "-"}</span>
            <span>{item.path || "-"}</span>
            <span>{ui.rawChars}: {formatCount(item.raw_chars)}</span>
            <button
              type="button"
              className="btn btn-secondary btn-small"
              onClick={() => onInspectRaw(item)}
              disabled={!item.source_id || rawInspections?.[item.source_id]?.loading}
            >
              {rawInspections?.[item.source_id]?.loading ? ui.loadingRaw : ui.inspectRaw}
            </button>
          </div>
          {item.summary && <p>{item.summary}</p>}
          {item.preview_text && <p className="github-preview-text">{item.preview_text}</p>}
          <GitHubRawPanel
            rawState={rawInspections?.[item.source_id]}
            onClose={() => onCloseRaw(item.source_id)}
            ui={ui}
          />
        </div>
      ))}
    </div>
  );
}

function GitHubContextStatusPanel({
  status,
  loading,
  error,
  onRefresh,
  onPreview,
  previews,
  rawInspections,
  onInspectRaw,
  onCloseRaw,
  language,
}) {
  const ui = githubDiagnosticsText(language);
  const sources = status?.sources || {};
  const projects = Array.isArray(status?.projects) ? status.projects.slice(0, 20) : [];
  const hiddenProjectCount = Math.max(0, Number(status?.projects?.length || 0) - projects.length);

  return (
    <section className="card">
      <div className="section-toolbar">
        <h2 className="card-title">{ui.statusTitle}</h2>
        <button type="button" className="btn btn-secondary btn-small" onClick={onRefresh} disabled={loading}>
          {ui.refresh}
        </button>
      </div>

      {loading && <p className="helper-text">{ui.statusLoading}</p>}

      {!loading && error && (
        <div className="github-status-message github-status-error">
          {ui.statusError}
        </div>
      )}

      {!loading && !error && status?.enabled === false && (
        <div className="github-status-message">
          {ui.statusDisabled}
        </div>
      )}

      {!loading && !error && status?.enabled && (
        <>
          <p className="helper-text">{ui.safetyNote}</p>
          <div className="github-status-grid">
            <div className="github-status-metric">
              <span>{ui.endpointEnabled}</span>
              <strong>{yesNo(status.enabled, ui)}</strong>
            </div>
            <div className="github-status-metric">
              <span>{ui.saved}</span>
              <strong>{yesNo(status.saved, ui)}</strong>
            </div>
            <div className="github-status-metric">
              <span>{ui.lastSync}</span>
              <strong>{formatStatusTimestamp(status.last_sync_at, language)}</strong>
            </div>
            <div className="github-status-metric">
              <span>{ui.repoCount}</span>
              <strong>{formatCount(status.repo_count)}</strong>
            </div>
            <div className="github-status-metric">
              <span>{ui.recordCount}</span>
              <strong>{formatCount(status.record_count)}</strong>
            </div>
            <div className="github-status-metric">
              <span>{ui.rawChars}</span>
              <strong>{formatCount(status.raw_chars)}</strong>
            </div>
            <div className="github-status-metric">
              <span>{ui.indexedCount}</span>
              <strong>{formatCount(status.indexed_count)}</strong>
            </div>
          </div>

          <h3 className="github-status-subtitle">{ui.sources}</h3>
          <div className="github-status-sources">
            {[
              statusSourceLine("project_memory", sources.project_memory, ui),
              statusSourceLine("project_compact_facts", sources.project_compact_facts, ui),
              statusSourceLine("chroma_github_evidence", sources.chroma_github_evidence, ui, ui.recordCount),
            ].filter(Boolean)}
          </div>

          <h3 className="github-status-subtitle">{ui.projects}</h3>
          {projects.length ? (
            <>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>{ui.projectId}</th>
                      <th>{ui.projectName}</th>
                      <th>{ui.repository}</th>
                      <th>{ui.saved}</th>
                      <th>{ui.records}</th>
                      <th>{ui.rawChars}</th>
                      <th>{ui.previewAvailable}</th>
                      <th>{ui.action}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {projects.map((project, index) => {
                      const projectKey = previewProjectKey(project);
                      const preview = previews?.[projectKey] || null;
                      return (
                        <Fragment key={`${project.project_id || project.repo || "project"}-${index}`}>
                          <tr key={`${project.project_id || project.repo || "project"}-${index}`}>
                            <td>{project.project_id || "-"}</td>
                            <td>{project.project_name || "-"}</td>
                            <td>{project.repo || "-"}</td>
                            <td>{yesNo(project.saved, ui)}</td>
                            <td>{formatCount(project.record_count)}</td>
                            <td>{formatCount(project.raw_chars)}</td>
                            <td>{yesNo(project.preview_available, ui)}</td>
                            <td>
                              <button
                                type="button"
                                className="btn btn-secondary btn-small"
                                onClick={() => onPreview(project)}
                                disabled={!projectKey || preview?.loading}
                              >
                                {preview?.loading ? ui.loading : ui.preview}
                              </button>
                            </td>
                          </tr>
                          {preview && (
                            <tr key={`${project.project_id || project.repo || "project"}-${index}-preview`}>
                              <td colSpan={8}>
                                <GitHubPreviewPanel
                                  preview={preview}
                                  rawInspections={rawInspections}
                                  onInspectRaw={onInspectRaw}
                                  onCloseRaw={onCloseRaw}
                                  ui={ui}
                                />
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {hiddenProjectCount > 0 && (
                <p className="helper-text">{ui.firstProjects} {formatCount(hiddenProjectCount)} {ui.moreProjects}</p>
              )}
            </>
          ) : (
            <p className="empty-state">{ui.emptyProjects}</p>
          )}

          {Array.isArray(status.errors) && status.errors.length > 0 && (
            <p className="warning-line">{ui.diagnosticsIssuesPrefix} {formatCount(status.errors.length)} {ui.diagnosticsIssuesSuffix}</p>
          )}
          <p className="helper-text">{ui.rawSafetyNote}</p>
        </>
      )}
    </section>
  );
}

function Phase2KeyValueGrid({ items }) {
  return (
    <div className="github-status-grid phase2-status-grid">
      {items.map((item) => (
        <div className="github-status-metric" key={item.label}>
          <span>{item.label}</span>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

function Phase2ProjectSummaries({ projects, ui }) {
  if (!projects.length) {
    return <p className="empty-state phase2-empty-state">{ui.emptyProjects}</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>{ui.projectId}</th>
            <th>{ui.repo}</th>
            <th>{ui.rawSources}</th>
            <th>{ui.chunks}</th>
            <th>{ui.changeSummaries}</th>
            <th>{ui.evidenceCards}</th>
            <th>{ui.capabilityFacts}</th>
            <th>{ui.rawChars}</th>
          </tr>
        </thead>
        <tbody>
          {projects.map((project, index) => (
            <tr key={`${project.project_id || "project"}-${project.repo || "repo"}-${index}`}>
              <td>{project.project_id || "-"}</td>
              <td>{project.repo || "-"}</td>
              <td>{formatCount(project.raw_sources)}</td>
              <td>{formatCount(project.chunks)}</td>
              <td>{formatCount(project.raw_change_summaries)}</td>
              <td>{formatCount(project.evidence_cards)}</td>
              <td>{formatCount(project.capability_facts)}</td>
              <td>{formatCount(project.raw_chars)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Phase2BuildResult({ result, error, ui }) {
  if (error) {
    return <div className="github-status-message github-status-error">{error}</div>;
  }
  if (!result) return null;
  return (
    <div className="phase2-build-result">
      <h3 className="github-status-subtitle">{ui.buildResult}</h3>
      <p className="helper-text">{ui.buildCompleted}</p>
      <Phase2KeyValueGrid
        items={[
          { label: ui.ranStages, value: phase2StageList(result.ran_stages, ui) },
          { label: ui.errors, value: formatCount(safeArray(result.errors).length) },
          { label: ui.warnings, value: formatCount(safeArray(result.warnings).length) },
        ]}
      />
      <div className="phase2-count-columns">
        {[
          [ui.countsBefore, result.counts_before],
          [ui.countsAfter, result.counts_after],
          [ui.deltas, result.deltas],
        ].map(([title, counts]) => (
          <div className="phase2-count-box" key={title}>
            <strong>{title}</strong>
            {Object.entries(counts || {}).map(([key, value]) => (
              <div key={key}>
                <span>{phase2FieldLabel(key, ui)}</span>
                <b>{formatCount(value)}</b>
              </div>
            ))}
          </div>
        ))}
      </div>
      {safeArray(result.stage_results).length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{ui.stage}</th>
                <th>{ui.ok}</th>
                <th>{ui.processed}</th>
                <th>{ui.createdUpdated}</th>
                <th>{ui.skipped}</th>
                <th>{ui.message}</th>
              </tr>
            </thead>
            <tbody>
              {safeArray(result.stage_results).map((stage) => (
                <tr key={stage.stage}>
                  <td>{phase2StageLabel(stage.stage, ui)}</td>
                  <td>{stage.ok ? ui.yes : ui.no}</td>
                  <td>{formatCount(stage.processed)}</td>
                  <td>{formatCount(stage.created_or_updated)}</td>
                  <td>{formatCount(stage.skipped)}</td>
                  <td>{phase2StageMessage(stage, ui)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Phase2InspectSamples({ samples, ui }) {
  const sampleEntries = Object.keys(PHASE2_SAMPLE_LABELS).map((sampleKey) => [
    sampleKey,
    ui.sampleLabels?.[sampleKey] || PHASE2_SAMPLE_LABELS[sampleKey],
  ]);
  return (
    <div className="phase2-samples">
      {sampleEntries.map(([sampleKey, label]) => {
        const items = safeArray(samples?.[sampleKey]).map((item) => sanitizePhase2SampleForDisplay(item));
        return (
          <div className="phase2-sample-group" key={sampleKey}>
            <h4>{label}</h4>
            {!items.length ? (
              <p className="helper-text">{ui.noSafeSamples}</p>
            ) : (
              items.map((item, index) => (
                <div className="phase2-sample-item" key={`${sampleKey}-${index}`}>
                  {Object.entries(item).map(([key, value]) => (
                    <div className="phase2-sample-row" key={key}>
                      <span>{phase2FieldLabel(key, ui)}</span>
                      <strong>{phase2DisplayValue(value, ui)}</strong>
                    </div>
                  ))}
                </div>
              ))
            )}
          </div>
        );
      })}
    </div>
  );
}

function GitHubPhase2Panel({
  status,
  health,
  inspect,
  loading,
  error,
  buildForm,
  setBuildForm,
  building,
  buildError,
  buildResult,
  onRefresh,
  onBuild,
  language,
}) {
  const ui = githubEvidenceMemoryText(language);
  const statusLoaded = Boolean(status);
  const enabled = status?.enabled === true;
  const counts = status || {};
  const healthFlags = health?.health || {};
  const projects = safeArray(status?.projects).length ? safeArray(status?.projects) : safeArray(inspect?.projects);
  const projectOptions = phase2ProjectOptions(status, inspect);
  const noRawSources = enabled && Number(counts.raw_sources_count || 0) === 0;
  const pipelineComplete = Boolean(status?.pipeline_complete || health?.pipeline_complete);
  const statusMessage = !statusLoaded
    ? ui.loadingMessage
    : !enabled
    ? ui.disabledMessage
    : noRawSources
      ? ui.noRawSourcesMessage
      : pipelineComplete
        ? ui.completeMessage
        : ui.availableMessage;

  const handleBuildClick = () => {
    const selectedStage = buildForm.stage === "all" ? null : [buildForm.stage];
    onBuild({
      projectId: buildForm.projectId.trim(),
      stages: selectedStage,
      limit: buildForm.limit,
      continueOnError: true,
    });
  };

  return (
    <section className="card phase2-panel">
      <div className="section-toolbar">
        <h2 className="card-title">{ui.panelTitle}</h2>
        <button type="button" className="btn btn-secondary btn-small" onClick={onRefresh} disabled={loading || building}>
          {ui.refresh}
        </button>
      </div>

      {loading && <p className="helper-text">{ui.loadingStatus}</p>}
      {error && <div className="github-status-message github-status-error">{error}</div>}
      <div className="github-status-message">{statusMessage}</div>

      <h3 className="github-status-subtitle">{ui.status}</h3>
      <Phase2KeyValueGrid
        items={[
          { label: ui.enabled, value: status?.enabled ? ui.yes : ui.no },
          { label: ui.available, value: status?.available ? ui.yes : ui.no },
          { label: ui.rawSources, value: formatCount(counts.raw_sources_count) },
          { label: ui.chunks, value: formatCount(counts.chunks_count) },
          { label: ui.changeSummaries, value: formatCount(counts.raw_change_summaries_count) },
          { label: ui.evidenceCards, value: formatCount(counts.evidence_cards_count) },
          { label: ui.capabilityFacts, value: formatCount(counts.capability_facts_count) },
          { label: ui.rawChars, value: formatCount(counts.raw_chars) },
          { label: ui.repos, value: formatCount(counts.repos_count) },
          { label: ui.pipelineComplete, value: pipelineComplete ? ui.yes : ui.no },
          { label: ui.nextAction, value: phase2ActionLabel(status?.next_recommended_action || health?.next_recommended_action, ui) },
        ]}
      />

      <h3 className="github-status-subtitle">{ui.health}</h3>
      <Phase2KeyValueGrid
        items={[
          { label: ui.hasRawSources, value: healthFlags.has_raw_sources ? ui.yes : ui.no },
          { label: ui.hasChunks, value: healthFlags.has_chunks ? ui.yes : ui.no },
          { label: ui.hasChangeSummaries, value: healthFlags.has_raw_change_summaries ? ui.yes : ui.no },
          { label: ui.hasEvidenceCards, value: healthFlags.has_evidence_cards ? ui.yes : ui.no },
          { label: ui.hasCapabilityFacts, value: healthFlags.has_capability_facts ? ui.yes : ui.no },
          { label: ui.missingStages, value: phase2StageList(health?.missing_stages, ui) },
          { label: ui.recommended, value: phase2ActionLabel(health?.next_recommended_action, ui) },
        ]}
      />

      <h3 className="github-status-subtitle">{ui.projectSummaries}</h3>
      <Phase2ProjectSummaries projects={projects} ui={ui} />

      <h3 className="github-status-subtitle">{ui.manualBuild}</h3>
      <div className="phase2-build-controls">
        <div className="field compact-field">
          <label>{ui.projectId}</label>
          <input
            list="phase2-project-options"
            value={buildForm.projectId}
            onChange={(event) => setBuildForm((current) => ({ ...current, projectId: event.target.value }))}
            placeholder={ui.projectPlaceholder}
            disabled={!statusLoaded || !enabled || building}
          />
          <datalist id="phase2-project-options">
            {projectOptions.map((projectId) => (
              <option key={projectId} value={projectId} />
            ))}
          </datalist>
        </div>
        <div className="field compact-field">
          <label>{ui.stage}</label>
          <select
            value={buildForm.stage}
            onChange={(event) => setBuildForm((current) => ({ ...current, stage: event.target.value }))}
            disabled={!statusLoaded || !enabled || building}
          >
            {PHASE2_STAGE_OPTIONS.map((stage) => (
              <option key={stage.value} value={stage.value}>{phase2StageLabel(stage.value, ui)}</option>
            ))}
          </select>
        </div>
        <div className="field compact-field">
          <label>{ui.limit}</label>
          <input
            type="number"
            min="0"
            value={buildForm.limit}
            onChange={(event) => setBuildForm((current) => ({ ...current, limit: event.target.value }))}
            placeholder={ui.optional}
            disabled={!statusLoaded || !enabled || building}
          />
        </div>
      </div>
      {noRawSources && (
        <p className="warning-line">{ui.noRawSourcesWarning}</p>
      )}
      <div className="btn-row">
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleBuildClick}
          disabled={!statusLoaded || !enabled || building}
        >
          {building ? ui.runningBuild : ui.runBuild}
        </button>
        {statusLoaded && !enabled && <span className="helper-text">{ui.enableBeforeBuild}</span>}
      </div>
      <Phase2BuildResult result={buildResult} error={buildError} ui={ui} />

      <h3 className="github-status-subtitle">{ui.safeSamples}</h3>
      {inspect?.enabled === false ? (
        <div className="github-status-message">{ui.inspectDisabled}</div>
      ) : (
        <Phase2InspectSamples samples={inspect?.samples || {}} ui={ui} />
      )}
    </section>
  );
}

function resolveProjectMemoryUpdatedAt(githubConfig, status) {
  return (
    githubConfig?.project_memory_updated_at ||
    status?.file_metadata?.project_memory?.mtime ||
    null
  );
}

function formatGithubEvidenceStatus(result, language) {
  const cacheStatus = result?.cache_status || "";
  const reason = result?.change_reason || "";
  const zhCacheLabels = {
    fetch: "全量获取 GitHub 证据",
    incremental: "增量获取 GitHub 证据",
    reused: "复用本地 GitHub 证据缓存",
    "remote-state-error": "远端状态检查失败",
  };
  const enCacheLabels = {
    fetch: "full GitHub evidence fetch",
    incremental: "incremental GitHub evidence fetch",
    reused: "reused local GitHub evidence cache",
    "remote-state-error": "remote state check failed",
  };
  const zhReasonLabels = {
    "latest commit changed": "检测到最新 commit 变化",
    "latest commit unchanged": "最新 commit 未变化",
    unchanged: "远端未变化",
    "default branch changed": "默认分支变化",
    "remote state check failed": "远端状态检查失败",
  };
  const enReasonLabels = {
    "latest commit changed": "latest commit changed",
    "latest commit unchanged": "latest commit unchanged",
    unchanged: "remote unchanged",
    "default branch changed": "default branch changed",
    "remote state check failed": "remote state check failed",
  };
  const cacheLabel = language === "en"
    ? (enCacheLabels[cacheStatus] || cacheStatus || "unknown")
    : (zhCacheLabels[cacheStatus] || cacheStatus || "未知状态");
  const reasonLabel = language === "en"
    ? (enReasonLabels[reason] || reason)
    : (zhReasonLabels[reason] || reason);
  return reasonLabel ? `${cacheLabel} · ${reasonLabel}` : cacheLabel;
}

function getProjectMemoryStatus(context, language) {
  const summary = context?.project_memory_status || context?.project_memory_update?.status_summary;
  if (!summary) return null;
  return {
    status: summary.status || "unknown",
    label: language === "en"
      ? (summary.label_en || summary.label)
      : (summary.label_zh || summary.label),
    detail: language === "en"
      ? (summary.detail_en || summary.detail)
      : (summary.detail_zh || summary.detail),
  };
}

function formatMapReduceSummary(context, language) {
  const repos = context?.project_memory_update?.map_reduce_repositories || [];
  if (!Array.isArray(repos) || !repos.length) return "";
  const totalChunks = repos.reduce((sum, repo) => sum + Number(repo.chunkCount || 0), 0);
  const title = language === "en"
    ? `Project-level Map-Reduce used for ${repos.length} repo(s), ${totalChunks || repos.length} chunk(s).`
    : `已启用项目级 Map-Reduce：${repos.length} 个仓库，${totalChunks || repos.length} 个分块。`;
  const lines = repos.slice(0, 6).map((repo) => {
    const repoName = repo.repoName || "repo";
    const projectName = repo.projectName || "";
    const chunkCount = repo.chunkCount || 1;
    const reason = repo.reason || "";
    return `- ${projectName ? `${projectName} / ` : ""}${repoName}: ${chunkCount} chunk(s)${reason ? ` (${reason})` : ""}`;
  });
  const more = repos.length > 6 ? [`- ... ${repos.length - 6} more repo(s)`] : [];
  return [title, ...lines, ...more].join("\n");
}

function normalizeProjectAlias(value) {
  let textValue = String(value || "").trim().toLowerCase();
  const githubMatch = textValue.match(/https?:\/\/(?:www\.)?github\.com\/([a-z0-9_.-]+)\/([a-z0-9_.-]+)/);
  if (githubMatch) textValue = `${githubMatch[1]}/${githubMatch[2]}`;
  textValue = textValue.replace(/^https?:\/\/(?:www\.)?github\.com\//, "");
  const repoMatch = textValue.match(/^([a-z0-9_.-]+)\/([a-z0-9_.-]+)$/);
  if (repoMatch) textValue = repoMatch[2];
  textValue = textValue.replace(/\.git$/, "");
  return textValue.replace(/[^a-z0-9]+/g, "");
}

function appendProjectOption(options, usedKeys, aliases, preferredValue) {
  const aliasValues = aliases.map((alias) => String(alias || "").trim()).filter(Boolean);
  if (!aliasValues.length) return;
  const keys = Array.from(new Set(aliasValues.map(normalizeProjectAlias).filter(Boolean)));
  if (!keys.length) return;
  const existing = options.find((option) => option.keys.some((key) => keys.includes(key)));
  if (existing) {
    keys.forEach((key) => {
      if (!usedKeys.has(key)) existing.keys.push(key);
      usedKeys.add(key);
    });
    return;
  }
  const label = String(preferredValue || aliasValues[0]).trim();
  options.push({ label, keys });
  keys.forEach((key) => usedKeys.add(key));
}

function collectProjectOptions(projectMemory) {
  const options = [];
  const usedKeys = new Set();
  const rawProjects = projectMemory?.projects;
  const projects = Array.isArray(rawProjects)
    ? rawProjects
    : rawProjects && typeof rawProjects === "object"
      ? [rawProjects]
      : [];

  projects.forEach((project) => {
    if (!project || typeof project !== "object") return;
    const aliases = [
      project.project_name,
      project.project_id,
      project.name,
      project.title,
      project.repository,
    ];

    const identity = project.identity;
    if (identity && typeof identity === "object") {
      aliases.push(identity.project_name, identity.project_id, identity.name);
    }

    const evidenceNotes = String(project.evidence_notes || "");
    evidenceNotes.replace(
      /(?:Repository|repository|repo)\s*:\s*([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)/g,
      (_, repository) => {
        aliases.push(repository.replace(/\.git$/, ""));
        return "";
      },
    );

    appendProjectOption(
      options,
      usedKeys,
      aliases,
      project.project_name || project.name || project.title || project.project_id || project.repository,
    );
  });

  return options.sort((left, right) => left.label.localeCompare(right.label));
}

export default function GitHubContext() {
  const { language } = useLanguage();
  const copy = text[language].github;
  const [scan, setScan] = useState(null);
  const [context, setContext] = useState(null);
  const [source, setSource] = useState("tailored_resume_and_resume_and_memory");
  const [projectScope, setProjectScope] = useState("");
  const [forceRefresh, setForceRefresh] = useState(false);
  const [reanalyzeCached, setReanalyzeCached] = useState(false);
  const [githubForm, setGithubForm] = useState({
    usernames: "",
    author_names: "",
    author_emails: "",
    token: "",
  });
  const [tokenConfigured, setTokenConfigured] = useState(false);
  const [memoryRepositories, setMemoryRepositories] = useState([]);
  const [projectOptions, setProjectOptions] = useState([]);
  const [projectMemoryUpdatedAt, setProjectMemoryUpdatedAt] = useState(null);
  const [contextStatus, setContextStatus] = useState(null);
  const [contextStatusLoading, setContextStatusLoading] = useState(false);
  const [contextStatusError, setContextStatusError] = useState("");
  const [contextPreviews, setContextPreviews] = useState({});
  const [rawInspections, setRawInspections] = useState({});
  const [phase2Status, setPhase2Status] = useState(null);
  const [phase2Health, setPhase2Health] = useState(null);
  const [phase2Inspect, setPhase2Inspect] = useState(null);
  const [phase2Loading, setPhase2Loading] = useState(false);
  const [phase2Error, setPhase2Error] = useState("");
  const [phase2BuildForm, setPhase2BuildForm] = useState({
    projectId: "",
    stage: "all",
    limit: "",
  });
  const [phase2Building, setPhase2Building] = useState(false);
  const [phase2BuildError, setPhase2BuildError] = useState("");
  const [phase2BuildResult, setPhase2BuildResult] = useState(null);
  const { loading, error, success, run } = useAsyncAction();
  const { active: agentActive, runAgentWithProgress } = useAgentProgress();

  const loadGithubConfig = () =>
    run(async () => {
      const [data, status, projectMemoryFile] = await Promise.all([
        api.getGithubConfig(),
        api.getStatus(),
        api.getFile("project_memory").catch(() => ({ content: "" })),
      ]);
      setGithubForm((current) => ({
        usernames: listToText(data.identities?.usernames),
        author_names: listToText(data.identities?.author_names),
        author_emails: listToText(data.identities?.author_emails),
        token: current.token,
      }));
      setTokenConfigured(data.token_configured);
      setMemoryRepositories(data.memory_repositories || []);
      try {
        setProjectOptions(collectProjectOptions(JSON.parse(projectMemoryFile.content || "{}")));
      } catch {
        setProjectOptions([]);
      }
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(data, status));
      return data;
    });

  const loadGithubContextStatus = async () => {
    setContextStatusLoading(true);
    setContextStatusError("");
    try {
      const data = await api.getGithubContextStatus();
      setContextStatus(data);
      return data;
    } catch (statusError) {
      setContextStatus(null);
      setContextStatusError(statusError?.message || "Unable to load GitHub context status.");
      return null;
    } finally {
      setContextStatusLoading(false);
    }
  };

  const loadGithubContextPreview = async (project) => {
    const projectKey = previewProjectKey(project);
    if (!projectKey) return null;
    setContextPreviews((current) => ({
      ...current,
      [projectKey]: { loading: true, error: "", data: current[projectKey]?.data || null },
    }));
    try {
      const data = await api.getGithubContextPreview(projectKey, 5);
      setContextPreviews((current) => ({
        ...current,
        [projectKey]: { loading: false, error: "", data },
      }));
      return data;
    } catch (previewError) {
      setContextPreviews((current) => ({
        ...current,
        [projectKey]: {
          loading: false,
          error: previewError?.message || "Unable to load GitHub context preview.",
          data: null,
        },
      }));
      return null;
    }
  };

  const inspectGithubContextRaw = async (item) => {
    const sourceId = String(item?.source_id || "").trim();
    if (!sourceId) return null;
    setRawInspections((current) => ({
      ...current,
      [sourceId]: { loading: true, error: "", data: current[sourceId]?.data || null },
    }));
    try {
      const data = await api.getGithubContextRaw(sourceId, 10000);
      setRawInspections((current) => ({
        ...current,
        [sourceId]: { loading: false, error: "", data },
      }));
      return data;
    } catch (rawError) {
      setRawInspections((current) => ({
        ...current,
        [sourceId]: {
          loading: false,
          error: rawError?.message || "Unable to inspect raw GitHub context.",
          data: null,
        },
      }));
      return null;
    }
  };

  const closeGithubContextRaw = (sourceId) => {
    setRawInspections((current) => {
      const next = { ...current };
      delete next[sourceId];
      return next;
    });
  };

  const loadGithubPhase2Debug = async (projectId = phase2BuildForm.projectId) => {
    const evidenceMemoryCopy = githubEvidenceMemoryText(language);
    const requestedProjectId = String(projectId || "").trim();
    setPhase2Loading(true);
    setPhase2Error("");
    try {
      const [statusData, healthData, inspectData] = await Promise.all([
        api.getGitHubContextPhase2Status(),
        api.getGitHubContextPhase2Health(requestedProjectId),
        api.getGitHubContextPhase2Inspect({
          projectId: requestedProjectId,
          limit: 10,
          includeSamples: true,
        }),
      ]);
      setPhase2Status(statusData || null);
      setPhase2Health(healthData || null);
      setPhase2Inspect(inspectData || null);
      return { statusData, healthData, inspectData };
    } catch (phase2LoadError) {
      setPhase2Error(evidenceMemoryCopy.loadError);
      return null;
    } finally {
      setPhase2Loading(false);
    }
  };

  const runGithubPhase2Build = async (payload) => {
    const evidenceMemoryCopy = githubEvidenceMemoryText(language);
    setPhase2Building(true);
    setPhase2BuildError("");
    setPhase2BuildResult(null);
    try {
      const data = await api.buildGitHubContextPhase2(payload);
      setPhase2BuildResult(data);
      await loadGithubPhase2Debug(payload?.projectId || "");
      return data;
    } catch (phase2ErrorResponse) {
      setPhase2BuildError(evidenceMemoryCopy.buildError);
      if (phase2ErrorResponse?.detail && typeof phase2ErrorResponse.detail === "object") {
        setPhase2BuildResult(phase2ErrorResponse.detail);
      }
      await loadGithubPhase2Debug(payload?.projectId || "");
      return null;
    } finally {
      setPhase2Building(false);
    }
  };

  useEffect(() => {
    loadGithubConfig();
    loadGithubContextStatus();
    loadGithubPhase2Debug("");
  }, []);

  const saveGithubConfig = () =>
    run(async () => {
      const data = await api.saveGithubConfig({
        usernames: textToList(githubForm.usernames),
        author_names: textToList(githubForm.author_names),
        author_emails: textToList(githubForm.author_emails),
        token: githubForm.token,
      });
      setGithubForm((current) => ({ ...current, token: "" }));
      setTokenConfigured(data.token_configured);
      setMemoryRepositories(data.memory_repositories || []);
      const status = await api.getStatus();
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(data, status));
      setScan((current) =>
        current
          ? { ...current, identities: data.identities, token_configured: data.token_configured }
          : current
      );
      return data;
    }, copy.saved);

  const scanRepos = () =>
    run(async () => {
      const scopeLabel = projectScope.trim() || "全部可识别项目";
      const data = await runAgentWithProgress({
        title: language === "zh" ? "正在扫描 GitHub 仓库" : "Scanning GitHub repositories",
        initialMessage: `Agent：我会从 ${source} 中扫描 GitHub 链接，项目范围：${scopeLabel}。`,
        stages: [
          { id: "scan", label: `扫描仓库链接：${source}` },
          { id: "apply", label: "整理 owner/repo、身份和 token 状态" },
        ],
        action: async (progress) => {
          const data = await progress.runStage("scan", `正在发送 resume_source=${source}、project_name=${projectScope.trim() || "(空)"}`, () =>
            api.scanGithub(source, {
              project_name: projectScope.trim(),
            }, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          progress.setStageStatus("apply", "running", "正在整理 repos 列表、GitHub identities 和 token_configured");
          progress.assertActive();
          progress.setStageStatus("apply", "done");
          progress.addAgentMessage("GitHub 仓库扫描完成。");
          return data;
        },
      });
      setScan(data);
      setContext(null);
      setTokenConfigured(data.token_configured);
      return data;
    }, copy.scanned);

  const approveFetchContext = () =>
    run(async () => {
      const scopeLabel = projectScope.trim() || "全部已扫描仓库";
      const cacheMode = forceRefresh
        ? "强制刷新远端仓库"
        : reanalyzeCached
          ? "复用缓存并重新分析 Project Memory"
          : "优先复用 ETag/SHA 未变化的缓存";
      const repositoryLines = (scan?.repos || [])
        .map((repo) => `- ${repo.owner}/${repo.repo}`)
        .join("\n");
      const identityLines = [
        ...((scan?.identities?.usernames || []).map((value) => `- GitHub username: ${value}`)),
        ...((scan?.identities?.author_names || []).map((value) => `- Commit author name: ${value}`)),
        ...((scan?.identities?.author_emails || []).map((value) => `- Commit author email: ${value}`)),
      ].join("\n");
      const fetchIntro = [
        `Agent：我会读取 ${scopeLabel} 的 GitHub evidence。`,
        `策略：${cacheMode}。`,
        repositoryLines ? `将处理这些仓库：\n${repositoryLines}` : "",
        `Token 状态：${tokenConfigured ? "已就绪" : "未配置"}`,
        identityLines ? `将用这些身份匹配你的提交：\n${identityLines}` : "",
      ].filter(Boolean).join("\n\n");
      const { data, githubConfig, status } = await runAgentWithProgress({
        title: language === "zh" ? "正在读取 GitHub 上下文" : "Fetching GitHub context",
        initialMessage: fetchIntro,
        stages: [
          { id: "fetch", label: `获取 README/语言/提交/diff 证据：${scopeLabel}` },
          { id: "refresh", label: "读取 GitHub 配置和 Project Memory 更新时间" },
          { id: "apply", label: "整理缓存状态和仓库证据预览" },
        ],
        modelStageIds: ["fetch"],
        action: async (progress) => {
          const data = await progress.runStage("fetch", `正在发送 approved=true、source=${source}、force_refresh=${forceRefresh}、reanalyze_cached=${reanalyzeCached}`, () =>
            api.fetchGithubContext(true, source, {
              project_name: projectScope.trim(),
              force_refresh: forceRefresh,
              reanalyze_cached: reanalyzeCached,
            }, {
              signal: progress.signal,
              agentProgressMessages: progress.getUserMessages(),
              agentTaskId: progress.agentTaskId,
            }),
          );
          const [githubConfig, status] = await progress.runStage("refresh", "正在读取 /github/config 和 /status，用于刷新 evidence DB 与 project_memory 时间", () =>
            Promise.all([api.getGithubConfig(), api.getStatus()]),
          );
          progress.setStageStatus("apply", "running", "正在整理 scan_results、cache_status、context JSON 预览");
          progress.assertActive();
          progress.setStageStatus("apply", "done");
          progress.addAgentMessage("GitHub 上下文已更新。");
          const mapReduceSummary = formatMapReduceSummary(data, language);
          if (mapReduceSummary) {
            progress.addSystemMessage(mapReduceSummary);
          }
          const memoryStatus = getProjectMemoryStatus(data, language);
          if (memoryStatus?.label) {
            progress.addAgentMessage(memoryStatus.detail ? `${memoryStatus.label}。${memoryStatus.detail}` : memoryStatus.label);
          }
          return { data, githubConfig, status };
        },
      });
      const { context: _rawContext, ...safeContext } = data || {};
      setContext(safeContext);
      setMemoryRepositories(githubConfig.memory_repositories || []);
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(githubConfig, status));
      await loadGithubContextStatus();
      await loadGithubPhase2Debug("");
      return data;
    }, copy.fetched);

  const identities = scan?.identities || {
    usernames: textToList(githubForm.usernames),
    author_names: textToList(githubForm.author_names),
    author_emails: textToList(githubForm.author_emails),
  };
  const identityItems = [
    ...(identities.usernames || []).map((value) => `GitHub username: ${value}`),
    ...(identities.author_names || []).map((value) => `Commit author name: ${value}`),
    ...(identities.author_emails || []).map((value) => `Commit author email: ${value}`),
  ];
  const projectScopeLabels = [];
  const projectScopeKeys = new Set();
  projectOptions.forEach((option) => {
    if (option.label && !projectScopeLabels.includes(option.label)) {
      projectScopeLabels.push(option.label);
    }
    (option.keys || []).forEach((key) => projectScopeKeys.add(key));
  });
  [...(scan?.repos || []).map((repo) => `${repo.owner}/${repo.repo}`), ...memoryRepositories.map((repo) => repo.repository)]
    .filter(Boolean)
    .forEach((option) => {
      const key = normalizeProjectAlias(option);
      if (key && !projectScopeKeys.has(key) && !projectScopeLabels.includes(option)) {
        projectScopeKeys.add(key);
        projectScopeLabels.push(option);
      }
    });
  const projectScopeOptions = projectScopeLabels
    .sort((left, right) => left.localeCompare(right));
  const repositoryEvidenceTitle =
    language === "en" ? (copy.memoryRepositories || "Repositories in Chroma Evidence DB") : "Chroma 证据库中的仓库";
  const repositoryEvidenceHint =
    language === "en"
      ? (copy.memoryRepositoriesHint || "These repositories already have local Chroma evidence records. Reading this list does not access GitHub.")
      : "这些仓库已经有本地 Chroma 证据库记录。读取列表不会访问 GitHub 云端。";
  const chromaEvidenceUpdatedAt =
    language === "en" ? (copy.chromaEvidenceUpdatedAt || "Chroma evidence DB updated: ") : "Chroma 证据库更新：";
  const projectMemoryUpdatedAtLabel =
    language === "en" ? (copy.projectMemoryUpdatedAt || "Project Memory JSON updated: ") : "Project Memory JSON 更新：";
  const noRepositoryEvidence =
    language === "en" ? (copy.noMemoryRepositories || "No Chroma repository evidence yet") : "暂无 Chroma 仓库证据";
  const projectMemoryStatus = getProjectMemoryStatus(context, language);
  const diagnosticsCopy = githubDiagnosticsText(language);

  return (
    <>
      <PageHeader title={copy.title} description={copy.description} />
      <LoadingBar loading={loading} />
      <Alert type="error" message={error} />
      <Alert type="success" message={success} />

      <section className="card">
        <h2 className="card-title">{copy.config}</h2>
        <div className="grid-2">
          <div className="field">
            <label>{copy.username}</label>
            <textarea
              className="short"
              value={githubForm.usernames}
              onChange={(event) => setGithubForm((current) => ({ ...current, usernames: event.target.value }))}
              placeholder="e.g. liammaxp"
            />
          </div>
          <div className="field">
            <label>{copy.token}</label>
            <input
              type="password"
              autoComplete="off"
              value={githubForm.token}
              onChange={(event) => setGithubForm((current) => ({ ...current, token: event.target.value }))}
              placeholder={tokenConfigured ? copy.tokenConfigured : copy.pasteToken}
            />
            <div className="status-line">
              {copy.tokenStatus}<StatusBadge ready={tokenConfigured} />
            </div>
          </div>
        </div>
        <div className="grid-2">
          <div className="field">
            <label>{copy.authorName}</label>
            <textarea
              className="short"
              value={githubForm.author_names}
              onChange={(event) => setGithubForm((current) => ({ ...current, author_names: event.target.value }))}
              placeholder="e.g. liammmmax"
            />
          </div>
          <div className="field">
            <label>{copy.authorEmail}</label>
            <textarea
              className="short"
              value={githubForm.author_emails}
              onChange={(event) => setGithubForm((current) => ({ ...current, author_emails: event.target.value }))}
              placeholder="e.g. name@example.com"
            />
          </div>
        </div>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={forceRefresh}
            onChange={(event) => setForceRefresh(event.target.checked)}
            disabled={loading}
          />
          {copy.forceRefresh || "Force refresh from GitHub"}
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            checked={reanalyzeCached}
            onChange={(event) => setReanalyzeCached(event.target.checked)}
            disabled={loading}
          />
          {copy.reanalyzeCached || "Reanalyze cached evidence only"}
        </label>
        <p className="helper-text">
          {copy.cacheHint || "By default, unchanged repositories reuse cached GitHub evidence and skip full README/commit/diff fetching."}
        </p>
        <div className="btn-row">
          <button type="button" className="btn btn-primary" onClick={saveGithubConfig} disabled={loading}>
            {copy.saveConfig}
          </button>
          <span className="helper-text">{copy.configHint}</span>
        </div>
      </section>

      <section className="card">
        <h2 className="card-title">{copy.scanSettings}</h2>
        <div className="field">
          <label>{copy.resumeSource}</label>
          <select value={source} onChange={(e) => setSource(e.target.value)}>
            <option value="tailored_resume_and_resume_and_memory">
              {copy.allResumeSources || "定制简历、基础简历与记忆项目"}
            </option>
            <option value="resume_and_memory">{copy.resumeAndMemory || "简历与记忆中的项目"}</option>
            <option value="resume">{copy.baseResume}</option>
            <option value="tailored_resume">{copy.tailoredResume}</option>
            <option value="memory">{copy.memoryProjects || "仅记忆中的项目"}</option>
          </select>
        </div>
        <div className="field compact-field">
          <label>{copy.projectScopeLabel || "Project scope"}</label>
          <input
            list="github-project-scope-options"
            value={projectScope}
            onChange={(event) => setProjectScope(event.target.value)}
            disabled={loading}
            placeholder={copy.projectScopePlaceholder || "Choose a project, or type a project name / ID"}
          />
          <datalist id="github-project-scope-options">
            {projectScopeOptions.map((option) => (
              <option key={option} value={option} />
            ))}
          </datalist>
          <p className="helper-text">
            {copy.projectScopeHint || "Optional. Suggestions come from Project Memory, scanned repositories, and local evidence; leave blank to scan every repository in the selected sources."}
          </p>
        </div>
        <div className="btn-row">
          <button type="button" className="btn btn-secondary" onClick={scanRepos} disabled={loading || agentActive}>
            {copy.scanRepos}
          </button>
          <button type="button" className="btn btn-primary" onClick={approveFetchContext} disabled={loading || agentActive || !scan?.repos?.length}>
            {copy.confirmFetch}
          </button>
        </div>
      </section>

      <GitHubContextStatusPanel
        status={contextStatus}
        loading={contextStatusLoading}
        error={contextStatusError}
        onRefresh={loadGithubContextStatus}
        onPreview={loadGithubContextPreview}
        previews={contextPreviews}
        rawInspections={rawInspections}
        onInspectRaw={inspectGithubContextRaw}
        onCloseRaw={closeGithubContextRaw}
        language={language}
      />

      <GitHubPhase2Panel
        status={phase2Status}
        health={phase2Health}
        inspect={phase2Inspect}
        loading={phase2Loading}
        error={phase2Error}
        buildForm={phase2BuildForm}
        setBuildForm={setPhase2BuildForm}
        building={phase2Building}
        buildError={phase2BuildError}
        buildResult={phase2BuildResult}
        onRefresh={() => loadGithubPhase2Debug(phase2BuildForm.projectId)}
        onBuild={runGithubPhase2Build}
        language={language}
      />

      <section className="card">
        <h2 className="card-title">{repositoryEvidenceTitle}</h2>
        <p className="helper-text">{repositoryEvidenceHint}</p>
        <p className="status-line">
          {projectMemoryUpdatedAtLabel}{formatUnixUpdatedAt(projectMemoryUpdatedAt, language)}
        </p>
        {memoryRepositories.length ? (
          <div className="repo-list">
            {memoryRepositories.map((repo) => (
              <div key={repo.repository} className="repo-item">
                <span>{repo.repository}</span>
                <span className="status-line">
                  {chromaEvidenceUpdatedAt}{formatMemoryUpdatedAt(repo.updated_at, language)}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="empty-state">{noRepositoryEvidence}</p>
        )}
      </section>

      {scan && (
        <section className="card">
          <h2 className="card-title">{copy.scanResult}</h2>
          <p className="status-line">
            {copy.tokenStatus}<StatusBadge ready={scan.token_configured} />
          </p>
          {identityItems.length > 0 && (
            <ul className="output-list" style={{ marginBottom: 16 }}>
              {identityItems.map((item) => <li key={item}>{item}</li>)}
            </ul>
          )}
          {scan.repos?.length ? (
            <div className="repo-list">
              {scan.repos.map((repo) => (
                <div key={repo.url} className="repo-item">
                  <span>{repo.owner}/{repo.repo}</span>
                  <a href={repo.url} target="_blank" rel="noreferrer">{copy.open}</a>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty-state">{copy.noRepos}</p>
          )}
        </section>
      )}

      {context && (
        <section className="card">
          <h2 className="card-title">{copy.contextSummary}</h2>
          {projectMemoryStatus && (
            <div className={`github-memory-status github-memory-status-${projectMemoryStatus.status}`}>
              <div className="github-memory-status-title">{projectMemoryStatus.label}</div>
              {projectMemoryStatus.detail && (
                <div className="github-memory-status-detail">{projectMemoryStatus.detail}</div>
              )}
            </div>
          )}
          {context.scan_results?.length ? (
            <div className="repo-list" style={{ marginBottom: 16 }}>
              {context.scan_results.map((result) => (
                <div key={result.repository} className="repo-item">
                  <span>{result.repository}</span>
                  <span className="status-line">
                    {formatGithubEvidenceStatus(result, language)}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
          <p className="helper-text">
            {diagnosticsCopy.rawNotRendered}
          </p>
        </section>
      )}

    </>
  );
}
