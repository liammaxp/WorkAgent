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
      statusDisabled: "GitHub context status v2 is disabled. Enable USE_GITHUB_CONTEXT_STATUS_V2=1 to view saved/indexed context diagnostics.",
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
      previewDisabled: "GitHub context preview v2 is disabled. Enable USE_GITHUB_CONTEXT_STATUS_V2=1 to view bounded previews.",
      previewEmpty: "No bounded preview items are available for this project.",
      inspectRaw: "Inspect Raw",
      loadingRaw: "Loading Raw",
      rawLoading: "Loading bounded raw content...",
      rawError: "Unable to inspect raw GitHub context.",
      rawDisabled: "GitHub raw inspect v2 is disabled. Enable USE_GITHUB_CONTEXT_STATUS_V2=1 to inspect bounded raw context.",
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
    statusDisabled: "GitHub 上下文诊断功能未启用。如需查看保存/索引状态，请在启动后端前设置：USE_GITHUB_CONTEXT_STATUS_V2=1",
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
    previewDisabled: "GitHub 上下文预览功能未启用。请设置 USE_GITHUB_CONTEXT_STATUS_V2=1 后查看有界预览。",
    previewEmpty: "该项目暂无可预览内容。",
    inspectRaw: "查看原始内容",
    loadingRaw: "正在加载原文",
    rawLoading: "正在加载有界原始内容...",
    rawError: "无法加载原始内容。",
    rawDisabled: "GitHub 原始内容查看功能未启用。请设置 USE_GITHUB_CONTEXT_STATUS_V2=1 后查看有界原文。",
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

function phase2DisplayValue(value) {
  if (Array.isArray(value)) {
    if (!value.length) return "-";
    return value.map((item) => phase2DisplayValue(item)).join(", ");
  }
  if (value && typeof value === "object") {
    return truncateDisplayValue(JSON.stringify(sanitizePhase2SampleForDisplay(value)), 400);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
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

function Phase2ProjectSummaries({ projects }) {
  if (!projects.length) {
    return <p className="empty-state phase2-empty-state">No Phase 2 project summaries yet.</p>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Project ID</th>
            <th>Repo</th>
            <th>Raw sources</th>
            <th>Chunks</th>
            <th>Change summaries</th>
            <th>Evidence cards</th>
            <th>Capability facts</th>
            <th>Raw chars</th>
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

function Phase2BuildResult({ result, error }) {
  if (error) {
    return <div className="github-status-message github-status-error">{error}</div>;
  }
  if (!result) return null;
  return (
    <div className="phase2-build-result">
      <h3 className="github-status-subtitle">Build result</h3>
      <p className="helper-text">{result.message || "Phase 2 build completed."}</p>
      <Phase2KeyValueGrid
        items={[
          { label: "Ran stages", value: safeArray(result.ran_stages).join(", ") || "-" },
          { label: "Errors", value: formatCount(safeArray(result.errors).length) },
          { label: "Warnings", value: formatCount(safeArray(result.warnings).length) },
        ]}
      />
      <div className="phase2-count-columns">
        {[
          ["Counts before", result.counts_before],
          ["Counts after", result.counts_after],
          ["Deltas", result.deltas],
        ].map(([title, counts]) => (
          <div className="phase2-count-box" key={title}>
            <strong>{title}</strong>
            {Object.entries(counts || {}).map(([key, value]) => (
              <div key={key}>
                <span>{key}</span>
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
                <th>Stage</th>
                <th>OK</th>
                <th>Processed</th>
                <th>Created/updated</th>
                <th>Skipped</th>
                <th>Message</th>
              </tr>
            </thead>
            <tbody>
              {safeArray(result.stage_results).map((stage) => (
                <tr key={stage.stage}>
                  <td>{stage.stage || "-"}</td>
                  <td>{String(Boolean(stage.ok))}</td>
                  <td>{formatCount(stage.processed)}</td>
                  <td>{formatCount(stage.created_or_updated)}</td>
                  <td>{formatCount(stage.skipped)}</td>
                  <td>{stage.message || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {safeArray(result.errors).length > 0 && (
        <pre className="phase2-json-preview">{JSON.stringify(sanitizePhase2SampleForDisplay(result.errors), null, 2)}</pre>
      )}
      {safeArray(result.warnings).length > 0 && (
        <pre className="phase2-json-preview">{JSON.stringify(sanitizePhase2SampleForDisplay(result.warnings), null, 2)}</pre>
      )}
    </div>
  );
}

function Phase2InspectSamples({ samples }) {
  const sampleEntries = Object.entries(PHASE2_SAMPLE_LABELS);
  return (
    <div className="phase2-samples">
      {sampleEntries.map(([sampleKey, label]) => {
        const items = safeArray(samples?.[sampleKey]).map((item) => sanitizePhase2SampleForDisplay(item));
        return (
          <div className="phase2-sample-group" key={sampleKey}>
            <h4>{label}</h4>
            {!items.length ? (
              <p className="helper-text">No safe samples available.</p>
            ) : (
              items.map((item, index) => (
                <div className="phase2-sample-item" key={`${sampleKey}-${index}`}>
                  {Object.entries(item).map(([key, value]) => (
                    <div className="phase2-sample-row" key={key}>
                      <span>{key}</span>
                      <strong>{phase2DisplayValue(value)}</strong>
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
}) {
  const statusLoaded = Boolean(status);
  const enabled = status?.enabled === true;
  const counts = status || {};
  const healthFlags = health?.health || {};
  const projects = safeArray(status?.projects).length ? safeArray(status?.projects) : safeArray(inspect?.projects);
  const projectOptions = phase2ProjectOptions(status, inspect);
  const noRawSources = enabled && Number(counts.raw_sources_count || 0) === 0;
  const pipelineComplete = Boolean(status?.pipeline_complete || health?.pipeline_complete);
  const statusMessage = !statusLoaded
    ? "Loading Phase 2 evidence memory diagnostics..."
    : !enabled
    ? "GitHub context Phase 2 is disabled. Enable USE_GITHUB_CONTEXT_PHASE2=1 to view and build Phase 2 evidence memory."
    : noRawSources
      ? "Phase 2 is enabled, but no raw sources are available yet. Sync GitHub context first, then run Phase 2 build."
      : pipelineComplete
        ? "Phase 2 pipeline complete: raw sources, chunks, change summaries, evidence cards, and capability facts are available."
        : status?.message || "Phase 2 evidence memory is available for inspection.";

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
        <h2 className="card-title">Phase 2 Evidence Memory</h2>
        <button type="button" className="btn btn-secondary btn-small" onClick={onRefresh} disabled={loading || building}>
          Refresh
        </button>
      </div>

      {loading && <p className="helper-text">Loading Phase 2 status, health, and inspect samples...</p>}
      {error && <div className="github-status-message github-status-error">{error}</div>}
      <div className="github-status-message">{statusMessage}</div>

      <h3 className="github-status-subtitle">Status</h3>
      <Phase2KeyValueGrid
        items={[
          { label: "Enabled", value: String(Boolean(status?.enabled)) },
          { label: "Available", value: String(Boolean(status?.available)) },
          { label: "Phase", value: status?.phase || "phase2" },
          { label: "Raw sources", value: formatCount(counts.raw_sources_count) },
          { label: "Chunks", value: formatCount(counts.chunks_count) },
          { label: "Change summaries", value: formatCount(counts.raw_change_summaries_count) },
          { label: "Evidence cards", value: formatCount(counts.evidence_cards_count) },
          { label: "Capability facts", value: formatCount(counts.capability_facts_count) },
          { label: "Raw chars", value: formatCount(counts.raw_chars) },
          { label: "Repos", value: formatCount(counts.repos_count) },
          { label: "Pipeline complete", value: String(pipelineComplete) },
          { label: "Next action", value: status?.next_recommended_action || health?.next_recommended_action || "-" },
        ]}
      />

      <h3 className="github-status-subtitle">Health</h3>
      <Phase2KeyValueGrid
        items={[
          { label: "Has raw sources", value: String(Boolean(healthFlags.has_raw_sources)) },
          { label: "Has chunks", value: String(Boolean(healthFlags.has_chunks)) },
          { label: "Has change summaries", value: String(Boolean(healthFlags.has_raw_change_summaries)) },
          { label: "Has evidence cards", value: String(Boolean(healthFlags.has_evidence_cards)) },
          { label: "Has capability facts", value: String(Boolean(healthFlags.has_capability_facts)) },
          { label: "Missing stages", value: safeArray(health?.missing_stages).join(", ") || "-" },
          { label: "Recommended", value: health?.next_recommended_action || "-" },
        ]}
      />

      <h3 className="github-status-subtitle">Project Summaries</h3>
      <Phase2ProjectSummaries projects={projects} />

      <h3 className="github-status-subtitle">Manual Build</h3>
      <div className="phase2-build-controls">
        <div className="field compact-field">
          <label>Project ID</label>
          <input
            list="phase2-project-options"
            value={buildForm.projectId}
            onChange={(event) => setBuildForm((current) => ({ ...current, projectId: event.target.value }))}
            placeholder="Leave empty for all projects"
            disabled={!statusLoaded || !enabled || building}
          />
          <datalist id="phase2-project-options">
            {projectOptions.map((projectId) => (
              <option key={projectId} value={projectId} />
            ))}
          </datalist>
        </div>
        <div className="field compact-field">
          <label>Stage</label>
          <select
            value={buildForm.stage}
            onChange={(event) => setBuildForm((current) => ({ ...current, stage: event.target.value }))}
            disabled={!statusLoaded || !enabled || building}
          >
            {PHASE2_STAGE_OPTIONS.map((stage) => (
              <option key={stage.value} value={stage.value}>{stage.label}</option>
            ))}
          </select>
        </div>
        <div className="field compact-field">
          <label>Limit</label>
          <input
            type="number"
            min="0"
            value={buildForm.limit}
            onChange={(event) => setBuildForm((current) => ({ ...current, limit: event.target.value }))}
            placeholder="Optional"
            disabled={!statusLoaded || !enabled || building}
          />
        </div>
      </div>
      {noRawSources && (
        <p className="warning-line">No raw sources are saved yet. Sync GitHub context first; this button will not trigger sync.</p>
      )}
      <div className="btn-row">
        <button
          type="button"
          className="btn btn-primary"
          onClick={handleBuildClick}
          disabled={!statusLoaded || !enabled || building}
        >
          {building ? "Running Phase 2 Build" : "Run Phase 2 Build"}
        </button>
        {statusLoaded && !enabled && <span className="helper-text">Enable USE_GITHUB_CONTEXT_PHASE2=1 before building.</span>}
      </div>
      <Phase2BuildResult result={buildResult} error={buildError} />

      <h3 className="github-status-subtitle">Safe Inspect Samples</h3>
      {inspect?.enabled === false ? (
        <div className="github-status-message">Phase 2 inspect is disabled.</div>
      ) : (
        <Phase2InspectSamples samples={inspect?.samples || {}} />
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
      setPhase2Error(phase2LoadError?.message || "Unable to load Phase 2 evidence memory diagnostics.");
      return null;
    } finally {
      setPhase2Loading(false);
    }
  };

  const runGithubPhase2Build = async (payload) => {
    setPhase2Building(true);
    setPhase2BuildError("");
    setPhase2BuildResult(null);
    try {
      const data = await api.buildGitHubContextPhase2(payload);
      setPhase2BuildResult(data);
      await loadGithubPhase2Debug(payload?.projectId || "");
      return data;
    } catch (phase2ErrorResponse) {
      setPhase2BuildError(phase2ErrorResponse?.message || "Unable to run Phase 2 build.");
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
