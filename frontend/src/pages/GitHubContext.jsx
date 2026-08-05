import { useEffect, useState } from "react";
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
import RepositoryAssociationSection from "../components/github/RepositoryAssociationSection.jsx";
import EvidencePreparationSection from "../components/github/EvidencePreparationSection.jsx";

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

const EVIDENCE_PIPELINE_STAGE_OPTIONS = [
  { value: "all" },
  { value: "chunk" },
  { value: "summarize_changes" },
  { value: "build_evidence_cards" },
  { value: "build_capability_facts" },
];

function formatCount(value) {
  return Number(value || 0).toLocaleString();
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function evidencePipelineText(language) {
  if (language === "en") {
    return {
      title: "Evidence Processing Pipeline",
      yes: "Yes",
      no: "No",
      hint: "Process saved local GitHub raw sources into chunks, change summaries, evidence cards, and capability facts. This does not sync GitHub again.",
      projectId: "Project ID",
      projectPlaceholder: "Leave empty for all saved sources",
      stage: "Stage",
      limit: "Limit",
      optional: "Optional",
      run: "Run Pipeline",
      running: "Running Pipeline",
      buildError: "Unable to run evidence pipeline.",
      resultTitle: "Pipeline Result",
      completed: "Pipeline completed.",
      completedWithIssues: "Pipeline finished with issues.",
      disabled: "Evidence processing is disabled. Enable USE_GITHUB_EVIDENCE_MEMORY=1 before running.",
      ranStages: "Ran stages",
      warnings: "Warnings",
      errors: "Errors",
      countsAfter: "Current counts",
      stageResults: "Stage results",
      ok: "OK",
      processed: "Processed",
      created: "Created",
      updated: "Updated",
      unchanged: "Unchanged",
      skipped: "Skipped",
      message: "Message",
      summaryOnly: "Only pipeline summaries are shown here; raw text, patches, and inspect samples are not rendered.",
      rawNotRendered: "Saved GitHub raw context is summarized here; full raw content is not rendered on this page.",
      stageLabels: {
        all: "All stages",
        chunk: "Chunk raw sources",
        summarize_changes: "Summarize changes",
        build_evidence_cards: "Build evidence cards",
        build_capability_facts: "Build capability facts",
      },
      fieldLabels: {
        raw_sources_count: "Raw sources",
        chunks_count: "Chunks",
        raw_change_summaries_count: "Change summaries",
        evidence_cards_count: "Evidence cards",
        capability_facts_count: "Capability facts",
        raw_chars: "Raw chars",
        repos_count: "Repos",
      },
    };
  }
  return {
    title: "证据处理流水线",
    yes: "是",
    no: "否",
    hint: "把本地已保存的 GitHub 原始来源处理成分块、变更摘要、证据卡片和能力事实；此操作不会重新同步 GitHub。",
    projectId: "项目 ID",
    projectPlaceholder: "留空表示处理全部已保存来源",
    stage: "处理阶段",
    limit: "数量上限",
    optional: "可选",
    run: "运行流水线",
    running: "正在运行",
    buildError: "无法运行证据处理流水线。",
    resultTitle: "流水线结果",
    completed: "流水线已完成。",
    completedWithIssues: "流水线完成，但存在问题。",
    disabled: "证据处理未启用。运行前请设置 USE_GITHUB_EVIDENCE_MEMORY=1。",
    ranStages: "已运行阶段",
    warnings: "警告",
    errors: "错误",
    countsAfter: "当前数量",
    stageResults: "阶段结果",
    ok: "成功",
      processed: "已处理",
      created: "新增",
      updated: "实际更新",
      unchanged: "未变化",
    skipped: "已跳过",
    message: "消息",
    summaryOnly: "这里只显示流水线摘要；不会渲染原文、patch 或 inspect 样例。",
    rawNotRendered: "已保存的 GitHub 原始上下文只在这里显示摘要；页面不会渲染完整原文。",
    stageLabels: {
      all: "全部阶段",
      chunk: "生成原始来源分块",
      summarize_changes: "生成变更摘要",
      build_evidence_cards: "生成证据卡片",
      build_capability_facts: "生成能力事实",
    },
    fieldLabels: {
      raw_sources_count: "原始来源",
      chunks_count: "分块",
      raw_change_summaries_count: "变更摘要",
      evidence_cards_count: "证据卡片",
      capability_facts_count: "能力事实",
      raw_chars: "原始字符数",
      repos_count: "仓库数",
    },
  };
}

function evidencePipelineStageLabel(stage, ui) {
  return ui.stageLabels?.[stage] || stage || "-";
}

function evidencePipelineFieldLabel(field, ui) {
  return ui.fieldLabels?.[field] || field;
}

function formatPipelineIssue(issue) {
  if (!issue) return "";
  if (typeof issue === "string") return issue;
  if (issue.message) return String(issue.message);
  if (issue.type) return String(issue.type);
  return JSON.stringify(issue);
}

function EvidencePipelineBuildResult({ result, error, ui }) {
  if (error && !result) {
    return <div className="evidence-pipeline-message evidence-pipeline-error">{error}</div>;
  }
  if (!result) return null;

  const ranStages = safeArray(result.ran_stages);
  const stageResults = safeArray(result.stage_results);
  const warnings = safeArray(result.warnings);
  const errors = safeArray(result.errors);
  const countsAfter = result.counts_after || {};
  const statusText = result.enabled === false
    ? ui.disabled
    : result.ok === false
      ? ui.completedWithIssues
      : ui.completed;

  return (
    <div className="evidence-pipeline-result">
      <h3>{ui.resultTitle}</h3>
      <div className={`evidence-pipeline-message ${result.ok === false || error ? "evidence-pipeline-error" : ""}`}>
        {error || statusText}
      </div>
      <div className="evidence-pipeline-meta">
        <span>{ui.ranStages}</span>
        <strong>{ranStages.map((stage) => evidencePipelineStageLabel(stage, ui)).join(", ") || "-"}</strong>
      </div>
      {Object.keys(countsAfter).length > 0 && (
        <div className="evidence-pipeline-counts">
          {Object.entries(countsAfter).map(([key, value]) => (
            <div key={key}>
              <span>{evidencePipelineFieldLabel(key, ui)}</span>
              <strong>{formatCount(value)}</strong>
            </div>
          ))}
        </div>
      )}
      {warnings.length > 0 && (
        <div className="evidence-pipeline-issues">
          <strong>{ui.warnings}</strong>
          <ul>
            {warnings.map((warning, index) => (
              <li key={`warning-${index}`}>{formatPipelineIssue(warning)}</li>
            ))}
          </ul>
        </div>
      )}
      {errors.length > 0 && (
        <div className="evidence-pipeline-issues evidence-pipeline-error-list">
          <strong>{ui.errors}</strong>
          <ul>
            {errors.map((issue, index) => (
              <li key={`error-${index}`}>{formatPipelineIssue(issue)}</li>
            ))}
          </ul>
        </div>
      )}
      {stageResults.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{ui.stage}</th>
                <th>{ui.ok}</th>
                <th>{ui.processed}</th>
                <th>{ui.created}</th>
                <th>{ui.updated}</th>
                <th>{ui.unchanged}</th>
                <th>{ui.skipped}</th>
                <th>{ui.message}</th>
              </tr>
            </thead>
            <tbody>
              {stageResults.map((stage) => (
                <tr key={stage.stage}>
                  <td>{evidencePipelineStageLabel(stage.stage, ui)}</td>
                  <td>{stage.ok ? ui.yes : ui.no}</td>
                  <td>{formatCount(stage.processed)}</td>
                  <td>{formatCount(stage.created)}</td>
                  <td>{formatCount(stage.updated)}</td>
                  <td>{formatCount(stage.unchanged)}</td>
                  <td>{formatCount(stage.skipped)}</td>
                  <td>{safeArray(stage.errors).join(", ") || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="helper-text">{ui.summaryOnly}</p>
    </div>
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
  const showEvidenceDebug = import.meta.env.DEV;
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
  const [evidencePipelineForm, setEvidencePipelineForm] = useState({
    projectId: "",
    stage: "all",
    limit: "",
  });
  const [evidencePipelineBuilding, setEvidencePipelineBuilding] = useState(false);
  const [evidencePipelineError, setEvidencePipelineError] = useState("");
  const [evidencePipelineResult, setEvidencePipelineResult] = useState(null);
  const [repositoryAssociationRevision, setRepositoryAssociationRevision] = useState(0);
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

  useEffect(() => {
    loadGithubConfig();
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
      return data;
    }, copy.fetched);

  const runEvidencePipeline = async () => {
    const pipelineCopy = evidencePipelineText(language);
    const selectedStages = evidencePipelineForm.stage === "all"
      ? null
      : [evidencePipelineForm.stage];
    setEvidencePipelineBuilding(true);
    setEvidencePipelineError("");
    setEvidencePipelineResult(null);
    try {
      const data = await api.buildGitHubContextEvidencePipeline({
        projectId: evidencePipelineForm.projectId.trim(),
        stages: selectedStages,
        limit: evidencePipelineForm.limit,
        continueOnError: true,
      });
      setEvidencePipelineResult(data);
      const [githubConfig, status] = await Promise.all([api.getGithubConfig(), api.getStatus()]);
      setMemoryRepositories(githubConfig.memory_repositories || []);
      setProjectMemoryUpdatedAt(resolveProjectMemoryUpdatedAt(githubConfig, status));
      return data;
    } catch (pipelineError) {
      const detail = pipelineError?.detail && typeof pipelineError.detail === "object"
        ? pipelineError.detail
        : null;
      if (detail) {
        setEvidencePipelineResult(detail);
      }
      setEvidencePipelineError(pipelineError?.message || pipelineCopy.buildError);
      return null;
    } finally {
      setEvidencePipelineBuilding(false);
    }
  };

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
  const pipelineCopy = evidencePipelineText(language);
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

      <RepositoryAssociationSection
        onAssociationChanged={() => setRepositoryAssociationRevision((revision) => revision + 1)}
      />
      <EvidencePreparationSection refreshSignal={repositoryAssociationRevision} />

      {showEvidenceDebug && <section className="card evidence-pipeline-panel">
        <h2 className="card-title">{pipelineCopy.title}</h2>
        <p className="helper-text">{pipelineCopy.hint}</p>
        <div className="evidence-pipeline-controls">
          <div className="field compact-field">
            <label>{pipelineCopy.projectId}</label>
            <input
              list="github-evidence-pipeline-projects"
              value={evidencePipelineForm.projectId}
              onChange={(event) => setEvidencePipelineForm((current) => ({ ...current, projectId: event.target.value }))}
              placeholder={pipelineCopy.projectPlaceholder}
              disabled={evidencePipelineBuilding}
            />
            <datalist id="github-evidence-pipeline-projects">
              {projectScopeOptions.map((option) => (
                <option key={option} value={option} />
              ))}
            </datalist>
          </div>
          <div className="field compact-field">
            <label>{pipelineCopy.stage}</label>
            <select
              value={evidencePipelineForm.stage}
              onChange={(event) => setEvidencePipelineForm((current) => ({ ...current, stage: event.target.value }))}
              disabled={evidencePipelineBuilding}
            >
              {EVIDENCE_PIPELINE_STAGE_OPTIONS.map((stage) => (
                <option key={stage.value} value={stage.value}>
                  {evidencePipelineStageLabel(stage.value, pipelineCopy)}
                </option>
              ))}
            </select>
          </div>
          <div className="field compact-field">
            <label>{pipelineCopy.limit}</label>
            <input
              type="number"
              min="0"
              value={evidencePipelineForm.limit}
              onChange={(event) => setEvidencePipelineForm((current) => ({ ...current, limit: event.target.value }))}
              placeholder={pipelineCopy.optional}
              disabled={evidencePipelineBuilding}
            />
          </div>
        </div>
        <div className="btn-row">
          <button
            type="button"
            className="btn btn-primary"
            onClick={runEvidencePipeline}
            disabled={loading || agentActive || evidencePipelineBuilding}
          >
            {evidencePipelineBuilding ? pipelineCopy.running : pipelineCopy.run}
          </button>
        </div>
        <EvidencePipelineBuildResult
          result={evidencePipelineResult}
          error={evidencePipelineError}
          ui={pipelineCopy}
        />
      </section>}

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
            {pipelineCopy.rawNotRendered}
          </p>
        </section>
      )}

    </>
  );
}
