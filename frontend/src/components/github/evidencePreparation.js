export const PREPARABLE_STATUSES = new Set(["ready_to_prepare", "partial"]);

const COPY = {
  en: {
    disabled: "Project evidence preparation is currently unavailable.",
    mapping_required: "Connect all detected GitHub repositories to projects before preparing evidence.",
    ready_to_prepare: "GitHub project evidence is ready to prepare.",
    prepared: "GitHub project evidence is up to date.",
    partial: "Some GitHub project evidence could not be prepared.",
    blocked: "GitHub project evidence cannot be prepared right now.",
    error: "Project evidence status could not be loaded.",
    created: "GitHub project evidence was prepared.",
    updated: "GitHub project evidence was refreshed.",
    unchanged: "GitHub project evidence is already up to date.",
    empty: "No saved GitHub evidence is available to prepare.",
    busy: "Evidence preparation is already in progress.",
    degraded: "Project evidence was prepared, but its status could not be refreshed.",
    unknown: "GitHub project evidence could not be prepared.",
  },
  zh: {
    disabled: "项目证据准备功能当前不可用。",
    mapping_required: "请先将检测到的 GitHub 仓库全部关联到项目。",
    ready_to_prepare: "GitHub 项目证据已可准备。",
    prepared: "GitHub 项目证据已是最新状态。",
    partial: "部分 GitHub 项目证据未能完成准备。",
    blocked: "当前无法准备 GitHub 项目证据。",
    error: "暂时无法加载项目证据状态。",
    created: "GitHub 项目证据已准备完成。",
    updated: "GitHub 项目证据已刷新。",
    unchanged: "GitHub 项目证据已经是最新状态。",
    empty: "没有可供准备的已保存 GitHub 信息。",
    busy: "项目证据正在准备中。",
    degraded: "项目证据已准备，但暂时无法刷新状态。",
    unknown: "GitHub 项目证据未能完成准备。",
  },
};

export function statusMessage(status, language = "en") {
  const copy = COPY[language] || COPY.en;
  return copy[status] || copy.error;
}

export function outcomeMessage(status, language = "en") {
  const copy = COPY[language] || COPY.en;
  return copy[status] || copy.unknown;
}

export function canStartPreparation(status) {
  return Boolean(status?.can_prepare && PREPARABLE_STATUSES.has(status?.status));
}

export function shouldReconcileAfterOutcome(status) {
  return !["busy", "disabled"].includes(status);
}

export function safeRemainingRepositoryText(status, language = "en") {
  const count = Number(status?.remaining_repository_count);
  if (status?.status !== "mapping_required" || !Number.isInteger(count) || count <= 0) return "";
  return language === "zh"
    ? `仍有 ${count} 个仓库需要关联。`
    : `${count} ${count === 1 ? "repository still needs" : "repositories still need"} to be connected.`;
}
