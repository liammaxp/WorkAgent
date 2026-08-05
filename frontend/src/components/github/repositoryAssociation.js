export const MAX_REPOSITORY_INPUT_CHARS = 500;

const REPOSITORY_PATTERN = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

export function isCanonicalRepositoryInput(value) {
  const repository = String(value || "").trim();
  if (!repository || repository.length > MAX_REPOSITORY_INPUT_CHARS) return false;
  if (!REPOSITORY_PATTERN.test(repository)) return false;
  const [owner, name] = repository.split("/");
  return ![owner, name].some((part) => part === "." || part === "..");
}

export function repositoryItemKey(item, index = 0) {
  if (item?.canonical && item?.repository) return `repository:${item.repository}`;
  return `alias:${String(item?.repository_alias || "unknown").toLowerCase()}:${index}`;
}

export function repositoryDisplayName(item) {
  return item?.repository || item?.repository_alias || "";
}

export function buildConfirmationPayload({ projectId, repository, repositoryAlias = "" }) {
  const payload = {
    project_id: String(projectId || "").trim(),
    repository: String(repository || "").trim(),
    confirmed: true,
  };
  const alias = String(repositoryAlias || "").trim();
  if (alias) payload.aliases = [alias];
  return payload;
}

export function associationExists({ unresolved = [], projects = [], repository, projectId }) {
  const canonical = String(repository || "").trim().toLowerCase();
  const stillUnresolved = unresolved.some(
    (item) => String(item?.repository || "").toLowerCase() === canonical,
  );
  if (stillUnresolved) return false;
  return projects.some(
    (project) => project?.project_id === projectId
      && (project?.already_linked_repositories || []).some(
        (linked) => String(linked).toLowerCase() === canonical,
      ),
  );
}
