const APP_OPENED_AT = Date.now();

export function fileChangedSinceAppOpened(status, name) {
  const mtime = status?.file_metadata?.[name]?.mtime_ms;
  return typeof mtime === "number" && mtime >= APP_OPENED_AT;
}

export function readStoredBoolean(key, fallback) {
  const value = localStorage.getItem(key);
  if (value === "true") return true;
  if (value === "false") return false;
  return fallback;
}

export function writeStoredBoolean(key, value) {
  localStorage.setItem(key, value ? "true" : "false");
}
