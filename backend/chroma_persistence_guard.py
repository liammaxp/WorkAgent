"""Authoritative runtime and static guard for Chroma persistence ownership."""

from __future__ import annotations

import ast
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

try:
    from backend.chroma_config import (
        ChromaDeploymentConfig,
        ChromaDeploymentMode,
        load_chroma_deployment_config,
    )
    from backend.chroma_persistence_guard_models import (
        ChromaEmbeddedProductionAccessDenied,
        ChromaMaintenanceAccessNotApproved,
        ChromaPersistenceContext,
        ChromaPersistenceDecision,
        ChromaPersistenceOwnershipAmbiguous,
        ChromaPersistenceRole,
        ChromaPersistenceStatus,
        ChromaPersistentClientStaticGuardError,
        ChromaProtectedPathAccessDenied,
        InvalidChromaPersistenceContext,
        PersistentClientAccessRecord,
        PersistentClientGuardReport,
    )
    from backend.chroma_server_lifecycle_models import (
        ChromaServerLifecycleConfig,
        ChromaServerLifecycleResult,
        build_chroma_server_lifecycle_config,
    )
except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch
    from chroma_config import (
        ChromaDeploymentConfig,
        ChromaDeploymentMode,
        load_chroma_deployment_config,
    )
    from chroma_persistence_guard_models import (
        ChromaEmbeddedProductionAccessDenied,
        ChromaMaintenanceAccessNotApproved,
        ChromaPersistenceContext,
        ChromaPersistenceDecision,
        ChromaPersistenceOwnershipAmbiguous,
        ChromaPersistenceRole,
        ChromaPersistenceStatus,
        ChromaPersistentClientStaticGuardError,
        ChromaProtectedPathAccessDenied,
        InvalidChromaPersistenceContext,
        PersistentClientAccessRecord,
        PersistentClientGuardReport,
    )
    from chroma_server_lifecycle_models import (
        ChromaServerLifecycleConfig,
        ChromaServerLifecycleResult,
        build_chroma_server_lifecycle_config,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_PERSISTENT_TARGET = (
    "tests/chroma_persistence_test_support.py",
    "create_test_owned_persistent_client",
)
_EXCLUDED_SCAN_DIRECTORIES = frozenset(
    {
        ".git",
        ".agents",
        ".codex",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "information",
        "logs",
        "outputs",
        "dist",
        "build",
        "venv",
        ".venv",
    }
)
_FALLBACK_CALL_MARKERS = (
    "persistentclient",
    "persistent_client",
    "embedded_client",
    "embedded_chroma",
    "ensure_client",
)


def _default_lifecycle_observer(
    config: ChromaServerLifecycleConfig,
) -> ChromaServerLifecycleResult:
    try:
        from backend.chroma_server_lifecycle import inspect_chroma_server_ownership
    except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch
        from chroma_server_lifecycle import inspect_chroma_server_ownership
    return inspect_chroma_server_ownership(config)


def _scan_repository(repository_root: str | Path) -> dict[str, Any]:
    try:
        from backend.chroma_access_inventory import scan_repository
    except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch
        from chroma_access_inventory import scan_repository
    return scan_repository(repository_root)


def _resolve_path(path: str | Path) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise InvalidChromaPersistenceContext("invalid_chroma_persistence_path")
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        raise InvalidChromaPersistenceContext(
            "invalid_chroma_persistence_path"
        ) from None


def _normalized_path(path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _path_is_within(candidate: Path, root: Path) -> bool:
    candidate_value = _normalized_path(candidate)
    root_value = _normalized_path(root)
    try:
        return os.path.commonpath((candidate_value, root_value)) == root_value
    except ValueError:
        return False


def _path_is_root(path: Path) -> bool:
    try:
        anchor = Path(path.anchor).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return True
    return _normalized_path(path) == _normalized_path(anchor)


def _safe_server_state(result: Any) -> str:
    if not isinstance(result, ChromaServerLifecycleResult):
        return "ambiguous"
    if result.state == "ready" and (
        not result.process_owned or not result.server_reachable
    ):
        return "ambiguous"
    if result.state in {"foreign_port_conflict", "ownership_mismatch"}:
        return "ambiguous"
    return result.state


class ChromaPersistenceGuard:
    """Decide direct persistence access without constructing a Chroma client."""

    __slots__ = ("_config", "_lifecycle_observer", "_server_owned_path")

    def __init__(
        self,
        config: ChromaServerLifecycleConfig,
        *,
        test_lifecycle_observer: Callable[
            [ChromaServerLifecycleConfig], ChromaServerLifecycleResult
        ]
        | None = None,
    ) -> None:
        if not isinstance(config, ChromaServerLifecycleConfig):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_lifecycle_config"
            )
        if test_lifecycle_observer is not None and not config.test_owned:
            raise InvalidChromaPersistenceContext(
                "lifecycle_observer_injection_requires_test_ownership"
            )
        if test_lifecycle_observer is not None and not callable(
            test_lifecycle_observer
        ):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_lifecycle_observer"
            )
        self._config = config
        self._server_owned_path = config.persistence_path.resolve(strict=False)
        self._lifecycle_observer = (
            test_lifecycle_observer or _default_lifecycle_observer
        )

    @property
    def lifecycle_config(self) -> ChromaServerLifecycleConfig:
        return self._config

    def is_server_owned_path(self, path: str | Path) -> bool:
        """Return whether a path resolves into the lifecycle-owned persistence tree."""

        return _path_is_within(_resolve_path(path), self._server_owned_path)

    def _observe_server(self) -> str:
        try:
            result = self._lifecycle_observer(self._config)
        except Exception:
            return "ambiguous"
        return _safe_server_state(result)

    def _test_path_owned(
        self, candidate: Path, context: ChromaPersistenceContext
    ) -> bool:
        root = context.test_storage_root
        if root is None or _path_is_root(candidate):
            return False
        resolved_root = root.resolve(strict=True)
        return (
            candidate != resolved_root
            and _path_is_within(candidate, resolved_root)
            and (
                self._config.test_owned
                or not _path_is_within(candidate, self._server_owned_path)
            )
        )

    def _decision(
        self,
        context: ChromaPersistenceContext,
        *,
        allowed: bool,
        scope: str,
        server_state: str,
        disposition: str,
        reason: str,
    ) -> ChromaPersistenceDecision:
        return ChromaPersistenceDecision(
            allowed=allowed,
            role=context.role.value,
            deployment_mode=context.deployment.mode.value,
            persistence_scope=scope,
            server_ownership_state=server_state,
            disposition=disposition,
            reason=reason,
        )

    def evaluate_embedded_access(
        self,
        *,
        path: str | Path,
        context: ChromaPersistenceContext,
    ) -> ChromaPersistenceDecision:
        if not isinstance(context, ChromaPersistenceContext):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_context"
            )
        candidate = _resolve_path(path)
        server_owned = _path_is_within(candidate, self._server_owned_path)

        if context.role is ChromaPersistenceRole.UNKNOWN:
            raise InvalidChromaPersistenceContext(
                "unknown_chroma_persistence_role"
            )
        if context.role is ChromaPersistenceRole.SERVER_OWNER:
            return self._decision(
                context,
                allowed=False,
                scope="server_owned" if server_owned else "unrelated",
                server_state="not_observed",
                disposition="blocked",
                reason="server_owner_role_cannot_authorize_embedded_access",
            )
        if context.role in {
            ChromaPersistenceRole.MAINTENANCE,
            ChromaPersistenceRole.MIGRATION,
        }:
            return self._decision(
                context,
                allowed=False,
                scope="server_owned" if server_owned else "unrelated",
                server_state="not_observed",
                disposition="maintenance_not_approved",
                reason="offline_embedded_access_not_approved",
            )

        if server_owned:
            if not self._config.test_owned:
                mode_reason = {
                    ChromaDeploymentMode.DISABLED: "disabled_mode_has_no_embedded_fallback",
                    ChromaDeploymentMode.LOCAL_HTTP: "local_persistence_is_server_owned",
                    ChromaDeploymentMode.REMOTE_HTTP: "remote_mode_forbids_local_embedded_access",
                    ChromaDeploymentMode.EPHEMERAL_TEST: "protected_persistence_forbidden_in_test_mode",
                }[context.deployment.mode]
                return self._decision(
                    context,
                    allowed=False,
                    scope="server_owned",
                    server_state="not_observed",
                    disposition="blocked",
                    reason=mode_reason,
                )

            server_state = self._observe_server()
            if (
                context.role is ChromaPersistenceRole.TEST_ONLY
                and server_state == "not_running"
                and self._test_path_owned(candidate, context)
            ):
                return self._decision(
                    context,
                    allowed=True,
                    scope="test_owned",
                    server_state=server_state,
                    disposition="approved_test_only",
                    reason="test_owned_server_stopped",
                )
            return self._decision(
                context,
                allowed=False,
                scope="server_owned",
                server_state=server_state,
                disposition="blocked",
                reason=(
                    "test_server_ownership_ambiguous"
                    if server_state in {
                        "ambiguous",
                        "stale_state",
                        "foreign_port_conflict",
                        "ownership_mismatch",
                    }
                    else "server_owned_persistence_cannot_be_opened_embedded"
                ),
            )

        if (
            context.role is ChromaPersistenceRole.TEST_ONLY
            and self._test_path_owned(candidate, context)
        ):
            return self._decision(
                context,
                allowed=True,
                scope="test_owned",
                server_state="not_applicable",
                disposition="approved_test_only",
                reason="isolated_test_owned_persistence",
            )

        return self._decision(
            context,
            allowed=False,
            scope="unsafe" if _path_is_root(candidate) else "unrelated",
            server_state="not_applicable",
            disposition="blocked",
            reason="embedded_access_requires_explicit_test_ownership",
        )

    def assert_embedded_access_allowed(
        self,
        *,
        path: str | Path,
        context: ChromaPersistenceContext,
    ) -> ChromaPersistenceDecision:
        decision = self.evaluate_embedded_access(path=path, context=context)
        if decision.allowed:
            return decision
        if decision.disposition == "maintenance_not_approved":
            raise ChromaMaintenanceAccessNotApproved(decision.reason)
        if decision.server_ownership_state in {
            "ambiguous",
            "stale_state",
            "foreign_port_conflict",
            "ownership_mismatch",
        }:
            raise ChromaPersistenceOwnershipAmbiguous(decision.reason)
        if decision.persistence_scope == "server_owned":
            raise ChromaProtectedPathAccessDenied(decision.reason)
        raise ChromaEmbeddedProductionAccessDenied(decision.reason)

    def verify_dedicated_server_owner(self) -> ChromaPersistenceDecision:
        deployment = self._config.deployment
        if deployment.mode is not ChromaDeploymentMode.LOCAL_HTTP:
            raise ChromaPersistenceOwnershipAmbiguous(
                "dedicated_server_ownership_requires_local_http"
            )
        state = self._observe_server()
        context = ChromaPersistenceContext(
            role=ChromaPersistenceRole.SERVER_OWNER,
            deployment=deployment,
        )
        if state != "ready":
            raise ChromaPersistenceOwnershipAmbiguous(
                "dedicated_server_ownership_not_verified"
            )
        return self._decision(
            context,
            allowed=True,
            scope="server_owned",
            server_state=state,
            disposition="verified_server_owner",
            reason="lifecycle_verified_server_owner",
        )

    def inspect_status(self, *, legacy_embedded_targets: int = 0) -> ChromaPersistenceStatus:
        if (
            not isinstance(legacy_embedded_targets, int)
            or isinstance(legacy_embedded_targets, bool)
            or legacy_embedded_targets < 0
        ):
            raise InvalidChromaPersistenceContext(
                "invalid_legacy_embedded_target_count"
            )
        return ChromaPersistenceStatus(
            production_persistence_role="server_owned",
            embedded_production_access="blocked",
            server_ownership_state=self._observe_server(),
            legacy_embedded_targets=legacy_embedded_targets,
        )


def build_production_chroma_persistence_guard(
    deployment: ChromaDeploymentConfig | None = None,
) -> ChromaPersistenceGuard:
    resolved_deployment = deployment or load_chroma_deployment_config()
    return ChromaPersistenceGuard(
        build_chroma_server_lifecycle_config(resolved_deployment)
    )


def assert_embedded_chroma_access_allowed(
    *,
    path: str | Path,
    context: ChromaPersistenceContext,
    lifecycle_config: ChromaServerLifecycleConfig | None = None,
) -> ChromaPersistenceDecision:
    guard = (
        ChromaPersistenceGuard(lifecycle_config)
        if lifecycle_config is not None
        else build_production_chroma_persistence_guard(context.deployment)
    )
    return guard.assert_embedded_access_allowed(path=path, context=context)


def _iter_python_sources(root: Path) -> Iterable[tuple[Path, str]]:
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = [
            name for name in sorted(directory_names) if name not in _EXCLUDED_SCAN_DIRECTORIES
        ]
        for name in sorted(file_names):
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                relative = path.resolve(strict=True).relative_to(root.resolve(strict=True))
            except (OSError, ValueError):
                raise ChromaPersistentClientStaticGuardError(
                    "persistent_client_source_path_escape"
                ) from None
            yield path, relative.as_posix()


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _scope_symbols(tree: ast.AST) -> dict[int, str]:
    symbols: dict[int, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _scope(self, node: ast.AST, name: str) -> None:
            self.stack.append(name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self._scope(node, node.name)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._scope(node, node.name)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._scope(node, node.name)

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            symbols[id(node)] = ".".join(self.stack) or "<module>"
            self.generic_visit(node)

    Visitor().visit(tree)
    return symbols


def _fallback_candidates(root: Path) -> tuple[dict[str, str | int], ...]:
    candidates: list[dict[str, str | int]] = []
    for path, module in _iter_python_sources(root):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=module)
        except (OSError, UnicodeError, SyntaxError):
            raise ChromaPersistentClientStaticGuardError(
                "persistent_client_source_unreadable"
            ) from None
        symbols = _scope_symbols(tree)
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            for call in (node for node in ast.walk(handler) if isinstance(node, ast.Call)):
                name = _call_name(call.func).casefold()
                if name.endswith(("error", "violation", "denied")):
                    continue
                if not any(marker in name for marker in _FALLBACK_CALL_MARKERS):
                    continue
                candidates.append(
                    {
                        "module": module,
                        "symbol": symbols.get(id(call), "<module>"),
                        "line": int(call.lineno),
                        "reason": "embedded_call_inside_exception_handler",
                    }
                )
    return tuple(
        sorted(candidates, key=lambda item: (str(item["module"]), int(item["line"])))
    )


def _source_line(root: Path, module: str, line: int) -> str:
    try:
        values = (root / Path(module)).read_text(encoding="utf-8").splitlines()
        return values[line - 1] if 0 < line <= len(values) else ""
    except (OSError, UnicodeError):
        return ""


def inspect_persistent_client_access(
    repository_root: str | Path = PROJECT_ROOT,
) -> PersistentClientGuardReport:
    root = Path(repository_root)
    scan = _scan_repository(root)
    records: list[PersistentClientAccessRecord] = []
    for item in scan["discoveries"]:
        if (
            item["client_type"] != "persistent_embedded"
            or item["operation"] != "client construction"
        ):
            continue
        identity = (item["module"], item["symbol"])
        disposition = (
            "approved_test_only" if identity == _TEST_PERSISTENT_TARGET else "forbidden"
        )
        records.append(
            PersistentClientAccessRecord(
                module=item["module"],
                symbol=item["symbol"],
                line=item["line"],
                constructor_form=item["semantic_role"],
                disposition=disposition,
            )
        )
    for item in scan["review_candidates"]:
        line_text = _source_line(root, item["module"], item["line"])
        if "PersistentClient" not in line_text:
            continue
        records.append(
            PersistentClientAccessRecord(
                module=item["module"],
                symbol=item["symbol"],
                line=item["line"],
                constructor_form=item["reason"],
                disposition="unknown",
            )
        )
    ordered = tuple(
        sorted(records, key=lambda item: (item.module, item.symbol, item.line, item.disposition))
    )
    return PersistentClientGuardReport(
        records=ordered,
        fallback_candidates=_fallback_candidates(root),
    )


def verify_persistent_client_access(
    repository_root: str | Path = PROJECT_ROOT,
) -> PersistentClientGuardReport:
    report = inspect_persistent_client_access(repository_root)
    summary = report.safe_summary()
    if summary["production_legacy_persistent_client_count"] != 0:
        raise ChromaPersistentClientStaticGuardError(
            "production_persistent_client_present"
        )
    if summary["test_only_persistent_client_count"] != 1:
        raise ChromaPersistentClientStaticGuardError(
            "test_persistent_client_target_drift"
        )
    if summary["approved_maintenance_persistent_client_count"] != 0:
        raise ChromaPersistentClientStaticGuardError(
            "unexpected_maintenance_persistent_client_access"
        )
    if summary["forbidden_persistent_client_count"]:
        raise ChromaPersistentClientStaticGuardError(
            "forbidden_persistent_client_access"
        )
    if summary["unknown_persistent_client_count"]:
        raise ChromaPersistentClientStaticGuardError(
            "unknown_persistent_client_access"
        )
    if summary["embedded_fallback_candidate_count"]:
        raise ChromaPersistentClientStaticGuardError(
            "embedded_http_fallback_detected"
        )
    return report
