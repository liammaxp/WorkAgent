"""Strict privacy-safe models for Chroma persistence ownership decisions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch
    from chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode


PERSISTENCE_GUARD_SCHEMA = "chroma_persistence_guard.v1"
PERSISTENT_CLIENT_GUARD_SCHEMA = "chroma_persistent_client_guard.v1"
MAX_GUARD_RECORDS = 500


class ChromaPersistenceGuardError(RuntimeError):
    """Stable guard failure that never includes paths, commands, or environment."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class InvalidChromaPersistenceContext(ChromaPersistenceGuardError):
    pass


class ChromaPersistenceOwnershipViolation(ChromaPersistenceGuardError):
    pass


class ChromaProtectedPathAccessDenied(ChromaPersistenceOwnershipViolation):
    pass


class ChromaEmbeddedProductionAccessDenied(ChromaPersistenceOwnershipViolation):
    pass


class ChromaPersistenceOwnershipAmbiguous(ChromaPersistenceOwnershipViolation):
    pass


class ChromaMaintenanceAccessNotApproved(ChromaPersistenceOwnershipViolation):
    pass


class ChromaPersistentClientStaticGuardError(ChromaPersistenceGuardError):
    pass


class ChromaPersistenceRole(str, Enum):
    SERVER_OWNER = "server_owner"
    PRODUCTION_CLIENT = "production_client"
    LEGACY_EMBEDDED = "legacy_embedded"
    MAINTENANCE = "maintenance"
    MIGRATION = "migration"
    TEST_ONLY = "test_only"
    UNKNOWN = "unknown"


def _resolved_directory(path: Any, code: str) -> Path:
    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise InvalidChromaPersistenceContext(code)
    try:
        candidate = Path(path).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise InvalidChromaPersistenceContext(code) from None
    if not candidate.is_dir():
        raise InvalidChromaPersistenceContext(code)
    try:
        anchor = Path(candidate.anchor).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise InvalidChromaPersistenceContext(code) from None
    if os.path.normcase(str(candidate)) == os.path.normcase(str(anchor)):
        raise InvalidChromaPersistenceContext(code)
    return candidate


@dataclass(frozen=True, slots=True, repr=False)
class ChromaPersistenceContext:
    role: ChromaPersistenceRole
    deployment: ChromaDeploymentConfig
    explicit_test_context: bool = False
    test_storage_root: Path | None = None
    operator_invoked: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.role, ChromaPersistenceRole):
            raise InvalidChromaPersistenceContext("invalid_chroma_persistence_role")
        if not isinstance(self.deployment, ChromaDeploymentConfig):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_deployment"
            )
        if not isinstance(self.explicit_test_context, bool) or not isinstance(
            self.operator_invoked, bool
        ):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_context_boolean"
            )
        if self.role is ChromaPersistenceRole.TEST_ONLY:
            if (
                not self.explicit_test_context
                or self.deployment.mode is not ChromaDeploymentMode.EPHEMERAL_TEST
                or self.test_storage_root is None
            ):
                raise InvalidChromaPersistenceContext(
                    "invalid_test_owned_chroma_persistence_context"
                )
            object.__setattr__(
                self,
                "test_storage_root",
                _resolved_directory(
                    self.test_storage_root,
                    "invalid_test_owned_chroma_storage_root",
                ),
            )
        elif self.explicit_test_context or self.test_storage_root is not None:
            raise InvalidChromaPersistenceContext(
                "test_chroma_context_requires_test_only_role"
            )
        if self.operator_invoked and self.role not in {
            ChromaPersistenceRole.MAINTENANCE,
            ChromaPersistenceRole.MIGRATION,
        }:
            raise InvalidChromaPersistenceContext(
                "operator_invocation_requires_offline_role"
            )

    @classmethod
    def legacy_embedded(
        cls, deployment: ChromaDeploymentConfig
    ) -> ChromaPersistenceContext:
        return cls(role=ChromaPersistenceRole.LEGACY_EMBEDDED, deployment=deployment)

    @classmethod
    def test_owned(
        cls,
        deployment: ChromaDeploymentConfig,
        *,
        storage_root: str | Path,
    ) -> ChromaPersistenceContext:
        return cls(
            role=ChromaPersistenceRole.TEST_ONLY,
            deployment=deployment,
            explicit_test_context=True,
            test_storage_root=Path(storage_root),
        )

    def safe_summary(self) -> dict[str, str | bool]:
        return {
            "role": self.role.value,
            "deployment_mode": self.deployment.mode.value,
            "explicit_test_context": self.explicit_test_context,
            "operator_invoked": self.operator_invoked,
            "storage_scope": "test_owned" if self.test_storage_root else "none",
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaPersistenceContext("
            f"role={summary['role']!r}, "
            f"deployment_mode={summary['deployment_mode']!r}, "
            f"explicit_test_context={summary['explicit_test_context']!r}, "
            f"storage_scope={summary['storage_scope']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaPersistenceDecision:
    allowed: bool
    role: str
    deployment_mode: str
    persistence_scope: str
    server_ownership_state: str
    disposition: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_decision"
            )
        if self.role not in {item.value for item in ChromaPersistenceRole}:
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_decision_role"
            )
        if self.deployment_mode not in {item.value for item in ChromaDeploymentMode}:
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_decision_mode"
            )
        if self.persistence_scope not in {
            "server_owned",
            "test_owned",
            "unrelated",
            "unsafe",
        }:
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_persistence_decision_scope"
            )
        for value, code in (
            (self.server_ownership_state, "invalid_chroma_server_ownership_state"),
            (self.disposition, "invalid_chroma_persistence_disposition"),
            (self.reason, "invalid_chroma_persistence_reason"),
        ):
            if not isinstance(value, str) or not value or len(value) > 96:
                raise InvalidChromaPersistenceContext(code)

    def safe_summary(self) -> dict[str, str | bool]:
        return {
            "allowed": self.allowed,
            "role": self.role,
            "deployment_mode": self.deployment_mode,
            "persistence_scope": self.persistence_scope,
            "server_ownership_state": self.server_ownership_state,
            "disposition": self.disposition,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaPersistenceDecision("
            f"allowed={summary['allowed']!r}, "
            f"role={summary['role']!r}, "
            f"persistence_scope={summary['persistence_scope']!r}, "
            f"disposition={summary['disposition']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ChromaPersistenceStatus:
    production_persistence_role: str
    embedded_production_access: str
    server_ownership_state: str
    legacy_embedded_targets: int

    def __post_init__(self) -> None:
        if self.production_persistence_role != "server_owned":
            raise InvalidChromaPersistenceContext(
                "invalid_production_chroma_persistence_role"
            )
        if self.embedded_production_access != "blocked":
            raise InvalidChromaPersistenceContext(
                "invalid_embedded_production_access_state"
            )
        if (
            not isinstance(self.server_ownership_state, str)
            or not self.server_ownership_state
            or len(self.server_ownership_state) > 96
        ):
            raise InvalidChromaPersistenceContext(
                "invalid_chroma_server_ownership_state"
            )
        if (
            not isinstance(self.legacy_embedded_targets, int)
            or isinstance(self.legacy_embedded_targets, bool)
            or self.legacy_embedded_targets < 0
        ):
            raise InvalidChromaPersistenceContext(
                "invalid_legacy_embedded_target_count"
            )

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "production_persistence_role": self.production_persistence_role,
            "embedded_production_access": self.embedded_production_access,
            "server_ownership_state": self.server_ownership_state,
            "legacy_embedded_targets": self.legacy_embedded_targets,
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "ChromaPersistenceStatus("
            f"production_persistence_role={summary['production_persistence_role']!r}, "
            f"embedded_production_access={summary['embedded_production_access']!r}, "
            f"server_ownership_state={summary['server_ownership_state']!r}, "
            f"legacy_embedded_targets={summary['legacy_embedded_targets']!r})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class PersistentClientAccessRecord:
    module: str
    symbol: str
    line: int
    constructor_form: str
    disposition: str

    def __post_init__(self) -> None:
        for value, code in (
            (self.module, "invalid_persistent_client_module"),
            (self.symbol, "invalid_persistent_client_symbol"),
            (self.constructor_form, "invalid_persistent_client_constructor_form"),
            (self.disposition, "invalid_persistent_client_disposition"),
        ):
            if not isinstance(value, str) or not value or len(value) > 240:
                raise ChromaPersistentClientStaticGuardError(code)
        if Path(self.module).is_absolute() or ".." in Path(self.module).parts:
            raise ChromaPersistentClientStaticGuardError(
                "unsafe_persistent_client_module"
            )
        if not isinstance(self.line, int) or isinstance(self.line, bool) or self.line <= 0:
            raise ChromaPersistentClientStaticGuardError(
                "invalid_persistent_client_line"
            )
        if self.disposition not in {
            "legacy_migration_target",
            "approved_test_only",
            "approved_maintenance",
            "forbidden",
            "unknown",
        }:
            raise ChromaPersistentClientStaticGuardError(
                "unknown_persistent_client_disposition"
            )

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "module": self.module,
            "symbol": self.symbol,
            "line": self.line,
            "constructor_form": self.constructor_form,
            "disposition": self.disposition,
        }


@dataclass(frozen=True, slots=True, repr=False)
class PersistentClientGuardReport:
    records: tuple[PersistentClientAccessRecord, ...]
    fallback_candidates: tuple[dict[str, str | int], ...] = ()

    def __post_init__(self) -> None:
        if len(self.records) > MAX_GUARD_RECORDS:
            raise ChromaPersistentClientStaticGuardError(
                "persistent_client_record_limit_exceeded"
            )
        ordered = tuple(
            sorted(
                self.records,
                key=lambda item: (item.module, item.symbol, item.line, item.disposition),
            )
        )
        if self.records != ordered:
            raise ChromaPersistentClientStaticGuardError(
                "non_deterministic_persistent_client_order"
            )
        for candidate in self.fallback_candidates:
            if frozenset(candidate) != {"module", "symbol", "line", "reason"}:
                raise ChromaPersistentClientStaticGuardError(
                    "invalid_embedded_fallback_candidate"
                )

    def count(self, disposition: str) -> int:
        return sum(item.disposition == disposition for item in self.records)

    def safe_summary(self) -> dict[str, str | int]:
        return {
            "schema": PERSISTENT_CLIENT_GUARD_SCHEMA,
            "production_legacy_persistent_client_count": self.count(
                "legacy_migration_target"
            ),
            "test_only_persistent_client_count": self.count("approved_test_only"),
            "approved_maintenance_persistent_client_count": self.count(
                "approved_maintenance"
            ),
            "forbidden_persistent_client_count": self.count("forbidden"),
            "unknown_persistent_client_count": self.count("unknown"),
            "embedded_fallback_candidate_count": len(self.fallback_candidates),
        }

    def __repr__(self) -> str:
        summary = self.safe_summary()
        return (
            "PersistentClientGuardReport("
            f"production_legacy={summary['production_legacy_persistent_client_count']!r}, "
            f"test_only={summary['test_only_persistent_client_count']!r}, "
            f"forbidden={summary['forbidden_persistent_client_count']!r}, "
            f"unknown={summary['unknown_persistent_client_count']!r})"
        )
