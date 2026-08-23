"""Static provenance guard for production Chroma collection-name literals."""

from __future__ import annotations

import ast
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.chroma_collection_registry import (
    COMPATIBILITY_COLLECTION_ALIASES,
    REGISTERED_COLLECTIONS,
    resolve_collection_name,
)


MAX_LITERAL_SOURCE_FILES = 20_000
MAX_LITERAL_SOURCE_BYTES = 2_000_000
AUTHORITATIVE_REGISTRY_MODULE = "backend/chroma_collection_registry.py"
_COLLECTION_CALL_NAMES = frozenset(
    {
        "create_collection",
        "delete_collection",
        "get_collection",
        "get_or_create_collection",
    }
)
_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".agents",
        ".codex",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "frontend",
        "information",
        "logs",
        "node_modules",
        "outputs",
        "venv",
    }
)


class ChromaCollectionLiteralGuardError(ValueError):
    """Stable static-guard failure without source bodies or absolute paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class UnregisteredProductionCollectionLiteral(ChromaCollectionLiteralGuardError):
    pass


@dataclass(frozen=True, slots=True)
class CollectionLiteralCandidate:
    module: str
    line: int
    symbol: str
    literal: str
    source_kind: str
    classification: str
    allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "line": self.line,
            "symbol": self.symbol,
            "literal": self.literal,
            "source_kind": self.source_kind,
            "classification": self.classification,
            "allowed": self.allowed,
        }


def _assignment_symbols(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            symbol
            for child in target.elts
            for symbol in _assignment_symbols(child)
        )
    return ()


def _looks_like_collection_symbol(symbol: str) -> bool:
    folded = symbol.casefold()
    return (
        symbol == "COLLECTION"
        or folded == "collection_name"
        or folded.endswith("_collection")
        or folded.endswith("_collection_name")
    )


def _string_literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


class _CollectionLiteralVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.candidates: list[tuple[int, str, str, str]] = []

    def _assignment(self, targets: Iterable[ast.AST], value: ast.AST) -> None:
        literal = _string_literal(value)
        if literal is None:
            return
        for target in targets:
            for symbol in _assignment_symbols(target):
                if _looks_like_collection_symbol(symbol):
                    self.candidates.append(
                        (int(getattr(value, "lineno", 0)), symbol, literal, "assignment")
                    )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self._assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is not None:
            self._assignment((node.target,), node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
        else:
            call_name = ""
        if call_name in _COLLECTION_CALL_NAMES:
            literal = _string_literal(node.args[0]) if node.args else None
            if literal is None:
                for keyword in node.keywords:
                    if keyword.arg in {"name", "collection_name"}:
                        literal = _string_literal(keyword.value)
                        if literal is not None:
                            break
            if literal is not None:
                self.candidates.append(
                    (int(node.lineno), call_name, literal, "direct_collection_call")
                )
        self.generic_visit(node)


def _iter_python_sources(repository_root: Path) -> Iterable[tuple[Path, str]]:
    root = repository_root.resolve(strict=True)
    source_count = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name for name in directory_names if name not in _EXCLUDED_DIRECTORIES
        )
        for file_name in sorted(file_names):
            if not file_name.endswith(".py"):
                continue
            source_count += 1
            if source_count > MAX_LITERAL_SOURCE_FILES:
                raise ChromaCollectionLiteralGuardError("collection_literal_source_limit_exceeded")
            path = Path(directory) / file_name
            try:
                relative = path.resolve(strict=True).relative_to(root).as_posix()
                size = path.stat().st_size
            except (OSError, ValueError) as error:
                raise ChromaCollectionLiteralGuardError(
                    "collection_literal_source_inspection_failed"
                ) from error
            if size > MAX_LITERAL_SOURCE_BYTES:
                raise ChromaCollectionLiteralGuardError("collection_literal_source_too_large")
            yield path, relative


def _compatibility_alias(module: str, symbol: str):
    matches = [
        alias
        for alias in COMPATIBILITY_COLLECTION_ALIASES
        if alias.module == module and alias.symbol == symbol
    ]
    return matches[0] if len(matches) == 1 else None


def _classify_candidate(
    *,
    module: str,
    line: int,
    symbol: str,
    literal: str,
    source_kind: str,
    allow_test_fixtures: bool,
) -> CollectionLiteralCandidate:
    registered_names = {definition.collection_name for definition in REGISTERED_COLLECTIONS}
    is_test = module == "tests" or module.startswith("tests/")
    alias = _compatibility_alias(module, symbol)
    if module == AUTHORITATIVE_REGISTRY_MODULE and literal in registered_names:
        classification, allowed = "authoritative_registry", True
    elif alias is not None and literal == resolve_collection_name(alias.semantic_id):
        classification, allowed = "compatibility_alias", True
    elif alias is not None:
        classification, allowed = "invalid_compatibility_alias", False
    elif is_test and allow_test_fixtures:
        classification, allowed = "sanitized_test_fixture", True
    elif literal in registered_names:
        classification, allowed = "duplicate_production_literal", False
    else:
        classification, allowed = "unregistered_production_literal", False
    return CollectionLiteralCandidate(
        module=module,
        line=line,
        symbol=symbol,
        literal=literal,
        source_kind=source_kind,
        classification=classification,
        allowed=allowed,
    )


def audit_collection_name_literals(
    repository_root: str | Path,
    *,
    allow_test_fixtures: bool = True,
) -> dict[str, Any]:
    """Inspect Python syntax only; scanned modules are never imported or executed."""

    root = Path(repository_root)
    candidates: list[CollectionLiteralCandidate] = []
    try:
        sources = list(_iter_python_sources(root))
    except (OSError, RuntimeError) as error:
        raise ChromaCollectionLiteralGuardError("collection_literal_repository_unavailable") from error
    for path, module in sources:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=module)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise ChromaCollectionLiteralGuardError("collection_literal_source_parse_failed") from error
        visitor = _CollectionLiteralVisitor()
        visitor.visit(tree)
        for line, symbol, literal, source_kind in visitor.candidates:
            candidates.append(
                _classify_candidate(
                    module=module,
                    line=line,
                    symbol=symbol,
                    literal=literal,
                    source_kind=source_kind,
                    allow_test_fixtures=allow_test_fixtures,
                )
            )
    candidates.sort(key=lambda item: (item.module, item.line, item.symbol, item.literal))
    classifications = Counter(item.classification for item in candidates)
    violations = [item for item in candidates if not item.allowed]
    return {
        "candidate_count": len(candidates),
        "allowed_count": len(candidates) - len(violations),
        "violation_count": len(violations),
        "unknown_count": sum(
            count
            for classification, count in classifications.items()
            if classification
            in {
                "invalid_compatibility_alias",
                "unregistered_production_literal",
            }
        ),
        "classification_counts": dict(sorted(classifications.items())),
        "candidates": [item.to_dict() for item in candidates],
        "violations": [item.to_dict() for item in violations],
        "validation_state": "valid" if not violations else "invalid",
    }


def validate_collection_name_literals(
    repository_root: str | Path,
    *,
    allow_test_fixtures: bool = True,
) -> dict[str, Any]:
    report = audit_collection_name_literals(
        repository_root,
        allow_test_fixtures=allow_test_fixtures,
    )
    if report["violation_count"]:
        raise UnregisteredProductionCollectionLiteral(
            "unregistered_production_collection_literal"
        )
    return report
