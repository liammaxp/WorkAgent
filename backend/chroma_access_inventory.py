"""Deterministically discover and verify reviewed repository Chroma access paths."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from backend.chroma_access_models import (
    CHROMA_ACCESS_INVENTORY_SCHEMA,
    ChromaAccessValidationError,
    inventory_digest,
    production_access_policy_violations,
    stable_access_id,
    validate_chroma_access_inventory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "backend" / "chroma_access_manifest.py"
MAX_SOURCE_FILES = 20_000
MAX_SOURCE_BYTES = 2_000_000

_EXCLUDED_DIRECTORIES = frozenset(
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
_DOCUMENT_EXTENSIONS = frozenset({".md", ".rst"})
_COMMAND_EXTENSIONS = frozenset({".ps1", ".sh", ".bat", ".cmd"})
_REFERENCE_EXTENSIONS = frozenset({".js", ".jsx", ".ts", ".tsx", ".toml", ".yaml", ".yml"})
_COLLECTION_OPERATIONS = frozenset(
    {
        "count",
        "get",
        "peek",
        "query",
        "add",
        "upsert",
        "update",
        "delete",
    }
)
_CLIENT_OPERATIONS = {
    "request": "http request",
    "heartbeat": "heartbeat",
    "list_collections": "list collections",
    "get_collection": "get collection",
    "get_or_create_collection": "get or create collection",
    "create_collection": "create collection",
    "delete_collection": "delete collection",
    "reset": "reset",
}
_COLLECTION_RETURNING_OPERATIONS = frozenset(
    {"get_collection", "get_or_create_collection", "create_collection"}
)
_EXECUTABLE_CHROMA_RE = re.compile(
    r"(?i)(?:\bchroma\s+run\b|\b(?:PersistentClient|HttpClient)\s*\(|"
    r"\b(?:get_collection|get_or_create_collection|create_collection|delete_collection)\s*\()"
)


class ChromaAccessInventoryError(RuntimeError):
    """A bounded inventory failure that does not include source or absolute paths."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _error(code: str) -> None:
    raise ChromaAccessInventoryError(code)


def _relative_module(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        _error("source_path_escape")
    if relative.startswith("../") or relative.startswith("/"):
        _error("source_path_escape")
    return relative


def _iter_source_files(root: Path) -> Iterable[tuple[Path, str]]:
    count = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _EXCLUDED_DIRECTORIES
            and not (Path(directory) / name).is_symlink()
        )
        for name in sorted(file_names):
            suffix = Path(name).suffix.casefold()
            if suffix != ".py" and suffix not in (
                _DOCUMENT_EXTENSIONS | _COMMAND_EXTENSIONS | _REFERENCE_EXTENSIONS
            ):
                continue
            candidate = Path(directory) / name
            if candidate.is_symlink():
                continue
            count += 1
            if count > MAX_SOURCE_FILES:
                _error("source_file_limit_exceeded")
            yield candidate, _relative_module(candidate, root)


def _read_source(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            _error("source_file_size_limit_exceeded")
        return path.read_text(encoding="utf-8")
    except ChromaAccessInventoryError:
        raise
    except (OSError, UnicodeError):
        _error("source_file_unreadable")


def _expr_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = _expr_name(node)
        return [name] if name else []
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for child in node.elts for name in _target_names(child)]
    return []


def _string_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


class _ModuleProvenance:
    def __init__(self, tree: ast.Module, module: str):
        self.module = module
        self.module_aliases: set[str] = set()
        self.httpx_module_aliases: set[str] = set()
        self.requests_module_aliases: set[str] = set()
        self.urllib_request_aliases: set[str] = set()
        self.direct_http_function_aliases: set[str] = set()
        self.constructor_aliases: dict[str, str] = {}
        self.constructor_roles: dict[str, str] = {}
        self.constants = _string_constants(tree)
        self.function_client_returns: dict[str, str] = {}
        self.function_collection_returns: dict[str, tuple[str, str, str]] = {}
        self.chroma_http_context = self._is_chroma_http_context(tree)
        self._collect_imports(tree)
        self._collect_constructor_aliases(tree)
        self.module_client_types = self._module_client_types(tree)
        self.default_client_type = (
            next(iter(self.module_client_types)) if len(self.module_client_types) == 1 else "unknown"
        )
        self._collect_function_returns(tree)

    def _is_chroma_http_context(self, tree: ast.Module) -> bool:
        if self.module.startswith(("backend/chroma_", "tests/chroma_")):
            return True
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "backend.chroma_config" or node.module.startswith(
                    "backend.chroma_"
                ):
                    return True
        return False

    def _collect_imports(self, tree: ast.Module) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "chromadb":
                        self.module_aliases.add(alias.asname or "chromadb")
                    elif alias.name == "httpx":
                        self.httpx_module_aliases.add(alias.asname or "httpx")
                    elif alias.name == "requests":
                        self.requests_module_aliases.add(alias.asname or "requests")
                    elif alias.name == "urllib.request":
                        self.urllib_request_aliases.add(alias.asname or "urllib.request")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "chromadb" or node.module.startswith("chromadb."):
                    for alias in node.names:
                        if alias.name in {"PersistentClient", "HttpClient"}:
                            local_name = alias.asname or alias.name
                            self.constructor_aliases[local_name] = (
                                "persistent_embedded"
                                if alias.name == "PersistentClient"
                                else "http"
                            )
                            self.constructor_roles[local_name] = (
                                "default_persistent_client"
                                if alias.name == "PersistentClient"
                                else "direct_chromadb_http_client"
                            )
                elif node.module == "httpx" and self.chroma_http_context:
                    for alias in node.names:
                        local_name = alias.asname or alias.name
                        if alias.name in {"Client", "AsyncClient"}:
                            self.constructor_aliases[local_name] = "http"
                            self.constructor_roles[local_name] = "low_level_http_client"
                        elif alias.name in {"request", "get", "post", "put", "patch", "delete"}:
                            self.direct_http_function_aliases.add(local_name)
                elif node.module == "requests" and self.chroma_http_context:
                    for alias in node.names:
                        local_name = alias.asname or alias.name
                        if alias.name == "Session":
                            self.constructor_aliases[local_name] = "http"
                            self.constructor_roles[local_name] = "independent_http_session"
                        elif alias.name in {"request", "get", "post", "put", "patch", "delete"}:
                            self.direct_http_function_aliases.add(local_name)
                elif node.module == "urllib" and self.chroma_http_context:
                    for alias in node.names:
                        if alias.name == "request":
                            self.urllib_request_aliases.add(alias.asname or alias.name)
                elif node.module == "urllib.request" and self.chroma_http_context:
                    for alias in node.names:
                        local_name = alias.asname or alias.name
                        if alias.name in {"urlopen", "Request"}:
                            self.direct_http_function_aliases.add(local_name)

    def resolve_constructor(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.constructor_aliases.get(node.id)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in self.module_aliases:
                if node.attr == "PersistentClient":
                    return "persistent_embedded"
                if node.attr == "HttpClient":
                    return "http"
            if (
                self.chroma_http_context
                and node.value.id in self.httpx_module_aliases
                and node.attr in {"Client", "AsyncClient"}
            ):
                return "http"
            if (
                self.chroma_http_context
                and node.value.id in self.requests_module_aliases
                and node.attr == "Session"
            ):
                return "http"
        return None

    def resolve_constructor_role(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.constructor_roles.get(node.id)
        name = _expr_name(node)
        if isinstance(node, ast.Attribute):
            receiver = _expr_name(node.value)
            if receiver in self.module_aliases:
                if node.attr == "PersistentClient":
                    return "default_persistent_client"
                if node.attr == "HttpClient":
                    return "direct_chromadb_http_client"
            if self.chroma_http_context and receiver in self.httpx_module_aliases:
                if node.attr in {"Client", "AsyncClient"}:
                    return "low_level_http_client"
            if self.chroma_http_context and receiver in self.requests_module_aliases:
                if node.attr == "Session":
                    return "independent_http_session"
        return self.constructor_roles.get(name)

    def is_direct_http_request(self, node: ast.AST) -> bool:
        if not self.chroma_http_context:
            return False
        if isinstance(node, ast.Name):
            return node.id in self.direct_http_function_aliases
        if not isinstance(node, ast.Attribute):
            return False
        receiver = _expr_name(node.value)
        return (
            receiver in self.httpx_module_aliases | self.requests_module_aliases
            and node.attr in {"request", "get", "post", "put", "patch", "delete"}
        ) or (
            receiver in self.urllib_request_aliases
            and node.attr in {"urlopen", "Request"}
        )

    def _collect_constructor_aliases(self, tree: ast.Module) -> None:
        assignments: list[tuple[list[str], ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assignments.append(
                    ([name for target in node.targets for name in _target_names(target)], node.value)
                )
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                assignments.append((_target_names(node.target), node.value))
        for _ in range(len(assignments) + 1):
            changed = False
            for targets, value in assignments:
                client_type = self.resolve_constructor(value)
                if client_type is None:
                    continue
                for target in targets:
                    if "." not in target and self.constructor_aliases.get(target) != client_type:
                        self.constructor_aliases[target] = client_type
                        role = self.resolve_constructor_role(value)
                        if role:
                            self.constructor_roles[target] = role
                        changed = True
            if not changed:
                break

    def _module_client_types(self, tree: ast.Module) -> set[str]:
        return {
            client_type
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for client_type in [self.resolve_constructor(node.func)]
            if client_type is not None
        }

    def collection_argument(self, call: ast.Call) -> tuple[str, str]:
        argument: ast.AST | None = call.args[0] if call.args else None
        for keyword in call.keywords:
            if keyword.arg == "name":
                argument = keyword.value
                break
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            return argument.value, "literal"
        if isinstance(argument, ast.Name):
            if argument.id in self.constants:
                return self.constants[argument.id], "shared_constant"
            return "dynamic_collection", "dynamic"
        if isinstance(argument, ast.Subscript):
            return "registry_collection", "registry"
        return "dynamic_collection", "dynamic"

    def _collect_function_returns(self, tree: ast.Module) -> None:
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for function in functions:
            local_clients: dict[str, str] = {}
            local_collections: dict[str, tuple[str, str, str]] = {}
            for node in ast.walk(function):
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                    targets = (
                        [name for target in node.targets for name in _target_names(target)]
                        if isinstance(node, ast.Assign)
                        else _target_names(node.target)
                    )
                    value = node.value
                    if isinstance(value, ast.Call):
                        client_type = self.resolve_constructor(value.func)
                        if client_type:
                            for target in targets:
                                local_clients[target] = client_type
                        if isinstance(value.func, ast.Attribute):
                            method = value.func.attr
                            receiver = _expr_name(value.func.value)
                            if method in _COLLECTION_RETURNING_OPERATIONS and (
                                receiver in local_clients
                                or receiver.endswith("client")
                                or receiver.endswith("._client")
                            ):
                                collection, resolution = self.collection_argument(value)
                                receiver_type = local_clients.get(receiver, self.default_client_type)
                                for target in targets:
                                    local_collections[target] = (
                                        receiver_type,
                                        collection,
                                        resolution,
                                    )
            for node in ast.walk(function):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                if isinstance(node.value, ast.Call):
                    client_type = self.resolve_constructor(node.value.func)
                    if client_type:
                        self.function_client_returns[function.name] = client_type
                name = _expr_name(node.value)
                if name in local_clients:
                    self.function_client_returns[function.name] = local_clients[name]
                if name in local_collections:
                    self.function_collection_returns[function.name] = local_collections[name]


class _PythonAccessVisitor(ast.NodeVisitor):
    def __init__(self, module: str, tree: ast.Module, source: str):
        self.module = module
        self.provenance = _ModuleProvenance(tree, module)
        self.source_mentions_chroma_sqlite = "chroma.sqlite3" in source
        self.scope: list[str] = []
        self.client_scopes: list[dict[str, str]] = [{}]
        self.collection_scopes: list[dict[str, tuple[str, str, str]]] = [{}]
        self.operation_scopes: list[dict[str, tuple[str, str]]] = [{}]
        self.attribute_clients: dict[str, str] = {}
        self.attribute_collections: dict[str, tuple[str, str, str]] = {}
        self.raw: list[dict[str, Any]] = []
        self.review_candidates: list[dict[str, Any]] = []

    @property
    def symbol(self) -> str:
        return ".".join(self.scope) or "<module>"

    def _client(self, node: ast.AST) -> str | None:
        name = _expr_name(node)
        if not name:
            return None
        if name.startswith("self.") and name in self.attribute_clients:
            return self.attribute_clients[name]
        for scope in reversed(self.client_scopes):
            if name in scope:
                return scope[name]
        return None

    def _collection(self, node: ast.AST) -> tuple[str, str, str] | None:
        name = _expr_name(node)
        if not name:
            return None
        if name.startswith("self.") and name in self.attribute_collections:
            return self.attribute_collections[name]
        for scope in reversed(self.collection_scopes):
            if name in scope:
                return scope[name]
        return None

    def _set_client(self, name: str, client_type: str) -> None:
        if name.startswith("self."):
            self.attribute_clients[name] = client_type
        else:
            self.client_scopes[-1][name] = client_type

    def _set_collection(self, name: str, value: tuple[str, str, str]) -> None:
        if name.startswith("self."):
            self.attribute_collections[name] = value
        else:
            self.collection_scopes[-1][name] = value

    def _operation_alias(self, node: ast.AST) -> tuple[str, str] | None:
        if isinstance(node, ast.Name):
            for scope in reversed(self.operation_scopes):
                if node.id in scope:
                    return scope[node.id]
            return None
        if isinstance(node, ast.Attribute) and node.attr in _CLIENT_OPERATIONS:
            client_type = self._client(node.value)
            if client_type:
                return client_type, node.attr
        return None

    def _set_operation_alias(self, name: str, value: tuple[str, str]) -> None:
        if not name.startswith("self."):
            self.operation_scopes[-1][name] = value

    def _add(
        self,
        node: ast.AST,
        *,
        client_type: str,
        collection: str,
        collection_resolution: str,
        operation: str,
        role: str,
    ) -> None:
        self.raw.append(
            {
                "module": self.module,
                "symbol": self.symbol,
                "line": int(getattr(node, "lineno", 0)),
                "client_type": client_type,
                "collection": collection,
                "collection_resolution": collection_resolution,
                "operation": operation,
                "semantic_role": role,
            }
        )

    def _candidate(self, node: ast.AST, reason: str) -> None:
        self.review_candidates.append(
            {
                "module": self.module,
                "symbol": self.symbol,
                "line": int(getattr(node, "lineno", 0)),
                "reason": reason,
            }
        )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        clients: dict[str, str] = {}
        collections: dict[str, tuple[str, str, str]] = {}
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for argument in arguments:
            name = argument.arg
            annotation = _expr_name(argument.annotation)
            if name == "client" or name.endswith("_client") or annotation.endswith("ClientAPI"):
                clients[name] = self.provenance.default_client_type
            if name == "collection" or name.endswith("_collection") or annotation.endswith("Collection"):
                collections[name] = (
                    self.provenance.default_client_type,
                    "dynamic_collection",
                    "dynamic",
                )
        self.client_scopes.append(clients)
        self.collection_scopes.append(collections)
        self.operation_scopes.append({})
        for statement in node.body:
            self.visit(statement)
        self.collection_scopes.pop()
        self.client_scopes.pop()
        self.operation_scopes.pop()
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def _assignment_provenance(
        self, value: ast.AST
    ) -> tuple[str | None, tuple[str, str, str] | None]:
        client_type = self._client(value)
        collection = self._collection(value)
        if isinstance(value, ast.Call):
            direct = self.provenance.resolve_constructor(value.func)
            if direct:
                client_type = direct
            function_name = _expr_name(value.func).split(".")[-1]
            if function_name in self.provenance.function_client_returns:
                client_type = self.provenance.function_client_returns[function_name]
            if function_name in self.provenance.function_collection_returns:
                collection = self.provenance.function_collection_returns[function_name]
            if isinstance(value.func, ast.Attribute):
                receiver_type = self._client(value.func.value)
                if receiver_type and value.func.attr in _COLLECTION_RETURNING_OPERATIONS:
                    semantic_name, resolution = self.provenance.collection_argument(value)
                    collection = (receiver_type, semantic_name, resolution)
        return client_type, collection

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        client_type, collection = self._assignment_provenance(node.value)
        operation_alias = self._operation_alias(node.value)
        targets = [name for target in node.targets for name in _target_names(target)]
        if client_type is None and self.provenance.default_client_type != "unknown":
            if any(target.casefold().endswith("_client") for target in targets):
                client_type = self.provenance.default_client_type
        for target in targets:
            if client_type:
                self._set_client(target, client_type)
            if collection:
                self._set_collection(target, collection)
            if operation_alias:
                self._set_operation_alias(target, operation_alias)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if node.value is None:
            return
        client_type, collection = self._assignment_provenance(node.value)
        operation_alias = self._operation_alias(node.value)
        for target in _target_names(node.target):
            if client_type:
                self._set_client(target, client_type)
            if collection:
                self._set_collection(target, collection)
            if operation_alias:
                self._set_operation_alias(target, operation_alias)
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        client_type, collection = self._assignment_provenance(node.value)
        operation_alias = self._operation_alias(node.value)
        for target in _target_names(node.target):
            if client_type:
                self._set_client(target, client_type)
            if collection:
                self._set_collection(target, collection)
            if operation_alias:
                self._set_operation_alias(target, operation_alias)
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        direct_type = self.provenance.resolve_constructor(node.func)
        if direct_type:
            self._add(
                node,
                client_type=direct_type,
                collection="not_applicable",
                collection_resolution="literal",
                operation="client construction",
                role=self.provenance.resolve_constructor_role(node.func)
                or "unresolved_client_constructor",
            )
        elif self.provenance.is_direct_http_request(node.func):
            self._add(
                node,
                client_type="http",
                collection="not_applicable",
                collection_resolution="literal",
                operation="http request",
                role="independent_http_request",
            )
        elif isinstance(node.func, ast.Name) and self._operation_alias(node.func):
            client_type, method = self._operation_alias(node.func) or ("unknown", "unknown")
            semantic_name, resolution = (
                self.provenance.collection_argument(node)
                if method in _COLLECTION_RETURNING_OPERATIONS
                else ("not_applicable", "literal")
            )
            self._add(
                node,
                client_type=client_type,
                collection=semantic_name,
                collection_resolution=resolution,
                operation=_CLIENT_OPERATIONS[method],
                role=f"aliased_{method}",
            )
        elif isinstance(node.func, ast.Name) and node.func.id.endswith("client_factory"):
            current_function = self.scope[-1] if self.scope else ""
            if current_function in self.provenance.function_client_returns:
                self._add(
                    node,
                    client_type="fake_http",
                    collection="not_applicable",
                    collection_resolution="literal",
                    operation="client construction",
                    role="injected_client_factory",
                )
        elif isinstance(node.func, ast.Name) and node.func.id == "PersistentClient":
            self._candidate(node, "unresolved_constructor_alias")
        elif (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Name)
            and node.func.func.id == "getattr"
            and any(
                isinstance(argument, ast.Constant)
                and argument.value == "PersistentClient"
                for argument in node.func.args
            )
        ):
            self._candidate(node, "dynamic_constructor_resolution")
        elif isinstance(node.func, ast.Attribute):
            method = node.func.attr
            receiver_name = _expr_name(node.func.value).casefold()
            client_type = self._client(node.func.value)
            collection = self._collection(node.func.value)
            if client_type and method in _CLIENT_OPERATIONS:
                semantic_name, resolution = (
                    self.provenance.collection_argument(node)
                    if method in _COLLECTION_RETURNING_OPERATIONS
                    else ("not_applicable", "literal")
                )
                self._add(
                    node,
                    client_type=client_type,
                    collection=semantic_name,
                    collection_resolution=resolution,
                    operation=_CLIENT_OPERATIONS[method],
                    role=method,
                )
            elif collection and method in _COLLECTION_OPERATIONS:
                collection_client, semantic_name, resolution = collection
                if collection_client == "unknown":
                    self._candidate(node, "unresolved_receiver_provenance")
                else:
                    self._add(
                        node,
                        client_type=collection_client,
                        collection=semantic_name,
                        collection_resolution=resolution,
                        operation=method,
                        role=method,
                    )
            elif (
                self.module == "backend/chroma_http_client_factory.py"
                and method == "get_collection"
                and receiver_name.endswith("transport")
            ):
                semantic_name, resolution = self.provenance.collection_argument(node)
                self._add(
                    node,
                    client_type="http",
                    collection=semantic_name,
                    collection_resolution=resolution,
                    operation="get collection",
                    role="get_collection",
                )
            elif (
                self.module == "backend/chroma_server_lifecycle.py"
                and method == "heartbeat"
                and receiver_name.endswith("adapter")
            ):
                self._add(
                    node,
                    client_type="http",
                    collection="not_applicable",
                    collection_resolution="literal",
                    operation="heartbeat",
                    role="heartbeat",
                )
            elif (
                self.module == "backend/chroma_operational_reader.py"
                and method == "heartbeat"
                and receiver_name.endswith("transport")
            ):
                self._add(
                    node,
                    client_type="http",
                    collection="not_applicable",
                    collection_resolution="literal",
                    operation="heartbeat",
                    role="heartbeat",
                )
            elif (method in _CLIENT_OPERATIONS and method != "request") or (
                method in _COLLECTION_OPERATIONS
                and receiver_name.endswith(("client", "collection"))
            ):
                self._candidate(node, "unresolved_receiver_provenance")
            elif method in {"PersistentClient", "HttpClient"}:
                self._candidate(node, "unresolved_constructor_alias")

            if (
                self.source_mentions_chroma_sqlite
                and method == "connect"
                and _expr_name(node.func.value) == "sqlite3"
            ):
                is_test_fixture = self.module.startswith("tests/")
                if self.symbol.endswith("inspect_github_vector_metadata"):
                    semantic_name, resolution = "github_evidence", "shared_constant"
                else:
                    semantic_name, resolution = "dynamic_collection", "dynamic"
                self._add(
                    node,
                    client_type="ephemeral_embedded" if is_test_fixture else "persistent_embedded",
                    collection=semantic_name,
                    collection_resolution=resolution,
                    operation="index" if is_test_fixture else "backup/recovery inspection",
                    role="synthetic_sqlite_fixture" if is_test_fixture else "immutable_sqlite_inspection",
                )
        self.generic_visit(node)


def _finalize_discoveries(raw: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (dict(item) for item in raw),
        key=lambda item: (
            item["module"],
            item["symbol"],
            item["line"],
            item["operation"],
            item["collection"],
            item["client_type"],
        ),
    )
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in ordered:
        grouped[
            (
                item["module"],
                item["symbol"],
                item["operation"],
                item["collection"],
                item["semantic_role"],
            )
        ].append(item)
    for items in grouped.values():
        if len(items) > 1:
            for index, item in enumerate(items, start=1):
                item["semantic_role"] = f"{item['semantic_role']}:{index}"
    for item in ordered:
        item["access_id"] = stable_access_id(item)
    return sorted(ordered, key=lambda item: item["access_id"])


def _scan_executable_text(
    source: str, module: str, *, document: bool
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    in_fence = not document
    for line_number, line in enumerate(source.splitlines(), start=1):
        if document and line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence and _EXECUTABLE_CHROMA_RE.search(line):
            candidates.append(
                {
                    "module": module,
                    "symbol": "<executable_example>",
                    "line": line_number,
                    "reason": "executable_chroma_reference_requires_review",
                }
            )
    return candidates


def scan_repository(repository_root: str | Path) -> dict[str, Any]:
    """Parse source without imports, application execution, or Chroma I/O."""

    root = Path(repository_root)
    if not root.is_dir():
        _error("repository_unavailable")
    raw: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for path, module in _iter_source_files(root):
        source = _read_source(path)
        suffix = path.suffix.casefold()
        if suffix == ".py":
            if not any(
                marker in source
                for marker in (
                    "chromadb",
                    "httpx",
                    "import requests",
                    "from requests",
                    "urllib.request",
                    "PersistentClient",
                    "HttpClient",
                    "BoundedChromaHttpTransport",
                    "ClientAPI",
                    "Collection",
                )
            ):
                continue
            try:
                tree = ast.parse(source, filename=module)
            except SyntaxError:
                _error("source_parse_failed")
            visitor = _PythonAccessVisitor(module, tree, source)
            visitor.visit(tree)
            raw.extend(visitor.raw)
            candidates.extend(visitor.review_candidates)
        else:
            candidates.extend(
                _scan_executable_text(source, module, document=suffix in _DOCUMENT_EXTENSIONS)
            )
    candidates.sort(key=lambda item: (item["module"], item["line"], item["reason"]))
    return {"discoveries": _finalize_discoveries(raw), "review_candidates": candidates}


_SEMANTIC_CLIENT_MODULES = frozenset(
    {
        "backend/chroma_operational_reader.py",
        "backend/chroma_read_client.py",
        "backend/chroma_write_client.py",
    }
)
_PERSISTENCE_FREE_MODULES = _SEMANTIC_CLIENT_MODULES | frozenset(
    {
        "backend/chroma_http_client_factory.py",
        "backend/chroma_http_transport.py",
        "backend/chroma_http_vector_search.py",
        "backend/memory_store.py",
        "backend/project_retrieval_v2.py",
    }
)
_LOW_LEVEL_HTTP_ROOTS = frozenset({"httpx", "requests"})


def scan_production_source_policy(
    repository_root: str | Path,
) -> list[dict[str, str | int]]:
    """Inspect production dependency ownership without importing application code."""

    root = Path(repository_root)
    violations: list[dict[str, str | int]] = []

    def add(module: str, node: ast.AST, symbol: str, category: str) -> None:
        item = {
            "module": module,
            "line": max(1, int(getattr(node, "lineno", 1) or 1)),
            "symbol": symbol,
            "category": category,
        }
        if item not in violations:
            violations.append(item)

    for path, module in _iter_source_files(root):
        if path.suffix.casefold() != ".py" or not module.startswith("backend/"):
            continue
        source = _read_source(path)
        try:
            tree = ast.parse(source, filename=module)
        except SyntaxError:
            _error("source_parse_failed")

        imported_modules: list[tuple[ast.AST, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend((node, alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append((node, node.module))
        imports_chroma_boundary = any(
            name == "backend.chroma_config" or name.startswith("backend.chroma_")
            for _, name in imported_modules
        )
        chroma_context = module.startswith("backend/chroma_") or imports_chroma_boundary

        class ScopeVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.stack: list[str] = []

            @property
            def symbol(self) -> str:
                return ".".join(self.stack) or "<module>"

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
                name = _expr_name(node.func)
                leaf = name.rsplit(".", 1)[-1]
                if chroma_context and leaf in {
                    "create_collection",
                    "get_or_create_collection",
                }:
                    add(module, node, self.symbol, "production_collection_creation_call")
                if module in _SEMANTIC_CLIENT_MODULES and leaf in {
                    "PersistentClient",
                    "HttpClient",
                    "BoundedChromaHttpTransport",
                }:
                    add(module, node, self.symbol, "semantic_client_bypasses_factory")
                if module in _SEMANTIC_CLIENT_MODULES and leaf in {
                    "start_chroma_server",
                    "start_local_chroma_server",
                }:
                    add(module, node, self.symbol, "automatic_server_start_call")
                self.generic_visit(node)

        ScopeVisitor().visit(tree)

        for node, imported in imported_modules:
            if imported == "tests.chroma_persistence_test_support" or imported.startswith(
                "tests.chroma_persistence_test_support."
            ):
                add(
                    module,
                    node,
                    "<module>",
                    "production_imports_test_persistence_helper",
                )
            root_name = imported.split(".", 1)[0]
            raw_urllib = imported == "urllib.request" or imported.startswith(
                "urllib.request."
            )
            if chroma_context and (
                root_name in _LOW_LEVEL_HTTP_ROOTS or raw_urllib
            ):
                approved = module == "backend/chroma_http_transport.py" and imported == "httpx"
                if not approved:
                    add(module, node, "<module>", "independent_chroma_http_dependency")

        if module in _SEMANTIC_CLIENT_MODULES and not any(
            imported == "backend.chroma_http_client_factory"
            for _, imported in imported_modules
        ):
            add(module, tree, "<module>", "semantic_client_missing_central_factory")

        if module in _PERSISTENCE_FREE_MODULES:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                normalized = node.value.replace("\\", "/").casefold()
                if "information/chroma" in normalized:
                    add(module, node, "<module>", "consumer_production_persistence_reference")

    return sorted(
        violations,
        key=lambda item: (
            str(item["module"]),
            int(item["line"]),
            str(item["symbol"]),
            str(item["category"]),
        ),
    )


def load_reviewed_inventory(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    candidate = Path(path)
    try:
        is_default = candidate.resolve() == DEFAULT_MANIFEST_PATH.resolve()
    except OSError:
        is_default = False
    if is_default:
        try:
            from backend.chroma_access_manifest import EXPECTED_INVENTORY_DIGEST, INVENTORY

            payload = json.loads(json.dumps(INVENTORY))
        except Exception as error:
            if isinstance(error, ChromaAccessInventoryError):
                raise
            _error("reviewed_inventory_unavailable")
        if not EXPECTED_INVENTORY_DIGEST:
            _error("reviewed_inventory_digest_missing")
        if payload.get("inventory_digest") != EXPECTED_INVENTORY_DIGEST:
            _error("reviewed_inventory_digest_mismatch")
        try:
            validate_chroma_access_inventory(payload)
        except ChromaAccessValidationError as error:
            _error(error.code)
        return payload
    try:
        if not candidate.is_file() or candidate.stat().st_size > MAX_SOURCE_BYTES:
            _error("reviewed_inventory_unavailable")
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except ChromaAccessInventoryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        _error("reviewed_inventory_unavailable")
    try:
        validate_chroma_access_inventory(payload)
    except ChromaAccessValidationError as error:
        _error(error.code)
    return payload


_DISCOVERY_FIELDS = (
    "access_id",
    "module",
    "symbol",
    "line",
    "semantic_role",
    "client_type",
    "collection",
    "collection_resolution",
    "operation",
)


def compare_discovery_to_inventory(
    scan: Mapping[str, Any], inventory: Mapping[str, Any]
) -> dict[str, Any]:
    discoveries = {item["access_id"]: item for item in scan["discoveries"]}
    records = {item["access_id"]: item for item in inventory["records"]}
    discovered_ids = set(discoveries)
    reviewed_ids = set(records)
    unresolved = sorted(discovered_ids - reviewed_ids)
    stale = sorted(reviewed_ids - discovered_ids)
    mismatched = []
    for access_id in sorted(discovered_ids & reviewed_ids):
        discovered = discoveries[access_id]
        reviewed = records[access_id]
        if any(discovered[field] != reviewed[field] for field in _DISCOVERY_FIELDS):
            mismatched.append(access_id)
    candidates = list(scan.get("review_candidates", []))
    return {
        "status": "verified" if not (unresolved or stale or mismatched or candidates) else "mismatch",
        "discovered_count": len(discoveries),
        "classified_count": len(records),
        "unresolved_access_ids": unresolved,
        "stale_access_ids": stale,
        "mismatched_access_ids": mismatched,
        "review_candidates": candidates,
    }


def inspect_repository(
    repository_root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    inventory = load_reviewed_inventory(manifest_path)
    scan = scan_repository(repository_root)
    comparison = compare_discovery_to_inventory(scan, inventory)
    records = inventory["records"]
    comparison["schema"] = inventory["schema"]
    comparison["inventory_digest"] = inventory["inventory_digest"]
    comparison["runtime_counts"] = dict(sorted(Counter(item["runtime"] for item in records).items()))
    comparison["client_counts"] = dict(
        sorted(Counter(item["client_type"] for item in records).items())
    )
    comparison["lifecycle_counts"] = dict(
        sorted(Counter(item["lifecycle"] for item in records).items())
    )
    comparison["unknown_count"] = sum(
        item["client_type"] == "unknown" or item["collection_resolution"] == "unknown"
        for item in records
    )
    comparison["policy_violations"] = sorted(
        [
            *production_access_policy_violations(records),
            *scan_production_source_policy(repository_root),
        ],
        key=lambda item: (
            str(item["module"]),
            int(item["line"]),
            str(item["symbol"]),
            str(item["category"]),
        ),
    )
    comparison["forbidden_count"] = len(comparison["policy_violations"])
    return comparison


def baseline_constructor_summary(
    repository_root: str | Path = PROJECT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Project reviewed constructors into the unchanged migration-baseline schema."""

    report = inspect_repository(repository_root, manifest_path)
    if report["status"] != "verified" or report["unknown_count"]:
        _error("unclassified_chroma_call_site")
    inventory = load_reviewed_inventory(manifest_path)
    call_sites = []
    for record in inventory["records"]:
        if record["operation"] != "client construction":
            continue
        if record["client_type"] not in {"persistent_embedded", "http"}:
            continue
        classification = {
            "production": "production",
            "maintenance_only": "approved_maintenance",
            "test_only": "test_only",
            "migration_only": "approved_maintenance",
        }[record["runtime"]]
        call_sites.append(
            {
                "client_type": "persistent" if record["client_type"] == "persistent_embedded" else "http",
                "classification": classification,
                "relative_path": record["module"],
                "line": record["line"],
                "callable": record["symbol"],
            }
        )
    call_sites.sort(key=lambda item: (item["relative_path"], item["line"], item["client_type"]))
    from backend.chroma_baseline_models import sha256_json

    return {
        "production_persistent_client_call_count": sum(
            item["client_type"] == "persistent" and item["classification"] == "production"
            for item in call_sites
        ),
        "approved_maintenance_persistent_client_call_count": sum(
            item["client_type"] == "persistent"
            and item["classification"] == "approved_maintenance"
            for item in call_sites
        ),
        "test_only_persistent_client_call_count": sum(
            item["client_type"] == "persistent" and item["classification"] == "test_only"
            for item in call_sites
        ),
        "http_client_call_count": sum(item["client_type"] == "http" for item in call_sites),
        "unknown_unclassified_call_count": 0,
        "call_sites": call_sites,
        "aggregate_sha256": sha256_json(call_sites),
    }


def _bounded_counts(values: Mapping[str, int]) -> str:
    return " ".join(f"{key}={values[key]}" for key in sorted(values)) or "none=0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the reviewed Chroma access inventory.")
    parser.add_argument("command", choices=("inspect", "verify"))
    parser.add_argument("--repository-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = inspect_repository(arguments.repository_root, arguments.manifest)
    except ChromaAccessInventoryError as error:
        print(f"inventory {arguments.command} failed code={error.code}", file=sys.stderr)
        return 1
    print(
        f"inventory {arguments.command} status={report['status']} "
        f"discovered={report['discovered_count']} classified={report['classified_count']} "
        f"unresolved={len(report['unresolved_access_ids'])} "
        f"stale={len(report['stale_access_ids'])} "
        f"mismatched={len(report['mismatched_access_ids'])} "
        f"review_candidates={len(report['review_candidates'])} "
        f"unknown={report['unknown_count']} "
        f"forbidden={report['forbidden_count']}"
    )
    print(f"runtime {_bounded_counts(report['runtime_counts'])}")
    print(f"client {_bounded_counts(report['client_counts'])}")
    print(f"lifecycle {_bounded_counts(report['lifecycle_counts'])}")
    for violation in report["policy_violations"]:
        print(
            "policy_violation "
            f"module={violation['module']} line={violation['line']} "
            f"symbol={violation['symbol']} category={violation['category']}"
        )
    return (
        0
        if report["status"] == "verified"
        and report["unknown_count"] == 0
        and report["forbidden_count"] == 0
        else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
