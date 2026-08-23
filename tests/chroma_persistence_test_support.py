"""Explicit subprocess-only embedded Chroma probe for temporary test storage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import chromadb

from backend.chroma_config import ChromaDeploymentConfig, ChromaDeploymentMode
from backend.chroma_collection_registry import get_collection_definition
from backend.memory_store import LocalHashEmbedding
from backend.chroma_persistence_guard import ChromaPersistenceGuard
from backend.chroma_persistence_guard_models import (
    ChromaPersistenceContext,
    ChromaPersistenceGuardError,
)
from backend.chroma_server_lifecycle_models import build_chroma_server_lifecycle_config


def create_test_owned_persistent_client(
    *,
    information_root: str | Path,
    persistence_path: str | Path,
    runtime_state_directory: str | Path,
    port: int,
    return_client: bool = False,
) -> Any:
    """Open only after the authoritative guard proves isolated stopped test storage."""

    information = Path(information_root)
    persistence = Path(persistence_path)
    runtime = Path(runtime_state_directory)
    lifecycle_deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.LOCAL_HTTP,
        "127.0.0.1",
        port,
        False,
        0.5,
    )
    lifecycle_config = build_chroma_server_lifecycle_config(
        lifecycle_deployment,
        information_root=information,
        persistence_path=persistence,
        runtime_state_directory=runtime,
        startup_timeout_seconds=5.0,
        shutdown_timeout_seconds=2.0,
        endpoint_release_timeout_seconds=2.0,
        poll_interval_seconds=0.05,
        test_owned=True,
    )
    test_deployment = ChromaDeploymentConfig(
        ChromaDeploymentMode.EPHEMERAL_TEST,
        "127.0.0.1",
        port,
        False,
        0.5,
    )
    context = ChromaPersistenceContext.test_owned(
        test_deployment,
        storage_root=information,
    )
    decision = ChromaPersistenceGuard(lifecycle_config).assert_embedded_access_allowed(
        path=persistence,
        context=context,
    )
    client = chromadb.PersistentClient(path=str(persistence))
    if return_client:
        return client
    del client
    return decision.safe_summary()


def read_test_owned_collection_snapshot(
    *,
    information_root: str | Path,
    persistence_path: str | Path,
    runtime_state_directory: str | Path,
    port: int,
    query: str,
    n_results: int,
) -> dict[str, Any]:
    """Return content hashes and projected metadata from a disposable legacy reader."""

    client = create_test_owned_persistent_client(
        information_root=information_root,
        persistence_path=persistence_path,
        runtime_state_directory=runtime_state_directory,
        port=port,
        return_client=True,
    )
    embedding = LocalHashEmbedding().embed(query)
    collections: dict[str, Any] = {}
    try:
        for semantic_id in ("github_evidence", "profile_facts"):
            definition = get_collection_definition(semantic_id)
            collection = client.get_collection(name=definition.collection_name)
            count = collection.count()
            records = collection.get(
                limit=max(count, 1),
                include=["documents", "metadatas"],
            )
            query_result = collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results, count),
                include=["distances", "documents", "metadatas"],
            ) if count else {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
            fields = definition.logical_integrity_metadata_allowlist

            def projected(metadata: Any) -> dict[str, Any]:
                if not isinstance(metadata, Mapping):
                    return {}
                return {key: metadata[key] for key in fields if key in metadata}

            documents = records.get("documents", [])
            metadatas = records.get("metadatas", [])
            collections[semantic_id] = {
                "count": count,
                "records": sorted(
                    [
                        {
                            "id": record_id,
                            "content_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest()
                            if isinstance(document, str)
                            else "",
                            "metadata": projected(metadata),
                        }
                        for record_id, document, metadata in zip(
                            records.get("ids", []), documents, metadatas
                        )
                    ],
                    key=lambda item: item["id"],
                ),
                "query": [
                    {
                        "id": record_id,
                        "distance": float(distance),
                        "content_sha256": hashlib.sha256(document.encode("utf-8")).hexdigest()
                        if isinstance(document, str)
                        else "",
                        "metadata": projected(metadata),
                    }
                    for record_id, distance, document, metadata in zip(
                        query_result.get("ids", [[]])[0],
                        query_result.get("distances", [[]])[0],
                        query_result.get("documents", [[]])[0],
                        query_result.get("metadatas", [[]])[0],
                    )
                ],
            }
    finally:
        del client
    return {"collections": collections}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("information_root")
    parser.add_argument("persistence_path")
    parser.add_argument("runtime_state_directory")
    parser.add_argument("port", type=int)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--query", default="retrieval evidence")
    parser.add_argument("--n-results", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.snapshot:
            result = read_test_owned_collection_snapshot(
                information_root=arguments.information_root,
                persistence_path=arguments.persistence_path,
                runtime_state_directory=arguments.runtime_state_directory,
                port=arguments.port,
                query=arguments.query,
                n_results=arguments.n_results,
            )
        else:
            result = create_test_owned_persistent_client(
                information_root=arguments.information_root,
                persistence_path=arguments.persistence_path,
                runtime_state_directory=arguments.runtime_state_directory,
                port=arguments.port,
            )
    except ChromaPersistenceGuardError as error:
        print(
            json.dumps(
                {"allowed": False, "error": error.code},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 4
    print(
        json.dumps(
            result,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
