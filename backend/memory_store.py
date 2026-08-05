"""Chroma-backed vector storage for durable profile memory and GitHub evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError as error:  # pragma: no cover - exercised when dependencies are missing
    chromadb = None
    CHROMA_IMPORT_ERROR = error
else:
    CHROMA_IMPORT_ERROR = None


EMBEDDING_DIMENSIONS = 384
PROFILE_COLLECTION = "profile_facts"
GITHUB_COLLECTION = "github_evidence"
PROFILE_MIGRATION_MARKER = ".legacy_profile_migrated"
TOKEN_PATTERN = re.compile(r"[\w.+#-]+", re.UNICODE)
GITHUB_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)
GITHUB_REPOSITORY_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def canonical_github_repository(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    url_match = GITHUB_URL_PATTERN.search(text)
    if url_match:
        owner, repo = url_match.groups()
        return f"{owner}/{repo.removesuffix('.git').rstrip('.,;:)]}>')}"
    repo_match = GITHUB_REPOSITORY_PATTERN.match(text)
    if repo_match:
        owner, repo = repo_match.groups()
        return f"{owner}/{repo.removesuffix('.git').rstrip('.,;:)]}>')}"
    return text.removesuffix(".git").rstrip(".,;:)]}>")


class LocalHashEmbedding:
    """Small deterministic local embedder so Chroma works without network downloads."""

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        words = TOKEN_PATTERN.findall(lowered)
        compact = re.sub(r"\s+", "", lowered)
        grams = [compact[index : index + 3] for index in range(max(0, len(compact) - 2))]
        return words + grams

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * EMBEDDING_DIMENSIONS
        for token in self._tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSIONS
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if not magnitude:
            return vector
        return [value / magnitude for value in vector]

    def embed_many(self, documents: list[str]) -> list[list[float]]:
        return [self.embed(document) for document in documents]


class MemoryVectorStore:
    def __init__(self, persist_directory: Path, legacy_memory_path: Path, legacy_github_dir: Path):
        self.persist_directory = persist_directory
        self.legacy_memory_path = legacy_memory_path
        self.legacy_github_dir = legacy_github_dir
        self.embedder = LocalHashEmbedding()
        self._client = None
        self._profile = None
        self._github = None

    def _ensure_client(self, migrate: bool = True) -> None:
        if self._client is not None:
            return
        if chromadb is None:
            raise RuntimeError(
                "Chroma is not installed. Run: python -m pip install -r backend/requirements.txt"
            ) from CHROMA_IMPORT_ERROR

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._profile = self._client.get_or_create_collection(
            PROFILE_COLLECTION,
            metadata={"description": "Durable user profile facts", "hnsw:space": "cosine"},
        )
        self._github = self._client.get_or_create_collection(
            GITHUB_COLLECTION,
            metadata={"description": "Approved GitHub repository and commit evidence", "hnsw:space": "cosine"},
        )
        if migrate:
            self._migrate_legacy_profile()
            self._migrate_legacy_github()

    def _migrate_legacy_profile(self) -> None:
        marker_path = self.persist_directory / PROFILE_MIGRATION_MARKER
        if marker_path.exists():
            return
        if not self._profile.count() and self.legacy_memory_path.exists():
            content = self.legacy_memory_path.read_text(encoding="utf-8").strip()
            if content:
                try:
                    memory = json.loads(content)
                except json.JSONDecodeError:
                    memory = {"notes": content}
                if isinstance(memory, dict):
                    self._replace_profile(memory, source="legacy-memory-json")
        marker_path.touch()

    def _migrate_legacy_github(self) -> None:
        if self._github.count() or not self.legacy_github_dir.exists():
            return
        files = sorted(
            self.legacy_github_dir.glob("github_context_*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return
        try:
            contexts = json.loads(files[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(contexts, list):
            self._store_github_contexts(contexts, source="legacy-github-json")

    @staticmethod
    def _record_id(prefix: str, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return f"{prefix}-{digest}"

    @staticmethod
    def _profile_item_key(section: str, value: Any, index: int) -> str:
        if isinstance(value, dict):
            identity = (
                value.get("name")
                or value.get("organization")
                or value.get("institution")
                or value.get("role")
            )
            if identity:
                return f"{section}:{identity}"
        return f"{section}:{index}"

    def _profile_records(self, memory: dict[str, Any], source: str) -> list[dict[str, Any]]:
        records = []
        for section, value in memory.items():
            values = value if isinstance(value, list) else [value]
            is_list = isinstance(value, list)
            for index, item in enumerate(values):
                key = self._profile_item_key(section, item, index) if is_list else section
                payload = {"section": section, "index": index, "is_list": is_list, "value": item}
                document = f"Profile memory section: {section}\n{normalized_json(item)}"
                records.append(
                    {
                        "id": self._record_id("profile", key),
                        "document": document,
                        "payload": payload,
                        "metadata": {
                            "section": section,
                            "index": index,
                            "is_list": int(is_list),
                            "source": source,
                            "updated_at": timestamp_slug(),
                        },
                    }
                )
        return records

    def _upsert_with_similarity(self, collection, records: list[dict[str, Any]]) -> dict[str, int]:
        inserted = 0
        updated = 0
        unchanged = 0
        deduplicated = 0
        for record in records:
            embedding = self.embedder.embed(record["document"])
            existing = collection.get(ids=[record["id"]], include=["documents"])
            previous = existing.get("documents", [])
            if previous and previous[0] == record["document"]:
                unchanged += 1
                continue

            # Query before write so additions can be compared with semantically related facts.
            if collection.count():
                similar = collection.query(
                    query_embeddings=[embedding],
                    n_results=min(3, collection.count()),
                    include=["documents", "distances", "metadatas"],
                )
                similar_ids = similar.get("ids", [[]])[0]
                similar_distances = similar.get("distances", [[]])[0]
                similar_metadatas = similar.get("metadatas", [[]])[0]
                for similar_id, distance, metadata in zip(
                    similar_ids, similar_distances, similar_metadatas
                ):
                    if similar_id == record["id"] or distance > 0.12:
                        continue
                    same_section = metadata.get("section") == record["metadata"].get("section")
                    same_repository = metadata.get("repository") == record["metadata"].get("repository")
                    if same_section or same_repository:
                        collection.delete(ids=[similar_id])
                        deduplicated += 1
                        break

            collection.upsert(
                ids=[record["id"]],
                embeddings=[embedding],
                documents=[record["document"]],
                metadatas=[record["metadata"]],
            )
            if previous:
                updated += 1
            else:
                inserted += 1
        return {
            "inserted": inserted,
            "updated": updated,
            "unchanged": unchanged,
            "deduplicated": deduplicated,
        }

    def _replace_profile(self, memory: dict[str, Any], source: str) -> dict[str, int]:
        records = self._profile_records(memory, source)
        expected_ids = {record["id"] for record in records}
        existing_ids = set(self._profile.get().get("ids", []))
        stale_ids = sorted(existing_ids - expected_ids)
        if stale_ids:
            self._profile.delete(ids=stale_ids)
        result = self._upsert_with_similarity(self._profile, records)
        result["deleted"] = len(stale_ids)
        return result

    def replace_profile(self, memory: dict[str, Any], source: str = "profile-update") -> dict[str, int]:
        self._ensure_client()
        return self._replace_profile(memory, source)

    def _collection_count_read_only(self, collection_name: str, live_collection: Any) -> int:
        database_path = (self.persist_directory / "chroma.sqlite3").resolve()
        if database_path.is_file():
            try:
                connection = sqlite3.connect(
                    f"file:{database_path.as_posix()}?mode=ro&immutable=1", uri=True,
                )
                try:
                    row = connection.execute(
                        """
                        SELECT COUNT(*)
                        FROM embeddings
                        WHERE segment_id = (
                            SELECT segments.id
                            FROM segments
                            JOIN collections ON collections.id = segments.collection
                            WHERE collections.name = ? AND segments.scope = 'METADATA'
                            LIMIT 1
                        )
                        """,
                        (collection_name,),
                    ).fetchone()
                finally:
                    connection.close()
            except (OSError, sqlite3.Error):
                return 0
            return int(row[0]) if row and isinstance(row[0], int) and row[0] > 0 else 0
        if live_collection is None:
            return 0
        try:
            count = live_collection.count()
        except Exception:
            return 0
        return count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0

    def profile_count(self) -> int:
        return self._collection_count_read_only(PROFILE_COLLECTION, self._profile)

    def read_profile(self, query: str = "", limit: int | None = None) -> dict[str, Any]:
        self._ensure_client()
        if not self._profile.count():
            return {}

        if query:
            result = self._profile.query(
                query_embeddings=[self.embedder.embed(query)],
                n_results=min(limit or 6, self._profile.count()),
                include=["documents", "metadatas"],
            )
            metadatas = result.get("metadatas", [[]])[0]
        else:
            metadatas = self._profile.get(include=["metadatas"]).get("metadatas", [])

        memory: dict[str, Any] = {}
        list_sections: dict[str, list[tuple[int, Any]]] = {}
        for metadata in metadatas:
            section = metadata["section"]
            document_id = self._record_id("profile", section)
            if metadata.get("is_list"):
                candidates = self._profile.get(
                    where={"section": section},
                    include=["documents", "metadatas"],
                )
                items = []
                for item_metadata, document in zip(
                    candidates.get("metadatas", []), candidates.get("documents", [])
                ):
                    payload = json.loads(document.split("\n", 1)[1])
                    items.append((int(item_metadata["index"]), payload))
                list_sections[section] = items
                continue

            result = self._profile.get(ids=[document_id], include=["documents"])
            documents = result.get("documents", [])
            if documents:
                memory[section] = json.loads(documents[0].split("\n", 1)[1])

        for section, items in list_sections.items():
            memory[section] = [value for _, value in sorted(items)]
        return memory

    def delete_profile(
        self,
        section: str,
        item_index: int | None = None,
        delete_section: bool = False,
    ) -> dict[str, Any]:
        self._ensure_client()
        section = section.strip()
        if not section:
            raise ValueError("Memory section is required.")
        if item_index is not None and item_index < 0:
            raise ValueError("Memory item_index must be zero or greater.")
        if item_index is None and not delete_section:
            raise ValueError(
                "Specify item_index to delete one list item, or set delete_section=true "
                "to delete the whole memory section."
            )

        candidates = self._profile.get(
            where={"section": section},
            include=["documents", "metadatas"],
        )
        deleted_ids = []
        deleted_values = []
        for record_id, document, metadata in zip(
            candidates.get("ids", []),
            candidates.get("documents", []),
            candidates.get("metadatas", []),
        ):
            is_target_item = metadata.get("is_list") and int(metadata["index"]) == item_index
            if delete_section or is_target_item:
                deleted_ids.append(record_id)
                deleted_values.append(json.loads(document.split("\n", 1)[1]))

        if deleted_ids:
            self._profile.delete(ids=deleted_ids)
        return {
            "deleted": len(deleted_ids),
            "section": section,
            "item_index": item_index,
            "deleted_values": deleted_values,
        }

    @staticmethod
    def _repo_key(context: dict[str, Any], index: int) -> str:
        key = canonical_github_repository(context.get("repository"))
        if key:
            return key
        key = canonical_github_repository(context.get("url"))
        return key or f"repo-{index}"

    def _store_github_contexts(self, contexts: list[dict[str, Any]], source: str) -> dict[str, Any]:
        run_id = timestamp_slug()
        records = []
        for index, context in enumerate(contexts):
            key = self._repo_key(context, index)
            records.append(
                {
                    "id": self._record_id("github", key),
                    "document": f"Approved GitHub evidence for {key}\n{normalized_json(context)}",
                    "metadata": {
                        "repository": key,
                        "run_id": run_id,
                        "source": source,
                        "updated_at": run_id,
                    },
                }
            )
        result = self._upsert_with_similarity(self._github, records)
        result["run_id"] = run_id
        result["cleanup"] = self.cleanup_github_repositories()
        return result

    def store_github_contexts(
        self, contexts: list[dict[str, Any]], source: str = "github-fetch"
    ) -> dict[str, Any]:
        self._ensure_client()
        return self._store_github_contexts(contexts, source)

    def read_github_contexts(self, query: str = "", limit: int | None = None) -> list[dict[str, Any]]:
        self._ensure_client()
        if not self._github.count():
            return []
        if query:
            result = self._github.query(
                query_embeddings=[self.embedder.embed(query)],
                n_results=min(limit or 8, self._github.count()),
                include=["documents"],
            )
            documents = result.get("documents", [[]])[0]
        else:
            documents = self._github.get(include=["documents"]).get("documents", [])
        return [json.loads(document.split("\n", 1)[1]) for document in documents]

    def search_github_vector_records(
        self,
        query: str,
        n_results: int = 5,
        *,
        project_id: str = "",
        authority: Any = None,
    ) -> list[dict[str, Any]]:
        """Use the default-off HTTP bridge without opening local persistence."""

        if not isinstance(query, str) or not query.strip() or isinstance(n_results, bool):
            return []
        if not isinstance(n_results, int) or n_results <= 0:
            return []
        try:
            try:
                from backend.chroma_http_vector_search import search_github_evidence_vectors_http
            except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch mode
                from chroma_http_vector_search import search_github_evidence_vectors_http
            return search_github_evidence_vectors_http(
                query=query,
                n_results=n_results,
                project_id=project_id,
                embedder=self.embedder,
                authority=authority,
            )
        except Exception:
            return []

    def inspect_github_vector_metadata(self, limit: int = 10000) -> list[dict[str, Any]]:
        """Inspect existing GitHub vector IDs and metadata without documents or embeddings."""

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return []
        database_path = (self.persist_directory / "chroma.sqlite3").resolve()
        if database_path.is_file():
            try:
                connection = sqlite3.connect(
                    f"file:{database_path.as_posix()}?mode=ro&immutable=1", uri=True,
                )
                try:
                    collection_row = connection.execute(
                        "SELECT id FROM collections WHERE name = ? LIMIT 1", (GITHUB_COLLECTION,),
                    ).fetchone()
                    if not collection_row:
                        return []
                    segment_row = connection.execute(
                        "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA' LIMIT 1",
                        (collection_row[0],),
                    ).fetchone()
                    if not segment_row:
                        return []
                    safe_keys = (
                        "github_repository", "project_id", "project_name", "repo",
                        "repository", "repository_url", "source", "updated_at",
                    )
                    placeholders = ",".join("?" for _ in safe_keys)
                    rows = connection.execute(
                        f"""
                        SELECT selected.id, selected.embedding_id, metadata.key, metadata.string_value
                        FROM (
                            SELECT id, embedding_id FROM embeddings
                            WHERE segment_id = ? ORDER BY id LIMIT ?
                        ) AS selected
                        LEFT JOIN embedding_metadata AS metadata
                          ON metadata.id = selected.id AND metadata.key IN ({placeholders})
                        ORDER BY selected.id, metadata.key
                        """,
                        (segment_row[0], min(limit, 10000), *safe_keys),
                    ).fetchall()
                finally:
                    connection.close()
            except (OSError, sqlite3.Error):
                return []
            records: dict[int, dict[str, Any]] = {}
            for internal_id, record_id, key, string_value in rows:
                record = records.setdefault(internal_id, {
                    "vector_record_id": str(record_id), "metadata": {},
                })
                if isinstance(key, str) and isinstance(string_value, str):
                    record["metadata"][key.lower()] = string_value
            return list(records.values())
        collection = self._github
        if collection is None:
            return []
        count = collection.count()
        if not count:
            return []
        result = collection.get(limit=min(limit, count), include=["metadatas"])
        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        safe_keys = {
            "github_repository", "project_id", "project_name", "repo", "repository",
            "repository_url", "source", "updated_at",
        }
        return [
            {
                "vector_record_id": str(record_id),
                "metadata": {
                    str(key).lower(): value
                    for key, value in metadata.items()
                    if isinstance(metadata, dict)
                    and str(key).lower() in safe_keys
                    and isinstance(value, str)
                } if isinstance(metadata, dict) else {},
            }
            for record_id, metadata in zip(ids, metadatas)
        ]

    def list_github_repositories(self) -> list[dict[str, str]]:
        repository_map: dict[str, dict[str, str]] = {}
        for record in self.inspect_github_vector_metadata():
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            repository = canonical_github_repository(metadata.get("repository"))
            if not repository:
                continue
            updated_at = str(metadata.get("updated_at", ""))
            current = repository_map.get(repository)
            if current is None or updated_at > current["updated_at"]:
                repository_map[repository] = {"repository": repository, "updated_at": updated_at}
        repositories = list(repository_map.values())
        return sorted(repositories, key=lambda item: item["updated_at"], reverse=True)

    def github_metadata_status(self) -> dict[str, Any]:
        if not self.persist_directory.exists() and self._github is None:
            return {"available": False, "count": 0, "repositories": []}
        return {
            "available": True,
            "count": self.github_count(),
            "repositories": self.list_github_repositories(),
        }

    def github_preview_metadata(self, limit: int = 5) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return []
        records = []
        for record in self.inspect_github_vector_metadata(limit=limit):
            metadata = record.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            repository = canonical_github_repository(metadata.get("repository"))
            records.append(
                {
                    "id": str(record.get("vector_record_id", "")),
                    "repository": repository,
                    "updated_at": str(metadata.get("updated_at", "")),
                    "source": str(metadata.get("source", "")),
                }
            )
        return records

    def read_github_document(self, record_id: str) -> dict[str, Any] | None:
        if chromadb is None:
            raise RuntimeError(
                "Chroma is not installed. Run: python -m pip install -r backend/requirements.txt"
            ) from CHROMA_IMPORT_ERROR
        if not self.persist_directory.exists():
            return None

        self._ensure_client(migrate=False)
        result = self._github.get(ids=[record_id], include=["documents", "metadatas"])
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        if not ids:
            return None
        metadata = metadatas[0] if metadatas else {}
        return {
            "id": str(ids[0]),
            "document": documents[0] if documents else "",
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    def cleanup_github_repositories(self) -> dict[str, int]:
        self._ensure_client()
        if not self._github.count():
            return {"canonicalized": 0, "deleted": 0}

        existing = self._github.get(include=["documents", "metadatas"])
        groups: dict[str, list[dict[str, Any]]] = {}
        for record_id, document, metadata in zip(
            existing.get("ids", []),
            existing.get("documents", []),
            existing.get("metadatas", []),
        ):
            repository = canonical_github_repository(metadata.get("repository"))
            if not repository:
                continue
            groups.setdefault(repository, []).append(
                {"id": record_id, "document": document, "metadata": metadata}
            )

        canonicalized = 0
        deleted_ids = []
        for repository, records in groups.items():
            records.sort(key=lambda record: str(record["metadata"].get("updated_at", "")), reverse=True)
            keep = records[0]
            canonical_id = self._record_id("github", repository)
            if keep["id"] != canonical_id or keep["metadata"].get("repository") != repository:
                payload = json.loads(keep["document"].split("\n", 1)[1])
                document = f"Approved GitHub evidence for {repository}\n{normalized_json(payload)}"
                metadata = {**keep["metadata"], "repository": repository}
                self._github.upsert(
                    ids=[canonical_id],
                    embeddings=[self.embedder.embed(document)],
                    documents=[document],
                    metadatas=[metadata],
                )
                canonicalized += 1
            for record in records:
                if record["id"] != canonical_id:
                    deleted_ids.append(record["id"])

        if deleted_ids:
            self._github.delete(ids=sorted(set(deleted_ids)))
        return {"canonicalized": canonicalized, "deleted": len(set(deleted_ids))}

    def github_count(self) -> int:
        return self._collection_count_read_only(GITHUB_COLLECTION, self._github)
