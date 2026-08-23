"""Centralized Chroma storage facade for profile memory and GitHub evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from backend.chroma_http_client_factory import ChromaAccessLifecycle
    from backend.chroma_http_transport import ChromaCollectionMissing
    from backend.chroma_operational_reader import ChromaOperationalReader
    from backend.chroma_read_client import ChromaReadClient
    from backend.chroma_write_client import ChromaWriteClient, ChromaWriteAuthorityViolation
    from backend.chroma_write_models import ChromaWriteRecord
    from backend.project_repository_identity import (
        authority_to_repository_mapping,
        load_project_repository_identity_authority,
        normalize_project_id,
        normalize_repository_identity,
    )
except ModuleNotFoundError:  # pragma: no cover - legacy backend-directory launch
    from chroma_http_client_factory import ChromaAccessLifecycle
    from chroma_http_transport import ChromaCollectionMissing
    from chroma_operational_reader import ChromaOperationalReader
    from chroma_read_client import ChromaReadClient
    from chroma_write_client import ChromaWriteClient, ChromaWriteAuthorityViolation
    from chroma_write_models import ChromaWriteRecord
    from project_repository_identity import (
        authority_to_repository_mapping,
        load_project_repository_identity_authority,
        normalize_project_id,
        normalize_repository_identity,
    )

EMBEDDING_DIMENSIONS = 384
PROFILE_COLLECTION = "profile_facts"
GITHUB_COLLECTION = "github_evidence"
TOKEN_PATTERN = re.compile(r"[\w.+#-]+", re.UNICODE)
GITHUB_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)
GITHUB_REPOSITORY_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$")
MAX_PROFILE_READ_RECORDS = 1_000
MAX_GITHUB_CONTEXT_READ_RECORDS = 1_000
MAX_GITHUB_METADATA_READ_RECORDS = 10_000
PROFILE_READ_METADATA_FIELDS = ("index", "is_list", "section")
GITHUB_READ_METADATA_FIELDS = (
    "github_repository",
    "project_id",
    "project_name",
    "repo",
    "repository",
    "repository_project_id",
    "repository_url",
    "source",
    "updated_at",
)
GITHUB_MUTATION_METADATA_FIELDS = tuple(
    sorted(
        set(GITHUB_READ_METADATA_FIELDS)
        | {
            "chunk_type",
            "commit_sha",
            "path",
            "run_id",
            "source_id",
            "source_type",
        }
    )
)


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
    def __init__(
        self,
        persist_directory: Path,
        legacy_memory_path: Path,
        legacy_github_dir: Path,
        *,
        operational_reader: Any | None = None,
        read_client: Any | None = None,
        write_client: Any | None = None,
        repository_authority_provider: Any | None = None,
    ):
        self.persist_directory = persist_directory
        self.legacy_memory_path = legacy_memory_path
        self.legacy_github_dir = legacy_github_dir
        self.operational_reader = operational_reader if operational_reader is not None else ChromaOperationalReader()
        self.read_client = read_client if read_client is not None else ChromaReadClient()
        self.write_client = write_client if write_client is not None else ChromaWriteClient()
        self.repository_authority_provider = (
            repository_authority_provider
            if repository_authority_provider is not None
            else load_project_repository_identity_authority
        )
        if not callable(self.repository_authority_provider):
            raise TypeError("invalid_repository_authority_provider")
        self.embedder = LocalHashEmbedding()

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

    def _repository_authority(self) -> tuple[Any, dict[str, Any]]:
        authority = self.repository_authority_provider()
        mapping = authority_to_repository_mapping(authority)
        if not mapping.get("mapping_count") or mapping.get("conflicts"):
            raise ChromaWriteAuthorityViolation("github_write_authority_unavailable")
        return authority, mapping

    @staticmethod
    def _github_authority_metadata(
        repository: Any,
        *,
        authority_mapping: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        canonical = normalize_repository_identity(repository)
        project_id = authority_mapping.get("repository_to_project", {}).get(canonical)
        if not canonical or not isinstance(project_id, str) or not project_id:
            raise ChromaWriteAuthorityViolation("github_write_authority_violation")
        for field in ("project_id", "repository_project_id"):
            explicit = normalize_project_id((context or {}).get(field))
            if (context or {}).get(field) is not None and explicit != project_id:
                raise ChromaWriteAuthorityViolation("github_write_authority_violation")
        return {
            "project_id": project_id,
            "repository": canonical,
            "repository_project_id": project_id,
        }

    def _upsert_with_similarity(
        self,
        semantic_collection_id: str,
        records: list[dict[str, Any]],
        *,
        read_consumer_id: str,
        vector_consumer_id: str,
        index_consumer_id: str,
        repository_authority: Any = None,
        authority_mapping: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        inserted = 0
        updated = 0
        unchanged = 0
        deduplicated = 0
        for record in records:
            embedding = self.embedder.embed(record["document"])
            existing = self.read_client.read_records(
                semantic_collection_id,
                consumer_id=read_consumer_id,
                ids=[record["id"]],
                include_documents=True,
                metadata_fields=(),
                max_records=1,
            )
            previous = [item.document for item in existing.records if item.document is not None]
            if previous and previous[0] == record["document"]:
                unchanged += 1
                continue

            similar = self.read_client.vector_query(
                semantic_collection_id,
                consumer_id=vector_consumer_id,
                query_embedding=embedding,
                n_results=3,
                include_documents=True,
                metadata_fields=("repository", "section"),
                include_distances=True,
            )
            for hit in similar.hits:
                if hit.record_id == record["id"] or hit.distance is None or hit.distance > 0.12:
                    continue
                same_section = hit.metadata.get("section") == record["metadata"].get("section")
                same_repository = hit.metadata.get("repository") == record["metadata"].get(
                    "repository"
                )
                if not (same_section or same_repository):
                    continue
                delete_authority = None
                if semantic_collection_id == GITHUB_COLLECTION:
                    if authority_mapping is None:
                        raise ChromaWriteAuthorityViolation("github_write_authority_unavailable")
                    delete_authority = [
                        self._github_authority_metadata(
                            hit.metadata.get("repository"),
                            authority_mapping=authority_mapping,
                        )
                    ]
                self.write_client.delete_records(
                    semantic_collection_id,
                    consumer_id=index_consumer_id,
                    ids=[hit.record_id],
                    lifecycle=ChromaAccessLifecycle.INDEX,
                    authority_metadata=delete_authority,
                    repository_authority=repository_authority,
                )
                deduplicated += 1
                break

            upsert_authority = None
            if semantic_collection_id == GITHUB_COLLECTION:
                upsert_authority = [record["authority_metadata"]]
            self.write_client.upsert_records(
                semantic_collection_id,
                consumer_id=index_consumer_id,
                records=[
                    ChromaWriteRecord(
                        record_id=record["id"],
                        document=record["document"],
                        metadata=record["metadata"],
                        embedding=embedding,
                    )
                ],
                authority_metadata=upsert_authority,
                repository_authority=repository_authority,
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
        existing = self.read_client.read_records(
            PROFILE_COLLECTION,
            consumer_id="profile_memory_reader",
            include_documents=False,
            metadata_fields=(),
            max_records=MAX_PROFILE_READ_RECORDS,
        )
        stale_ids = sorted({record.record_id for record in existing.records} - expected_ids)
        if stale_ids:
            self.write_client.delete_records(
                PROFILE_COLLECTION,
                consumer_id="profile_memory_writer",
                ids=stale_ids,
                lifecycle=ChromaAccessLifecycle.WRITE,
            )
        result = self._upsert_with_similarity(
            PROFILE_COLLECTION,
            records,
            read_consumer_id="profile_memory_reader",
            vector_consumer_id="profile_memory_vector_reader",
            index_consumer_id="profile_memory_indexer",
        )
        result["deleted"] = len(stale_ids)
        return result

    def replace_profile(self, memory: dict[str, Any], source: str = "profile-update") -> dict[str, int]:
        return self._replace_profile(memory, source)

    def profile_count(self) -> int:
        try:
            return self.operational_reader.safe_count(PROFILE_COLLECTION)
        except Exception:
            return 0

    def read_profile(self, query: str = "", limit: int | None = None) -> dict[str, Any]:
        try:
            if query:
                result = self.read_client.vector_query(
                    PROFILE_COLLECTION,
                    consumer_id="profile_memory_vector_reader",
                    query_embedding=self.embedder.embed(query),
                    n_results=limit or 6,
                    include_documents=False,
                    metadata_fields=PROFILE_READ_METADATA_FIELDS,
                    include_distances=False,
                )
                metadatas = [hit.metadata for hit in result.hits]
            else:
                result = self.read_client.read_records(
                    PROFILE_COLLECTION,
                    consumer_id="profile_memory_reader",
                    include_documents=False,
                    metadata_fields=PROFILE_READ_METADATA_FIELDS,
                    max_records=MAX_PROFILE_READ_RECORDS,
                )
                metadatas = [record.metadata for record in result.records]
        except ChromaCollectionMissing:
            return {}
        if not metadatas:
            return {}

        memory: dict[str, Any] = {}
        list_sections: dict[str, list[tuple[int, Any]]] = {}
        for metadata in metadatas:
            section = metadata["section"]
            document_id = self._record_id("profile", section)
            if metadata.get("is_list"):
                candidates = self.read_client.read_records(
                    PROFILE_COLLECTION,
                    consumer_id="profile_memory_reader",
                    where={"section": section},
                    include_documents=True,
                    metadata_fields=PROFILE_READ_METADATA_FIELDS,
                    max_records=MAX_PROFILE_READ_RECORDS,
                )
                items = []
                for candidate in candidates.records:
                    item_metadata = candidate.metadata
                    document = candidate.document or ""
                    payload = json.loads(document.split("\n", 1)[1])
                    items.append((int(item_metadata["index"]), payload))
                list_sections[section] = items
                continue

            selected = self.read_client.read_records(
                PROFILE_COLLECTION,
                consumer_id="profile_memory_reader",
                ids=[document_id],
                include_documents=True,
                metadata_fields=(),
                max_records=1,
            )
            if selected.records and selected.records[0].document is not None:
                memory[section] = json.loads(selected.records[0].document.split("\n", 1)[1])

        for section, items in list_sections.items():
            memory[section] = [value for _, value in sorted(items)]
        return memory

    def delete_profile(
        self,
        section: str,
        item_index: int | None = None,
        delete_section: bool = False,
    ) -> dict[str, Any]:
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

        candidates = self.read_client.read_records(
            PROFILE_COLLECTION,
            consumer_id="profile_memory_reader",
            where={"section": section},
            include_documents=True,
            metadata_fields=PROFILE_READ_METADATA_FIELDS,
            max_records=MAX_PROFILE_READ_RECORDS,
        )
        deleted_ids = []
        deleted_values = []
        for record in candidates.records:
            record_id = record.record_id
            document = record.document or ""
            metadata = record.metadata
            is_target_item = metadata.get("is_list") and int(metadata["index"]) == item_index
            if delete_section or is_target_item:
                deleted_ids.append(record_id)
                deleted_values.append(json.loads(document.split("\n", 1)[1]))

        if deleted_ids:
            self.write_client.delete_records(
                PROFILE_COLLECTION,
                consumer_id="profile_memory_writer",
                ids=deleted_ids,
                lifecycle=ChromaAccessLifecycle.WRITE,
            )
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

    def _store_github_contexts(
        self, contexts: list[dict[str, Any]], source: str
    ) -> dict[str, Any]:
        authority, authority_mapping = self._repository_authority()
        run_id = timestamp_slug()
        records = []
        for index, context in enumerate(contexts):
            if not isinstance(context, dict):
                raise ChromaWriteAuthorityViolation("github_write_authority_violation")
            key = self._repo_key(context, index)
            authority_metadata = self._github_authority_metadata(
                key,
                authority_mapping=authority_mapping,
                context=context,
            )
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
                    "authority_metadata": authority_metadata,
                }
            )
        if len({record["authority_metadata"]["project_id"] for record in records}) > 1:
            raise ChromaWriteAuthorityViolation("github_cross_project_batch_rejected")
        result = self._upsert_with_similarity(
            GITHUB_COLLECTION,
            records,
            read_consumer_id="github_evidence_metadata_reader",
            vector_consumer_id="github_evidence_vector_reader",
            index_consumer_id="github_evidence_materializer",
            repository_authority=authority,
            authority_mapping=authority_mapping,
        )
        result["run_id"] = run_id
        result["cleanup"] = self.cleanup_github_repositories(
            repository_authority=authority,
            authority_mapping=authority_mapping,
        )
        return result

    def store_github_contexts(
        self, contexts: list[dict[str, Any]], source: str = "github-fetch"
    ) -> dict[str, Any]:
        return self._store_github_contexts(contexts, source)

    def read_github_contexts(self, query: str = "", limit: int | None = None) -> list[dict[str, Any]]:
        try:
            if query:
                result = self.read_client.vector_query(
                    GITHUB_COLLECTION,
                    consumer_id="github_evidence_vector_reader",
                    query_embedding=self.embedder.embed(query),
                    n_results=limit or 8,
                    include_documents=True,
                    metadata_fields=(),
                    include_distances=False,
                )
                documents = [hit.document for hit in result.hits if hit.document is not None]
            else:
                result = self.read_client.read_records(
                    GITHUB_COLLECTION,
                    consumer_id="github_evidence_metadata_reader",
                    include_documents=True,
                    metadata_fields=(),
                    max_records=MAX_GITHUB_CONTEXT_READ_RECORDS,
                )
                documents = [
                    record.document for record in result.records if record.document is not None
                ]
        except ChromaCollectionMissing:
            return []
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
        try:
            result = self.read_client.read_records(
                GITHUB_COLLECTION,
                consumer_id="github_evidence_metadata_reader",
                include_documents=False,
                metadata_fields=GITHUB_READ_METADATA_FIELDS,
                max_records=min(limit, MAX_GITHUB_METADATA_READ_RECORDS),
            )
        except Exception:
            return []
        return [
            {"vector_record_id": record.record_id, "metadata": dict(record.metadata)}
            for record in result.records
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
        try:
            status = self.operational_reader.read_collection_status(
                GITHUB_COLLECTION,
                include_repository_inventory=True,
            )
        except Exception:
            return {"available": False, "count": 0, "repositories": []}
        repositories = [
            {
                "repository": item.repository,
                "updated_at": getattr(item, "updated_at", None) or "",
            }
            for item in status.repositories
        ]
        repositories.sort(key=lambda item: item["updated_at"], reverse=True)
        return {
            "available": status.available,
            "count": status.safe_record_count if status.available else 0,
            "repositories": repositories,
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
        if not isinstance(record_id, str) or not record_id:
            return None
        try:
            result = self.read_client.read_records(
                GITHUB_COLLECTION,
                consumer_id="github_evidence_metadata_reader",
                ids=[record_id],
                include_documents=True,
                metadata_fields=GITHUB_READ_METADATA_FIELDS,
                max_records=1,
            )
        except ChromaCollectionMissing:
            return None
        if not result.records:
            return None
        record = result.records[0]
        return {
            "id": record.record_id,
            "document": record.document or "",
            "metadata": dict(record.metadata),
        }

    def cleanup_github_repositories(
        self,
        *,
        repository_authority: Any = None,
        authority_mapping: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        if repository_authority is None or authority_mapping is None:
            repository_authority, authority_mapping = self._repository_authority()
        existing = self.read_client.read_records(
            GITHUB_COLLECTION,
            consumer_id="github_evidence_metadata_reader",
            include_documents=True,
            metadata_fields=GITHUB_MUTATION_METADATA_FIELDS,
            max_records=MAX_GITHUB_CONTEXT_READ_RECORDS,
        )
        if not existing.records:
            return {"canonicalized": 0, "deleted": 0}

        groups: dict[str, list[dict[str, Any]]] = {}
        for item in existing.records:
            repository = canonical_github_repository(item.metadata.get("repository"))
            if not repository:
                continue
            groups.setdefault(repository, []).append(
                {
                    "id": item.record_id,
                    "document": item.document or "",
                    "metadata": dict(item.metadata),
                }
            )

        plans = []
        for repository, records in groups.items():
            authority_metadata = self._github_authority_metadata(
                repository,
                authority_mapping=authority_mapping,
            )
            records.sort(
                key=lambda record: str(record["metadata"].get("updated_at", "")),
                reverse=True,
            )
            keep = records[0]
            canonical_id = self._record_id("github", repository)
            canonical_record = None
            if keep["id"] != canonical_id or keep["metadata"].get("repository") != repository:
                payload = json.loads(keep["document"].split("\n", 1)[1])
                document = f"Approved GitHub evidence for {repository}\n{normalized_json(payload)}"
                canonical_record = ChromaWriteRecord(
                    record_id=canonical_id,
                    document=document,
                    metadata={**keep["metadata"], "repository": repository},
                    embedding=self.embedder.embed(document),
                )
            delete_ids = sorted(
                {record["id"] for record in records if record["id"] != canonical_id}
            )
            plans.append(
                {
                    "authority": authority_metadata,
                    "canonical_record": canonical_record,
                    "delete_ids": delete_ids,
                }
            )

        canonicalized = 0
        delete_groups: dict[str, list[tuple[str, dict[str, str]]]] = {}
        for plan in plans:
            if plan["canonical_record"] is not None:
                self.write_client.upsert_records(
                    GITHUB_COLLECTION,
                    consumer_id="github_evidence_materializer",
                    records=[plan["canonical_record"]],
                    authority_metadata=[plan["authority"]],
                    repository_authority=repository_authority,
                )
                canonicalized += 1
            project_id = plan["authority"]["project_id"]
            delete_groups.setdefault(project_id, []).extend(
                (record_id, plan["authority"]) for record_id in plan["delete_ids"]
            )

        deleted = 0
        for items in delete_groups.values():
            if not items:
                continue
            ids = [record_id for record_id, _ in items]
            self.write_client.delete_records(
                GITHUB_COLLECTION,
                consumer_id="github_evidence_materializer",
                ids=ids,
                lifecycle=ChromaAccessLifecycle.INDEX,
                authority_metadata=[metadata for _, metadata in items],
                repository_authority=repository_authority,
            )
            deleted += len(ids)
        return {"canonicalized": canonicalized, "deleted": deleted}

    def github_count(self) -> int:
        try:
            return self.operational_reader.safe_count(GITHUB_COLLECTION)
        except Exception:
            return 0
