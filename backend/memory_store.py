"""Chroma-backed vector storage for durable profile memory and GitHub evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
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
TOKEN_PATTERN = re.compile(r"[\w.+#-]+", re.UNICODE)


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


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

    def _ensure_client(self) -> None:
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
        self._migrate_legacy_profile()
        self._migrate_legacy_github()

    def _migrate_legacy_profile(self) -> None:
        if self._profile.count() or not self.legacy_memory_path.exists():
            return
        content = self.legacy_memory_path.read_text(encoding="utf-8").strip()
        if not content:
            return
        try:
            memory = json.loads(content)
        except json.JSONDecodeError:
            memory = {"notes": content}
        if isinstance(memory, dict):
            self._replace_profile(memory, source="legacy-memory-json")

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
                        updated += 1
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
        return {"inserted": inserted, "updated": updated, "unchanged": unchanged}

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

    def profile_count(self) -> int:
        self._ensure_client()
        return self._profile.count()

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

    @staticmethod
    def _repo_key(context: dict[str, Any], index: int) -> str:
        return str(context.get("repository") or context.get("url") or f"repo-{index}")

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

    def github_count(self) -> int:
        self._ensure_client()
        return self._github.count()
