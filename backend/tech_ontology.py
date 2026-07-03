"""Compact technical terminology ontology helpers for resume tailoring."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_TECH_ONTOLOGY_PATH = Path(__file__).resolve().parent / "data" / "tech_ontology.jsonl"
TECH_ONTOLOGY_MAX_PROMPT_ENTRIES = 10
TECH_ONTOLOGY_MAX_ENTRY_CHARS = 800
TECH_ONTOLOGY_MAX_SAFE_PHRASES = 5
TECH_ONTOLOGY_MAX_WEAK_PHRASES = 3
TECH_ONTOLOGY_MAX_FORBIDDEN_CLAIMS = 5


def _string_list(value: Any, limit: int = 20, max_chars: int = 120) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip()
        if text not in items:
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _clean_entry(entry: dict[str, Any]) -> dict[str, Any] | None:
    term = re.sub(r"\s+", " ", str(entry.get("term") or "")).strip()
    if not term:
        return None
    return {
        "term": term,
        "aliases": _string_list(entry.get("aliases"), 12, 80),
        "category": re.sub(r"\s+", " ", str(entry.get("category") or "")).strip(),
        "resume_category": re.sub(r"\s+", " ", str(entry.get("resume_category") or "")).strip(),
        "role_families": _string_list(entry.get("role_families"), 8, 60),
        "related_terms": _string_list(entry.get("related_terms"), 10, 80),
        "safe_resume_phrases": _string_list(
            entry.get("safe_resume_phrases"),
            TECH_ONTOLOGY_MAX_SAFE_PHRASES,
            120,
        ),
        "weak_evidence_phrases": _string_list(
            entry.get("weak_evidence_phrases"),
            TECH_ONTOLOGY_MAX_WEAK_PHRASES,
            120,
        ),
        "do_not_claim_without_evidence": _string_list(
            entry.get("do_not_claim_without_evidence"),
            TECH_ONTOLOGY_MAX_FORBIDDEN_CLAIMS,
            140,
        ),
        "common_jd_contexts": _string_list(entry.get("common_jd_contexts"), 8, 100),
    }


@lru_cache(maxsize=8)
def _load_tech_ontology_cached(path_key: str) -> tuple[tuple[tuple[str, Any], ...], ...]:
    target = Path(path_key)
    entries: list[dict] = []
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return tuple()
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        cleaned = _clean_entry(payload)
        if cleaned:
            entries.append(cleaned)
    return tuple(tuple(entry.items()) for entry in entries)


def load_tech_ontology(path: str | None = None) -> list[dict]:
    """
    Load backend/data/tech_ontology.jsonl.
    Return [] safely if the file is missing or malformed.
    Do not crash resume generation if ontology loading fails.
    """

    target = str(Path(path) if path else DEFAULT_TECH_ONTOLOGY_PATH)
    entries = []
    for cached_entry in _load_tech_ontology_cached(target):
        entry = dict(cached_entry)
        for key in [
            "aliases",
            "role_families",
            "related_terms",
            "safe_resume_phrases",
            "weak_evidence_phrases",
            "do_not_claim_without_evidence",
            "common_jd_contexts",
        ]:
            if isinstance(entry.get(key), list):
                entry[key] = list(entry[key])
        entries.append(entry)
    return entries


def normalize_tech_term(term: str) -> str:
    """
    Normalize technology terms for matching.
    Lowercase, strip punctuation, normalize whitespace.
    Preserve meaningful symbols like C++, C#, HTML/CSS when needed.
    """

    text = str(term or "").lower()
    text = text.replace("\\", "/")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-z0-9+#./\s-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    return text


def build_tech_ontology_index(entries: list[dict]) -> dict:
    """
    Build exact-match and alias lookup maps.
    Include term and aliases.
    """

    exact: dict[str, dict] = {}
    alias_to_term: dict[str, str] = {}
    canonical: dict[str, dict] = {}
    for raw_entry in entries or []:
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        term = str(entry.get("term") or "").strip()
        if not term:
            continue
        canonical[term.lower()] = entry
        labels = [term] + [str(alias) for alias in entry.get("aliases", []) if str(alias).strip()]
        for label in labels:
            normalized = normalize_tech_term(label)
            if not normalized:
                continue
            exact.setdefault(normalized, entry)
            alias_to_term[normalized] = term
    return {
        "exact": exact,
        "alias_to_term": alias_to_term,
        "canonical": canonical,
    }


def _normalized_term_in_text(normalized_term: str, normalized_text: str) -> bool:
    if not normalized_term or not normalized_text:
        return False
    if normalized_term == "c":
        pattern = r"(?<![a-z0-9+#./-])c(?![a-z0-9+#./-])"
    else:
        pattern = rf"(?<![a-z0-9+#./-]){re.escape(normalized_term)}(?![a-z0-9+#./-])"
    return bool(re.search(pattern, normalized_text))


def extract_possible_tech_terms(text: str) -> list[str]:
    """
    Extract possible technology/tool/platform terms from JD, resume, project evidence, and skills.
    Use exact known ontology terms and aliases first.
    Avoid treating every capitalized business word as a technology.
    """

    entries = load_tech_ontology()
    index = build_tech_ontology_index(entries)
    normalized_text = normalize_tech_term(str(text or ""))
    terms: list[str] = []
    for normalized, entry in sorted(index.get("exact", {}).items(), key=lambda item: (-len(item[0]), item[0])):
        if _normalized_term_in_text(normalized, normalized_text):
            term = str(entry.get("term") or "").strip()
            if term and term not in terms:
                terms.append(term)
    return terms


def _compact_tech_entry(entry: dict[str, Any]) -> dict[str, Any]:
    compact = _clean_entry(entry) or {}
    compact["safe_resume_phrases"] = compact.get("safe_resume_phrases", [])[:TECH_ONTOLOGY_MAX_SAFE_PHRASES]
    compact["weak_evidence_phrases"] = compact.get("weak_evidence_phrases", [])[:TECH_ONTOLOGY_MAX_WEAK_PHRASES]
    compact["do_not_claim_without_evidence"] = compact.get("do_not_claim_without_evidence", [])[
        :TECH_ONTOLOGY_MAX_FORBIDDEN_CLAIMS
    ]
    while len(json.dumps(compact, ensure_ascii=False)) > TECH_ONTOLOGY_MAX_ENTRY_CHARS:
        trimmed = False
        for key in ["common_jd_contexts", "related_terms", "safe_resume_phrases", "do_not_claim_without_evidence"]:
            if isinstance(compact.get(key), list) and len(compact[key]) > 1:
                compact[key] = compact[key][:-1]
                trimmed = True
                break
        if not trimmed:
            break
    return compact


def retrieve_tech_context_for_terms(
    terms: list[str],
    ontology_entries: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """
    Retrieve matching ontology entries using exact term and alias matching.
    Return compact entries only.
    Deduplicate by canonical term.
    """

    index = build_tech_ontology_index(ontology_entries or [])
    results: list[dict] = []
    seen: set[str] = set()
    for term in terms or []:
        normalized = normalize_tech_term(str(term or ""))
        entry = index.get("exact", {}).get(normalized)
        if not entry:
            continue
        canonical = str(entry.get("term") or "").strip()
        if not canonical or canonical.lower() in seen:
            continue
        seen.add(canonical.lower())
        results.append(_compact_tech_entry(entry))
        if len(results) >= top_k:
            break
    return results


def build_tech_ontology_documents(entries: list[dict]) -> list[dict]:
    """Convert ontology entries into compact searchable documents."""

    documents = []
    for entry in entries or []:
        compact = _compact_tech_entry(entry)
        if not compact.get("term"):
            continue
        text = "\n".join(
            [
                f"Term: {compact.get('term')}",
                f"Category: {compact.get('category')}",
                f"Resume Category: {compact.get('resume_category')}",
                "Related: " + ", ".join(compact.get("related_terms", [])),
                "Safe Phrases: " + ", ".join(compact.get("safe_resume_phrases", [])),
                "Do Not Claim Without Evidence: " + ", ".join(compact.get("do_not_claim_without_evidence", [])),
            ]
        )
        documents.append({"term": compact["term"], "text": text, "metadata": compact})
    return documents


def enrich_jd_profile_with_tech_ontology(
    jd_profile: dict,
    jd_text: str,
    ontology_entries: list[dict] | None = None,
) -> dict:
    """
    Attach technology categories, resume categories, role-family relevance,
    safe phrases, weak-evidence phrases, and forbidden unsupported claims
    to the JD profile.
    """

    enriched = dict(jd_profile or {})
    entries = ontology_entries if ontology_entries is not None else load_tech_ontology()
    if not entries:
        enriched.setdefault("detected_technologies", [])
        enriched.setdefault("forbidden_unsupported_claims", [])
        enriched.setdefault("safe_weak_wording", {})
        enriched.setdefault("tech_ontology", {"detected_technologies": []})
        return enriched

    query_text = "\n".join(
        [
            str(jd_text or ""),
            json.dumps(jd_profile or {}, ensure_ascii=False),
        ]
    )
    terms = extract_possible_tech_terms(query_text)
    tech_context = retrieve_tech_context_for_terms(terms, entries, TECH_ONTOLOGY_MAX_PROMPT_ENTRIES)
    detected = []
    forbidden: list[str] = []
    safe_weak: dict[str, dict[str, list[str]]] = {}
    for entry in tech_context:
        term = str(entry.get("term") or "").strip()
        if not term:
            continue
        detected.append(
            {
                "term": term,
                "category": entry.get("category", ""),
                "resume_category": entry.get("resume_category", ""),
                "role_families": entry.get("role_families", [])[:6],
                "safe_resume_phrases": entry.get("safe_resume_phrases", [])[:TECH_ONTOLOGY_MAX_SAFE_PHRASES],
                "weak_evidence_phrases": entry.get("weak_evidence_phrases", [])[:TECH_ONTOLOGY_MAX_WEAK_PHRASES],
                "do_not_claim_without_evidence": entry.get("do_not_claim_without_evidence", [])[
                    :TECH_ONTOLOGY_MAX_FORBIDDEN_CLAIMS
                ],
            }
        )
        for claim in entry.get("do_not_claim_without_evidence", [])[:TECH_ONTOLOGY_MAX_FORBIDDEN_CLAIMS]:
            if claim not in forbidden:
                forbidden.append(claim)
        safe_weak[term] = {
            "safe_resume_phrases": entry.get("safe_resume_phrases", [])[:TECH_ONTOLOGY_MAX_SAFE_PHRASES],
            "weak_evidence_phrases": entry.get("weak_evidence_phrases", [])[:TECH_ONTOLOGY_MAX_WEAK_PHRASES],
        }

    tech_payload = {
        "detected_technologies": detected,
        "forbidden_unsupported_claims": forbidden[: TECH_ONTOLOGY_MAX_PROMPT_ENTRIES * 2],
        "safe_weak_wording": safe_weak,
    }
    enriched["detected_technologies"] = detected
    enriched["forbidden_unsupported_claims"] = tech_payload["forbidden_unsupported_claims"]
    enriched["safe_weak_wording"] = safe_weak
    enriched["tech_ontology"] = tech_payload
    return enriched
