"""Pure offline intelligence for explicit job context and role families.

The public builder adapts existing normalized job-analysis values and bounded
offline organization rules into the authoritative hiring-context contracts.
It intentionally does not parse raw job-description text, inspect candidate
data, rank projects or stories, or perform I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from backend.hiring_context_models import (
    MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS,
    MAX_HIRING_CONTEXT_NAME_LENGTH,
    MAX_HIRING_CONTEXT_RANKING_EFFECTS,
    MAX_HIRING_CONTEXT_ROLE_TITLE_LENGTH,
    MAX_HIRING_CONTEXT_SIGNALS,
    MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS,
    MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
    HiringContextConfidence,
    HiringContextProfile,
    HiringContextSignal,
    HiringContextSignalKind,
    HiringContextSourceKind,
    HiringContextSourceRef,
    RankingEffect,
    RoleFamily,
)
from backend.hiring_context_organization import (
    DEFAULT_ORGANIZATION_CONTEXT_REGISTRY,
    OrganizationContextRegistry,
    OrganizationContextResolution,
    OrganizationContextScope,
    resolve_organization_context,
)


MAX_NORMALIZED_JOB_VALUES_PER_FIELD = 128
MAX_NORMALIZED_JOB_CONTEXT_VALUES = 512

_CONTROL_VALUES = frozenset({
    "description",
    "experience with",
    "metadata",
    "nice to have",
    "preferred",
    "project",
    "requirement",
    "requirements",
    "responsibilities",
    "responsibility",
    "skill",
    "skills",
    "technologies",
    "technology",
    "value",
})

_SKILL_REQUIREMENT_FIELDS = (
    "languages",
    "frameworks",
    "cloudInfra",
    "databases",
    "devops",
    "testing",
    "automation",
    "aiMl",
)

_SOFT_SKILL_REQUIREMENT_FIELDS = ("softSkills",)

_TECHNOLOGY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("JavaScript", ("javascript", "java script", "ecmascript", "js")),
    ("TypeScript", ("typescript", "type script")),
    ("C++", ("c++", "cpp", "c plus plus")),
    ("C#", ("c#", "c sharp", "csharp")),
    ("C", ("c", "c language")),
    ("Java", ("java", "java se")),
    ("Python", ("python", "python3")),
    ("Go", ("go", "golang")),
    ("R", ("r", "r language")),
    ("Rails", ("rails", "ruby on rails")),
    ("React", ("react", "react.js", "reactjs")),
    ("Vue", ("vue", "vue.js", "vuejs")),
    ("Angular", ("angular",)),
    ("Node.js", ("node.js", "nodejs", "node js")),
    ("FastAPI", ("fastapi", "fast api")),
    ("Django", ("django",)),
    ("Flask", ("flask",)),
    ("Spring", ("spring", "spring boot")),
    ("REST API", ("rest api", "restful api", "rest apis", "restful apis")),
    ("GraphQL", ("graphql",)),
    ("SQL", ("sql", "structured query language")),
    ("PostgreSQL", ("postgresql", "postgres")),
    ("MySQL", ("mysql",)),
    ("SQLite", ("sqlite", "sqlite3")),
    ("MongoDB", ("mongodb", "mongo db")),
    ("Redis", ("redis",)),
    ("AWS", ("aws", "amazon web services")),
    ("Azure", ("azure",)),
    ("GCP", ("gcp", "google cloud", "google cloud platform")),
    ("Docker", ("docker",)),
    ("Kubernetes", ("kubernetes", "k8s")),
    ("Terraform", ("terraform",)),
    ("CI/CD", ("ci/cd", "continuous integration", "continuous delivery")),
    ("GitHub Actions", ("github actions",)),
    ("Jenkins", ("jenkins",)),
    ("Linux", ("linux",)),
    ("Android", ("android",)),
    ("iOS", ("ios",)),
    ("Swift", ("swift",)),
    ("Kotlin", ("kotlin",)),
    ("Unity", ("unity",)),
    ("Unreal Engine", ("unreal engine", "unreal")),
    ("Airflow", ("airflow", "apache airflow")),
    ("Spark", ("spark", "apache spark")),
    ("Kafka", ("kafka", "apache kafka")),
    ("dbt", ("dbt",)),
    ("Tableau", ("tableau",)),
    ("Power BI", ("power bi",)),
    ("TensorFlow", ("tensorflow",)),
    ("PyTorch", ("pytorch",)),
    ("Machine learning", ("machine learning", "ml")),
    ("AI", ("artificial intelligence", "ai")),
    ("LLM", ("large language model", "large language models", "llm", "llms")),
)

_TRAIT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("API design", ("api design", "api development", "rest api", "restful api")),
    ("Collaboration", ("collaboration", "cross functional", "cross-functional")),
    ("Data pipelines", ("data pipeline", "data pipelines", "etl")),
    ("Debugging", ("debugging", "troubleshooting")),
    ("Performance", ("performance", "latency", "optimization", "optimisation")),
    ("Real-time systems", ("real time systems", "real-time systems", "real time interactive")),
    ("Reliability", ("reliability", "resilience", "fault tolerance", "high availability")),
    ("Testing", ("testing", "unit test", "unit testing", "integration test", "integration testing")),
)

_TITLE_RULES: dict[RoleFamily, tuple[tuple[str, int], ...]] = {
    RoleFamily.FULL_STACK_ENGINEERING: (("full stack", 9), ("fullstack", 9)),
    RoleFamily.BACKEND_ENGINEERING: (("backend", 8), ("back end", 8), ("server side", 7), ("api engineer", 8)),
    RoleFamily.FRONTEND_ENGINEERING: (("frontend", 8), ("front end", 8), ("ui engineer", 7)),
    RoleFamily.DATA_ENGINEERING: (("data engineer", 9), ("data platform", 8)),
    RoleFamily.DATA_ANALYTICS: (("data analyst", 9), ("business intelligence", 8), ("analytics analyst", 8), ("analytics engineer", 8)),
    RoleFamily.MACHINE_LEARNING_AI: (("machine learning", 9), ("ml engineer", 9), ("ai engineer", 9), ("data scientist", 8)),
    RoleFamily.DEVOPS_CLOUD: (("devops", 9), ("site reliability", 9), ("sre", 8), ("cloud engineer", 8), ("infrastructure engineer", 8), ("platform engineer", 7)),
    RoleFamily.GAME_DEVELOPMENT: (("game developer", 9), ("game engineer", 9), ("gameplay", 9), ("engine programmer", 8)),
    RoleFamily.MOBILE: (("mobile engineer", 9), ("mobile developer", 9), ("android developer", 9), ("ios developer", 9)),
    RoleFamily.EMBEDDED_SYSTEMS: (("embedded", 9), ("firmware", 9)),
    RoleFamily.SECURITY: (("security engineer", 9), ("cybersecurity", 9), ("technology risk", 8), ("privacy engineer", 8)),
    RoleFamily.CONSULTING_STRATEGY: (("consultant", 8), ("consulting", 8), ("strategy", 8), ("advisory", 7), ("business analyst", 7)),
    RoleFamily.SOFTWARE_ENGINEERING: (("software engineer", 5), ("software engineering", 5), ("software developer", 5), ("application developer", 4), ("sde", 5)),
}

_JOB_RULES: dict[RoleFamily, tuple[tuple[str, int], ...]] = {
    RoleFamily.BACKEND_ENGINEERING: (("backend", 3), ("server side", 3), ("rest api", 3), ("api development", 3), ("microservices", 3), ("fastapi", 2), ("django", 2), ("flask", 2), ("node.js", 1), ("database", 1)),
    RoleFamily.FRONTEND_ENGINEERING: (("frontend", 3), ("browser ui", 3), ("user interface", 2), ("react", 2), ("vue", 2), ("angular", 2), ("typescript", 1), ("javascript", 1), ("html", 1), ("css", 1)),
    RoleFamily.FULL_STACK_ENGINEERING: (("full stack", 7), ("fullstack", 7), ("end to end web", 4), ("end-to-end web", 4)),
    RoleFamily.DATA_ENGINEERING: (("data engineering", 4), ("data pipeline", 3), ("etl", 3), ("data warehouse", 3), ("data platform", 3), ("airflow", 2), ("spark", 2), ("kafka", 2), ("dbt", 2)),
    RoleFamily.DATA_ANALYTICS: (("data analytics", 4), ("data analysis", 3), ("analytics", 2), ("reporting", 2), ("dashboard", 2), ("business intelligence", 3), ("tableau", 2), ("power bi", 2), ("kpi", 2)),
    RoleFamily.MACHINE_LEARNING_AI: (("machine learning", 4), ("artificial intelligence", 4), ("ml", 3), ("ai", 3), ("data science", 3), ("pytorch", 2), ("tensorflow", 2), ("llm", 2), ("nlp", 2), ("computer vision", 2)),
    RoleFamily.DEVOPS_CLOUD: (("devops", 4), ("infrastructure", 3), ("kubernetes", 3), ("terraform", 3), ("ci/cd", 3), ("observability", 3), ("site reliability", 4), ("docker", 2), ("cloud", 2), ("deployment", 2)),
    RoleFamily.GAME_DEVELOPMENT: (("game development", 5), ("gameplay", 4), ("game engine", 4), ("unity", 3), ("unreal engine", 3), ("real time interactive", 3), ("real-time interactive", 3)),
    RoleFamily.MOBILE: (("mobile development", 4), ("android", 3), ("ios", 3), ("swift", 2), ("react native", 3), ("flutter", 3)),
    RoleFamily.EMBEDDED_SYSTEMS: (("embedded systems", 5), ("embedded", 4), ("firmware", 4), ("microcontroller", 3), ("rtos", 3), ("device driver", 3)),
    RoleFamily.SECURITY: (("cybersecurity", 5), ("security", 3), ("privacy", 3), ("technology risk", 4), ("cyber transformation", 4), ("data controls", 3), ("vulnerability", 3), ("threat", 2), ("zero trust", 3)),
    RoleFamily.CONSULTING_STRATEGY: (("consulting", 4), ("strategy", 4), ("advisory", 3), ("transformation", 3), ("stakeholder", 2), ("business requirements", 2), ("decision support", 2)),
    RoleFamily.SOFTWARE_ENGINEERING: (("software development", 3), ("software engineering", 3), ("application development", 2), ("programming", 2)),
}

_SPECIFIC_FAMILY_ORDER = (
    RoleFamily.FULL_STACK_ENGINEERING,
    RoleFamily.BACKEND_ENGINEERING,
    RoleFamily.FRONTEND_ENGINEERING,
    RoleFamily.DATA_ENGINEERING,
    RoleFamily.DATA_ANALYTICS,
    RoleFamily.MACHINE_LEARNING_AI,
    RoleFamily.DEVOPS_CLOUD,
    RoleFamily.GAME_DEVELOPMENT,
    RoleFamily.MOBILE,
    RoleFamily.EMBEDDED_SYSTEMS,
    RoleFamily.SECURITY,
    RoleFamily.CONSULTING_STRATEGY,
    RoleFamily.SOFTWARE_ENGINEERING,
)


@dataclass(frozen=True, slots=True)
class _NormalizedJobContext:
    responsibilities: tuple[str, ...]
    required_qualifications: tuple[str, ...]
    required_skills: tuple[str, ...]
    technologies: tuple[str, ...]
    preferred_qualifications: tuple[str, ...]
    context_values: tuple[str, ...]

    @property
    def all_values(self) -> tuple[str, ...]:
        return _stable_texts(
            self.responsibilities
            + self.required_qualifications
            + self.required_skills
            + self.technologies
            + self.preferred_qualifications
            + self.context_values
        )

    def fingerprint_payload(self) -> dict[str, list[str]]:
        return {
            "responsibilities": list(self.responsibilities),
            "required_qualifications": list(self.required_qualifications),
            "required_skills": list(self.required_skills),
            "technologies": list(self.technologies),
            "preferred_qualifications": list(self.preferred_qualifications),
            "context_values": list(self.context_values),
        }


@dataclass(frozen=True, slots=True)
class _RoleClassification:
    primary: RoleFamily
    secondary: tuple[RoleFamily, ...]
    confidence: HiringContextConfidence
    title_scores: Mapping[RoleFamily, int]
    job_scores: Mapping[RoleFamily, int]


@dataclass(frozen=True, slots=True)
class _SignalCandidate:
    priority: int
    value: str
    kind: HiringContextSignalKind
    confidence: HiringContextConfidence
    effects: tuple[RankingEffect, ...]
    source_refs: tuple[HiringContextSourceRef, ...]


def _normalized_text(value: Any, name: str, maximum: int, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(value.split())
    if required and not normalized:
        raise ValueError(f"{name} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"{name} exceeds maximum length {maximum}")
    return normalized


def _optional_text(value: Any, name: str, maximum: int) -> str | None:
    if value is None:
        return None
    normalized = _normalized_text(value, name, maximum)
    return normalized or None


def _stable_texts(values: Sequence[str]) -> tuple[str, ...]:
    normalized: dict[str, str] = {}
    for value in values:
        key = value.casefold()
        current = normalized.get(key)
        if current is None or value < current:
            normalized[key] = value
    return tuple(normalized[key] for key in sorted(normalized))


def _sequence_field(container: Mapping[str, Any], key: str, path: str) -> tuple[str, ...]:
    value = container.get(key)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path}.{key} must be an array of strings")
    if len(value) > MAX_NORMALIZED_JOB_VALUES_PER_FIELD:
        raise ValueError(
            f"{path}.{key} exceeds maximum item count "
            f"{MAX_NORMALIZED_JOB_VALUES_PER_FIELD}"
        )
    items = []
    for item in value:
        text = _normalized_text(
            item,
            f"{path}.{key}",
            MAX_HIRING_CONTEXT_SIGNAL_VALUE_LENGTH,
            required=True,
        )
        if text.casefold() not in _CONTROL_VALUES:
            items.append(text)
    return _stable_texts(items)


def _mapping_field(container: Mapping[str, Any], key: str, path: str) -> Mapping[str, Any]:
    value = container.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{path}.{key} must be an object")
    return value


def _merged_fields(*values: tuple[str, ...]) -> tuple[str, ...]:
    return _stable_texts(tuple(item for group in values for item in group))


def _coalesce_responsibilities(
    core_values: Sequence[tuple[str, tuple[str, ...]]],
    alias_values: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[str, ...]:
    populated_core = [(name, value) for name, value in core_values if value]
    populated_alias = [(name, value) for name, value in alias_values if value]
    all_populated = populated_core + populated_alias
    if not all_populated:
        return ()
    expected = all_populated[0][1]
    expected_keys = tuple(item.casefold() for item in expected)
    conflicts = [
        name
        for name, value in all_populated[1:]
        if tuple(item.casefold() for item in value) != expected_keys
    ]
    if conflicts:
        names = ", ".join(name for name, _ in all_populated)
        raise ValueError(f"conflicting responsibilities aliases: {names}")
    return expected


def _normalize_job_context(value: Mapping[str, Any]) -> _NormalizedJobContext:
    if not isinstance(value, Mapping):
        raise TypeError("normalized_job_context must be an object")
    target = _mapping_field(value, "target_role", "normalized_job_context")
    skill_requirements = _mapping_field(value, "skill_requirements", "normalized_job_context")
    requirements_value = value.get("requirements")
    requirements = requirements_value if isinstance(requirements_value, Mapping) else {}
    direct_requirements = (
        _sequence_field(value, "requirements", "normalized_job_context")
        if requirements_value is not None and not isinstance(requirements_value, Mapping)
        else ()
    )

    core_sources = (
        ("normalized_job_context.core_responsibilities", _sequence_field(value, "core_responsibilities", "normalized_job_context")),
        ("normalized_job_context.target_role.core_responsibilities", _sequence_field(target, "core_responsibilities", "normalized_job_context.target_role")),
    )
    alias_sources = (
        ("normalized_job_context.responsibilities", _sequence_field(value, "responsibilities", "normalized_job_context")),
        ("normalized_job_context.target_role.responsibilities", _sequence_field(target, "responsibilities", "normalized_job_context.target_role")),
        ("normalized_job_context.requirements.responsibilities", _sequence_field(requirements, "responsibilities", "normalized_job_context.requirements")),
    )
    responsibilities = _coalesce_responsibilities(core_sources, alias_sources)

    required_qualifications = _merged_fields(
        direct_requirements,
        _sequence_field(value, "required_qualifications", "normalized_job_context"),
        _sequence_field(value, "qualifications", "normalized_job_context"),
        _sequence_field(requirements, "required_qualifications", "normalized_job_context.requirements"),
        _sequence_field(requirements, "qualifications", "normalized_job_context.requirements"),
    )
    required_skills = _merged_fields(
        _sequence_field(value, "skills", "normalized_job_context"),
        _sequence_field(value, "required_skills", "normalized_job_context"),
        _sequence_field(value, "must_have_skills", "normalized_job_context"),
        _sequence_field(requirements, "must_have_skills", "normalized_job_context.requirements"),
        _sequence_field(target, "must_have_keywords", "normalized_job_context.target_role"),
    )
    technologies = _merged_fields(
        _sequence_field(value, "technologies", "normalized_job_context"),
        _sequence_field(requirements, "tools_platforms", "normalized_job_context.requirements"),
        _sequence_field(target, "testing_ci_infrastructure_keywords", "normalized_job_context.target_role"),
        *(
            _sequence_field(
                skill_requirements,
                key,
                "normalized_job_context.skill_requirements",
            )
            for key in _SKILL_REQUIREMENT_FIELDS
        ),
    )
    preferred_qualifications = _merged_fields(
        _sequence_field(value, "preferred_qualifications", "normalized_job_context"),
        _sequence_field(value, "preferred_skills", "normalized_job_context"),
        _sequence_field(requirements, "preferred_qualifications", "normalized_job_context.requirements"),
        _sequence_field(requirements, "preferred_skills", "normalized_job_context.requirements"),
    )
    context_values = _merged_fields(
        _sequence_field(value, "engineering_traits", "normalized_job_context"),
        _sequence_field(requirements, "domain_knowledge", "normalized_job_context.requirements"),
        _sequence_field(requirements, "soft_skills", "normalized_job_context.requirements"),
        *(
            _sequence_field(
                skill_requirements,
                key,
                "normalized_job_context.skill_requirements",
            )
            for key in _SOFT_SKILL_REQUIREMENT_FIELDS
        ),
    )
    normalized = _NormalizedJobContext(
        responsibilities=responsibilities,
        required_qualifications=required_qualifications,
        required_skills=required_skills,
        technologies=technologies,
        preferred_qualifications=preferred_qualifications,
        context_values=context_values,
    )
    if sum(len(group) for group in (
        normalized.responsibilities,
        normalized.required_qualifications,
        normalized.required_skills,
        normalized.technologies,
        normalized.preferred_qualifications,
        normalized.context_values,
    )) > MAX_NORMALIZED_JOB_CONTEXT_VALUES:
        raise ValueError(
            "normalized_job_context exceeds maximum semantic item count "
            f"{MAX_NORMALIZED_JOB_CONTEXT_VALUES}"
        )
    return normalized


def _match_text(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"[\u2010-\u2015_]", "-", text)
    text = re.sub(r"[^a-z0-9+#./\s-]+", " ", text)
    return " ".join(text.split())


def _phrase_present(phrase: str, value: str) -> bool:
    normalized_phrase = _match_text(phrase)
    normalized_value = _match_text(value)
    if not normalized_phrase or not normalized_value:
        return False
    escaped = re.escape(normalized_phrase).replace(r"\ ", r"[\s-]+")
    pattern = rf"(?<![a-z0-9+#]){escaped}(?![a-z0-9+#])"
    return bool(re.search(pattern, normalized_value))


def _ambiguous_alias_present(canonical: str, alias: str, value: str, *, atomic: bool) -> bool:
    normalized = _match_text(value)
    if atomic and normalized == _match_text(alias):
        return True
    if canonical == "AI":
        return (
            _phrase_present(alias, value)
            if alias != "ai"
            else bool(re.search(r"(?<![A-Za-z0-9])AI(?![A-Za-z0-9])", value))
        )
    if canonical == "C":
        return alias == "c language" and _phrase_present(alias, value)
    if canonical == "Go":
        return (
            _phrase_present(alias, value)
            if alias == "golang"
            else bool(re.search(r"(?<![A-Za-z0-9])Go(?![A-Za-z0-9])", value))
        )
    if canonical == "R":
        return alias == "r language" and _phrase_present(alias, value)
    return False


def _canonical_terms(value: str, *, atomic: bool) -> tuple[str, ...]:
    matches = []
    for canonical, aliases in _TECHNOLOGY_ALIASES:
        for alias in aliases:
            if canonical in {"AI", "C", "Go", "R"}:
                present = _ambiguous_alias_present(canonical, alias, value, atomic=atomic)
            else:
                present = _phrase_present(alias, value)
            if present:
                matches.append(canonical)
                break
    if "REST API" in matches and "API" in matches:
        matches.remove("API")
    return _stable_texts(matches)


def _canonical_atomic_value(value: str) -> str:
    normalized = _match_text(value)
    for canonical, aliases in _TECHNOLOGY_ALIASES:
        if any(normalized == _match_text(alias) for alias in aliases):
            return canonical
    return value


def _trait_values(values: Sequence[str]) -> tuple[str, ...]:
    traits = []
    for trait, aliases in _TRAIT_ALIASES:
        if any(_phrase_present(alias, value) for value in values for alias in aliases):
            traits.append(trait)
    return _stable_texts(traits)


def _score_rules(
    values: Sequence[str],
    rules: Mapping[RoleFamily, tuple[tuple[str, int], ...]],
) -> tuple[dict[RoleFamily, int], dict[RoleFamily, int]]:
    scores: dict[RoleFamily, int] = {family: 0 for family in _SPECIFIC_FAMILY_ORDER}
    support_counts: dict[RoleFamily, int] = {family: 0 for family in _SPECIFIC_FAMILY_ORDER}
    for family, family_rules in rules.items():
        for phrase, weight in family_rules:
            matching_values = sum(1 for value in values if _phrase_present(phrase, value))
            if matching_values:
                scores[family] += weight + min(matching_values - 1, 2)
                support_counts[family] += matching_values
    return scores, support_counts


def _classify_role(
    role_title: str | None,
    context: _NormalizedJobContext,
) -> _RoleClassification:
    title_values = (role_title,) if role_title else ()
    title_scores, _ = _score_rules(title_values, _TITLE_RULES)
    job_scores, job_support = _score_rules(context.all_values, _JOB_RULES)
    combined = {
        family: title_scores.get(family, 0) + job_scores.get(family, 0)
        for family in _SPECIFIC_FAMILY_ORDER
    }

    backend = combined[RoleFamily.BACKEND_ENGINEERING]
    frontend = combined[RoleFamily.FRONTEND_ENGINEERING]
    if backend >= 4 and frontend >= 4:
        combined[RoleFamily.FULL_STACK_ENGINEERING] = max(
            combined[RoleFamily.FULL_STACK_ENGINEERING],
            min(backend, frontend) + 4,
        )
        job_scores[RoleFamily.FULL_STACK_ENGINEERING] = max(
            job_scores[RoleFamily.FULL_STACK_ENGINEERING],
            min(job_scores[RoleFamily.BACKEND_ENGINEERING], job_scores[RoleFamily.FRONTEND_ENGINEERING]) + 3,
        )

    order = {family: index for index, family in enumerate(_SPECIFIC_FAMILY_ORDER)}
    ranked = sorted(
        _SPECIFIC_FAMILY_ORDER,
        key=lambda family: (-combined[family], order[family]),
    )
    if combined[ranked[0]] > 0:
        primary = ranked[0]
    elif role_title and any(_phrase_present(term, role_title) for term in ("engineer", "developer", "analyst", "consultant")):
        primary = RoleFamily.GENERAL
    else:
        primary = RoleFamily.UNKNOWN

    secondary = []
    for family in ranked:
        if family is primary or combined[family] < 3:
            continue
        secondary.append(family)
        if len(secondary) >= 6:
            break
    if primary is RoleFamily.FULL_STACK_ENGINEERING:
        for family in (RoleFamily.BACKEND_ENGINEERING, RoleFamily.FRONTEND_ENGINEERING):
            if combined[family] >= 3 and family not in secondary:
                secondary.append(family)
    secondary = secondary[:6]

    if primary in {RoleFamily.GENERAL, RoleFamily.UNKNOWN}:
        confidence = HiringContextConfidence.LOW
    else:
        title_score = title_scores.get(primary, 0)
        job_score = job_scores.get(primary, 0)
        support = job_support.get(primary, 0)
        if title_score >= 7 and job_score >= 3 and support >= 1:
            confidence = HiringContextConfidence.HIGH
        elif title_score >= 7 or job_score >= 5 or combined[primary] >= 7:
            confidence = HiringContextConfidence.MEDIUM
        else:
            confidence = HiringContextConfidence.LOW
    return _RoleClassification(
        primary=primary,
        secondary=tuple(secondary),
        confidence=confidence,
        title_scores=title_scores,
        job_scores=job_scores,
    )


def classify_hiring_context_role_families(
    *,
    role_title: str | None,
    normalized_job_context: Mapping[str, Any],
) -> tuple[RoleFamily, tuple[RoleFamily, ...], HiringContextConfidence]:
    """Classify a bounded role family from normalized title and JD values."""

    title = _optional_text(
        role_title,
        "role_title",
        MAX_HIRING_CONTEXT_ROLE_TITLE_LENGTH,
    )
    classification = _classify_role(title, _normalize_job_context(normalized_job_context))
    return classification.primary, classification.secondary, classification.confidence


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_ref(source_kind: HiringContextSourceKind, payload: Any) -> HiringContextSourceRef:
    return HiringContextSourceRef(
        source_kind=source_kind,
        source_fingerprint=_digest(payload),
    )


def _candidate(
    priority: int,
    value: str,
    kind: HiringContextSignalKind,
    confidence: HiringContextConfidence,
    effects: tuple[RankingEffect, ...],
    source_refs: tuple[HiringContextSourceRef, ...],
) -> _SignalCandidate:
    return _SignalCandidate(
        priority=priority,
        value=value,
        kind=kind,
        confidence=confidence,
        effects=effects,
        source_refs=source_refs,
    )


def _explicit_signal_candidates(
    context: _NormalizedJobContext,
    jd_source: HiringContextSourceRef | None,
) -> list[_SignalCandidate]:
    if jd_source is None:
        return []
    refs = (jd_source,)
    candidates: list[_SignalCandidate] = []
    groups = (
        (0, context.responsibilities, HiringContextSignalKind.EXPLICIT_JD, HiringContextConfidence.HIGH, False),
        (0, context.required_qualifications, HiringContextSignalKind.EXPLICIT_JD, HiringContextConfidence.HIGH, False),
        (1, context.required_skills, HiringContextSignalKind.EXPLICIT_JD, HiringContextConfidence.HIGH, True),
        (1, context.technologies, HiringContextSignalKind.EXPLICIT_JD, HiringContextConfidence.HIGH, True),
        (2, context.preferred_qualifications, HiringContextSignalKind.PREFERRED_QUALIFICATION, HiringContextConfidence.MEDIUM, False),
        (4, context.context_values, HiringContextSignalKind.EXPLICIT_JD, HiringContextConfidence.MEDIUM, False),
    )
    for priority, values, kind, confidence, atomic in groups:
        for value in values:
            candidates.append(_candidate(
                priority,
                _canonical_atomic_value(value) if atomic else value,
                kind,
                confidence,
                (RankingEffect.EXPLICIT_ALIGNMENT,),
                refs,
            ))
            for term in _canonical_terms(value, atomic=atomic):
                candidates.append(_candidate(
                    priority,
                    term,
                    kind,
                    confidence,
                    (RankingEffect.EXPLICIT_ALIGNMENT,),
                    refs,
                ))
    return candidates


def _role_signal_candidates(
    classification: _RoleClassification,
    role_source: HiringContextSourceRef | None,
    jd_source: HiringContextSourceRef | None,
) -> list[_SignalCandidate]:
    candidates = []
    for family in (classification.primary,) + classification.secondary:
        if family in {RoleFamily.GENERAL, RoleFamily.UNKNOWN}:
            continue
        refs = []
        if role_source is not None and classification.title_scores.get(family, 0) > 0:
            refs.append(role_source)
        if jd_source is not None and classification.job_scores.get(family, 0) > 0:
            refs.append(jd_source)
        if not refs:
            refs = [source for source in (role_source, jd_source) if source is not None]
        candidates.append(_candidate(
            3,
            family.value.replace("_", " "),
            HiringContextSignalKind.ROLE_FAMILY,
            classification.confidence,
            (RankingEffect.ROLE_FAMILY_ALIGNMENT,),
            tuple(refs),
        ))
    return candidates


def _organization_signal_candidates(
    resolution: OrganizationContextResolution,
) -> list[_SignalCandidate]:
    priorities = {
        OrganizationContextScope.TEAM: 4,
        OrganizationContextScope.COMPANY: 5,
        OrganizationContextScope.PARENT_ORGANIZATION: 6,
    }
    return [
        _candidate(
            priorities[resolved.scope],
            resolved.signal.value,
            resolved.signal.kind,
            resolved.signal.confidence,
            resolved.signal.ranking_effects,
            resolved.signal.source_refs,
        )
        for resolved in resolution.signals
    ]


def _candidate_sort_key(item: _SignalCandidate) -> tuple[Any, ...]:
    return (
        item.priority,
        item.kind.value,
        item.value.casefold(),
        item.value,
        tuple(source.reference_id for source in item.source_refs),
    )


def _merge_signal_candidates(
    values: Sequence[_SignalCandidate],
) -> _SignalCandidate:
    ordered = sorted(values, key=_candidate_sort_key)
    representative = ordered[0]
    sources = {
        source.reference_id: source
        for item in ordered
        for source in item.source_refs
    }
    if len(sources) > MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS:
        raise ValueError(
            "merged signal provenance exceeds maximum source count "
            f"{MAX_HIRING_CONTEXT_SIGNAL_SOURCE_REFS}"
        )
    effect_order = {effect: index for index, effect in enumerate(RankingEffect)}
    effects = tuple(sorted(
        {effect for item in ordered for effect in item.effects},
        key=effect_order.__getitem__,
    ))
    if len(effects) > MAX_HIRING_CONTEXT_RANKING_EFFECTS:
        raise ValueError(
            "merged signal effects exceed maximum effect count "
            f"{MAX_HIRING_CONTEXT_RANKING_EFFECTS}"
        )
    confidence_order = {
        HiringContextConfidence.LOW: 0,
        HiringContextConfidence.MEDIUM: 1,
        HiringContextConfidence.HIGH: 2,
    }
    confidence = max(
        (item.confidence for item in ordered),
        key=confidence_order.__getitem__,
    )
    return _SignalCandidate(
        priority=representative.priority,
        value=representative.value,
        kind=representative.kind,
        confidence=confidence,
        effects=effects,
        source_refs=tuple(sources[key] for key in sorted(sources)),
    )


def _select_signals(candidates: Sequence[_SignalCandidate]) -> tuple[HiringContextSignal, ...]:
    grouped: dict[tuple[str, str], list[_SignalCandidate]] = {}
    for item in candidates:
        truth_group = (
            item.kind.value
            if item.kind in {
                HiringContextSignalKind.ROLE_FAMILY,
                HiringContextSignalKind.ENGINEERING_TRAIT,
            }
            else "hiring_context_value"
        )
        grouped.setdefault((truth_group, item.value.casefold()), []).append(item)
    merged = sorted(
        (_merge_signal_candidates(values) for values in grouped.values()),
        key=_candidate_sort_key,
    )
    selected = merged[:MAX_HIRING_CONTEXT_SIGNALS]
    return tuple(
        HiringContextSignal(
            value=item.value,
            kind=item.kind,
            confidence=item.confidence,
            ranking_effects=item.effects,
            source_refs=item.source_refs,
        )
        for item in selected
    )


def build_hiring_context_profile(
    *,
    company: str | None,
    role_title: str | None,
    normalized_job_context: Mapping[str, Any],
    team: str | None = None,
    parent_organization: str | None = None,
    organization_registry: OrganizationContextRegistry = DEFAULT_ORGANIZATION_CONTEXT_REGISTRY,
) -> HiringContextProfile:
    """Build explicit JD, role-family, and offline organization context."""

    normalized_company = _optional_text(
        company,
        "company",
        MAX_HIRING_CONTEXT_NAME_LENGTH,
    )
    normalized_team = _optional_text(
        team,
        "team",
        MAX_HIRING_CONTEXT_NAME_LENGTH,
    )
    normalized_parent = _optional_text(
        parent_organization,
        "parent_organization",
        MAX_HIRING_CONTEXT_NAME_LENGTH,
    )
    normalized_title = _optional_text(
        role_title,
        "role_title",
        MAX_HIRING_CONTEXT_ROLE_TITLE_LENGTH,
    )
    context = _normalize_job_context(normalized_job_context)
    classification = _classify_role(normalized_title, context)
    organization = resolve_organization_context(
        company=normalized_company,
        team=normalized_team,
        parent_organization=normalized_parent,
        registry=organization_registry,
    )

    source_refs = list(organization.source_refs)
    jd_source = None
    if context.all_values:
        jd_source = _source_ref(
            HiringContextSourceKind.JOB_DESCRIPTION,
            context.fingerprint_payload(),
        )
        source_refs.append(jd_source)
    role_source = None
    if normalized_title:
        role_source = _source_ref(
            HiringContextSourceKind.ROLE_TITLE,
            {"role_title": normalized_title},
        )
        source_refs.append(role_source)
    candidates = _explicit_signal_candidates(context, jd_source)
    candidates.extend(_role_signal_candidates(classification, role_source, jd_source))
    candidates.extend(_organization_signal_candidates(organization))
    signals = _select_signals(candidates)
    traits = _stable_texts(
        _trait_values(context.all_values) + organization.high_value_traits
    )
    if len(traits) > MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS:
        raise ValueError(
            "combined hiring context exceeds maximum high-value trait count "
            f"{MAX_HIRING_CONTEXT_HIGH_VALUE_TRAITS}"
        )
    selected_source_ids = {
        source.reference_id
        for signal in signals
        for source in signal.source_refs
    }
    source_refs = [
        source
        for source in source_refs
        if source.source_kind is not HiringContextSourceKind.INTERNAL_TAXONOMY
        or source.reference_id in selected_source_ids
    ]

    return HiringContextProfile(
        source_refs=tuple(source_refs),
        confidence=classification.confidence,
        company=organization.company,
        team=organization.team,
        parent_organization=organization.parent_organization,
        role_title=normalized_title,
        primary_role_family=classification.primary,
        secondary_role_families=classification.secondary,
        signals=signals,
        high_value_traits=traits,
    )


__all__ = [
    "MAX_NORMALIZED_JOB_CONTEXT_VALUES",
    "MAX_NORMALIZED_JOB_VALUES_PER_FIELD",
    "build_hiring_context_profile",
    "classify_hiring_context_role_families",
]
