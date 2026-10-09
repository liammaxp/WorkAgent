"""Safe bounded acquisition for registered official hiring-context pages.

This is the sole network boundary for employer context.  It performs GET-only
sequential acquisition, validates every redirect hop, streams into a byte cap,
and emits only bounded normalized text documents.  It never carries candidate
data, arbitrary caller URLs, response objects, cookies, or raw HTML onward.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
import codecs
import hashlib
from html.parser import HTMLParser
import re
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from backend.hiring_context_official_sources import (
    DnsResolver,
    OfficialSourceCandidate,
    OfficialSourceUrlError,
    default_public_dns_resolver,
    validate_official_source_url,
)


OFFICIAL_WEB_CONNECT_TIMEOUT_SECONDS = 3.0
OFFICIAL_WEB_READ_TIMEOUT_SECONDS = 5.0
OFFICIAL_WEB_WRITE_TIMEOUT_SECONDS = 2.0
OFFICIAL_WEB_POOL_TIMEOUT_SECONDS = 2.0
OFFICIAL_WEB_PAGE_DEADLINE_SECONDS = 8.0
OFFICIAL_WEB_CONTEXT_DEADLINE_SECONDS = 15.0
MAX_OFFICIAL_WEB_REDIRECTS = 3
MAX_OFFICIAL_WEB_RESPONSE_BYTES = 512 * 1024
MAX_OFFICIAL_WEB_TEXT_CHARACTERS = 40_000
MAX_OFFICIAL_WEB_TITLE_CHARACTERS = 300
MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT = 4
MAX_OFFICIAL_WEB_PAGES_PER_ORGANIZATION = 2
OFFICIAL_WEB_CONCURRENCY = 1
OFFICIAL_WEB_RETRY_COUNT = 0

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ACCEPTED_MEDIA_TYPES = frozenset({"text/html", "text/plain"})
_CHARSET_RE = re.compile(r"(?:^|;)\s*charset\s*=\s*['\"]?([^;'\"\s]+)", re.IGNORECASE)


class OfficialWebAcquisitionError(RuntimeError):
    """A stable sanitized acquisition failure; URLs and bodies are omitted."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class BoundedOfficialSourceDocument:
    source: OfficialSourceCandidate
    normalized_url: str
    title: str
    text: str
    content_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, OfficialSourceCandidate):
            raise TypeError("source must be an OfficialSourceCandidate")
        if not isinstance(self.normalized_url, str) or not self.normalized_url:
            raise ValueError("normalized_url must not be blank")
        if not isinstance(self.title, str) or len(self.title) > MAX_OFFICIAL_WEB_TITLE_CHARACTERS:
            raise ValueError("title exceeds its bound")
        if not isinstance(self.text, str) or not self.text or len(self.text) > MAX_OFFICIAL_WEB_TEXT_CHARACTERS:
            raise ValueError("text must be non-empty and bounded")
        if not re.fullmatch(r"[0-9a-f]{64}", self.content_fingerprint):
            raise ValueError("content_fingerprint must be a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class OfficialSourceAcquisitionFailure:
    source: OfficialSourceCandidate
    code: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, OfficialSourceCandidate):
            raise TypeError("source must be an OfficialSourceCandidate")
        if not isinstance(self.code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.code):
            raise ValueError("failure code must be a stable sanitized token")


@dataclass(frozen=True, slots=True)
class OfficialSourceAcquisition:
    documents: tuple[BoundedOfficialSourceDocument, ...]
    failures: tuple[OfficialSourceAcquisitionFailure, ...]

    def __post_init__(self) -> None:
        if len(self.documents) + len(self.failures) > MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT:
            raise ValueError("acquisition exceeds the source bound")
        if any(not isinstance(item, BoundedOfficialSourceDocument) for item in self.documents):
            raise TypeError("documents contains an invalid value")
        if any(not isinstance(item, OfficialSourceAcquisitionFailure) for item in self.failures):
            raise TypeError("failures contains an invalid value")


ClientBuilder = Callable[[httpx.Timeout], httpx.Client]
Clock = Callable[[], float]


def _default_client_builder(timeout: httpx.Timeout) -> httpx.Client:
    transport = httpx.HTTPTransport(retries=OFFICIAL_WEB_RETRY_COUNT, trust_env=False)
    return httpx.Client(
        timeout=timeout,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
        headers={
            "Accept": "text/html,text/plain;q=0.9",
            "Accept-Encoding": "identity",
            "User-Agent": "WorkAgentOfficialContext/1.0",
        },
    )


class _BoundedVisibleTextParser(HTMLParser):
    _CAPTURE_TAGS = frozenset({"title", "h1", "h2", "h3", "p", "li"})
    _IGNORED_TAGS = frozenset({"script", "style", "noscript", "template", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._capture_depth = 0
        self._title_depth = 0
        self._chunks: list[str] = []
        self._title_chunks: list[str] = []
        self._character_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in self._CAPTURE_TAGS:
            self._capture_depth += 1
        if tag == "title":
            self._title_depth += 1
        if tag == "meta":
            values = {key.casefold(): (value or "") for key, value in attrs}
            if values.get("name", "").casefold() == "description":
                self._append(values.get("content", ""), title=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return
        if tag in self._CAPTURE_TAGS:
            self._capture_depth = max(0, self._capture_depth - 1)
        if tag == "title":
            self._title_depth = max(0, self._title_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or not self._capture_depth:
            return
        self._append(data, title=bool(self._title_depth))

    def _append(self, value: str, *, title: bool) -> None:
        normalized = " ".join(value.split())
        if not normalized:
            return
        remaining = MAX_OFFICIAL_WEB_TEXT_CHARACTERS - self._character_count
        if remaining <= 0:
            return
        bounded = normalized[:remaining]
        self._chunks.append(bounded)
        self._character_count += len(bounded) + 1
        if title and sum(len(item) for item in self._title_chunks) < MAX_OFFICIAL_WEB_TITLE_CHARACTERS:
            self._title_chunks.append(bounded)

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_chunks).split())[:MAX_OFFICIAL_WEB_TITLE_CHARACTERS]

    @property
    def text(self) -> str:
        return " ".join(" ".join(self._chunks).split())[:MAX_OFFICIAL_WEB_TEXT_CHARACTERS]


def _decode_body(body: bytes, content_type: str) -> str:
    if b"\x00" in body:
        raise OfficialWebAcquisitionError("binary_content")
    match = _CHARSET_RE.search(content_type)
    encoding = match.group(1) if match else "utf-8"
    try:
        codecs.lookup(encoding)
        return body.decode(encoding, errors="strict")
    except (LookupError, UnicodeDecodeError) as exc:
        raise OfficialWebAcquisitionError("invalid_encoding") from exc


def _bounded_document(
    source: OfficialSourceCandidate,
    normalized_url: str,
    body: bytes,
    media_type: str,
    content_type: str,
) -> BoundedOfficialSourceDocument:
    decoded = _decode_body(body, content_type)
    if media_type == "text/html":
        parser = _BoundedVisibleTextParser()
        try:
            parser.feed(decoded)
            parser.close()
        except Exception as exc:
            raise OfficialWebAcquisitionError("malformed_content") from exc
        title = parser.title
        text = parser.text
    else:
        title = ""
        text = " ".join(decoded.split())[:MAX_OFFICIAL_WEB_TEXT_CHARACTERS]
    if not text:
        raise OfficialWebAcquisitionError("empty_content")
    fingerprint = hashlib.sha256(
        f"{title}\n{text}".encode("utf-8")
    ).hexdigest()
    return BoundedOfficialSourceDocument(
        source=source,
        normalized_url=normalized_url,
        title=title,
        text=text,
        content_fingerprint=fingerprint,
    )


def _content_length(headers: httpx.Headers) -> int | None:
    value = headers.get("content-length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OfficialWebAcquisitionError("invalid_content_length") from exc
    if parsed < 0:
        raise OfficialWebAcquisitionError("invalid_content_length")
    return parsed


def _fetch_one(
    client: httpx.Client,
    source: OfficialSourceCandidate,
    *,
    resolver: DnsResolver,
    clock: Clock,
    context_deadline: float,
) -> BoundedOfficialSourceDocument:
    current = source.url
    visited: set[str] = set()
    page_deadline = min(context_deadline, clock() + OFFICIAL_WEB_PAGE_DEADLINE_SECONDS)
    for redirect_count in range(MAX_OFFICIAL_WEB_REDIRECTS + 1):
        if clock() > page_deadline:
            raise OfficialWebAcquisitionError("deadline_exceeded")
        try:
            current = validate_official_source_url(source, current, resolver=resolver)
        except OfficialSourceUrlError as exc:
            raise OfficialWebAcquisitionError(exc.code) from exc
        if current in visited:
            raise OfficialWebAcquisitionError("redirect_loop")
        visited.add(current)
        try:
            with client.stream("GET", current) as response:
                status = response.status_code
                if status in _REDIRECT_STATUSES:
                    if redirect_count >= MAX_OFFICIAL_WEB_REDIRECTS:
                        raise OfficialWebAcquisitionError("too_many_redirects")
                    location = response.headers.get("location")
                    if not location or len(location) > 2_048:
                        raise OfficialWebAcquisitionError("invalid_redirect")
                    current = urljoin(current, location)
                    continue
                if not 200 <= status < 300:
                    raise OfficialWebAcquisitionError("http_status_rejected")
                content_type = response.headers.get("content-type", "")
                media_type = content_type.split(";", 1)[0].strip().casefold()
                if media_type not in _ACCEPTED_MEDIA_TYPES:
                    raise OfficialWebAcquisitionError("unsupported_media_type")
                length = _content_length(response.headers)
                if length is not None and length > MAX_OFFICIAL_WEB_RESPONSE_BYTES:
                    raise OfficialWebAcquisitionError("response_too_large")
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    if clock() > page_deadline:
                        raise OfficialWebAcquisitionError("deadline_exceeded")
                    size += len(chunk)
                    if size > MAX_OFFICIAL_WEB_RESPONSE_BYTES:
                        raise OfficialWebAcquisitionError("response_too_large")
                    chunks.append(chunk)
                return _bounded_document(
                    source,
                    current,
                    b"".join(chunks),
                    media_type,
                    content_type,
                )
        except OfficialWebAcquisitionError:
            raise
        except httpx.TimeoutException as exc:
            raise OfficialWebAcquisitionError("timeout") from exc
        except (httpx.HTTPError, OSError, TimeoutError) as exc:
            raise OfficialWebAcquisitionError("network_unavailable") from exc
        finally:
            client.cookies.clear()
    raise OfficialWebAcquisitionError("too_many_redirects")


def _bounded_candidates(
    candidates: Sequence[OfficialSourceCandidate],
) -> tuple[OfficialSourceCandidate, ...]:
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence")
    if any(not isinstance(item, OfficialSourceCandidate) for item in candidates):
        raise TypeError("candidates must contain only OfficialSourceCandidate values")
    ordered = sorted(
        {item.url: item for item in candidates}.values(),
        key=lambda item: (int(item.specificity), item.canonical_organization.casefold(), item.url),
    )
    counts: dict[str, int] = {}
    bounded = []
    for candidate in ordered:
        organization = candidate.canonical_organization.casefold()
        if counts.get(organization, 0) >= MAX_OFFICIAL_WEB_PAGES_PER_ORGANIZATION:
            continue
        bounded.append(candidate)
        counts[organization] = counts.get(organization, 0) + 1
        if len(bounded) >= MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT:
            break
    return tuple(bounded)


def fetch_official_hiring_context_sources(
    candidates: Sequence[OfficialSourceCandidate],
    *,
    resolver: DnsResolver = default_public_dns_resolver,
    client_builder: ClientBuilder = _default_client_builder,
    clock: Clock = time.monotonic,
) -> OfficialSourceAcquisition:
    """Fetch a deterministic, sequential, request-scoped set of official pages."""

    selected = _bounded_candidates(candidates)
    if not selected:
        return OfficialSourceAcquisition(documents=(), failures=())
    timeout = httpx.Timeout(
        connect=OFFICIAL_WEB_CONNECT_TIMEOUT_SECONDS,
        read=OFFICIAL_WEB_READ_TIMEOUT_SECONDS,
        write=OFFICIAL_WEB_WRITE_TIMEOUT_SECONDS,
        pool=OFFICIAL_WEB_POOL_TIMEOUT_SECONDS,
    )
    documents = []
    failures = []
    context_deadline = clock() + OFFICIAL_WEB_CONTEXT_DEADLINE_SECONDS
    try:
        client_context: AbstractContextManager[httpx.Client] = client_builder(timeout)
        with client_context as client:
            for source in selected:
                if clock() > context_deadline:
                    failures.append(OfficialSourceAcquisitionFailure(source, "deadline_exceeded"))
                    continue
                try:
                    documents.append(
                        _fetch_one(
                            client,
                            source,
                            resolver=resolver,
                            clock=clock,
                            context_deadline=context_deadline,
                        )
                    )
                except OfficialWebAcquisitionError as exc:
                    failures.append(OfficialSourceAcquisitionFailure(source, exc.code))
    except OfficialWebAcquisitionError:
        raise
    except Exception:
        failures = [
            OfficialSourceAcquisitionFailure(source, "client_unavailable")
            for source in selected
        ]
        documents = []
    return OfficialSourceAcquisition(tuple(documents), tuple(failures))


__all__ = [
    "BoundedOfficialSourceDocument",
    "MAX_OFFICIAL_WEB_PAGES_PER_ORGANIZATION",
    "MAX_OFFICIAL_WEB_REDIRECTS",
    "MAX_OFFICIAL_WEB_RESPONSE_BYTES",
    "MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT",
    "MAX_OFFICIAL_WEB_TEXT_CHARACTERS",
    "MAX_OFFICIAL_WEB_TITLE_CHARACTERS",
    "OFFICIAL_WEB_CONCURRENCY",
    "OFFICIAL_WEB_CONNECT_TIMEOUT_SECONDS",
    "OFFICIAL_WEB_CONTEXT_DEADLINE_SECONDS",
    "OFFICIAL_WEB_PAGE_DEADLINE_SECONDS",
    "OFFICIAL_WEB_POOL_TIMEOUT_SECONDS",
    "OFFICIAL_WEB_READ_TIMEOUT_SECONDS",
    "OFFICIAL_WEB_RETRY_COUNT",
    "OFFICIAL_WEB_WRITE_TIMEOUT_SECONDS",
    "OfficialSourceAcquisition",
    "OfficialSourceAcquisitionFailure",
    "OfficialWebAcquisitionError",
    "fetch_official_hiring_context_sources",
]
