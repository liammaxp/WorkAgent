from __future__ import annotations

import inspect

import httpx
import pytest

import backend.hiring_context_web_client as web_client
from backend.hiring_context_official_sources import (
    OfficialHiringContextSourceKind,
    OfficialSourceCandidate,
    OfficialSourceSpecificity,
)
from backend.hiring_context_web_client import (
    MAX_OFFICIAL_WEB_PAGES_PER_ORGANIZATION,
    MAX_OFFICIAL_WEB_REDIRECTS,
    MAX_OFFICIAL_WEB_RESPONSE_BYTES,
    MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT,
    MAX_OFFICIAL_WEB_TEXT_CHARACTERS,
    OFFICIAL_WEB_CONCURRENCY,
    OFFICIAL_WEB_RETRY_COUNT,
    OfficialSourceAcquisition,
    fetch_official_hiring_context_sources,
)


PUBLIC_IP = "93.184.216.34"


def _source(
    *,
    organization: str = "Example",
    url: str = "https://official.example/",
    specificity: OfficialSourceSpecificity = OfficialSourceSpecificity.COMPANY,
) -> OfficialSourceCandidate:
    host = httpx.URL(url).host
    return OfficialSourceCandidate(
        canonical_organization=organization,
        url=url,
        source_kind=OfficialHiringContextSourceKind.OFFICIAL_COMPANY_PAGE,
        specificity=specificity,
        allowed_domains=(host,),
    )


def _public(_: str) -> tuple[str, ...]:
    return (PUBLIC_IP,)


def _builder(handler, observed: list[httpx.Timeout] | None = None):
    def build(timeout: httpx.Timeout) -> httpx.Client:
        if observed is not None:
            observed.append(timeout)
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        )

    return build


def _fetch(handler, sources=None, **kwargs) -> OfficialSourceAcquisition:
    return fetch_official_hiring_context_sources(
        sources or (_source(),),
        resolver=kwargs.pop("resolver", _public),
        client_builder=_builder(handler),
        **kwargs,
    )


def test_successful_html_fetch_emits_only_bounded_document_fields():
    result = _fetch(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<title>Engineering</title><h1>Game development</h1>",
        )
    )
    assert not result.failures
    assert result.documents[0].title == "Engineering"
    assert result.documents[0].text == "Engineering Game development"
    assert set(result.documents[0].__slots__) == {
        "source",
        "normalized_url",
        "title",
        "text",
        "content_fingerprint",
    }


def test_plain_text_fetch_is_normalized():
    result = _fetch(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=b"  Data   infrastructure\nDeveloper platform  ",
        )
    )
    assert result.documents[0].text == "Data infrastructure Developer platform"


def test_script_style_noscript_template_and_svg_text_are_excluded():
    body = b"""
    <html><head><title>Team</title><style>Azure hidden</style></head>
    <body><script>C++ hidden</script><noscript>AWS hidden</noscript>
    <template>Risk hidden</template><svg><text>Unreal hidden</text></svg>
    <h1>Game development</h1><p>Online systems</p></body></html>
    """
    text = _fetch(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, content=body
        )
    ).documents[0].text
    assert text == "Team Game development Online systems"
    assert all(value not in text for value in ("Azure", "C++", "AWS", "Risk", "Unreal"))


def test_meta_description_is_included_without_other_head_boilerplate():
    result = _fetch(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b'<meta name="description" content="Software platforms"><link title="ignored">',
        )
    )
    assert result.documents[0].text == "Software platforms"


def test_malformed_html_is_tolerated_and_bounded():
    result = _fetch(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<title>Studio<h1>Game development<p>Online systems<<<",
        )
    )
    assert result.documents
    assert "Game development" in result.documents[0].text


@pytest.mark.parametrize(
    ("headers", "body", "code"),
    (
        ({"content-type": "application/pdf"}, b"%PDF", "unsupported_media_type"),
        ({"content-type": "application/octet-stream"}, b"binary", "unsupported_media_type"),
        ({"content-type": "text/html"}, b"hello\x00world", "binary_content"),
        ({"content-type": "text/html; charset=utf-8"}, b"\xff", "invalid_encoding"),
        ({"content-type": "text/html; charset=not-a-codec"}, b"hello", "invalid_encoding"),
        ({}, b"hello", "unsupported_media_type"),
    ),
)
def test_unsupported_binary_and_invalid_text_content_fail_safely(headers, body, code):
    result = _fetch(lambda request: httpx.Response(200, headers=headers, content=body))
    assert not result.documents
    assert result.failures[0].code == code


@pytest.mark.parametrize("status", (400, 401, 403, 404, 429, 500, 503))
def test_non_success_http_statuses_are_rejected(status):
    result = _fetch(
        lambda request: httpx.Response(
            status, headers={"content-type": "text/html"}, content=b"error"
        )
    )
    assert result.failures[0].code == "http_status_rejected"


def test_timeout_is_sanitized_and_bounded():
    def timeout(request):
        raise httpx.ReadTimeout("sensitive transport detail", request=request)

    result = _fetch(timeout)
    assert result.failures[0].code == "timeout"
    assert "sensitive" not in result.failures[0].code


def test_network_unavailable_is_sanitized_and_bounded():
    def unavailable(request):
        raise httpx.ConnectError("host detail", request=request)

    result = _fetch(unavailable)
    assert result.failures[0].code == "network_unavailable"


def test_client_construction_failure_returns_offline_safe_failures():
    def builder(timeout):
        raise RuntimeError("client failed")

    result = fetch_official_hiring_context_sources(
        (_source(),), resolver=_public, client_builder=builder
    )
    assert result.documents == ()
    assert result.failures[0].code == "client_unavailable"


def test_content_length_rejects_oversized_response_before_semantics():
    result = _fetch(
        lambda request: httpx.Response(
            200,
            headers={
                "content-type": "text/html",
                "content-length": str(MAX_OFFICIAL_WEB_RESPONSE_BYTES + 1),
            },
            content=b"small",
        )
    )
    assert result.failures[0].code == "response_too_large"


def test_streamed_byte_count_rejects_lying_response_length():
    result = _fetch(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"x" * (MAX_OFFICIAL_WEB_RESPONSE_BYTES + 1),
        )
    )
    assert result.failures[0].code == "response_too_large"


def test_visible_text_is_truncated_at_explicit_character_bound():
    body = ("<p>" + "word " * (MAX_OFFICIAL_WEB_TEXT_CHARACTERS // 2) + "</p>").encode()
    result = _fetch(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/html"}, content=body
        )
    )
    assert len(result.documents[0].text) <= MAX_OFFICIAL_WEB_TEXT_CHARACTERS


def test_same_official_host_redirect_is_manually_followed_and_preserved():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/":
            return httpx.Response(302, headers={"location": "/team"})
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"<h1>Game development</h1>",
        )

    result = _fetch(handler)
    assert seen == ["https://official.example/", "https://official.example/team"]
    assert result.documents[0].normalized_url == "https://official.example/team"


def test_redirect_to_unregistered_public_domain_is_rejected():
    result = _fetch(
        lambda request: httpx.Response(
            302, headers={"location": "https://attacker.example/"}
        )
    )
    assert result.failures[0].code == "unregistered_official_domain"


def test_redirect_to_private_resolution_is_rejected_on_second_hop():
    calls = 0

    def resolver(host):
        nonlocal calls
        calls += 1
        return (PUBLIC_IP,) if calls == 1 else ("127.0.0.1",)

    result = _fetch(
        lambda request: httpx.Response(302, headers={"location": "/private"}),
        resolver=resolver,
    )
    assert result.failures[0].code == "forbidden_destination"
    assert calls == 2


def test_excessive_redirects_are_rejected_at_fixed_bound():
    def handler(request):
        number = int(request.url.path.strip("/") or "0")
        return httpx.Response(302, headers={"location": f"/{number + 1}"})

    result = _fetch(handler)
    assert result.failures[0].code == "too_many_redirects"
    assert MAX_OFFICIAL_WEB_REDIRECTS == 3


def test_redirect_loop_is_rejected():
    result = _fetch(
        lambda request: httpx.Response(302, headers={"location": "/"})
    )
    assert result.failures[0].code == "redirect_loop"


@pytest.mark.parametrize("location", (None, "", "x" * 2_049))
def test_invalid_redirect_location_is_rejected(location):
    headers = {} if location is None else {"location": location}
    result = _fetch(lambda request: httpx.Response(302, headers=headers))
    assert result.failures[0].code == "invalid_redirect"


def test_duplicate_canonical_urls_are_deduplicated_request_locally():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"Software platforms"
        )

    first = _source(url="https://official.example:443/")
    second = _source(url="https://official.example/")
    result = _fetch(handler, (first, second))
    assert calls == 1
    assert len(result.documents) == 1


def test_deterministic_acquisition_order_uses_specificity_then_identity_then_url():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"Software platforms"
        )

    sources = (
        _source(organization="Zed", url="https://z.example/"),
        _source(
            organization="Team",
            url="https://team.example/",
            specificity=OfficialSourceSpecificity.TEAM_PRODUCT,
        ),
        _source(organization="Alpha", url="https://a.example/"),
    )
    _fetch(handler, sources)
    assert seen == ["https://team.example/", "https://a.example/", "https://z.example/"]


def test_page_count_bound_is_enforced_per_organization():
    sources = tuple(
        _source(organization="Example", url=f"https://official.example/page-{index}")
        for index in range(MAX_OFFICIAL_WEB_PAGES_PER_ORGANIZATION + 2)
    )
    result = _fetch(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"Software platforms"
        ),
        sources,
    )
    assert len(result.documents) == MAX_OFFICIAL_WEB_PAGES_PER_ORGANIZATION


def test_source_count_bound_is_enforced_per_context():
    sources = tuple(
        _source(organization=f"Org {index}", url=f"https://org-{index}.example/")
        for index in range(MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT + 2)
    )
    result = _fetch(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"Software platforms"
        ),
        sources,
    )
    assert len(result.documents) == MAX_OFFICIAL_WEB_SOURCES_PER_CONTEXT


def test_context_deadline_prevents_late_request():
    values = iter((0.0, 20.0))
    result = fetch_official_hiring_context_sources(
        (_source(),),
        resolver=_public,
        client_builder=_builder(lambda request: pytest.fail("network must not be called")),
        clock=lambda: next(values),
    )
    assert result.failures[0].code == "deadline_exceeded"


def test_timeout_retry_proxy_redirect_and_concurrency_bounds_are_explicit():
    observed = []
    result = fetch_official_hiring_context_sources(
        (_source(),),
        resolver=_public,
        client_builder=_builder(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"Software platforms"
            ),
            observed,
        ),
    )
    assert result.documents
    timeout = observed[0]
    assert timeout.connect == web_client.OFFICIAL_WEB_CONNECT_TIMEOUT_SECONDS
    assert timeout.read == web_client.OFFICIAL_WEB_READ_TIMEOUT_SECONDS
    assert timeout.write == web_client.OFFICIAL_WEB_WRITE_TIMEOUT_SECONDS
    assert timeout.pool == web_client.OFFICIAL_WEB_POOL_TIMEOUT_SECONDS
    assert OFFICIAL_WEB_RETRY_COUNT == 0
    assert OFFICIAL_WEB_CONCURRENCY == 1
    source = inspect.getsource(web_client._default_client_builder)
    assert "trust_env=False" in source
    assert "follow_redirects=False" in source


def test_acquisition_contract_is_immutable():
    result = _fetch(
        lambda request: httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"Software platforms"
        )
    )
    with pytest.raises(AttributeError):
        result.documents = ()
