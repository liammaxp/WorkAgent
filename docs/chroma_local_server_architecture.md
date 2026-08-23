# Chroma Local Server Architecture and Operations

## Production access deprecation policy

Production WorkAgent does not support embedded Chroma access. Application reads, vector queries, writes, indexes, and operational status use `ChromaHttpClientFactory` and `BoundedChromaHttpTransport`; `PersistentClient` is permitted only in `tests/chroma_persistence_test_support.py` for disposable, protected-path-rejecting parity verification. Production consumers may not construct `chromadb.HttpClient`, own a Chroma `httpx`, `requests`, or raw `urllib` session, import the test persistence helper, or recover from HTTP failure by opening embedded storage.

The reviewed access inventory is an enforcement boundary, not an allowlist that can authorize a different architecture. Its semantic policy rejects production embedded clients, direct public Chroma clients, low-level HTTP outside the bounded transport, production collection creation, deprecated embedded states, and unapproved owners even if a matching manifest record is added. Static failures contain only repository-relative module, line, symbol, and violation category.

Production collection creation is not automatic. Every registry definition has `automatic_creation=false`; `create_collection` and `get_or_create_collection` are allowed only in the disposable HTTP test fixture. A future production bootstrap requires a separately reviewed controlled operation and is not implemented here.

Starting or stopping the local server is an explicit operator action owned by lifecycle tooling. Semantic clients never start a server, read the production persistence directory, or fall back to another storage mode. The supported deployment modes remain exactly `disabled`, `local_http`, `remote_http`, and explicitly authorized `ephemeral_test`; embedded, persistent, automatic, and compatibility aliases fail closed. Remote HTTP compatibility remains available without changing the central ownership rule.

Real-data cutover remains a separate controlled operation. The deprecation guard neither starts Chroma against protected persistence nor mutates, restores, bootstraps, or accepts new production data.

## Protected pre-migration baseline

The migration baseline records a deterministic, privacy-safe snapshot of the current local Chroma persistence files before any client or collection migration begins. It exists to prove that later work preserves the protected installation; it does not change application retrieval behavior and it is not a server lifecycle tool.

Embedded Chroma inspection is prohibited for this baseline. Constructing `PersistentClient`, opening a collection, or issuing a local collection query can change SQLite or HNSW internals even when the caller intends to read. The capture therefore traverses the protected directory with ordinary filesystem reads only. It rejects symlinks, junctions, reparse points, unreadable files, path escapes, and any pre/post inventory mismatch.

The protected-storage aggregate is a historical byte baseline. It is appropriate before migration while the directory is quiescent, but it is not the future logical-integrity model for mutable server-owned storage. Logical inventory is recorded separately and is either obtained through an explicitly approved, already-running local HTTP boundary or marked unavailable. Capture never starts a server and never falls back to embedded database inspection.

Immutable evidence artifacts are represented only by semantic name, repository-relative path, byte size, SHA-256, and a bounded schema marker when one is safely detectable. Baselines never contain document bodies, embeddings, patches, raw metadata, secrets, environment values, or absolute paths. A static constructor inventory records the currently classified `PersistentClient` and `HttpClient` call sites without importing application modules.

Generate the default protected baseline from the repository root:

```powershell
python -m backend.chroma_migration_baseline capture
```

If an approved local HTTP server is already running and its existing environment configuration is intentionally enabled, logical metadata can be included without starting or stopping the server:

```powershell
python -m backend.chroma_migration_baseline capture --approved-http
```

Validate schema, privacy declarations, and deterministic digests without comparing current machine state:

```powershell
python -m backend.chroma_migration_baseline verify
```

Explicitly compare protected bytes and immutable artifact hashes when the directory is expected to be quiescent:

```powershell
python -m backend.chroma_migration_baseline verify --compare-protected --compare-artifacts
```

Real captures are written atomically under `information/chroma_migration_baselines/`. The existing `information/*` source-control rule ignores this directory, so machine-specific hashes, storage layout, and runtime observations are not committed. Verification never rewrites a baseline and offers no shortcut for accepting changed hashes.

Later migration work remains blocked until a valid baseline has zero unclassified Chroma client constructors, the protected pre/post byte inventory matches, immutable artifact hashes validate, and any required logical inventory is either safely captured through approved HTTP or explicitly documented as unavailable. Central HTTP-client ownership, service lifecycle, collection and writer migration, backup/restore, and production cutover are intentionally outside this implementation.

## Authoritative access inventory

The repository-wide Chroma access inventory is the machine-verifiable ownership map for current client construction, collection resolution, collection operations, and direct Chroma-storage inspection. A deterministic AST scanner discovers access without importing application modules, constructing clients, connecting to a server, or reading the protected persistence directory. A separate reviewed manifest supplies the classifications that static discovery cannot safely decide. Verification fails on new or stale access, changed operations or client types, unresolved receiver provenance, unknown classifications, or a manifest digest mismatch.

Each entry has one lifecycle category: `read`, `vector_query`, `write`, `index`, `migration`, `maintenance`, or `test_only`. Runtime is classified separately as `production`, `maintenance_only`, `migration_only`, or `test_only`. Client ownership is also explicit: `persistent_embedded`, `http`, `fake_http`, or isolated `ephemeral_embedded`. Unknown client types and unknown collection-name resolution are invalid.

Logical record mutation and storage-internal mutation risk are deliberately separate. An embedded query is a logical read and does not mutate records, but its local process may still change SQLite or HNSW internals. An HTTP query leaves those internals server-owned. Immutable SQLite inspection is recorded independently from embedded collection calls, and temporary test fixtures remain explicitly test-only.

The manifest maps every entry to a semantic owner such as configuration, collection registration, centralized HTTP transport, status reads, vector/business reads, writes/indexing, lifecycle health, or test infrastructure. The GitHub vector compatibility wrapper delegates to the central semantic read client and no longer owns a low-level client. No production embedded constructor or operation remains; embedded entries are test-only. This inventory is classification evidence, not authorization to add another call site.

Inspect discovery and reviewed classifications with bounded count-only output:

```powershell
python -m backend.chroma_access_inventory inspect
```

Validate deterministic identifiers, schema, privacy rules, manifest digest, and synchronization:

```powershell
python -m backend.chroma_access_inventory verify
```

The pre-migration baseline projects its unchanged constructor counts from this authoritative inventory when the reviewed manifest is present. Older captured baseline documents remain valid because the baseline schema and its bounded constructor summary shape are unchanged.

## Deployment configuration contract

`backend/chroma_config.py` is the authoritative typed deployment configuration for later Chroma server, client, migration, and operations work. The immutable model supports exactly `disabled`, `local_http`, `remote_http`, and `ephemeral_test`. There is no production embedded mode and no alias such as `auto`, `server`, `http`, or `persistent`. Missing or blank mode selects `disabled`; an unknown value raises a bounded semantic error instead of choosing another mode.

Configuration precedence is deterministic: explicit function overrides take priority over an injected or process environment mapping, followed by safe defaults. Only `CHROMA_DEPLOYMENT_MODE`, `CHROMA_HTTP_HOST`, `CHROMA_HTTP_PORT`, `CHROMA_HTTP_SSL`, and `CHROMA_HTTP_TIMEOUT_SECONDS` are recognized as Chroma overrides. The parser never loads `.env`; production startup may already have populated the process environment, while tests inject a plain mapping and therefore do not observe user configuration.

Disabled mode rejects the presence of any HTTP field, including blank values, so contradictory configuration cannot silently become a usable partial configuration. It returns no host or port and does not attempt health checks, collection access, or client construction.

Local HTTP mode requires the literal IPv4 loopback host `127.0.0.1`. Hostnames, wildcard bindings, IPv6 wildcard forms, public addresses, URLs, credentials, paths, queries, and fragments are rejected without DNS resolution or normalization. Its port is explicit and must be an integer from 1 through 65535. SSL defaults to false and must remain false because certificate handling is outside the loopback-only local policy.

Remote HTTP mode is an explicit configuration-only compatibility path for a future HTTP-hosted deployment. It requires a bounded safe hostname or IPv4 address, an explicit valid port, and an explicit SSL boolean. It adds no authentication, credentials, certificate management, cloud assumptions, DNS lookup, or permission to bind a local server publicly.

Ephemeral test mode requires both explicit test-context authorization and an assertion that the endpoint is test-owned. Host and port must come from an injected mapping or explicit override, the host remains `127.0.0.1`, SSL remains disabled, and the existing local port convention `8100` is rejected to prevent accidental reuse. The model has no persistence-path field, so it cannot resolve to protected production storage. Parsing does not start an ephemeral server or construct a fake or real client.

Timeout defaults to 5 seconds when missing or blank. Values must be finite and fall within the inclusive range 0.1 through 30 seconds; invalid values fail instead of being clamped or replaced. Boolean, zero, negative, malformed, NaN, and infinite values are rejected. SSL accepts native booleans and the existing explicit spellings `1/0`, `true/false`, `yes/no`, and `on/off`; arbitrary truthy strings are invalid.

Configuration errors expose stable codes only. They never include rejected values, environment dumps, credentials, or paths. The safe summary is limited to deployment mode, transport, host scope, SSL state, and timeout. The redacted representation omits the remote hostname and port.

Importing or parsing this module performs no Chroma, network, collection, server-lifecycle, or filesystem I/O. Ordinary application configuration intentionally has no Chroma persistence path: local persistence belongs to server operations rather than API/runtime consumers. The narrow vector wrapper retains its feature-specific enablement parser for compatibility but delegates enabled access to the centralized deployment/factory/transport boundary. Production code has no embedded Chroma client path.

## Deterministic collection registry

`backend/chroma_collection_registry.py` is the sole semantic authority for production Chroma collection names and contracts. Its schema is `chroma_collection_registry.v1`. Definitions are frozen, deterministically ordered, detached-serializable data; name resolution and validation do not read environment variables, deployment configuration, source files, Chroma storage, or the network. The registry contains no client or collection objects and provides no open, create, connect, or get-or-create operation.

Two real production collections are registered. The accepted access inventory and current consumers expose no additional production collection:

| Semantic ID | Canonical physical name | Expected schema | Owner | Allowed lifecycles |
| --- | --- | --- | --- | --- |
| `github_evidence` | `github_evidence` | `github_evidence.v1` | `github_evidence` | `read`, `vector_query`, `index`, `migration`, `maintenance`, `test_only` |
| `profile_facts` | `profile_facts` | `profile_facts.v1` | `profile_memory` | `read`, `vector_query`, `write`, `index`, `migration`, `maintenance`, `test_only` |

These schema versions describe the current expected semantic data contracts. Existing live collection metadata has no authoritative persisted version marker, so the registry does not claim that a live record was stamped with either marker. No database record is rewritten. Later migration validation may compare this expected authority with a safely obtained logical inventory.

Automatic creation is false for every production collection. Read, vector-query, status, count, inventory, and configuration paths therefore have no creation authority. Future creation requires a separate explicit bootstrap or cutover design; the registry only records the existing-only policy and implements no production creation lifecycle. Production code contains no create or get-or-create collection call.

Approved consumer categories are intentionally narrower than current access:

| Collection | Readers | Vector query | Writers | Indexers | Migration | Maintenance | Test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `github_evidence` | `central_http_collection_factory`, `github_evidence_metadata_reader` | `github_evidence_vector_reader` | none | `github_evidence_materializer` | `github_evidence_migration` | `github_evidence_maintenance` | `ephemeral_test_fixture` |
| `profile_facts` | `central_http_collection_factory`, `profile_memory_reader` | `profile_memory_vector_reader` | `profile_memory_writer` | `profile_memory_indexer` | `profile_memory_migration` | `profile_memory_maintenance` | `ephemeral_test_fixture` |

The GitHub vector compatibility wrapper maps to the GitHub metadata and vector-query consumer identities through `ChromaReadClient`, the central factory, and the bounded transport. It has no independent `chromadb.HttpClient`, socket preflight, or direct collection ownership. Runtime profile mutations use `profile_memory_writer` for explicit ID deletion and `profile_memory_indexer` for stable-ID embedding upsert. Runtime GitHub context indexing and its normal duplicate cleanup use only `github_evidence_materializer`; ordinary `github_evidence` writer authority remains empty. The registry has no remaining legacy consumer entries and grants no embedded migration authority.

`github_evidence` requires project identity, authoritative repository identity, the existing repository-to-project mapping authority, and project isolation. The registry records only those booleans and does not copy repository mappings, aliases, or raw repository data. `profile_facts` requires the existing profile identity and profile scope; it does not invent authentication or multi-user behavior.

Logical-integrity metadata is a future comparison input, not a fingerprint implementation. The stable allowlist is `index`, `is_list`, and `section` for `profile_facts`; and `chunk_type`, `commit_sha`, `project_id`, `repository`, `repository_project_id`, `source_id`, and `source_type` for `github_evidence`. Document and source bodies, embeddings, patches, raw metadata, raw text, absolute or filesystem paths, secret-bearing fields, run IDs, and transient timestamps are explicitly forbidden. The registry contains field names only, never field values.

The existing `PROFILE_COLLECTION` and `GITHUB_COLLECTION` constants in `backend/memory_store.py` remain runtime-compatible aliases for the registered semantic IDs across centralized runtime access. `backend/chroma_http_vector_search.py` imports its collection name and semantic ID from the registry. Compatibility names are not competing semantic authorities.

Registry synchronization validates every discovered low-level access entry without opening Chroma. Every entry is classified, every collection access resolves to registered authority, and every non-collection entry belongs to the bounded transport, lifecycle health, or isolated test infrastructure. Exact inventory cardinality is reported for audit but is not the architectural contract. The contract is zero unresolved, unknown, forbidden, production-embedded, direct production client, production creation, and fallback entries. Direct names must resolve to a registered collection. Each dynamic helper and temporary fixture has an explicit reviewed binding. Every production collection access maps to an approved consumer, while every embedded access entry is test-only. Read and vector consumers have no collection-creation authority. Non-collection client construction, bounded request, and explicit lifecycle heartbeat remain outside collection resolution.

`backend/chroma_collection_literal_guard.py` provides the separate filesystem-backed AST guard used by tests. It does not import or execute scanned modules and excludes protected storage. New production collection-name assignments and direct literal get/get-or-create calls fail unless they are registry authority values or one of the three reviewed compatibility aliases. Test literals are allowed only under the test source tree and only when test-fixture handling is explicitly enabled. Unrelated strings are ignored.

## Central HTTP client factory

`backend/chroma_http_client_factory.py` is the authoritative transport-construction and semantic collection-access gate for later Chroma consumer migration. It accepts an already validated `ChromaDeploymentConfig` and resolves collections only through `chroma_collection_registry.v1`. It does not parse environment variables, load `.env`, reinterpret deployment modes, derive collection names from user input, or fall back to embedded storage.

Factory construction is inert. `disabled` state rejects transport and collection access before invoking a constructor. `local_http`, `remote_http`, and validated `ephemeral_test` configurations remain `uninitialized` until `get_transport()`, the compatibility `get_client()` alias, or an approved collection-handle request needs transport. One `BoundedChromaHttpTransport` is cached per factory instance. Collection handles are not cached because each request carries distinct lifecycle and consumer authority. Neither initialization nor transport creation performs a heartbeat, port probe, readiness poll, server start, or automatic collection lookup.

The injectable constructor contract receives the complete validated configuration object, without separately re-parsing or copying its fields. The production constructor creates only the WorkAgent-owned bounded transport. There is no direct `chromadb.HttpClient` path in the central factory. Remote authentication, credentials, certificate customization, and cloud-specific behavior remain outside this factory.

`validate_collection_access()` performs no Chroma I/O. It resolves the semantic collection ID, exact lifecycle, and semantic consumer ID, then returns a frozen bounded descriptor. Approved consumers may proceed only within their registry category. Test consumers require both `ephemeral_test` mode and explicit test context. Removed legacy consumer identities are unknown and fail before client construction or collection lookup. Ordinary `github_evidence` write authority remains empty.

`get_collection_handle()` first applies that pure validation, then passes the semantic collection ID to the bounded transport's sole existing-only lookup. The factory never creates, get-or-creates, deletes, or recreates a collection. A validated handle exposes lifecycle-gated `safe_count()`, bounded metadata-only operational pages, bounded business record reads, bounded vector queries, index-only stable-ID upsert, and explicit-ID delete limited to write or index lifecycles. Document content is permitted only when an approved semantic business reader explicitly requests it or an approved indexer writes the existing document contract; operational pages remain document-free. Stored embeddings and URIs are never returned. The underlying bounded accessor is private and excluded from safe summaries and representations.

Errors are stable semantic codes. Disabled configuration, unknown collection authority, unsupported lifecycle, unknown or unapproved consumer, legacy migration requirement, forbidden creation, missing collection, unavailable transport, unexpected transport failure, and collection-authority mismatch remain distinct. Raw exception messages, hosts, ports, credentials, paths, documents, embeddings, and metadata never appear in an error or safe summary. No error triggers fallback, recovery, creation, or server lifecycle work.

The GitHub HTTP vector API remains as a compatibility wrapper, but low-level ownership is centralized. Production modules contain no `PersistentClient`; the sole remaining constructor is isolated test support for disposable parity checks. Server lifecycle, persistence ownership, backup/recovery, logical fingerprints, and cutover remain separate semantic boundaries.

## Ephemeral HTTP integration testing

`tests/chroma_http_test_support.py` owns the disposable real-HTTP test boundary. Importing the module and constructing its server controller start nothing. An explicitly invoked fixture creates a unique child directory under a pytest-owned temporary parent, asks the operating system for a dynamic loopback port other than `8100`, and launches the installed `chroma run` command as a separate process. The command binds only to `127.0.0.1`, receives its temporary persistence path explicitly, runs with that directory as its working directory, and receives a sanitized environment with Chroma deployment values removed and telemetry disabled. It cannot resolve to `information/chroma`, and both the protected root and its descendants are rejected before process creation.

Endpoint ownership is established by observing the port free before launch, retaining the spawned process, requiring that process to remain alive, and accepting the endpoint only after a successful Chroma heartbeat. Startup uses an explicit finite deadline. A child exit or readiness timeout terminates the child and removes test storage. Teardown first requests termination with a bounded wait and then uses a bounded kill fallback. Storage cleanup follows confirmed process exit. This process model uses Windows-compatible `subprocess` behavior and hides the child console on Windows; it remains independent disposable test infrastructure and is not the production lifecycle authority.

Collection preparation is a separate fixture-only admin boundary. It resolves the two existing semantic collection definitions through the production registry, creates them only in disposable server storage, and optionally seeds synthetic embeddings and bounded metadata. The central factory under test still performs only existing-collection lookup. Real HTTP coverage proves lazy client creation, existing and missing collection behavior, explicit non-production preparation, read/vector access, stable-ID upsert idempotency, explicit-ID deletion, post-write vector visibility, project-isolated GitHub deletion, unavailable endpoint handling, process teardown, temporary-directory cleanup, and port release. No temporary collection name is added to the production registry.

Real-server tests use the `chroma_http_integration` pytest marker. The isolated support tests run without a real server; a fast run can exclude the integration boundary explicitly:

```powershell
python -m pytest -m "not chroma_http_integration"
```

The installed package examined by the executable tests is Chroma `1.5.9`. Its public `chromadb.HttpClient` signature is `(host, port, ssl, headers, settings, tenant, database)` and has no timeout argument. Public Settings exposes HTTP keepalive and connection-limit controls, plus distributed log-service, system-database, and query-service timeout fields, but no general HTTP request-timeout field. In particular, `chroma_query_request_timeout_seconds` is not an HTTP transport deadline.

Runtime evidence matches the public surface. With `timeout_seconds=0.1`, a controlled loopback server delaying its response for `0.35` seconds was not interrupted at the configured deadline. Setting the public query-service timeout to one second likewise did not interrupt a controlled `1.2` second HTTP delay. A fast ephemeral Chroma response succeeds, while a closed test-owned endpoint and a server removed after client creation fail safely but take longer than the configured `0.1` second contract. The experiments use no public internet, global socket-timeout mutation, private session patch, method replacement, or thread-kill workaround.

The deterministic public-client capability classification remains `unsupported_by_current_public_client_api`. This historical finding is unchanged: Chroma's installed Python `HttpClient` still offers no generic transport deadline.

## Bounded Chroma HTTP transport

The accepted semantic work item is `Bounded Chroma HTTP Transport Adapter`.

`backend/chroma_http_transport.py` is the single WorkAgent-owned production network boundary behind the central factory. It uses public `httpx` `0.28.1` networking APIs and the public HTTP surface exposed by installed Chroma `1.5.9`; it imports no Chroma implementation internals, constructs no embedded client, applies no monkey patch, does not change global socket state, and has no persistence-path or server-lifecycle behavior. `httpx` is declared directly because production code now imports it rather than receiving it only as Chroma's transitive dependency.

The adapter consumes only an already validated `ChromaDeploymentConfig`. Its single `timeout_seconds` value is mapped explicitly and identically to `httpx` connect, read, write, and pool-acquisition timeouts. Redirect following is disabled, environment proxy and certificate-variable inheritance is disabled with `trust_env=false`, and transport retries are zero. Therefore every individual HTTP request has bounded transport phases. Each allowlisted operation performs one HTTP request, so it is also bounded in this implementation; this is a per-request guarantee, not a claim that a future logical workflow containing multiple sequential calls has one aggregate deadline.

The public HTTP assumptions are deliberately narrow:

- `GET /api/v2/heartbeat` for readiness;
- `GET /api/v2/tenants/{tenant}/databases/{database}/collections/{name}` for existing-only lookup;
- `GET /api/v2/tenants/{tenant}/databases/{database}/collections/{collection-id}/count` for count;
- `POST /api/v2/tenants/{tenant}/databases/{database}/collections/{collection-id}/get` for bounded metadata-only reads;
- `POST /api/v2/tenants/{tenant}/databases/{database}/collections/{collection-id}/query` for bounded vector queries returning only identifiers, distances, and bounded metadata;
- `POST /api/v2/tenants/{tenant}/databases/{database}/collections/{collection-id}/upsert` for bounded stable-ID index writes;
- `POST /api/v2/tenants/{tenant}/databases/{database}/collections/{collection-id}/delete` for bounded explicit-ID deletion.

The adapter explicitly uses `default_tenant` and `default_database`, matching the accepted public-client behavior. Collection names never enter from business callers: semantic IDs resolve through the production collection registry, the returned collection UUID/name/tenant/database is validated, and unknown semantic IDs fail before any request. A missing collection remains missing. Normal access has no create or get-or-create behavior.

Responses are converted into frozen `ChromaTransportCollection`, `ChromaTransportCount`, `ChromaTransportRecords`, `ChromaTransportQueryResult`, and content-free `ChromaTransportMutationResult` values. Request selectors, batch sizes, result counts, embedding dimensions, filters, metadata shape, nesting, documents, aggregate payloads, and strings are bounded. Documents, embeddings, URIs, vectors, metadata values, and response bodies never enter safe summaries or exceptions. Upsert accepts only the installed server's empty-object success or the compatible null success shape; delete accepts only a bounded `deleted` count. Unsupported includes and malformed response schemas fail closed.

Error mapping is deterministic. Read, write, and pool timeout failures become `ChromaTransportTimeout`; connection refusal and connect timeout become `ChromaTransportUnavailable`; malformed JSON, invalid content types, and schema mismatches become `ChromaTransportProtocolError`; authority/client/server HTTP statuses become `ChromaTransportResponseError`; and a lookup 404 becomes `ChromaCollectionMissing`. Underlying exception text, host, port, URL, response body, environment, and filesystem paths are not propagated. Diagnostics expose only deployment mode, the validated timeout, the enforced timeout dimensions, no-retry policy, closed state, and the last safe error category.

The implemented operation matrix is:

| Operation | Current consumer | Future migration owner | Required by adapter evidence now | Production permission | Creates collection | Timeout required |
| --- | --- | --- | --- | --- | --- | --- |
| heartbeat | test readiness | server lifecycle | yes | transport readiness only | no | yes |
| collection lookup | central factory | central factory | yes | existing-only, registry-gated | no | yes |
| count | temporary integration evidence | status/read migration | yes | read-only | no | yes |
| get | temporary integration evidence | reader migration | yes | bounded metadata-only | no | yes |
| query | temporary integration evidence | vector-reader migration | yes | bounded vector query | no | yes |
| add | test-only public Chroma setup outside adapter | none | no | not implemented | no | not applicable |
| upsert | approved indexers/materializer | semantic write client | yes | index lifecycle only | no | yes |
| delete | approved writer/indexers/materializer | semantic write client | yes | write or index lifecycle | no | yes |
| update | none | no established owner | no | not implemented | no | not applicable |
| peek | none | no established owner | no | not implemented | no | not applicable |
| list collections | none | maintenance, if later justified | no | not implemented | no | not applicable |
| create collection | explicit disposable test setup outside adapter | migration/index administration | no | forbidden for ordinary production access | yes, test-only outside adapter | not applicable here |

Executable evidence uses only dynamic loopback ports and disposable storage. With a configured `0.2` second deadline and a controlled `1.0` second delayed response, the bounded request raises `ChromaTransportTimeout` before the delayed response completes. A test-owned endpoint with no listener and an ephemeral server stopped after a successful operation both fail within the bounded tolerance as `ChromaTransportUnavailable`. Fast real-server collection lookup, count, metadata get, and vector query all succeed. Boundary mappings at `0.1`, a normal value, and `30.0` are verified without sleeping for the maximum.

The authoritative access inventory is now 15 discovered and 15 classified, with zero unresolved, stale, mismatched, review-candidate, or unknown entries. The bounded transport owns the sole production `httpx.Client` and request boundary, the factory lookup remains registry-bound, and the test-only public Chroma constructor remains in the shared fixture helper. The explicit server lifecycle and operational reader each contribute one classified heartbeat boundary. The persistence guard records one direct test-only PersistentClient constructor used only by verified-stopped temporary storage; its four parity operations are separately test-classified. Production collection definitions and automatic-creation policy are unchanged; business reads, runtime mutations, and the GitHub vector wrapper consume the central boundary.

The critical distinction is preserved:

```text
chromadb.HttpClient generic timeout = unsupported in installed public API
WorkAgent bounded HTTP transport = enforced by WorkAgent
```

The executable acceptance state is `transport_timeout_support = enforced`. The bounded transport is ready for later consumer work, so `production_consumer_migration_gate = transport_ready`. This releases only the transport prerequisite; it does not migrate the accepted GitHub bridge, `memory_store.py`, any reader, writer, indexer, maintenance path, or production data. Consumer migration, the single-owner persistence guard, backup/recovery, and cutover remain separate work.

## Windows local Chroma server lifecycle

`backend/chroma_server_lifecycle.py` is the sole production-local process authority for an explicitly managed dedicated Chroma server. It provides `start`, `health`, `stop`, and `restart` through one Python CLI:

```powershell
python -m backend.chroma_server_lifecycle start
python -m backend.chroma_server_lifecycle health
python -m backend.chroma_server_lifecycle stop
python -m backend.chroma_server_lifecycle restart
```

Each command also accepts `--json` and emits only the bounded lifecycle result. `windows/chroma_server.ps1` is a parameterized thin wrapper over this CLI; it selects the repository virtual-environment Python when present, adds no configuration or ownership logic, and returns the Python process exit code. Exit codes are deterministic: `0` is success or healthy, `2` is invalid CLI usage, `3` is a disabled or non-local deployment mode, `4` is ownership/state/port conflict, `5` is startup failure or timeout, `6` is shutdown timeout, `7` is a non-ready health result, and `8` is invalid configuration.

Lifecycle operations consume the existing `ChromaDeploymentConfig` and never parse a parallel set of deployment variables. `local_http` is the only supported mode and must remain literal `127.0.0.1`, non-SSL, with the configured authoritative port. `disabled` fails closed for mutating commands and reports disabled health. `remote_http` is not locally owned, while `ephemeral_test` remains under the separate disposable test infrastructure. The lifecycle never rewrites an unsafe host or selects a replacement production port.

Server-only configuration is held by `ChromaServerLifecycleConfig`, separate from application connection configuration. The deterministic production persistence path is `information/chroma`, and runtime state is outside the database at `information/runtime/chroma/runtime_state.json`; both are under the intended application information area. Ordinary `ChromaDeploymentConfig` still has no persistence path. Production paths are fixed. Tests must opt into test ownership, inject distinct temporary paths under a temporary information root, and cannot use any production lifecycle path. Symlinks, junctions, reparse points, dangerous roots, overlapping runtime/persistence paths, and path escapes are rejected. When explicitly invoked under the operator contract below, the lifecycle owns `information/chroma`; the module never initiates a cutover or grants embedded-owner access by itself.

Runtime state uses the strict schema `chroma_server_runtime_state.v1`. Its exact fields are schema, PID, loopback scope, port, process-start token, hashed executable identity, hashed command identity, hashed random ownership token, lifecycle state, and creation timestamp. It contains no persistence path, command text, raw token, environment, collection metadata, documents, or embeddings. Reads are size-bounded and reject corruption or extra fields. Writes use an exclusive same-directory temporary file, flush and `fsync`, then `os.replace`; failed writes remove only their own temporary file. State and its directory are already covered by the ignored `information/*` runtime boundary.

PID existence alone never proves ownership. Before treating a process as owned, the controller verifies the state PID, process creation time, executable identity, complete command identity, a cryptographically random lifecycle token inherited through the controlled process environment, the expected public `chroma run --path ... --host 127.0.0.1 --port ...` flag semantics, and endpoint ownership. On Windows, the `chroma.exe` launcher delegates the listener to a child process. Endpoint inspection therefore considers only descendants of the verified parent that inherited the exact same ownership token. An unrelated process reusing the PID, a creation-time mismatch, a changed command or executable, a missing or wrong token, or an unverified listener is never terminated or adopted.

`start` validates local mode and state before touching a process. A healthy verified existing server returns `ready` idempotently. A live owned but unhealthy server is reported distinctly and is not replaced. Dead state is cleared only when the configured endpoint is also free. An occupied endpoint without matching ownership is a foreign-port conflict; the controller neither kills it nor chooses another port. With a free endpoint, start validates the persistence directory, rejects local `.env` files there, resolves the installed public `chroma` executable, and launches argv-style with `shell=false`, a hidden Windows console, closed standard input, and stdout/stderr directed to the null device. This zero-persistence log strategy prevents unbounded logs and avoids intentionally recording server payloads. The child receives a small allowlisted environment, loopback-only proxy exclusions, telemetry disabled, and the ownership token.

The default overall startup deadline is 20 seconds, distinct from each deployment request timeout. The controller first captures verified process identity and atomically writes `starting`, then repeatedly re-verifies ownership and the listener and probes `GET /api/v2/heartbeat` through `BoundedChromaHttpTransport`. Each readiness request is capped at 0.5 seconds and the remaining overall deadline. Only a successful bounded heartbeat permits an atomic transition to `ready`. Early exit and timeout clean up only the just-spawned process after re-verifying ownership; an ambiguous identity fails closed instead of guessing.

`health` is read-only and bounded. Its semantic states are `disabled`, `unsupported_mode`, `not_running`, `starting`, `ready`, `unhealthy`, `stale_state`, `foreign_port_conflict`, and `ownership_mismatch`. A safe result contains deployment mode, endpoint scope, configured port, ownership/reachability booleans, a bounded detail code, and whether shutdown was forced. It contains no PID, path, command, token, process environment, response body, or collection data. A missing state plus an occupied port remains foreign and is never silently adopted, even if a Chroma heartbeat might respond.

`stop` loads state, proves process ownership, requests graceful termination, and waits for at most five seconds by default. If the process remains alive, it re-verifies the full identity before a force-kill and applies another bounded wait. It then requires the configured endpoint to become bindable within the separate five-second release deadline before removing runtime state. An absent process and free endpoint is deterministic success; dead stale state is removed only with a free endpoint. Ownership ambiguity, PID reuse, or an unowned listener blocks the operation and preserves diagnostic state.

`restart` has no alternate implementation: it performs verified `stop` followed by verified `start`. A failed ownership check or failed stop prevents a new process from starting. A successful restart obtains a new process identity. If a previously owned process crashes, health reports stale state; the next explicit start may clear it only after confirming the endpoint is free. Persistence contents are never rebuilt, migrated, or deleted during crash recovery.

Importing the lifecycle modules, constructing a controller, loading the ASGI application service, and handling ordinary application requests do not start or stop Chroma. The lifecycle is not wired into application lifespan, resume, GitHub, memory, or frontend routes because process ownership is an explicit operator concern rather than request behavior. This keeps operational process authority separate from application client creation, collection permissions, retrieval semantics, and evidence or memory pipelines. The GitHub HTTP vector bridge, registry definitions, and `automatic_creation=false` policy remain present; no production embedded constructor path exists.

Real lifecycle integration is marked `chroma_server_integration`. It uses the installed public `chroma run`, a dynamic non-production loopback port, and temporary persistence/runtime directories. It verifies start, bounded health, restart with a new identity, forced test-owned crash and stale-state recovery, sentinel preservation, stop, no orphan PID, endpoint release, Windows directory-handle release, and an unchanged protected production fingerprint. It never starts the server against `information/chroma`.

## Single-owner Chroma persistence guard

The production persistence invariant is `production_chroma_persistence_owner = dedicated_local_server`. `information/chroma` and every canonical alias or child beneath it are server-owned. An application process, worker, status path, ordinary CLI, or embedded library cannot open that location directly. This remains true whether deployment is `disabled`, `local_http`, `remote_http`, or `ephemeral_test`: disabled means unavailable rather than embedded fallback; local HTTP reserves the directory for the dedicated server; remote HTTP has no local embedded ownership; and test mode never authorizes production storage.

`backend/chroma_persistence_guard.py` is the runtime and repository-static authority. Its immutable context roles are `server_owner`, `production_client`, `legacy_embedded`, `maintenance`, `migration`, `test_only`, and `unknown`. A direct embedded request is evaluated before directory creation or `PersistentClient` construction. Safe decisions contain only role, deployment mode, semantic persistence scope, lifecycle ownership state, disposition, and a bounded reason. They never contain an absolute path, PID, command, token, environment, document, embedding, or Chroma response.

Protected-path authority comes from the validated `ChromaServerLifecycleConfig`; callers cannot nominate an arbitrary string as production storage. The production lifecycle configuration fixes the target to `information/chroma`. Comparison resolves `.` and `..`, follows existing filesystem links through `Path.resolve`, normalizes separators and case with Windows `normcase`, and treats children as protected. Existing symlink or junction aliases therefore resolve to the same target when the operating system exposes them. Dangerous filesystem roots cannot qualify as test-owned storage.

The obsolete `MemoryVectorStore` embedded bootstrap, automatic historical imports, and legacy read/write helpers have been removed after their runtime contracts were replaced by the centralized readers and writers. Disabled and HTTP deployment modes do not silently create an embedded fallback, and maintenance or migration labels cannot authorize one.

Temporary embedded access is the sole runtime exception. It requires the `test_only` role, explicit test context, `ephemeral_test` deployment, an existing non-root test storage authority, and a canonical target strictly beneath that authority. It cannot resolve into production persistence. A test-owned lifecycle persistence target additionally requires authoritative lifecycle health to be exactly `not_running`; ready, starting, unhealthy, stale, foreign-listener, ownership-mismatch, or ambiguous state blocks construction. The guarded direct test constructor runs in a disposable subprocess so process exit releases all embedded handles.

An arbitrary caller setting `role=server_owner` cannot authorize an embedded client. Dedicated server ownership is recognized only by `verify_dedicated_server_owner`, which consumes the lifecycle module's semantic `inspect_chroma_server_ownership` result. That lifecycle path retains PID creation-time, executable/command hashes, ownership token, descendant-listener, endpoint, and heartbeat verification. The persistence guard contains none of that process logic. Test observer injection is accepted only with a validated test-owned lifecycle configuration, so production callers cannot replace lifecycle proof.

Maintenance and migration labels grant no direct access today, even with an explicit operator flag. Future offline tooling would require a separately reviewed identity and an atomic exclusivity protocol in addition to verified server stop and a free, unambiguous endpoint. No current caller needs that exception, so both roles remain denied rather than establishing a broad escape hatch.

The static guard reuses the authoritative Chroma AST scanner instead of maintaining a competing constructor parser. It detects direct `chromadb.PersistentClient`, imported aliases, assignment aliases, wrapper/factory construction, unresolved constructor names, and dynamic constructor resolution. Production constructor count must remain zero and exactly one test-only constructor is allowed in the isolated subprocess helper. Approved maintenance, forbidden, unknown, and embedded-in-exception fallback counts must all remain zero. Any new production constructor, unclassified alias, or `try HTTP / except embedded` pattern fails repository verification.

HTTP transport, central-factory, collection lookup, and GitHub bridge failures never invoke the legacy constructor. The factory and transport modules contain no embedded constructor, and the static fallback audit rejects calls whose persistent/embedded semantics appear inside an exception handler. No server, reader, writer, indexer, or collection API gains an embedded fallback.

No filesystem lock or marker is added. For production this is deliberate: protected embedded access is always denied, so safety does not depend on a time-of-check/time-of-use decision that later permits an application open. The external server remains authorized only through explicit lifecycle launch. The stopped-to-embedded transition exists solely for a test-owned temporary directory controlled by one test orchestration process; a state change after observation could cause a conservative failure or a test-only race, but cannot widen production access. This semantic guard cannot stop an unrelated external program outside WorkAgent from opening files directly; ongoing production operation therefore requires the controlled process discipline and backup/integrity gates documented below.

## Backup, restore, and version compatibility gate

`backend/chroma_backup_recovery.py` is the authoritative byte-recovery boundary. Production capture is permitted only when lifecycle authority reports `not_running`, no WorkAgent-owned process or reachable endpoint exists, port `8100` is free, runtime state is absent, and the protected inventory matches the accepted baseline. Capture never starts, stops, or kills a server. It uses ordinary filesystem reads and copies only; it does not construct a Chroma client, make an HTTP request, open SQLite, or inspect a collection.

The production backup root is `information/backups/chroma`, outside `information/chroma` and already covered by the ignored `information/*` boundary. Test backups use a caller-declared temporary ownership root. Neither normal output nor manifest data contains an absolute path. Backup IDs combine a UTC capture timestamp with the first twelve characters of the source aggregate digest. An existing ID is never overwritten.

Each backup directory contains exactly an immutable `snapshot` and a sibling `manifest.json`. The strict manifest schema is `chroma_backup_manifest.v1`. It records the safe backup ID, semantic source kind and relative source name, sorted relative file paths, sizes, per-file SHA-256 values, deterministic aggregate SHA-256, installed Chroma package version, executable-probe compatibility policy, verified capture state, UTC time, and stopped server state. Extra fields, absolute or traversing paths, duplicate paths, unsupported policy/schema/state, malformed sizes or hashes, and document, embedding, raw-metadata, environment, or secret fields are rejected.

Capture applies a source A/B consistency protocol. It fingerprints the source, copies only inventoried regular files into a uniquely owned staging directory, independently fingerprints the staged snapshot, fingerprints the source again, and requires all three inventories to match. Immediately before manifest publication it rechecks lifecycle authority, endpoint freedom, runtime-state absence, and the source digest. A server that starts during copying therefore blocks finalization. It then atomically writes the verified manifest inside staging and atomically renames the complete directory to its final ID. Copy, hashing, manifest, source-drift, ownership-state, collision, or finalize failure exposes no verified backup and cleans only the invocation's explicitly named staging directory. Integrity depends on sorted relative path, size, and SHA-256, never timestamps, drive letters, or absolute paths. A semantic time-of-check/time-of-use interval remains between the final stopped-state observation and directory rename; the operation does not claim an operating-system lock and requires controlled operator discipline.

`verify` does not trust the manifest. It independently validates the exact two-entry backup layout, strict schema and verified state, recalculates every file size and hash, rejects missing or added files, and recomputes the aggregate. Ordinary commands never write into a verified snapshot. Restore and compatibility always copy from it.

Restore accepts only an absent target strictly beneath an explicit non-root, non-production target authority. It rejects `information/chroma`, canonical aliases and descendants, targets inside the backup root, traversal, existing targets, and symlink or junction escapes detectable through canonical resolution. Restore copies into a uniquely owned sibling staging directory, verifies it against the manifest, atomically finalizes the target, verifies again, and re-fingerprints the immutable snapshot. No command in this boundary can restore directly into production persistence or merge into an existing directory.

The restore drill uses an isolated temporary target and reports only file count, total bytes, aggregate digest, manifest match, and immutable-snapshot status. It proves byte recovery but does not authorize production replacement.

Version compatibility requires executable evidence rather than package-version string equality. The verified restore is copied again into a disposable compatibility workspace because Chroma may rewrite SQLite or HNSW internals merely by opening them. The accepted lifecycle starts a separate test-owned server on a dynamic loopback port against only this second copy, with bounded startup, heartbeat, shutdown, ownership checks, and cleanup. The central factory uses `ephemeral_test_fixture` authority to perform existing-only lookups for exactly `github_evidence` and `profile_facts`, followed by bounded counts. Missing collections are reported and never created. No documents, embeddings, record bodies, or raw metadata are needed.

The compatibility workspace is fingerprinted before server start and after verified shutdown. A difference is reported as `server_open_mutated_internal_storage=true` with only aggregate digests and a changed-file count. Such mutation does not modify the immutable snapshot or verified restore and is not by itself an incompatibility. Results are exactly `compatible`, `migration_required`, `incompatible`, or `unknown`. Only successful startup, heartbeat, both required existing-only lookups, and both safe counts produce `compatible`; package-version equality alone never does. Explicit supported migration evidence that has not been accepted produces `migration_required`, fatal open/use failure produces `incompatible`, and insufficient evidence produces `unknown`.

Rollback-source readiness requires independent backup verification, a passing restore drill, unchanged immutable bytes, and executable compatibility classified as `compatible`. Only that conjunction yields `production_cutover_recovery_gate=recovery_ready`. The gate is recovery evidence, not cutover permission.

Later rollback triggers include server startup failure after cutover, a required collection missing, logical-fingerprint or unexpected cardinality drift, failed read/vector or write/index validation, ownership ambiguity, or a new persistence corruption error. A future in-place restore must require explicit operator action, verified server stop, a free endpoint, unambiguous ownership, an approved verified backup, a pre-restore backup where possible, staged atomic replacement, and post-restore verification. This boundary deliberately implements none of those production replacement actions.

The single-owner rule is unchanged. Filesystem capture needs no embedded exception; approved maintenance `PersistentClient` access remains zero. Compatibility runs only against a disposable restored copy through bounded HTTP. Business read/vector migration and controlled cutover consume this recovery evidence but do not change it: no writer, indexer, mutation method, production `PersistentClient` constructor, collection definition, `automatic_creation=false` policy, or logical baseline is changed.

The bounded operator commands are:

```powershell
python -m backend.chroma_backup_recovery capture
python -m backend.chroma_backup_recovery verify --backup <backup-id>
python -m backend.chroma_backup_recovery restore-drill --backup <backup-id>
python -m backend.chroma_backup_recovery compatibility --backup <backup-id>
python -m backend.chroma_backup_recovery gate --backup <backup-id>
```

They print only semantic states, safe IDs, counts, versions, and hashes. They never print backup/source absolute paths or application data.

## Logical Chroma collection fingerprints

`backend/chroma_logical_fingerprint.py` is the single logical-integrity authority for registered production collections. It deliberately separates offline byte integrity from server-owned logical integrity. The immutable recovery snapshot continues to use sorted file inventories and SHA-256. A running Chroma server may legitimately rewrite SQLite or HNSW bytes, so those mutable internal bytes are never inputs to a logical collection fingerprint. Byte changes alone are not classified as logical corruption.

The strict per-collection schema is `chroma_logical_fingerprint.v1`. Each safe value contains only the semantic collection ID, canonical collection name, collection schema version, record count, record-ID digest, allowlisted-metadata digest, registry/authority digest, aggregate digest, and `valid` integrity state. It never contains record IDs, metadata rows, documents, embeddings, URIs, absolute paths, Chroma internal metadata, or source bodies. Models are immutable, reject extra fields, validate every SHA-256 value, and expose only bounded safe summaries.

The registry remains the collection authority. Fingerprinting resolves an existing registered semantic collection through `ChromaHttpClientFactory`, and the handle delegates metadata-only pages to `BoundedChromaHttpTransport`. Collection creation and get-or-create remain impossible. The authority digest covers the registry schema, semantic and physical collection identities, collection schema version, sorted logical metadata allowlist, automatic-creation policy, required authority flags, rejected unsafe metadata-field identity, and excluded volatile-field identity. For GitHub evidence it additionally hashes the existing authoritative repository-to-project and alias mapping; mapping rows are never emitted. For profile facts it records only the existing single-profile scope and invents no user, tenant, or authentication identity.

The accepted metadata allowlists are consumed directly from the registry. `profile_facts` uses `index`, `is_list`, and `section`. `github_evidence` uses `chunk_type`, `commit_sha`, `project_id`, `repository`, `repository_project_id`, `source_id`, and `source_type`. Every allowlisted key is represented as absent or present, with null, boolean, integer, finite float, and string encoded as distinct types. Keys and records are canonically sorted and JSON-encoded with length-safe structure before hashing. Unsupported nested allowlisted values, non-finite numbers, secrets, absolute paths, URIs, unsafe body containers, documents, embeddings, and raw metadata fail closed. Non-allowlisted ordinary fields and registry-declared volatile timestamps/run IDs are intentionally excluded, so changes to them do not alter the logical digest.

GitHub authority uses the existing project/repository identity artifact and never builds or fetches a second mapping. Repository metadata must resolve through one exact, confirmed, conflict-free authority entry. Explicit `project_id` and `repository_project_id` values, when present, must equal that mapped project. Legacy repository-only records may obtain normalized project authority only from the unique accepted mapping; repository-name guessing and ambiguous, unknown, unauthorized, or cross-project records fail. This preserves the previously accepted safe normalization boundary without weakening project isolation.

Record IDs are exact strings, sorted independently of Chroma response order, and represented only by a SHA-256 digest. Blank, oversized, or duplicate IDs fail. The count-before/paginated-get/count-after protocol requires both counts and retrieved unique cardinality to agree. Pages contain at most 200 records, collection cardinality is capped at 100,000, each canonical metadata row at 16,384 bytes, and total canonical metadata at 32,000,000 bytes. A short page, duplicate across pages, overrun, changing count, oversized collection, unsafe response include, or non-convergent read is an unstable snapshot and cannot yield a valid fingerprint. The Chroma API does not provide an atomic collection snapshot; this fail-closed consistency protocol is the explicit concurrency limitation.

The digest layers are deterministic. `record_id_digest` hashes the sorted ID array. `metadata_digest` hashes sorted ID-to-typed-allowlisted-field rows. `authority_digest` hashes stable registry and semantic authority. `aggregate_digest` hashes the fingerprint schema, collection identities, record count, and the three component digests. Logical equality is therefore independent of page boundaries, response order, process identity, server restart, and mutable SQLite/HNSW bytes. It is sensitive to record addition, removal, rename, allowlisted metadata changes, repository/project authority changes, and collection registry identity changes.

`compare_logical_fingerprints` returns only `match`, `record_count_mismatch`, `record_identity_mismatch`, `metadata_mismatch`, `authority_mismatch`, `collection_missing`, `schema_mismatch`, or `invalid`. It never emits a differing ID or metadata value. Logical fingerprints intentionally exclude documents and embeddings; they protect stable collection identity, record identity, safe metadata, and authority, not every stored byte. The immutable backup remains the byte-for-byte recovery proof for the excluded storage surface.

The pre-cutover baseline schema is `chroma_logical_baseline.v1` and is written atomically under the ignored `information/chroma_logical_fingerprints` operational boundary. Capture first independently verifies the immutable backup, restores it to an isolated target, copies that restore into a disposable server workspace, and starts a test-owned lifecycle server only against the disposable copy. It fingerprints both registered collections twice, stops and restarts the server, fingerprints again, and requires identical logical values. It separately records bounded before/after workspace byte hashes, verifies the immutable backup again, verifies that protected production bytes remain identical, executes bounded synthetic logical-mutation checks, and cleans the temporary server workspace. The baseline links only the safe backup ID and digest, captured Chroma version, registry schema, and fingerprint schema; it contains no absolute backup path.

The executable gate is `production_logical_integrity_gate=logical_integrity_ready` only when every registered collection is present and valid, repeated and restarted reads are deterministic, disposable internal bytes changed while logical values stayed stable, synthetic logical mutations are detected, excluded fields remain excluded, the immutable recovery source remains unchanged, and protected production persistence remains unchanged. Otherwise it is `blocked`. This gate is integrity evidence only. It does not authorize a production server start, consumer migration, write/index migration, legacy-constructor removal, restore, or cutover.

The bounded operational commands are:

```powershell
python -m backend.chroma_logical_fingerprint capture --backup <backup-id>
python -m backend.chroma_logical_fingerprint gate --baseline <ignored-baseline-artifact>
```

Neither command starts Chroma against `information/chroma`. No API route, frontend control, LLM, GitHub request, fallback reader, rebuild, backfill, or materialization path is involved.

## Operational status and count reads

`backend/chroma_operational_reader.py` is the approved backend-only adapter for collection readiness, existence, safe counts, and bounded operational repository inventory. Its path is strictly application status consumer → `ChromaOperationalReader` → `ChromaHttpClientFactory` → `BoundedChromaHttpTransport` → local Chroma server. It imports no `chromadb`, constructs no client directly, opens no persistence path or SQLite database, creates no collection, and has no HTTP-to-embedded fallback.

The adapter is registered as the read-only consumer `chroma_operational_reader` for both existing collection definitions. Registry names, schema versions, lifecycle allowlists, authority requirements, and `automatic_creation=false` remain unchanged. Every lookup is existing-only with `creation_requested=false`. A bounded heartbeat precedes collection lookup, followed by a safe count. GitHub status may additionally read bounded metadata-only pages to build repository summaries containing only repository, authoritative project ID, source type, and the already exposed safe update timestamp. Record IDs are used only for local duplicate detection and are never returned; documents, embeddings, URIs, raw metadata, content, paths, and server response bodies are never exposed.

`MemoryVectorStore.profile_count`, `github_count`, and `github_metadata_status` now delegate to this adapter. They do not inspect `information/chroma`, open immutable SQLite, initialize embedded collections, or call an embedded collection count. The existing status response shape is preserved. Unavailable, disabled, timed-out, missing, malformed, or integrity-failing server reads fail closed to a safe unavailable/empty result without an embedded fallback.

## Business read and vector access

`backend/chroma_read_client.py` is the lazy semantic boundary for business reads and vector queries. The path is business or retrieval consumer → immutable semantic result → central HTTP factory → bounded transport → dedicated server. Import and construction do not resolve configuration, connect, heartbeat, open a collection, read persistence, or start a server. Each invoked operation validates an existing registered collection, approved lifecycle, consumer identity, explicit selector or pagination bound, metadata projection, document policy, and response cardinality before returning immutable records or hits.

`MemoryVectorStore.read_profile`, `read_github_contexts`, `inspect_github_vector_metadata`, and `read_github_document` now use this boundary. Profile list membership and index ordering, scalar reconstruction, query selection, GitHub context ordering, optional missing-collection behavior, and bounded document lookup remain unchanged. Unavailable transport is not converted into fabricated empty profile/context success. The methods do not inspect `chroma.sqlite3`, require the local persistence directory, or access embedded collection fields.

The GitHub vector compatibility wrapper now delegates vector query, readiness metadata, and compatibility fingerprint reads to the same semantic client. Compatibility fingerprint reads remain metadata-only. Query embeddings, result limits, server order, distances, metadata allowlists, project isolation, repository mapping authority, unauthorized-record rejection, and fail-closed empty behavior are preserved. Retrieval mode selection remains external and default-off; disabled retrieval performs no vector I/O, and enabled HTTP failure never invokes legacy retrieval.

Business reads allow bounded documents only where current product reconstruction requires them. Operational/status readers remain metadata-only. Documents and embeddings remain intentionally excluded from the authoritative logical collection fingerprint. Read-migration parity tests may compare document hashes without changing that fingerprint scope. Stored embeddings and URIs are rejected, per-record and aggregate content limits are enforced, response bytes remain capped, metadata is allowlist-projected, pagination detects duplicates and changing snapshots, and safe summaries/errors contain no content. The record maximum is 1,000 for profile and GitHub context reads, 10,000 for metadata readiness, and vector queries remain capped by the transport at 100 and by GitHub retrieval at 20.

The authoritative low-level access inventory requires discovered and classified entries to remain equal, with zero unresolved, stale, mismatched, review-candidate, unknown, or forbidden entries. Production embedded read, vector, write, index, materialization, maintenance, migration, and recovery access are all zero. The remaining embedded inventory entries are test-only: one constructor and four logical parity operations against disposable storage. Direct production `PersistentClient`, `chromadb.HttpClient`, and independent business-owned HTTP Chroma access are also zero.

## Semantic write and index access

`backend/chroma_write_client.py` is the lazy semantic boundary for runtime mutations. The path is approved writer/indexer/materializer → immutable `ChromaWriteRecord` → `ChromaWriteClient` → central factory → bounded HTTP transport → dedicated server. Import and construction perform no configuration resolution, network, filesystem, collection lookup, or server start. Every operation validates its semantic collection, lifecycle, consumer, existing-only collection policy, request bounds, and authority before mutation. There is no implicit retry and no HTTP-to-embedded fallback.

Upsert is authorized only under the `index` lifecycle because current records include caller-generated embeddings. Delete requires an explicit non-empty bounded ID list and is authorized only under `write` or `index`. The transport implements neither `add` nor `update`, exposes no generic request method to business callers, and never creates or deletes a collection. Missing collections, unavailable transport, timeouts, authority violations, and malformed success responses are distinct failures rather than false success.

Requests are capped at 100 upsert records, 1,000 delete IDs, 512 characters per ID, 512,000 document characters per record, 1,500,000 document characters per request, 128 metadata fields, 32,768 metadata bytes per record, 1,000,000 metadata bytes per request, 8,192 finite embedding dimensions, and 2,000,000 serialized request bytes. Models are frozen. Safe results expose only semantic collection, operation, requested count, accepted count, and applied status. Documents, embeddings, metadata values, payloads, raw response bodies, endpoints, environment values, and paths are excluded from representations, summaries, and semantic errors.

`MemoryVectorStore.replace_profile`, `delete_profile`, `store_github_contexts`, `cleanup_github_repositories`, and the runtime similarity/upsert helper now combine only `ChromaReadClient` reads with `ChromaWriteClient` mutations. Stable record IDs, exact document construction, stored metadata, embedding generation, the `0.12` similarity threshold, one-match deduplication, stale profile deletion, empty profile replacement, selected profile-item deletion, and GitHub canonicalization remain unchanged. Replace and cleanup may require multiple bounded requests; delete-then-upsert or upsert-then-delete workflows are not claimed to be transactionally atomic, and any later failure is surfaced without fallback.

GitHub stored metadata is not expanded during transport migration. A separate validation-only authority row must provide exact `project_id`, canonical repository, and matching `repository_project_id` for every requested record or deleted ID. The write client accepts only the existing verified project/repository authority, rejects missing or conflicting mappings, unknown repositories, wrong project fields, and cross-project batches before network access. Cleanup prevalidates all repository groups and separates deletes by project, so one project's request cannot target another project's IDs. Ordinary GitHub writer authority remains empty; the approved materializer owns only this existing index/normal-cleanup workflow.

Historical automatic profile and GitHub JSON imports were unreachable from ordinary production and depended on embedded get-or-create bootstrap. They and their private legacy helpers were removed rather than converted into a hidden maintenance exception. No replacement import is auto-invoked. Any future historical import or collection bootstrap requires a separate explicit existing-collection design; neither the write/index boundary nor application startup starts a production server, mutates protected persistence automatically, or authorizes another cutover.

## Controlled production cutover and operator contract

The software default remains `CHROMA_DEPLOYMENT_MODE=disabled`. Importing or starting the WorkAgent backend does not start Chroma, select a production persistence path, or enable an embedded fallback. The controlled production deployment is an operator-owned, explicit `local_http` configuration:

```powershell
$env:CHROMA_DEPLOYMENT_MODE = "local_http"
$env:CHROMA_HTTP_HOST = "127.0.0.1"
$env:CHROMA_HTTP_PORT = "8100"
$env:CHROMA_HTTP_SSL = "false"
$env:CHROMA_HTTP_TIMEOUT_SECONDS = "5"
python -m backend.chroma_server_lifecycle start
```

The lifecycle command is the only production-local server owner. It starts the dedicated server against `information/chroma`, writes bounded ownership state under `information/runtime/chroma`, verifies the owned process and descendant listener, and requires a bounded heartbeat. WorkAgent remains HTTP-only through the central factory and bounded transport. Production `PersistentClient`, direct `chromadb.HttpClient`, independent HTTP clients, collection creation, and HTTP-to-embedded fallback remain prohibited.

Production startup order is:

1. Set the explicit loopback deployment configuration in the operator-owned process environment.
2. Start Chroma through `backend.chroma_server_lifecycle`.
3. Require lifecycle health to be `ready`, process ownership to be valid, and `127.0.0.1:8100` to be owned by that process or its verified descendant.
4. Resolve both registered existing collections through `ChromaHttpClientFactory` and compare their counts and logical fingerprints with the accepted baseline.
5. Start or use WorkAgent only after the health and logical-integrity gates pass.

Production shutdown and backup order is:

1. Stop WorkAgent write traffic and stop the backend when required.
2. Stop Chroma explicitly through `backend.chroma_server_lifecycle`.
3. Require the lifecycle state to be absent/not running, port `8100` to be free, and runtime ownership state to be removed.
4. Capture a backup only while those stopped-state conditions remain true.

The initial controlled real-data cutover used installed `chromadb 1.5.9`. Before first ownership, the protected directory contained 10 files and 8,499,528 bytes with aggregate SHA-256 `a47ec430cbf101e98c2560437ebf68c9dd64b11f1299215d0df9c9285472d70f`. The fresh pre-cutover recovery source is `20260819T190442Z-a47ec430cbf1`; the older accepted source `20260810T053709Z-a47ec430cbf1` remains preserved. After the first verified start and clean stop, Chroma had legitimately rewritten five SQLite/HNSW internal files; file count and byte count were unchanged, while the offline aggregate became `832aa293e0816ebe64784a2ac740bfa18038047b0ef1df3e52659535132ba05a`. The stopped post-cutover recovery source is `20260819T191432Z-832aa293e081`.

That internal byte change was not accepted as a new logical baseline. Before application smoke, after smoke, in a disposable restore of the post-cutover backup, and after the real restart, `profile_facts` remained at 13 records and `github_evidence` at 5 records. Their record-ID, metadata, authority, and aggregate digests matched the unchanged accepted logical baseline exactly. Real profile, GitHub metadata, vector, explicitly enabled retrieval-v2, and product-status reads completed through the centralized HTTP architecture with zero production record mutations or collection creation. Retrieval-v2 remains default-off after its bounded smoke.

The ordinary `capture` CLI remains pinned to the accepted pre-cutover byte baseline and therefore fails closed when presented with unexplained byte drift. Post-cutover backup capture is not a rebaseline command and is not exposed as a looser CLI mode. It requires a separately completed logical-integrity gate plus an exact full stopped-server inventory supplied to the existing backup engine; the engine compares that inventory byte-for-byte at both capture preflights, copies only validated regular files, and independently verifies the finalized backup. Historical and pre-cutover backups are never replaced.

If heartbeat or ownership fails, WorkAgent must not use Chroma. If logical fingerprints differ, stop the server and classify rollback as required without rewriting the baseline. If a read, vector, or retrieval smoke fails while logical state remains unchanged, stop the server and classify the cutover as blocked; do not restore merely to mask an application/runtime problem. Production replacement remains an explicit operator procedure using a verified backup, a fully stopped lifecycle, a free endpoint, staged restore, and independent byte and logical verification. No automatic application startup, automatic rollback, frontend control, debug endpoint, or production test-record write is part of this contract.
