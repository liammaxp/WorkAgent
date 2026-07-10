# Phase 2 GitHub Context Evidence Memory

Phase 2 prepares source-traceable evidence memory for later resume generation improvements. It is currently a separate persistence, extraction, inspection, and debugging pipeline. It is not connected to resume generation.

## Enable Phase 2

Set the feature flag before starting the backend:

```text
USE_GITHUB_CONTEXT_PHASE2=1
```

Phase 2 is disabled by default. When disabled, its build routes return safe disabled responses and do not write Phase 2 records.

## Storage

The default storage directory is:

```text
information/phase2_evidence_memory/
```

Override it with:

```text
PHASE2_EVIDENCE_MEMORY_DIR=<path>
```

Tests use temporary override directories and must not write fixtures to the default storage directory.

## Pipeline

The deterministic pipeline runs on already-saved Phase 2 raw sources:

```text
github_raw_sources
  -> evidence_chunks
  -> raw_change_summaries
  -> evidence_cards
  -> capability_facts
```

The build orchestrator does not trigger GitHub sync. Raw sources are populated only through the existing GitHub context sync flow while the Phase 2 flag is enabled.

## Debug Routes

- `GET /api/github/context/phase2/status`: feature state, counts, and compact project summaries.
- `GET /api/github/context/phase2/health`: pipeline completeness and the next recommended action.
- `GET /api/github/context/phase2/inspect`: bounded safe samples for every Phase 2 layer.
- `POST /api/github/context/phase2/build`: manually runs all or selected extraction stages.

Status, health, and inspect are read-only. Safe responses do not include `raw_text` or full chunk text.

## Frontend Panel

Open the GitHubContext page and find **Phase 2 Evidence Memory**. The panel displays status, health, project summaries, safe inspect samples, and manual build controls.

The build button calls only the Phase 2 build route. It does not call GitHub sync.

## Manual Test

1. Set `USE_GITHUB_CONTEXT_PHASE2=1` and start the backend.
2. Sync GitHub context through the existing GitHubContext flow.
3. Open the GitHubContext page.
4. Check Phase 2 status and health.
5. Click **Run Phase 2 Build**.
6. Inspect the bounded safe samples.

If health reports no raw sources, sync GitHub context first. Do not expect the build action to fetch GitHub data.

## Current Safety Boundary

- No resume generation integration yet.
- No bullet writer integration yet.
- No `current_project_compact_facts` integration yet.
- No Chroma indexing or retrieval changes yet.
- No automatic GitHub sync or automatic Phase 2 build.
- No raw text or full patch display in Phase 2 preview, inspect, or frontend samples.
- No LLM calls in the Phase 2 extraction pipeline.

