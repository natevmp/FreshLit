# Architecture Specification: FreshLit — Automated Literature Monitoring Agent

> **Revision note.** This is an adapted version of the original spec, incorporating: (1) project rename to **FreshLit** (package `freshlit`), (2) migration of the primary LLM provider to the official **openai-codex SDK** with ChatGPT OAuth (the former opencode-go integration is retained only for explicit rollback), (3) local **SPECTER2** embeddings, (4) an editable **research profile** Markdown file driving topics/keywords/vector filtering, (5) a central **configuration loader** (the original spec defined config file contents but no loading mechanism), and (6) robustness/security hardening surfaced by design review (request timeouts + pagination + retries, strict structured-output validation, isolated/disabled Codex capabilities, NULL-safe/DOI-normalized dedup, path-containment + atomic vault writes, secret hygiene).

This document describes the system architecture. The executable implementation and README setup instructions are authoritative. Search selectors are explicitly user-authored; there is no AI query-generation stage. Profile front matter supplies topics/source/venue choices, and the Markdown body supplies keywords and research context to generic scoring and synthesis prompts.

---

## 1. System Overview

FreshLit is a modular, four-stage ETL pipeline that can be scheduled externally (weekly or bi-weekly). It ingests newly published papers and preprints, filters out duplicates and off-topic items, performs LLM-driven structured synthesis, and outputs Markdown to a local directory, optionally an Obsidian vault. Current secure runtime/delivery support requires macOS or Linux/POSIX.

```text
1. INGESTION ──► 2. DEDUP & FILTERING ──► 3. LLM SYNTHESIS ──► 4. OBSIDIAN DELIVERY
   OpenAlex /         SQLite ID Check          Pydantic Schema      YAML Frontmatter
   Europe PMC         Title String Matching    Context Extraction   Dataview Metadata
                      Vector Similarity        Field Pulse Pass     Vault File Writer
                      LLM Scoring + Journal Multiplier
```

**Runtime stack**

- **LLM scoring & synthesis (primary):** the official `openai-codex==0.147.0` SDK using an existing ChatGPT OAuth profile. The default provider is `codex` and the default model is `gpt-5.6-luna`.
- **LLM provider boundary:** `freshlit/utils/llm.py` exposes one provider-neutral structured client. It constructs only the explicitly selected provider and never falls back. The opencode-go OpenAI-compatible client and `instructor` mode negotiation remain available only when `gateway` is explicitly selected for rollback.
- **Structured output:** Pydantic response schemas are copied and recursively normalized to OpenAI strict JSON Schema (all object properties required, defaults removed, extra properties forbidden, and unsupported map-like schemas rejected). The returned JSON then undergoes strict Pydantic validation before use.
- **Embeddings:** local `allenai/specter2_base` via `sentence-transformers` (SPECTER2 is a scientific-text domain model). Configurable device (`cpu`/`cuda`).
- **Configuration:** a central loader (`freshlit/utils/config.py`) validates and exposes all tunables at startup. See Section 5.

**Codex runtime isolation and lifecycle**

- Production startup requires an existing, current-user-owned, dedicated `FRESHLIT_CODEX_HOME` disjoint from the repository with exact mode `0700` permissions and forces both the ChatGPT login method and built-in OpenAI provider. FreshLit neither initiates login nor opens/parses auth files. It starts the app-server through `/usr/bin/env -i` with only explicit profile, temporary-directory, locale, and bundled-runtime path values, then verifies the effective safety configuration, absence of custom providers/endpoint redirects, account type, and requested model. This hardened launch currently requires a POSIX system with `/usr/bin/env` and has no inherited API-key/billed OpenAI API path.
- A pipeline run creates one shared Codex runtime. Every structured request (and every retry) starts a fresh ephemeral thread in a fresh temporary working directory outside the project. Request directories are deleted after use.
- Every thread uses deny-all approval and a read-only sandbox. Shell/exec tools, network/web search, MCP, agents, skills, hooks, apps, memory, history persistence, telemetry, feedback, update checks, and inherited shell environment are explicitly disabled. The client also rejects completed turns that report tool-like activity.
- The low-level approval callback explicitly declines server requests. A local override of the pinned SDK's private message router retains early `turn/completed` notifications and atomically replays buffered events before publishing the live queue. SDK upgrades require re-review and regression testing of this compatibility override.
- Batch workers may issue requests concurrently through the shared runtime, bounded by `llm.workers`. The orchestrator owns the runtime with a context manager and closes it cleanly and idempotently after filtering and synthesis, including exceptional exits.
- `llm.timeout_seconds` is one total deadline across thread startup, turns, and retries for a structured request. An over-deadline turn is interrupted; failed bounded cancellation poisons and terminates the shared runtime before the request fails after a short grace. Successful cancellation fails only that request and leaves the runtime available. `llm.max_retries` applies only to schema/Pydantic-invalid output and SDK-classified retryable failures. Other failures stop that request; none trigger per-paper replay, a provider change, or an API-key fallback.
- Explicit gateway rollback uses a parse-only Tenacity retry policy. Response-mode negotiation is limited to Pydantic validation or HTTP 400/404/422 errors with explicit mode-specific error metadata; generic request, model, endpoint, authentication, rate-limit, and transport failures are terminal after one HTTP attempt.
- Delivery trusts the existing configured vault root, then opens/creates nested digest directories using nofollow directory-relative operations. It holds in-process and advisory locks across the complete merge and atomic replacement, retaining a content-free `.freshlit-delivery.lock` entry. Unsupported platforms fail closed. Uncooperative external editors are outside the advisory-lock guarantee.

---

## 2. Directory Structure

The codebase is organized as follows on the local filesystem:

```text
freshlit/                        # project root
├── config/
│   ├── settings.yaml            # Query parameters, thresholds, vault paths, model selection
│   ├── journal_tiers.json       # Legacy ISSN weights, empty in new installs
│   └── research_profile.md      # Private front matter + Markdown profile
├── data/
│   ├── cache.db                 # SQLite database for state persistence
│   ├── profile_exemplars.json   # Generated target research profile vector (built from research_profile.md)
│   └── last_ingest.json         # Generated cache of the last ingestion batch (--from-cache)
├── freshlit/
│   ├── __init__.py
│   ├── main.py                  # Pipeline orchestrator / CLI entrypoint
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── ingestion.py         # Node 1: API fetchers (OpenAlex, Europe PMC)
│   │   ├── filtering.py         # Node 2: Dedup, embeddings, LLM scoring
│   │   ├── synthesis.py         # Node 3: Structured paper summarization
│   │   └── delivery.py          # Node 4: Obsidian Markdown rendering & I/O
│   └── utils/
│       ├── config.py            # Central configuration loader (Pydantic-validated)
│       ├── profile.py           # Typed user profile parsing and legacy migration
│       ├── onboarding.py        # Local initialization, diagnostics, explicit topic lookup
│       ├── db.py                # SQLite wrapper functions
│       ├── llm.py               # Codex structured client + explicit gateway rollback client
│       └── embeddings.py        # SPECTER2 embedding model loader (lazy singleton)
├── .env                         # Secrets (gitignored)
├── .env.example                 # Documented environment template (no credentials)
├── .gitignore
├── pyproject.toml               # Package metadata + `freshlit` console-script entry point
├── requirements.txt
└── README.md
```

---

## 3. Node Specifications

### Node 1: Ingestion Engine (`freshlit/nodes/ingestion.py`)

* **Primary Source:** OpenAlex REST API (`https://api.openalex.org/works`).
* **Optional Biomedical Source:** Europe PMC REST API; opt-in in modern profiles, not an automatic fallback.
* **Ingestion Logic:**
  1. Calculate date window: `[current_date - lookback_days]` to `[current_date]` (closed-open, timezone defined in `settings.yaml`).
  2. Issue topic requests for user-selected IDs in profile `search.openalex_topics`, resolved to runtime `ingestion.openalex_topics`, AND the publication-date window.
  3. Combine with independent keyword queries using the user's explicit `## Keywords` bullets — quoted keywords are OR-combined into chunked OpenAlex `search=` queries. Topic and keyword result streams form a union, not an intersection. An empty topic list disables that stream.
  4. Authenticate with the **OpenAlex API key** (`api_key=` query param, or `Authorization: Bearer`). *(The legacy `mailto=` polite-pool parameter was deprecated Feb 2026 and is no longer used.)*
* **Resilience (hardening):**
  * Set explicit connect/read timeouts on every request (`api.request_timeout_seconds`).
  * Follow cursor-based pagination (`meta.next_cursor`) until exhausted, bounded by `api.max_pages` and `ingestion.max_results_per_query`.
  * Retry transient failures (`429`/`403`/`5xx`) with exponential backoff, honoring `Retry-After` — but capped at 120 s per wait, after which the query is skipped (so an exhausted quota never parks the pipeline for hours).
  * Validate and normalize each record defensively (defaults for missing fields); isolate per-record errors so one malformed item never aborts the run.
* **Output Schema (Internal Standardized Model):**

```python
from pydantic import BaseModel, Field
from typing import Optional

class RawPaper(BaseModel):
    id: str                             # OpenAlex ID or source URI
    doi: Optional[str] = None
    title: str
    authors: list[str]
    publication_date: str
    venue_name: Optional[str] = None
    venue_issn: Optional[str] = None
    venue_impact_factor: Optional[float] = 0.0
    abstract: str
    pdf_url: Optional[str] = None
    source_type: str                    # 'journal' or 'preprint'
```

---

### Node 2: Deduplication, Vector Filtering, & Scoring (`freshlit/nodes/filtering.py`)

This node operates as a 4-step reduction funnel:

1. **Exact Database Check:**
   Query `cache.db` for existing DOIs or primary IDs. Drop matched records immediately. **Normalize DOIs before comparison** (lowercase, strip `https://doi.org/` prefix, strip trailing punctuation/whitespace). Use **NULL-safe matching** (`IS NOT DISTINCT FROM` semantics) so records lacking a DOI fall back to primary-ID comparison rather than never matching.
2. **Preprint/Journal Fuzzy Deduplication:**
   Normalize titles (strip punctuation, lowercase, stop-word removal). If title Jaro-Winkler similarity exceeds the configurable threshold (`filtering.dedup_title_similarity`, default `0.92`) AND first-author surnames match a historical record, treat as duplicate. Apply a **minimum-title-length guard** and robust surname parsing: the first-author surname extractor handles `Last, First`, `First Last`, and `Last Initial(s)` (e.g. Europe PMC's `"Parks M"`), plus organizational authors / "et al.". Precedence rule: peer-reviewed version wins over preprint; otherwise keep the newer `publication_date`.
3. **Vector Similarity Filter:**
   * Model: local `allenai/specter2_base`.
   * Embed `title + abstract`. Compute cosine similarity against the target profile vector (cached in `data/profile_exemplars.json`). Exact Markdown body and separate keyword text are the embedding inputs; front matter is excluded. Model/input fingerprints invalidate stale vectors automatically before a CLI run.
   * Drop records with missing/short abstracts (< 50 chars) and those below `filtering.vector_threshold` (default `0.65`).
   * Rank survivors by similarity and keep only the top `filtering.max_llm_candidates` (default `40`) — SPECTER2 similarities cluster in a narrow band, so ranking + a candidate budget is the effective filter; the threshold is only a floor.
4. **LLM Reasoning & Weighted Scoring:**
   Pass surviving items through the selected structured client against a rubric grounded in the research profile text. Scoring is **batched**: up to `filtering.llm_batch_size` papers per structured request (via the `LLMEvaluationBatch` wrapper model), run concurrently at `llm.workers`, with per-paper fallback if a batch request fails. Request-local opaque IDs (`P0001`, etc.) avoid exposing source IDs as control identifiers. The response must have exactly one item per requested ID; cardinality, unknown IDs, duplicates, and missing IDs are validated before results are restored to request order. Contract failures trigger only per-paper retry, never a provider change.

```python
from pydantic import BaseModel, Field

class LLMEvaluation(BaseModel):
    relevance_score: int = Field(..., description="Relevance score from 1 to 10 based on rubric")
    passes_rubric: bool = Field(..., description="True if score >= 7")
    methodology_tags: list[str]
    fit_rationale: str = Field(..., description="1-2 sentences explaining alignment with research focus")

class ScoredPaper(BaseModel):
    raw_paper: RawPaper
    llm_eval: LLMEvaluation
    journal_weight: float
    final_score: float
```

* **Scoring Formula:**

```text
Final Score = LLM Score * W_venue
```

Where `W_venue` is pulled from optional named profile `venue_weights` based on ISSN. Legacy `config/journal_tiers.json` is still supported for migration. All unlisted venues use `filtering.default_journal_weight` (default `1.0`).
* **Qualification Threshold:** raw LLM score >= 7 AND Final Score >= `filtering.score_cutoff` (default `7.0`).
* **Persistence:** dispositions are committed in stage-local transactions, not one transaction spanning delivery. Passed rows cannot be downgraded by later dedup rejection. An additive `history_contexts` table tracks observed selection fingerprints and warns on mixed/unknown history. `--reconsider-rejected` ignores rejected database entries for exact/fuzzy lookup but retains passed history and within-batch dedup; it only processes papers in the current input.

---

### Node 3: LLM Synthesis (`freshlit/nodes/synthesis.py`)

Processes qualified candidates into structured summary units and generates an executive overview. Both single and batch prompts include profile context, with no fixed discipline or modeling requirement. The actual method and findings must come from the abstract; quantitative metrics are included only if reported. The generic field-pulse prompt remains unchanged.

1. **Single-Paper Extraction:**

```python
from pydantic import BaseModel, Field

class ProcessedPaperSummary(BaseModel):
    paper_id: str
    title: str
    authors_formatted: str
    venue_and_year: str
    doi_url: str
    final_score: float
    core_question: str = Field(..., description="1 sentence on central question")
    framework_and_method: str = Field(..., description="Specific method actually used in the paper")
    key_finding: str = Field(..., description="Main result, quantitative metrics only when stated")
    code_data_link: str = Field(default="None stated", description="Code or data repository URL if present")
    relevance_rationale: str
```

2. **Field Pulse Generation:**
   Pass all extracted summaries in a single context window to generate 3 bullet points characterizing macro trends or overlapping methodologies in the current batch.

Extraction is **batched** the same way as scoring: up to `filtering.llm_batch_size` papers per structured request (via the `PaperExtractionBatch` wrapper model), run concurrently at `llm.workers`, with per-paper fallback. It uses the same opaque request-local IDs and validates exact cardinality, membership, uniqueness, and completeness before mapping results back to input order. Evaluation, extraction, batch-wrapper, and field-pulse models use strict Pydantic validation and reject extra fields.

---

### Node 4: Obsidian Delivery Engine (`freshlit/nodes/delivery.py`)

Formats the outputs from Node 3 and writes a `.md` file to the local Obsidian Vault.

* **Target Output Path:** `{OBSIDIAN_VAULT_PATH}/{DIGEST_FOLDER}/YYYY-W{ISO_week}.md` (ISO week **and** ISO year, so year-boundary `W53`/`W01` never collide). An optional `filename` override (exposed via `--output-name`) lets a run write to an alternate file for A/B model comparison.
* **Safety (hardening):**
  * **Path containment:** resolve the final path and assert it stays within `obsidian.vault_path` before writing (guards against `../` traversal from a misconfigured `digest_folder`).
  * **Atomic write:** write to a temp file in the target directory, then `os.replace()` into place (no mid-crash corruption).
  * **Overwrite handling:** if the weekly file already exists, warn and merge/append instead of clobbering (a `--force` flag may overwrite).
  * **Markdown safety:** sanitize/escape untrusted title text and validate DOI URLs before rendering.
* **Markdown File Structure:**

```markdown
---
date: YYYY-MM-DD
type: literature-digest
tags:
  - literature-digest
  - automated
papers_analyzed: 14
papers_selected: 4
top_score: 9.2
llm_model: gpt-5.6-luna
llm_provider: codex
---

# 🔬 Literature Digest: Week WW, YYYY

## Executive Pulse
- **Trend 1:** ...
- **Trend 2:** ...
- **Trend 3:** ...

---

## 🌟 Top Ranked Papers

### [Paper Title](DOI_URL)
- **Score:** `9.2/10` | **Venue:** Journal Name | **Authors:** First Author et al.
- **Core Question:** ...
- **Framework & Method:** ...
- **Key Finding:** ...
- **Code & Data:** [Repository](URL)
- **Why It Matters:** ...

---

## 📚 Secondary Selections

### [Paper Title](DOI_URL)
...
```

---

## 4. Database Schema (`data/cache.db`)

Implement using standard Python `sqlite3`.

```sql
CREATE TABLE IF NOT EXISTS processed_papers (
    paper_id TEXT PRIMARY KEY,
    doi TEXT,
    normalized_title TEXT,
    first_author TEXT,
    processed_date TEXT,
    disposition TEXT, -- 'dropped_dedup', 'dropped_vector', 'dropped_llm', 'passed'
    llm_score INTEGER,
    final_score REAL
);

CREATE INDEX IF NOT EXISTS idx_doi ON processed_papers(doi);
CREATE INDEX IF NOT EXISTS idx_norm_title ON processed_papers(normalized_title);

CREATE TABLE IF NOT EXISTS history_contexts (
    fingerprint TEXT PRIMARY KEY
);
```

* **DOI normalization:** store DOIs lowercase with the `https://doi.org/` prefix stripped.
* **NULL-safe dedup:** the DOI check uses `IS NOT DISTINCT FROM` semantics and falls back to `paper_id` matching when `doi` is NULL.
* **Atomicity:** inserts/dispositions are wrapped in stage-local transactions. Context tracking is additive and never deletes existing paper history.

---

## 5. Configuration & Environment Setup

### Configuration loading (central loader)

All configuration is loaded once at startup by `freshlit/utils/config.py`, which:

1. Parses `.env` via `python-dotenv` without mutating the process environment.
2. Parses `config/settings.yaml` into a Pydantic model with **fail-fast validation** (missing/blank required values raise a clear error before the pipeline runs).
3. Loads `config/journal_tiers.json` (default `{}` if absent).
4. Exposes a single `Settings` object consumed by every node; no node reads config files directly.

### `config/settings.yaml`

```yaml
obsidian:
  vault_path: "~/Documents/FreshLit"                       # ordinary Markdown folder is sufficient
  digest_folder: ""                                        # relative to vault_path ("" = root)

ingestion:
  lookback_days: 7
  timezone: "UTC"                                          # defines the date-window boundary
  max_results_per_query: 200                               # per-query cap (search is relevance-sorted)
  # User-chosen keywords, topics, and Europe PMC opt-in live in the profile.

api:
  request_timeout_seconds: 30
  max_pages: 50
  retry_backoff_seconds: 2

filtering:
  dedup_title_similarity: 0.92
  vector_threshold: 0.65
  max_llm_candidates: 40     # LLM-score only the top-N vector-ranked survivors
  llm_batch_size: 15         # papers scored per single LLM request
  score_cutoff: 7.0
  default_journal_weight: 1.0
  embedding_model: "allenai/specter2_base"
  embedding_device: "cpu"

llm:
  provider: "codex"          # "gateway" is explicit rollback only
  model: "gpt-5.6-luna"
  timeout_seconds: 120       # total deadline per structured request, across retries
  workers: 3                 # concurrent batches through the shared runtime
  max_retries: 2             # invalid output / SDK-classified transient failures
  codex_home: null           # set with FRESHLIT_CODEX_HOME for LLM runs
  base_url: "https://opencode.ai/zen/go/v1"  # rollback gateway only
```

### `config/journal_tiers.json`

```json
{
  "ISSN-HERE": 1.3
}
```

Legacy ISSN-to-weight mapping only. New installations have no overrides. Modern profiles store optional `{issn, name, weight}` records in `venue_weights`; weights must be positive and finite. Labels are never used for retrieval or embeddings.

### `config/research_profile.md`

An editable Markdown file with validated version-1 YAML front matter:

```yaml
version: 1
search:
  openalex_topics: []     # explicit {id, label} selections; no automatic inference
  europe_pmc:
    enabled: false
    sources: [PPR]
venue_weights: []
```

The body contains `## Keywords` bullets and a nonblank `## Research Description`. Keywords may be empty when topic IDs are selected. Both sections are required; unfinished templates are rejected. The whole body and a separate keyword text feed the vector profile, while only the body feeds the relevance/extraction prompts. Front matter never enters these inputs.

`freshlit init` creates missing private files without replacing them. `freshlit doctor` performs local read-only checks without downloading models, inspecting auth files, or making requests. `freshlit topics "phrase"` explicitly queries OpenAlex autocomplete and prints choices without modifying the profile or using an LLM. `freshlit migrate-profile` previews conversion from legacy settings, and `--apply` backs up the original before writing the profile; matching legacy settings can then be removed manually. Missing profiles never silently fall back to the example.

Profile vectors record model, dimensions, input fingerprint, and schema version. CLI runs rebuild stale/legacy vectors. Ingestion caches record effective query selections/date windows/limits; `--from-cache` warns when provenance differs and never fetches automatically when its cache is missing. Query cache metadata contains no credentials. Local files remain tied to the editable source checkout; standalone installation and new providers/sources are separate future work.

### `.env` File

```env
FRESHLIT_CODEX_HOME="/absolute/path/to/freshlit-codex-home"
FRESHLIT_LLM_PROVIDER="codex"
OPENALEX_API_KEY=""
# OPENCODE_GO_API_KEY=""
```

`FRESHLIT_CODEX_HOME` is the primary required LLM setting. It must resolve to an existing, user-owned dedicated Codex profile directory outside the repository with no group/other permissions. Create it without `sudo` (for example, `mkdir -p "$HOME/.freshlit-codex" && chmod 700 "$HOME/.freshlit-codex"`), then perform a one-time ChatGPT OAuth login using the [official Codex tooling and documentation](https://developers.openai.com/codex/auth/) with `CODEX_HOME` directed to that directory. FreshLit does not initiate login and does not inspect auth-file contents.

`FRESHLIT_LLM_PROVIDER` is optional and defaults to `codex`. `OPENCODE_GO_API_KEY` is neither read nor required in Codex mode; it is required only after the legacy `gateway` provider is explicitly selected by configuration or `--provider gateway`. Rotate any key that was previously exposed before rollback use. There is no automatic provider fallback. `OPENALEX_API_KEY` is **optional but recommended** (anonymous OpenAlex requests have a small daily budget that resets midnight UTC). `.env` is gitignored; `.env.example` documents settings without credentials.

---

## 6. Implementation Workflow for Executing Agent

An executing code agent must implement components in this exact sequence:

1. Create directory structure and set up a standard Python virtual environment with dependencies (`openai-codex==0.147.0`, `pydantic`, `pyyaml`, `requests`, `numpy`, `scikit-learn`, `sentence-transformers`, `jellyfish`, `python-dotenv`, `peft`; `instructor` and `openai` are retained for gateway rollback only). Add `.gitignore` (excluding `.env`, `data/cache.db`, `data/profile_exemplars.json`, `data/last_ingest.json`, `__pycache__/`, `.venv/`) and `.env.example`. Declare the package and the `freshlit = "freshlit.main:main"` console-script entry point in `pyproject.toml`, and `pip install -e .` so a global `freshlit` command runs the current source.
2. Implement `freshlit/utils/config.py` (central loader) and `freshlit/utils/db.py` (including the first-author surname extractor handling `Last Initial(s)`); run the migration script to build `data/cache.db`.
3. Implement `freshlit/utils/llm.py` with a provider-neutral protocol, the official Codex client (dedicated ChatGPT profile preflight, shared runtime, ephemeral request threads/temp directories, strict schema normalization and Pydantic validation, deadlines/retries, deny-all/read-only/disabled capabilities, context-managed shutdown), and the explicit opencode-go/Instructor rollback client; implement `freshlit/utils/embeddings.py` (lazy SPECTER2 loader).
4. Implement `freshlit/nodes/ingestion.py` to fetch and normalize JSON from OpenAlex (API key, timeouts, pagination, retries, OR-combined keyword search, ingest-cache save/load) and Europe PMC.
5. Implement `freshlit/nodes/filtering.py` including NULL-safe/DOI-normalized SQLite dedup, Jaro-Winkler title dedup with surname matching, vector ranking with candidate budget, and batched structured LLM rating with opaque-ID/cardinality validation.
6. Implement `freshlit/nodes/synthesis.py` to extract structured JSON summaries (batched) and build the executive pulse.
7. Implement `freshlit/nodes/delivery.py` to compile Markdown payloads and save to the Obsidian vault path with path containment, atomic writes, overwrite handling, and an optional `--output-name` override.
8. Implement `freshlit/main.py` as a CLI runner (`run`, `build-profile`, `init-db`) that connects Nodes 1–4 sequentially, validates the vault path at startup, and emits operational logs with secret redaction. `run` supports `--lookback`, `--limit`, `--max-candidates`, `--provider codex|gateway`, `--model`, `--output-name`, `--skip-llm` (stop after vector ranking), `--from-cache` (reuse `data/last_ingest.json`), `--dry-run`, and `--force`. It owns the selected structured client in a context manager so the shared runtime is always closed, and passes provider/model provenance to delivery.

### Performance notes

* **Keyword ingestion** OR-combines the profile keywords into chunked OpenAlex `search=` queries (under the ~4 KB URL limit) instead of one request per keyword.
* **LLM scoring and synthesis are batched**: up to `filtering.llm_batch_size` papers are evaluated in a single structured request, run concurrently at `llm.workers` through one runtime. This reduces structured-request round trips by an order of magnitude; each chunk falls back to per-paper calls if the batch request or opaque-ID contract fails.
* **Ingestion is cached** to `data/last_ingest.json` after every run; `--from-cache` and `--skip-llm` provide a fast (~1–2 min) troubleshooting loop that skips API fetching and LLM calls respectively.
