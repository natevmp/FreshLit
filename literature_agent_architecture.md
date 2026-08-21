# Architecture Specification: FreshLit — Automated Literature Monitoring Agent

> **Revision note.** This is an adapted version of the original spec, incorporating: (1) project rename to **FreshLit** (package `freshlit`), (2) LLM provider switched to the **opencode-go** OpenAI-compatible gateway, (3) local **SPECTER2** embeddings (the gateway is chat-only), (4) an editable **research profile** Markdown file driving topics/keywords/vector filtering, (5) a central **configuration loader** (the original spec defined config file contents but no loading mechanism), and (6) robustness/security hardening surfaced by design review (request timeouts + pagination + retries, NULL-safe/DOI-normalized dedup, path-containment + atomic vault writes, secret hygiene).

This document defines the system architecture, node interfaces, schemas, database tables, and execution flow for an automated literature monitoring agent named **FreshLit**. An executing code agent can read this specification directly to construct and run the system on the local filesystem.

---

## 1. System Overview

FreshLit is a modular, four-stage ETL pipeline executed on a scheduled basis (weekly or bi-weekly). It ingests newly published papers and preprints, filters out duplicates and off-topic items, performs LLM-driven structured synthesis, and outputs a formatted Markdown file directly into a local Obsidian vault.

```text
1. INGESTION ──► 2. DEDUP & FILTERING ──► 3. LLM SYNTHESIS ──► 4. OBSIDIAN DELIVERY
   OpenAlex /         SQLite ID Check          Pydantic Schema      YAML Frontmatter
   Europe PMC         Title String Matching    Context Extraction   Dataview Metadata
                      Vector Similarity        Field Pulse Pass     Vault File Writer
                      LLM Scoring + Journal Multiplier
```

**Runtime stack**

- **LLM scoring & synthesis:** the [opencode-go](https://opencode.ai/zen/go/v1) OpenAI-compatible gateway (default model `qwen3.5-plus`), driven through `instructor` for schema-enforced JSON output. Model and base URL are configurable in `config/settings.yaml`.
- **LLM structured-output negotiation:** `freshlit/utils/llm.py` auto-negotiates the instructor mode (TOOLS → JSON → MD_JSON) since gateway structured-output support varies by model, then locks the working mode for the rest of the run.
- **Embeddings:** local `allenai/specter2_base` via `sentence-transformers` (opencode-go is chat-only; SPECTER2 is a scientific-text domain model). Configurable device (`cpu`/`cuda`).
- **Configuration:** a central loader (`freshlit/utils/config.py`) validates and exposes all tunables at startup. See Section 5.

---

## 2. Directory Structure

The codebase is organized as follows on the local filesystem:

```text
freshlit/                        # project root
├── config/
│   ├── settings.yaml            # Query parameters, thresholds, vault paths, model selection
│   ├── journal_tiers.json       # ISSN to multiplier weight mapping
│   └── research_profile.md      # Editable keywords + research description (drives topics/keywords/vector profile)
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
│       ├── db.py                # SQLite wrapper functions
│       ├── llm.py               # opencode-go (OpenAI-compatible) / Instructor client setup
│       └── embeddings.py        # SPECTER2 embedding model loader (lazy singleton)
├── .env                         # Secrets (gitignored)
├── .env.example                 # Documented template for required secrets
├── .gitignore
├── pyproject.toml               # Package metadata + `freshlit` console-script entry point
├── requirements.txt
└── README.md
```

---

## 3. Node Specifications

### Node 1: Ingestion Engine (`freshlit/nodes/ingestion.py`)

* **Primary Source:** OpenAlex REST API (`https://api.openalex.org/works`).
* **Fallback/Direct Source:** Europe PMC REST API (for zero-lag bioRxiv/medRxiv preprints).
* **Ingestion Logic:**
  1. Calculate date window: `[current_date - lookback_days]` to `[current_date]` (closed-open, timezone defined in `settings.yaml`).
  2. Issue GET requests filtering by OpenAlex Topic IDs (from `ingestion.openalex_topics`) AND the publication-date window.
  3. Combine with keyword queries derived from `config/research_profile.md` — keywords are OR-combined into chunked OpenAlex `search=` queries (under the ~4 KB URL limit).
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
   * Embed `title + abstract`. Compute cosine similarity against the target profile vector (built from `config/research_profile.md`, cached in `data/profile_exemplars.json`).
   * Drop records with missing/short abstracts (< 50 chars) and those below `filtering.vector_threshold` (default `0.65`).
   * Rank survivors by similarity and keep only the top `filtering.max_llm_candidates` (default `40`) — SPECTER2 similarities cluster in a narrow band, so ranking + a candidate budget is the effective filter; the threshold is only a floor.
4. **LLM Reasoning & Weighted Scoring:**
   Pass surviving items to the LLM (via `instructor`) against a rubric grounded in the research profile text. Scoring is **batched**: up to `filtering.llm_batch_size` papers per structured request (via the `LLMEvaluationBatch` wrapper model), run concurrently at `llm.workers`, with per-paper fallback if a batch request fails.

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

Where `W_venue` is pulled from `config/journal_tiers.json` based on ISSN. Default for unlisted preprints is `filtering.default_journal_weight` (default `1.0`).
* **Qualification Threshold:** Final Score >= `filtering.score_cutoff` (default `7.0`).
* **Persistence:** all dispositions are written to `cache.db` inside a single transaction per run (atomic, no partial rows).

---

### Node 3: LLM Synthesis (`freshlit/nodes/synthesis.py`)

Processes qualified candidates (Final Score >= `score_cutoff`) into structured summary units and generates an executive overview.

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
    framework_and_method: str = Field(..., description="Specific mathematical/computational/empirical method")
    key_finding: str = Field(..., description="Main result including quantitative metrics")
    code_data_link: str = Field(default="None stated", description="GitHub or dataset URL if present")
    relevance_rationale: str
```

2. **Field Pulse Generation:**
   Pass all extracted summaries in a single context window to generate 3 bullet points characterizing macro trends or overlapping methodologies in the current batch.

Extraction is **batched** the same way as scoring: up to `filtering.llm_batch_size` papers per structured request (via the `PaperExtractionBatch` wrapper model), run concurrently at `llm.workers`, with per-paper fallback.

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
llm_model: qwen3.5-plus
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
```

* **DOI normalization:** store DOIs lowercase with the `https://doi.org/` prefix stripped.
* **NULL-safe dedup:** the DOI check uses `IS NOT DISTINCT FROM` semantics and falls back to `paper_id` matching when `doi` is NULL.
* **Atomicity:** batch inserts/dispositions are wrapped in a transaction per run.

---

## 5. Configuration & Environment Setup

### Configuration loading (central loader)

All configuration is loaded once at startup by `freshlit/utils/config.py`, which:

1. Loads `.env` via `python-dotenv`.
2. Parses `config/settings.yaml` into a Pydantic model with **fail-fast validation** (missing/blank required values raise a clear error before the pipeline runs).
3. Loads `config/journal_tiers.json` (default `{}` if absent).
4. Exposes a single `Settings` object consumed by every node; no node reads config files directly.

### `config/settings.yaml`

```yaml
obsidian:
  vault_path: "/Users/me/WorkVault/Work Vault/FreshLit"   # absolute path; digest written here
  digest_folder: ""                                        # relative to vault_path ("" = root)

ingestion:
  lookback_days: 7
  timezone: "UTC"                                          # defines the date-window boundary
  max_results_per_query: 200                               # per-query cap (search is relevance-sorted)
  openalex_topics:                                         # OpenAlex topic IDs
    - "T11764"                                             # Evolution and Genetic Dynamics
    - "T11287"                                             # Cancer Genomics and Diagnostics
    - "T10012"                                             # Genetic Diversity and Population Structure
  keywords: []                                             # populated from config/research_profile.md;
                                                           #   OR-combined into chunked `search=` queries
  europe_pmc:
    enabled: true
    sources: ["PPR"]                                       # preprints (bioRxiv/medRxiv)

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
  base_url: "https://opencode.ai/zen/go/v1"
  model: "qwen3.5-plus"
  timeout_seconds: 120       # thinking-mode responses can exceed 30s
  workers: 3                 # parallel LLM requests; keep low to avoid throttling
```

### `config/journal_tiers.json`

```json
{
  "ISSN-HERE": 1.3
}
```

ISSN (with dashes removed, as returned by OpenAlex) mapped to venue weight `W_venue`. Higher weight boosts high-impact venues; unlisted preprints default to `filtering.default_journal_weight`.

### `config/research_profile.md`

An editable Markdown file with two sections that drive the whole pipeline: a `## Keywords` list (used for OpenAlex/Europe PMC queries) and a `## Research Description` prose block (used to build the vector profile and the LLM relevance rubric). The pipeline reads this file at runtime and regenerates `data/profile_exemplars.json` via `freshlit build-profile`.

### `.env` File

```env
OPENCODE_GO_API_KEY="your-opencode-go-key"
OPENALEX_API_KEY=""
```

`OPENCODE_GO_API_KEY` is **required** (the loader fails fast if missing/blank); `OPENALEX_API_KEY` is **optional but recommended** (anonymous OpenAlex requests have a small daily budget that resets midnight UTC). `.env` is gitignored; `.env.example` documents the keys without values.

---

## 6. Implementation Workflow for Executing Agent

An executing code agent must implement components in this exact sequence:

1. Create directory structure and set up a standard Python virtual environment with dependencies (`pydantic`, `instructor`, `openai`, `pyyaml`, `requests`, `numpy`, `scikit-learn`, `sentence-transformers`, `jellyfish`, `python-dotenv`, `peft`). Add `.gitignore` (excluding `.env`, `data/cache.db`, `data/profile_exemplars.json`, `data/last_ingest.json`, `__pycache__/`, `.venv/`) and `.env.example`. Declare the package and the `freshlit = "freshlit.main:main"` console-script entry point in `pyproject.toml`, and `pip install -e .` so a global `freshlit` command runs the current source.
2. Implement `freshlit/utils/config.py` (central loader) and `freshlit/utils/db.py` (including the first-author surname extractor handling `Last Initial(s)`); run the migration script to build `data/cache.db`.
3. Implement `freshlit/utils/llm.py` (opencode-go OpenAI-compatible client + `instructor` wrapper with TOOLS→JSON→MD_JSON mode negotiation) and `freshlit/utils/embeddings.py` (lazy SPECTER2 loader).
4. Implement `freshlit/nodes/ingestion.py` to fetch and normalize JSON from OpenAlex (API key, timeouts, pagination, retries, OR-combined keyword search, ingest-cache save/load) and Europe PMC.
5. Implement `freshlit/nodes/filtering.py` including NULL-safe/DOI-normalized SQLite dedup, Jaro-Winkler title dedup with surname matching, vector ranking with candidate budget, and batched LLM rating via `instructor`.
6. Implement `freshlit/nodes/synthesis.py` to extract structured JSON summaries (batched) and build the executive pulse.
7. Implement `freshlit/nodes/delivery.py` to compile Markdown payloads and save to the Obsidian vault path with path containment, atomic writes, overwrite handling, and an optional `--output-name` override.
8. Implement `freshlit/main.py` as a CLI runner (`run`, `build-profile`, `init-db`) that connects Nodes 1–4 sequentially, validates the vault path at startup, and emits operational logs with secret redaction. `run` supports `--lookback`, `--limit`, `--max-candidates`, `--model`, `--output-name`, `--skip-llm` (stop after vector ranking), `--from-cache` (reuse `data/last_ingest.json`), `--dry-run`, and `--force`.

### Performance notes

* **Keyword ingestion** OR-combines the profile keywords into chunked OpenAlex `search=` queries (under the ~4 KB URL limit) instead of one request per keyword.
* **LLM scoring and synthesis are batched**: up to `filtering.llm_batch_size` papers are evaluated in a single structured request, run concurrently at `llm.workers`. This reduces the gateway round-trips (the dominant cost) by an order of magnitude; each chunk falls back to per-paper calls if the batch request fails.
* **Ingestion is cached** to `data/last_ingest.json` after every run; `--from-cache` and `--skip-llm` provide a fast (~1–2 min) troubleshooting loop that skips API fetching and LLM calls respectively.
