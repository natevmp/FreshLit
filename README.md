# FreshLit

An automated literature-monitoring agent: a four-stage ETL pipeline that ingests
newly published papers and preprints, filters out duplicates and off-topic items,
performs LLM-driven structured synthesis, and writes a formatted Markdown digest
directly into a local Obsidian vault.

See `literature_agent_architecture.md` for the full architecture specification.

```
INGESTION  ->  DEDUP & FILTERING  ->  LLM SYNTHESIS  ->  OBSIDIAN DELIVERY
OpenAlex        SQLite ID check       Pydantic schema     YAML frontmatter
Europe PMC      Title matching        Field pulse         Vault writer
                Vector (SPECTER2)
                LLM scoring + journal weight
```

## AI Disclosure

FreshLit's concept, architecture specification, and research profile are the
author's own work. The implementation — code, debugging, and documentation — was
developed with AI assistance (opencode, an AI coding agent) under human
supervision. All AI-generated code was reviewed and tested by the author before
being committed; no unreviewed AI output is included.

## Setup

1. Create the environment and install the current checkout and its dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -e .
   ```

   The editable install also creates `.venv/bin/freshlit` and keeps it pointed at
   this checkout. The official `openai-codex` SDK is pinned to `0.147.0`.

2. Create a dedicated Codex profile directory owned by your user. It must already
   exist, be outside this repository, and grant no group/other access. Do not use
   `sudo`, run Codex as root, or share this directory with another Codex use:

   ```bash
   mkdir -p "$HOME/.freshlit-codex"
   chmod 700 "$HOME/.freshlit-codex"
   ```

3. Install the official Codex tooling and complete a one-time **ChatGPT OAuth**
   login as described in the [official Codex authentication
   documentation](https://developers.openai.com/codex/auth/), directing Codex to
   that dedicated profile:

   ```bash
   CODEX_HOME="$HOME/.freshlit-codex" codex login
   ```

   Select ChatGPT login, not API-key authentication. FreshLit does not start the
   login flow and does not open, parse, or inspect auth files; the official Codex
   runtime uses the prepared profile. FreshLit verifies the effective safety
   configuration, built-in OpenAI provider, ChatGPT account, and requested model.

4. Copy `.env.example` to `.env` and configure the absolute profile path:

   ```env
   FRESHLIT_CODEX_HOME="/Users/you/.freshlit-codex"
   ```

   - `FRESHLIT_CODEX_HOME` — overrides `llm.codex_home`. Codex LLM runs require
     a dedicated authenticated profile configured through either setting.
   - `FRESHLIT_LLM_PROVIDER` — optional; defaults to `codex`.
   - `OPENALEX_API_KEY` — optional but recommended; the free OpenAlex API key.
     Anonymous requests have a small daily budget (resets midnight UTC).
   - `OPENCODE_GO_API_KEY` — rollback-only and ignored unless `gateway` is
     explicitly selected; see [Gateway rollback](#gateway-rollback).

5. Review `config/settings.yaml` (vault path is pre-set) and
   `config/research_profile.md` (keywords + research description, drafted from
   your vault notes).

6. Initialize the database and build the profile vector (downloads SPECTER2 on
   first run):

   ```bash
   .venv/bin/python -m freshlit.main init-db
   .venv/bin/python -m freshlit.main build-profile
   ```

## Updating an existing installation

Once the intended source changes are in this checkout, refresh the editable
installation and verify it from the project root:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
```

This refreshes FreshLit's dependency metadata and console command without a
blanket dependency upgrade. Preserve local source changes, the existing
environment configuration, data, and authenticated Codex profile; there is no
need to recreate them. Installing only `requirements.txt` does not refresh the
installed FreshLit package metadata.

FreshLit launches the SDK's bundled, version-matched Codex runtime. Updating a
separate global `codex` command does not update that runtime. Keep the SDK pin
until its private-router compatibility fix has been reviewed against a newer
release.

## Usage

An existing PATH launcher pointing to this checkout's virtual environment can be
used from any directory: `freshlit run`. Configuration and data are located from
the source checkout, not the terminal's working directory. Bare `freshlit` still
requires an explicit subcommand; it does not start the pipeline automatically.

```bash
# Full pipeline
.venv/bin/python -m freshlit.main run

# Useful flags
.venv/bin/python -m freshlit.main run --provider codex
.venv/bin/python -m freshlit.main run --model gpt-5.6-luna
.venv/bin/python -m freshlit.main run --lookback 14
.venv/bin/python -m freshlit.main run --limit 50
.venv/bin/python -m freshlit.main run --max-candidates 20
.venv/bin/python -m freshlit.main run --skip-llm
.venv/bin/python -m freshlit.main run --from-cache
.venv/bin/python -m freshlit.main run --dry-run
.venv/bin/python -m freshlit.main run --force
```

Other supported overrides include `--output-name`. `--from-cache` reuses
`data/last_ingest.json`; `--skip-llm` stops after vector ranking. CLI provider and
model options override their configured values for that run.

`--dry-run` skips paper-disposition updates and digest delivery, but it can still
initialize the database, update ingestion/profile caches, download embeddings,
and submit LLM requests. To avoid ingestion and LLM requests while testing local
ranking with an existing ingestion cache, combine
`--from-cache --skip-llm --dry-run`; this still needs the embedding model and may
download its weights if they are not cached.

Output: `{vault_path}/{digest_folder}/YYYY-Www.md` (ISO year + week), with YAML
frontmatter for Dataview, including the LLM provider and model used.

The vault must already exist. Delivery opens it as the trusted filesystem
boundary, securely creates nested digest directories, and uses directory-relative
atomic replacement without following swapped symlinks. In-process and advisory
locks serialize FreshLit rerun merges; the content-free `.freshlit-delivery.lock`
entry is retained in the digest folder. These guarantees require POSIX
directory-descriptor operations and advisory locking; unsupported platforms fail
closed. Other editors must cooperate with the lock to avoid conflicting edits.

## Configuration files

| File | Purpose |
|---|---|
| `config/settings.yaml` | Vault paths, lookback window, topics, thresholds, LLM provider/model, timeout/retries, batching/concurrency |
| `config/research_profile.md` | Editable keywords + research description driving queries and the vector/LLM profile |
| `config/journal_tiers.json` | ISSN -> venue weight multiplier |
| `data/cache.db` | SQLite dedup/disposition history (state) |
| `data/profile_exemplars.json` | Generated profile vector (from `build-profile`) |

## Tuning

- **Research focus** — edit `config/research_profile.md`, then
  `build-profile`. Keywords are OR-combined into chunked OpenAlex searches.
- **Selectivity** — `filtering.vector_threshold` (floor), `score_cutoff`
  (qualify at >= 7.0), `max_llm_candidates` (LLM-scoring budget).
- **Venue weighting** — add/remove ISSNs in `journal_tiers.json`.
- **LLM latency** — `llm.model`, `filtering.llm_batch_size`, `llm.workers`,
  `llm.timeout_seconds`, and `llm.max_retries`. The CLI can override the model,
  provider, and candidate count without changing configuration.

## Codex runtime and security model

- `codex` and `gpt-5.6-luna` are the default provider and model. Production
  startup forces ChatGPT login and the built-in OpenAI provider, then fails
  closed unless the effective safety configuration, account, and model pass
  preflight. The app-server starts through an allowlisted, empty environment, so
  inherited credentials cannot select an API-key-backed billed OpenAI API path.
- Each run uses one official Codex runtime and fresh ephemeral threads in fresh
  temporary working directories outside the repository. Approval is deny-all,
  the sandbox is read-only, and tools, shell execution, network/web search, MCP,
  agents, skills, hooks, apps, memories, history persistence, telemetry, feedback,
  and update checks are explicitly disabled. Temporary request directories are
  deleted and the shared runtime is closed when the LLM context exits.
- The low-level client explicitly declines approval requests rather than using
  the SDK's permissive default. A local compatibility override buffers early
  turn-completion notifications and replays pending events atomically. This
  override depends on the pinned SDK's private router and must be reviewed and
  regression-tested before upgrading `openai-codex`.
- `llm.timeout_seconds` is a total deadline for each structured Codex request,
  including startup of its thread and retries. An over-deadline turn is
  interrupted; if bounded cancellation fails, that runtime is terminated,
  permanently marked unusable, and the request fails after a short cancellation
  grace. Successful cancellation fails only that request, without poisoning a
  healthy shared runtime. Up to
  `llm.max_retries` retries are made only for invalid structured output or errors
  the SDK classifies as retryable. Other provider errors fail closed. A malformed
  batch may be retried as individual papers, but FreshLit never changes provider.
- `llm.workers` bounds concurrent batch requests through the shared runtime. All
  responses must satisfy normalized strict JSON schemas and strict Pydantic
  validation before entering the pipeline.

There is no automatic provider fallback or hidden billed API route. FreshLit will
not start OAuth, switch to an API key, or silently invoke the rollback gateway.
The hardened app-server launcher currently requires a POSIX system with the
standard `/usr/bin/env` utility (including macOS and mainstream Linux systems).

## Gateway rollback

The legacy opencode-go/Instructor integration remains only as an explicit
rollback. Before using it, rotate any key that was ever exposed and place the
replacement only in `.env`:

```env
OPENCODE_GO_API_KEY="replacement-key"
```

Select rollback for one run, with a gateway-compatible model:

```bash
.venv/bin/python -m freshlit.main run --provider gateway --model qwen3.5-plus
```

Alternatively, set `FRESHLIT_LLM_PROVIDER="gateway"` in `.env`, or configure the
rollback explicitly:

```yaml
llm:
  provider: "gateway"
  model: "qwen3.5-plus"
  base_url: "https://opencode.ai/zen/go/v1"
```

The key is read and required only after `gateway` is explicitly selected. A
gateway failure does not fall back to Codex, and a Codex failure does not fall
back to the gateway.

Gateway response-mode negotiation accepts Pydantic validation failures or
HTTP 400/404/422 errors whose structured error metadata explicitly identifies a
mode-specific field such as `response_format` or `tool_choice`. Generic request,
model, endpoint, authentication, rate-limit, and transport failures stop after
one HTTP attempt. An explicit parse-only retry policy prevents older Instructor
versions from retrying those terminal failures internally.

Creating a gateway client suppresses Instructor's raw retry diagnostics
process-wide, including direct library use without CLI logging setup. FreshLit's
sanitized errors and application logging remain enabled. Recheck the diagnostic
logger names when upgrading Instructor.

## Notes

- LLM latency varies by model and batch content; bounded concurrency and batching
  keep weekly runs practical without weakening request isolation.
- OpenAlex anonymous requests have a small daily budget; add an API key for
  reliable weekly runs.
