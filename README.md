# FreshLit

An automated literature-monitoring agent: a four-stage ETL pipeline that ingests
newly published papers and preprints, filters out duplicates and off-topic items,
performs LLM-driven structured synthesis, and writes a formatted Markdown digest
into a local folder (including an Obsidian vault). Obsidian is optional.

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
supervision.

## Setup

Requires **Python 3.11+ and macOS or Linux/POSIX**. Native Windows is not currently
supported by the hardened Codex launcher or secure digest writer. Installation
currently uses a source checkout, not a standalone wheel/pipx deployment.

1. From the repository root, create the environment and install FreshLit:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -e .
   ```

   The editable install also creates `.venv/bin/freshlit` and keeps it pointed at
   this checkout. The official `openai-codex` SDK is pinned to `0.147.0`.

2. Initialize the missing user files:

   ```bash
   .venv/bin/freshlit init
   ```

   This copies the neutral profile template and `.env.example` only if your local
   files do not exist. It never overwrites them. Edit `config/research_profile.md`
   with **your chosen search keywords/topics** and a research description; see
   [Research profile](#research-profile). An incomplete template cannot run.

   In `.env`, optionally set `FRESHLIT_VAULT_PATH` to your output directory and
   `OPENALEX_API_KEY` for a larger OpenAlex request budget. A normal Markdown
   folder works; no Obsidian installation or plugin is required.

3. Prepare the configured output directory and dedicated Codex home:

   ```bash
   .venv/bin/freshlit init --create-output-dir --prepare-codex-home
   ```

   These explicit flags create missing directories. The Codex home must be
   outside this repository, owned by you, and mode `0700`; existing unsafe
   permissions are rejected, not silently changed. Do not use `sudo` or share
   this directory with another Codex use. New `.env` files select
   `~/.freshlit-codex`; existing configured paths remain supported.

4. Install the official Codex CLI and complete **ChatGPT OAuth** login as described
   in the [official authentication documentation](https://developers.openai.com/codex/auth/).
   Initialization prints a shell-quoted login command for your configured path.
   With the default path:

   ```bash
   CODEX_HOME="$HOME/.freshlit-codex" codex login
   ```

   Select ChatGPT login, not API-key authentication. FreshLit never initiates
   login or reads authentication files. Full runs verify the effective safety
   configuration, account, and availability of the configured model. The explicit
   [gateway alternative](#gateway-rollback) does not require Codex setup.

5. Check local setup, then run:

   ```bash
   .venv/bin/freshlit doctor
   .venv/bin/freshlit run
   ```

   `doctor` is local and read-only: no requests, downloads, model loading, login,
   or auth-file inspection. It cannot verify account/model access. The first
   `run` initializes the database and builds the profile vector automatically;
   building may download the embedding model. Later profile/model changes trigger
   a rebuild. Manual `init-db` and `build-profile` remain available and do not
   require LLM credentials.

## Updating an existing installation

Before replacing old configuration files, retain copies of your existing
`config/settings.yaml`, `config/journal_tiers.json`, and private profile. The new
public defaults no longer contain the original author's topics or venue weights.
Migrate while your old local research settings are still present (restore them
from your copies if needed):

```bash
.venv/bin/freshlit migrate-profile          # preview only, printed to stdout
.venv/bin/freshlit migrate-profile --apply  # writes profile + exclusive .bak
```

Migration copies effective topics, Europe PMC choices, and venue weights into
the profile front matter without altering the existing Markdown body. It makes
`config/research_profile.md.bak` before the first write, refuses to overwrite an
existing backup, and does not touch the database, credentials, or other settings.
An already-modern profile is not rewritten. Conflicting settings keyword
overrides require manual resolution; they are not silently merged.

After migration, remove the matching research-only settings from the old YAML
and venue JSON. Nonempty conflicting legacy values fail with guidance. Retain
your output path, provider/model, thresholds, and other operational choices.

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
need to recreate them. `pyproject.toml` is the dependency source of truth;
`requirements.txt` is now a compatibility entry point for the same editable
installation, not a second list to maintain. No dependency versions were changed.

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
.venv/bin/python -m freshlit.main run --reconsider-rejected
.venv/bin/python -m freshlit.main run --dry-run
.venv/bin/python -m freshlit.main run --force
```

Other supported overrides include `--output-name`. `--from-cache` reuses
`data/last_ingest.json`; `--skip-llm` stops after vector ranking. CLI provider and
model options override their configured values for that run. `--from-cache` fails
if no cache exists rather than silently fetching. Legacy caches and caches from
other queries/date windows can be replayed explicitly, with a warning.

Changing the profile does not delete paper history. If FreshLit detects history
from another or unknown profile/selection context, it warns. Use
`--reconsider-rejected` to re-evaluate rejected papers **present in the current
ingestion or replay cache**. It retains passed-paper history and within-batch
deduplication; it does not fetch all previously rejected papers. Use a suitable
lookback or an explicit cache replay to supply them. Mixed-history warnings remain
conservative because older rejected rows are not erased.

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
| `config/settings.yaml` | Output paths, lookback, thresholds, LLM provider/model, timeout/retries, batching/concurrency |
| `config/research_profile.md` | User-chosen search selectors, optional venue preferences, and research context; gitignored |
| `config/journal_tiers.json` | Legacy compatibility only; new venue preferences live in the profile |
| `data/cache.db` | SQLite dedup/disposition history (state) |
| `data/profile_exemplars.json` | Generated profile vector (from `build-profile`) |

## Research profile

**Retrieval is not AI-generated.** You select the keywords and topic IDs. FreshLit
constructs API queries deterministically: quoted keywords joined by OR, plus a
separate OpenAlex topic query when topics are selected. These result streams are
combined, not intersected. Europe PMC is an explicit opt-in for biomedical
coverage; it uses the same keywords. Topic-only profiles work with OpenAlex, but
Europe PMC needs keywords to search.

The profile's YAML front matter contains structured choices; the Markdown body
contains the keywords and research context. For example:

```markdown
---
version: 1
search:
  openalex_topics: []
  europe_pmc:
    enabled: false
    sources: [PPR]
venue_weights: []
---
# Research Profile

## Keywords
- exoplanet atmospheres
- transmission spectroscopy

## Research Description
I study exoplanet atmospheres using observational spectroscopy. Include new
observational constraints and transferable retrieval methods. Exclude papers
concerned only with Solar System missions.
```

Optional topic lookup uses OpenAlex directly, not an LLM, and never edits your
profile automatically:

```bash
.venv/bin/freshlit topics "exoplanet atmospheres"
```

Copy selected `{id, label}` entries from its output into `search.openalex_topics`.
IDs control retrieval; labels are for you. Canonical OpenAlex topic URLs are also
accepted. Optional `venue_weights` entries have `issn`, `weight`, and an optional
`name`; unlisted venues use `filtering.default_journal_weight` (normally `1.0`).

The Markdown body drives the vector profile and both relevance/synthesis prompts.
The AI evaluates retrieved papers against your stated interests and exclusions;
it does not choose initial search terms. Front matter is excluded from embedding
and LLM text so changing a display label cannot change ranking. For compatibility,
embedding inputs retain the complete Markdown body and a separate keyword text;
editing that body can therefore change ranking. Never put credentials in it.

Both `## Keywords` and `## Research Description` are required for new profiles.
Use `- ` keyword bullets; provide at least one keyword or topic and a nonblank
research description. The unfinished example is never used as a fallback.

## Tuning

- **Research focus** — edit `config/research_profile.md`; the next run detects
  stale vectors and rebuilds them. `build-profile` can refresh explicitly.
- **Selectivity** — `filtering.vector_threshold` (floor), `score_cutoff`
  (qualify at >= 7.0), `max_llm_candidates` (LLM-scoring budget).
- **Venue weighting** — edit the optional named `venue_weights` in the profile.
- **LLM latency** — `llm.model`, `filtering.llm_batch_size`, `llm.workers`,
  `llm.timeout_seconds`, and `llm.max_retries`. The CLI can override the model,
  provider, and candidate count without changing configuration.

## Scheduling

FreshLit does not install a background service. After a successful manual run,
schedule the checkout's **absolute** `.venv/bin/freshlit run` path using cron on
Linux/macOS or launchd on macOS. For example, a cron entry running Mondays at
08:00 (scheduler-local time) is:

```cron
0 8 * * 1 /absolute/path/to/FreshLit/.venv/bin/freshlit run >> /absolute/path/to/freshlit.log 2>&1
```

Quote executable/log paths if they contain spaces. Configuration is located from
the source checkout, not the scheduler's working directory. Run as the same user
who owns the private Codex home; ensure no overlapping runs and arrange log
rotation. Ingestion dates use `ingestion.timezone` (UTC by default), independently
of the scheduler's timezone. Moving the checkout may require recreating its
virtual environment and updating the scheduled executable path.

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
