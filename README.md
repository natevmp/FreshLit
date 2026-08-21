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

1. Create the environment and install dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Configure secrets: copy `.env.example` to `.env` and fill in values.

   - `OPENCODE_GO_API_KEY` — required; your opencode-go key (also stored in
     `~/.local/share/opencode/auth.json` under the `opencode-go` provider).
   - `OPENALEX_API_KEY` — optional but recommended; the free OpenAlex API key.
     Anonymous requests have a small daily budget (resets midnight UTC).

3. Review `config/settings.yaml` (vault path is pre-set) and
   `config/research_profile.md` (keywords + research description, drafted from
   your vault notes).

4. Initialize the database and build the profile vector (downloads SPECTER2 on
   first run):

   ```bash
   .venv/bin/python -m freshlit.main init-db
   .venv/bin/python -m freshlit.main build-profile
   ```

## Usage

```bash
# Full pipeline
.venv/bin/python -m freshlit.main run

# Useful flags
run --lookback 14          # override the date window (days)
run --limit 50             # cap ingested papers (testing)
run --max-candidates 20    # override how many papers get LLM-scored
run --skip-llm             # stop after vector ranking; print top candidates
run --from-cache           # reuse data/last_ingest.json (skip refetching)
run --dry-run              # do not write to DB or vault
run --force                # overwrite an existing weekly digest
```

Output: `{vault_path}/{digest_folder}/YYYY-Www.md` (ISO year + week), with YAML
frontmatter for Dataview.

## Configuration files

| File | Purpose |
|---|---|
| `config/settings.yaml` | Vault paths, lookback window, topics, thresholds, models, batching/concurrency |
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
- **Cost/latency** — `llm.model`, `llm.batch_size`, `llm.workers`.

## Notes

- The LLM gateway (opencode-go) serves thinking-mode models with ~30–60s latency
  per request; batching keeps a weekly run in the low single-digit minutes.
- OpenAlex anonymous requests have a small daily budget; add an API key for
  reliable weekly runs.
