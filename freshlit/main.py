"""FreshLit pipeline orchestrator / CLI entrypoint.

Usage:
    freshlit init-db
    freshlit build-profile
    freshlit run [--lookback DAYS] [--limit N] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .utils import db, embeddings
from .utils.config import load_settings

log = logging.getLogger("freshlit")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Never propagate API request details (they may embed keys in URLs).
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # instructor logs expected mode-negotiation fallbacks as ERROR; silence them
    # (real failures surface through our own logging / RuntimeError).
    logging.getLogger("instructor").setLevel(logging.CRITICAL)


def _cmd_init_db(settings) -> None:
    db.init_db(settings.db_path)
    log.info("Database ready: %s", settings.db_path)


def _cmd_build_profile(settings) -> None:
    vector = embeddings.build_profile_vector(settings)
    embeddings.save_profile_vector(
        settings.profile_vector_path, vector, settings.filtering.embedding_model
    )
    log.info(
        "Profile vector saved (%d dims) -> %s",
        len(vector),
        settings.profile_vector_path,
    )


def _cmd_run(args) -> None:
    from .nodes import delivery, filtering, ingestion, synthesis
    from .utils.llm import build_client

    settings = load_settings(provider_override=getattr(args, "provider", None))
    if args.lookback is not None:
        settings.ingestion.lookback_days = args.lookback
    if args.max_candidates is not None:
        settings.filtering.max_llm_candidates = args.max_candidates
    if args.model is not None:
        settings.llm.model = args.model

    _cmd_init_db(settings)
    if not settings.profile_vector_path.exists():
        log.info("No profile vector found; building it now...")
        _cmd_build_profile(settings)

    cache_path = settings.project_root / "data" / "last_ingest.json"
    if args.from_cache and cache_path.exists():
        papers = ingestion.load_ingest_cache(cache_path)
        log.info("[--from-cache] loaded %d papers", len(papers))
    else:
        papers = ingestion.ingest(settings)
        ingestion.save_ingest_cache(cache_path, papers)

    if args.limit:
        papers = papers[: args.limit]
        log.info("[--limit] truncated to %d papers", len(papers))

    papers_analyzed = len(papers)
    if not papers:
        log.info("Nothing to process; exiting.")
        return

    if args.skip_llm:
        ranked = filtering.preprocess_and_rank(papers, settings, dry_run=True)
        log.info("[skip-llm] top %d candidates by vector similarity:", len(ranked))
        for i, (p, sim) in enumerate(ranked, 1):
            log.info("  %2d. %.3f  [%s] %s", i, sim, p.source_type, p.title[:90])
        return

    with build_client(settings) as client:
        qualified = filtering.filter_and_score(
            papers, settings, client, dry_run=args.dry_run
        )
        if not qualified:
            log.info("No papers passed all filters; nothing to deliver.")
            return

        summaries, pulse = synthesis.synthesize(client, settings, qualified)
    path = delivery.deliver(
        settings,
        summaries,
        pulse,
        papers_analyzed=papers_analyzed,
        force=args.force,
        dry_run=args.dry_run,
        filename=args.output_name,
    )
    top = summaries[0].final_score if summaries else 0.0
    log.info(
        "Done: %d analyzed -> %d selected (top score %.1f) -> %s",
        papers_analyzed, len(summaries), top, path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="freshlit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create data/cache.db schema")
    sub.add_parser(
        "build-profile",
        help="Embed config/research_profile.md -> data/profile_exemplars.json",
    )

    run = sub.add_parser("run", help="Run the full pipeline")
    run.add_argument("--lookback", type=int, default=None,
                     help="Override ingestion.lookback_days")
    run.add_argument("--limit", type=int, default=None,
                     help="Cap number of ingested papers (testing)")
    run.add_argument("--max-candidates", type=int, default=None,
                     help="Override filtering.max_llm_candidates")
    run.add_argument("--model", default=None,
                      help="Override llm.model (e.g. gpt-5.6-luna)")
    run.add_argument("--provider", choices=("codex", "gateway"), default=None,
                     help="Override llm.provider")
    run.add_argument("--output-name", default=None,
                     help="Override digest filename (for A/B comparison)")
    run.add_argument("--skip-llm", action="store_true",
                     help="Stop after vector ranking; print top candidates")
    run.add_argument("--from-cache", action="store_true",
                     help="Reuse data/last_ingest.json instead of refetching")
    run.add_argument("--dry-run", action="store_true",
                     help="Skip paper-disposition updates and digest delivery; "
                          "setup/cache writes and LLM requests still occur")
    run.add_argument("--force", action="store_true",
                     help="Overwrite an existing weekly digest")
    run.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    _setup_logging(getattr(args, "verbose", False))

    try:
        if args.command == "run":
            _cmd_run(args)
        else:
            settings = load_settings()
            if args.command == "init-db":
                _cmd_init_db(settings)
            elif args.command == "build-profile":
                _cmd_build_profile(settings)
    except Exception as exc:
        log.error("Fatal: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
