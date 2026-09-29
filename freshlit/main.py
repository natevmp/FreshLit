"""FreshLit pipeline orchestrator / CLI entrypoint.

Usage:
    freshlit init [--create-output-dir] [--prepare-codex-home]
    freshlit doctor
    freshlit topics "search phrase"
    freshlit migrate-profile [--apply]
    freshlit init-db
    freshlit build-profile
    freshlit run [--lookback DAYS] [--limit N] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys

from .utils import db, embeddings
from .utils.config import _project_root, load_settings

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
        settings.profile_vector_path, vector, settings.filtering.embedding_model,
        fingerprint=embeddings.profile_fingerprint(settings),
    )
    log.info(
        "Profile vector saved (%d dims) -> %s",
        len(vector),
        settings.profile_vector_path,
    )


def _cmd_run(args) -> None:
    from .nodes import delivery, filtering, ingestion, synthesis
    from .utils.llm import build_client

    settings = load_settings(
        provider_override=getattr(args, "provider", None),
        require_llm_credentials=not args.skip_llm,
    )
    if args.lookback is not None:
        settings.ingestion.lookback_days = args.lookback
    if args.max_candidates is not None:
        settings.filtering.max_llm_candidates = args.max_candidates
    if args.model is not None:
        settings.llm.model = args.model

    cache_path = settings.project_root / "data" / "last_ingest.json"
    if args.from_cache and not cache_path.is_file():
        raise FileNotFoundError(
            "No ingestion cache is available. Run without --from-cache to fetch "
            "papers; offline replay never silently starts a network fetch."
        )
    if not args.dry_run and not args.skip_llm:
        delivery.validate_destination(settings, filename=args.output_name)

    _cmd_init_db(settings)
    if not embeddings.profile_vector_is_current(settings):
        log.info("Profile vector is missing or stale; rebuilding it now...")
        _cmd_build_profile(settings)

    if args.from_cache:
        papers = ingestion.load_ingest_cache(cache_path, settings=settings)
        log.info("[--from-cache] loaded %d papers", len(papers))
    else:
        papers = ingestion.ingest(settings, cache_path=cache_path)

    if args.limit:
        papers = papers[: args.limit]
        log.info("[--limit] truncated to %d papers", len(papers))

    papers_analyzed = len(papers)
    if not papers:
        log.info("Nothing to process; exiting.")
        return

    if args.skip_llm:
        ranked = filtering.preprocess_and_rank(
            papers, settings, dry_run=True,
            reconsider_rejected=args.reconsider_rejected,
        )
        log.info("[skip-llm] top %d candidates by vector similarity:", len(ranked))
        for i, (p, sim) in enumerate(ranked, 1):
            log.info("  %2d. %.3f  [%s] %s", i, sim, p.source_type, p.title[:90])
        return

    with build_client(settings) as client:
        qualified = filtering.filter_and_score(
            papers, settings, client, dry_run=args.dry_run,
            reconsider_rejected=args.reconsider_rejected,
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


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(prog="freshlit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create missing user files, never overwrite them")
    init.add_argument("--create-output-dir", action="store_true",
                      help="Create the configured output root if missing")
    init.add_argument("--prepare-codex-home", action="store_true",
                      help="Prepare a private Codex directory; login stays manual")
    sub.add_parser("doctor", help="Read-only local setup checks; no network or downloads")
    topics = sub.add_parser("topics", help="Look up user-chosen OpenAlex topics (network)")
    topics.add_argument("query", help="Topic name or phrase to look up; no AI is used")
    migrate = sub.add_parser("migrate-profile", help="Preview legacy profile migration")
    migrate.add_argument("--apply", action="store_true",
                         help="Write the profile after making an exclusive .bak backup")
    sub.add_parser("init-db", help="Create data/cache.db schema")
    sub.add_parser(
        "build-profile",
        help="Embed config/research_profile.md -> data/profile_exemplars.json",
    )

    run = sub.add_parser("run", help="Run the full pipeline")
    run.add_argument("--lookback", type=_positive_int, default=None,
                     help="Override ingestion.lookback_days")
    run.add_argument("--limit", type=_positive_int, default=None,
                     help="Cap number of ingested papers (testing)")
    run.add_argument("--max-candidates", type=_positive_int, default=None,
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
    run.add_argument("--reconsider-rejected", action="store_true",
                     help="Re-evaluate rejected papers in this input; keep passed history")
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
        elif args.command in {"init", "doctor", "topics"}:
            from .utils import onboarding

            root = _project_root()
            if args.command == "init":
                messages = onboarding.initialize(
                    root, create_output_dir=args.create_output_dir,
                    prepare_codex_home=args.prepare_codex_home,
                )
            elif args.command == "doctor":
                messages = onboarding.doctor(root)
            else:
                import yaml

                results = onboarding.search_topics(args.query, root)
                print(yaml.safe_dump({"openalex_topics": results}, sort_keys=False), end="")
                return
            for message in messages:
                print(message)
            if any(message.startswith("ERROR:") for message in messages):
                sys.exit(1)
        elif args.command == "migrate-profile":
            from .utils.profile import migrate_profile

            preview = migrate_profile(_project_root(), apply=args.apply)
            if args.apply:
                log.info("Profile ready; first migration preserves the original in "
                         "config/research_profile.md.bak. Existing modern profiles are unchanged.")
            else:
                log.info("Migration preview only; use --apply to write with a backup.")
                print(preview, end="")
        else:
            settings = load_settings(require_llm_credentials=False)
            if args.command == "init-db":
                _cmd_init_db(settings)
            elif args.command == "build-profile":
                _cmd_build_profile(settings)
    except Exception as exc:
        log.error("Fatal: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
