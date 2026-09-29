"""Regression tests for persistent filtering history and reconsideration."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from freshlit.nodes.filtering import (
    LLMEvaluation,
    _selection_context_fingerprint,
    filter_and_score,
    preprocess_and_rank,
)
from freshlit.nodes.ingestion import RawPaper
from freshlit.utils import db


def _settings(root: Path, profile: str = "Coastal geomorphology"):
    return SimpleNamespace(
        db_path=root / "data" / "cache.db",
        profile_vector_path=root / "data" / "profile.json",
        research_profile_text=profile,
        ingestion=SimpleNamespace(
            keywords=["coastal erosion"], openalex_topics=["T-COAST"]
        ),
        filtering=SimpleNamespace(
            dedup_title_similarity=0.92,
            vector_threshold=0.65,
            max_llm_candidates=40,
            score_cutoff=7.0,
            default_journal_weight=1.0,
            embedding_model="test-embedding-model",
        ),
        llm=SimpleNamespace(model="test-llm-model"),
        journal_tiers={"12345678": 1.2},
    )


def _paper(paper_id: str, title: str, doi: str | None = None) -> RawPaper:
    return RawPaper(
        id=paper_id,
        doi=doi,
        title=title,
        authors=["Rivera, Ana"],
        publication_date="2026-09-01",
        venue_name="Earth Systems",
        abstract=(
            "Field observations quantify shoreline retreat across several sites "
            "using repeated surveys and sediment measurements."
        ),
        source_type="journal",
    )


class DatabaseHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.db_path = Path(self.tempdir.name) / "data" / "cache.db"
        db.init_db(self.db_path)

    def test_reconsideration_consults_only_passed_exact_and_fuzzy_history(self):
        rejected_title = "Experimental reconstruction of ancient ceramic firing"
        passed_title = "Satellite estimates of long term coastal erosion patterns"
        with db.connect(self.db_path) as conn:
            db.record_disposition(
                conn,
                "REJECTED",
                "10.1000/rejected",
                rejected_title,
                ["Rivera, Ana"],
                "dropped_llm",
            )
            db.record_disposition(
                conn,
                "PASSED",
                "10.1000/passed",
                passed_title,
                ["Rivera, Ana"],
                "passed",
                9,
                9.0,
            )

            self.assertTrue(db.paper_seen(conn, "REJECTED", None))
            self.assertTrue(db.paper_seen(conn, "OTHER", "10.1000/rejected"))
            self.assertFalse(
                db.paper_seen(
                    conn, "REJECTED", None, reconsider_rejected=True
                )
            )
            self.assertFalse(
                db.paper_seen(
                    conn,
                    "OTHER",
                    "https://doi.org/10.1000/rejected",
                    reconsider_rejected=True,
                )
            )
            self.assertTrue(
                db.paper_seen(conn, "PASSED", None, reconsider_rejected=True)
            )
            self.assertTrue(
                db.paper_seen(
                    conn, "OTHER", "10.1000/passed", reconsider_rejected=True
                )
            )

            self.assertEqual(len(db.fetch_known_titles(conn)), 2)
            self.assertEqual(
                db.fetch_known_titles(conn, reconsider_rejected=True),
                [(db.normalize_title(passed_title), "rivera")],
            )

    def test_rejection_cannot_downgrade_passed_disposition_for_same_id(self):
        with db.connect(self.db_path) as conn:
            db.record_disposition(
                conn,
                "P1",
                "10.1000/original",
                "A selected coastal paper",
                ["Rivera, Ana"],
                "passed",
                9,
                10.8,
            )
            for rejection in ("dropped_dedup", "dropped_vector", "dropped_llm"):
                db.record_disposition(
                    conn,
                    "P1",
                    "10.1000/replacement",
                    "Replacement metadata",
                    ["Other, Author"],
                    rejection,
                    2,
                    2.0,
                )

            row = conn.execute(
                "SELECT * FROM processed_papers WHERE paper_id = 'P1'"
            ).fetchone()

        self.assertEqual(row["disposition"], "passed")
        self.assertEqual(row["doi"], "10.1000/original")
        self.assertEqual(row["llm_score"], 9)
        self.assertEqual(row["final_score"], 10.8)

    def test_schema_upgrade_is_additive_for_legacy_database(self):
        legacy_path = Path(self.tempdir.name) / "legacy" / "cache.db"
        legacy_path.parent.mkdir()
        with closing(sqlite3.connect(legacy_path)) as conn:
            conn.execute(
                """
                CREATE TABLE processed_papers (
                    paper_id TEXT PRIMARY KEY, doi TEXT, normalized_title TEXT,
                    first_author TEXT, processed_date TEXT, disposition TEXT,
                    llm_score INTEGER, final_score REAL
                )
                """
            )
            conn.execute(
                "INSERT INTO processed_papers VALUES "
                "('LEGACY', NULL, 'legacy title', 'rivera', 'old', "
                "'dropped_llm', 3, 3.0)"
            )
            conn.commit()

        db.init_db(legacy_path)

        with db.connect(legacy_path) as conn:
            self.assertIsNotNone(
                conn.execute(
                    "SELECT 1 FROM processed_papers WHERE paper_id = 'LEGACY'"
                ).fetchone()
            )
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'history_contexts'"
            ).fetchone()
            self.assertIsNotNone(table)
            self.assertEqual(db.fetch_history_contexts(conn), set())


class FilteringHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.settings = _settings(self.root)
        db.init_db(self.settings.db_path)

    def _record(self, paper: RawPaper, disposition: str = "dropped_llm") -> None:
        with db.connect(self.settings.db_path) as conn:
            db.record_disposition(
                conn,
                paper.id,
                paper.doi,
                paper.title,
                paper.authors,
                disposition,
            )

    def _rank_without_model(self, papers, **kwargs):
        with patch(
            "freshlit.nodes.filtering.embeddings.load_profile_vector",
            return_value=[1.0],
        ), patch(
            "freshlit.nodes.filtering.embeddings.embed_texts",
            return_value=[[1.0] for _ in papers],
        ), patch(
            "freshlit.nodes.filtering.embeddings.cosine_to_profile",
            return_value=[0.9 for _ in papers],
        ):
            return preprocess_and_rank(papers, self.settings, dry_run=True, **kwargs)

    def test_reconsider_rejected_changes_exact_and_fuzzy_preprocessing(self):
        exact = _paper("REJECTED", "A long exact history title for shoreline surveys")
        self._record(exact)

        self.assertEqual(preprocess_and_rank([exact], self.settings, dry_run=True), [])
        self.assertEqual(
            [paper.id for paper, _ in self._rank_without_model(
                [exact], reconsider_rejected=True
            )],
            ["REJECTED"],
        )

        historical = _paper(
            "OLD", "Experimental reconstruction of ceramic production methods"
        )
        self._record(historical)
        current = _paper(
            "NEW", "Experimental reconstruction of ceramic production methods"
        )
        self.assertEqual(
            preprocess_and_rank([current], self.settings, dry_run=True), []
        )
        self.assertEqual(
            [paper.id for paper, _ in self._rank_without_model(
                [current], reconsider_rejected=True
            )],
            ["NEW"],
        )

    def test_reconsideration_still_deduplicates_within_current_batch(self):
        first = _paper("NEW-1", "Repeated field surveys of coastal erosion patterns")
        second = _paper("NEW-2", "Repeated field surveys of coastal erosion patterns")

        ranked = self._rank_without_model(
            [first, second], reconsider_rejected=True
        )

        self.assertEqual(len(ranked), 1)

    def test_reconsidered_paper_is_scored_but_dry_run_keeps_rejection(self):
        rejected = _paper("RETRY", "A long rejected title about coastal monitoring")
        self._record(rejected)
        evaluation = LLMEvaluation(
            relevance_score=8,
            passes_rubric=True,
            methodology_tags=["field survey"],
            fit_rationale="Matches the current profile.",
        )

        with patch(
            "freshlit.nodes.filtering.embeddings.load_profile_vector",
            return_value=[1.0],
        ), patch(
            "freshlit.nodes.filtering.embeddings.embed_texts", return_value=[[1.0]]
        ), patch(
            "freshlit.nodes.filtering.embeddings.cosine_to_profile",
            return_value=[0.9],
        ), patch(
            "freshlit.nodes.filtering._score_batch", return_value=[evaluation]
        ):
            qualified = filter_and_score(
                [rejected],
                self.settings,
                object(),
                dry_run=True,
                reconsider_rejected=True,
            )

        self.assertEqual([item.raw_paper.id for item in qualified], ["RETRY"])
        with db.connect(self.settings.db_path) as conn:
            row = conn.execute(
                "SELECT disposition FROM processed_papers WHERE paper_id = 'RETRY'"
            ).fetchone()
            self.assertEqual(row["disposition"], "dropped_llm")
            self.assertEqual(db.fetch_history_contexts(conn), set())

    def test_unknown_context_warns_and_dry_run_writes_no_history(self):
        rejected = _paper("LEGACY", "A long legacy title about coastal monitoring")
        self._record(rejected)
        with db.connect(self.settings.db_path) as conn:
            before = dict(
                conn.execute(
                    "SELECT * FROM processed_papers WHERE paper_id = 'LEGACY'"
                ).fetchone()
            )

        with self.assertLogs("freshlit.nodes.filtering", level="WARNING") as captured:
            preprocess_and_rank([rejected], self.settings, dry_run=True)

        self.assertIn("different or unknown", " ".join(captured.output))
        self.assertIn("current input only", " ".join(captured.output))
        self.assertIn("does not fetch", " ".join(captured.output))
        with db.connect(self.settings.db_path) as conn:
            self.assertEqual(db.fetch_history_contexts(conn), set())
            after = dict(
                conn.execute(
                    "SELECT * FROM processed_papers WHERE paper_id = 'LEGACY'"
                ).fetchone()
            )
        self.assertEqual(after, before)

    def test_non_dry_legacy_run_keeps_unknown_marker_and_current_context(self):
        rejected = _paper("LEGACY", "A long legacy title about coastal monitoring")
        self._record(rejected)

        with self.assertLogs("freshlit.nodes.filtering", level="WARNING"):
            preprocess_and_rank([rejected], self.settings, dry_run=False)

        fingerprint = _selection_context_fingerprint(self.settings)
        with db.connect(self.settings.db_path) as conn:
            self.assertEqual(
                db.fetch_history_contexts(conn),
                {db.LEGACY_HISTORY_CONTEXT, fingerprint},
            )

        with self.assertLogs("freshlit.nodes.filtering", level="WARNING") as captured:
            preprocess_and_rank([rejected], self.settings, dry_run=True)
        self.assertIn("unknown", " ".join(captured.output))

    def test_changed_context_warns_without_removing_older_context(self):
        rejected = _paper("P1", "A long context-aware coastal monitoring title")
        self._record(rejected)
        old_settings = _settings(self.root, profile="Earlier research profile")
        old_fingerprint = _selection_context_fingerprint(old_settings)
        with db.connect(self.settings.db_path) as conn:
            db.record_history_context(conn, old_fingerprint)

        with self.assertLogs("freshlit.nodes.filtering", level="WARNING") as captured:
            preprocess_and_rank([rejected], self.settings, dry_run=False)

        self.assertIn("selection context", " ".join(captured.output))
        current_fingerprint = _selection_context_fingerprint(self.settings)
        with db.connect(self.settings.db_path) as conn:
            self.assertEqual(
                db.fetch_history_contexts(conn),
                {old_fingerprint, current_fingerprint},
            )

        with self.assertLogs("freshlit.nodes.filtering", level="WARNING"):
            preprocess_and_rank([rejected], self.settings, dry_run=True)

    def test_matching_context_does_not_warn(self):
        rejected = _paper("P1", "A long context-aware coastal monitoring title")
        self._record(rejected)
        with db.connect(self.settings.db_path) as conn:
            db.record_history_context(
                conn, _selection_context_fingerprint(self.settings)
            )

        with patch("freshlit.nodes.filtering.log.warning") as warning:
            preprocess_and_rank([rejected], self.settings, dry_run=True)

        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
