from __future__ import annotations

import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from freshlit.nodes import ingestion
from freshlit.nodes.ingestion import RawPaper
from freshlit.utils import embeddings


def make_settings(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        profile_vector_path=root / "profile.json",
        research_profile_text=(
            "# Research profile\n\nNarrative kept verbatim.\n\n"
            "## Keywords\n- exoplanet atmospheres\n- transmission spectroscopy\n"
        ),
        filtering=SimpleNamespace(
            embedding_model="example/model-a",
            embedding_device="cpu",
            vector_threshold=0.65,
        ),
        ingestion=SimpleNamespace(
            lookback_days=7,
            timezone="UTC",
            max_results_per_query=125,
            openalex_topics=["T100", "T200"],
            keywords=["override one", "override two"],
            topic_labels={"T100": "Display label"},
            europe_pmc=SimpleNamespace(enabled=True, sources=["PPR", "MED"]),
        ),
        api=SimpleNamespace(
            max_pages=3,
            request_timeout_seconds=30,
            retry_backoff_seconds=2.0,
        ),
        openalex_api_key="private-openalex-key",
        opencode_go_api_key="private-llm-key",
        private_narrative="not cache metadata",
    )


def make_paper() -> RawPaper:
    return RawPaper(
        id="https://openalex.org/W1",
        doi="10.1000/example",
        title="A deterministic paper",
        authors=["Ada Lovelace"],
        publication_date="2026-09-27",
        venue_name="Example Journal",
        venue_issn="12345678",
        abstract="A sufficiently descriptive abstract for a cache fixture.",
        source_type="journal",
    )


class ProfileCacheProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.settings = make_settings(self.root)

    def save_current(self) -> None:
        embeddings.save_profile_vector(
            self.settings.profile_vector_path,
            np.array([3.0, 4.0]),
            self.settings.filtering.embedding_model,
            fingerprint=embeddings.profile_fingerprint(self.settings),
        )

    def test_fingerprint_is_deterministic_and_uses_actual_embedding_inputs(self) -> None:
        first = embeddings.profile_fingerprint(self.settings)
        self.assertEqual(first, embeddings.profile_fingerprint(self.settings))

        changed_query = deepcopy(self.settings)
        changed_query.ingestion.keywords.append("YAML-only query override")
        changed_query.ingestion.topic_labels["T100"] = "Renamed display label"
        self.assertEqual(first, embeddings.profile_fingerprint(changed_query))

        changed_text = deepcopy(self.settings)
        changed_text.research_profile_text += "\nExact body change.\n"
        self.assertNotEqual(first, embeddings.profile_fingerprint(changed_text))

        changed_model = deepcopy(self.settings)
        changed_model.filtering.embedding_model = "example/model-b"
        self.assertNotEqual(first, embeddings.profile_fingerprint(changed_model))

    def test_new_profile_cache_roundtrip_and_metadata_are_minimal(self) -> None:
        self.save_current()

        loaded = embeddings.load_profile_vector(
            self.settings.profile_vector_path, settings=self.settings
        )
        np.testing.assert_allclose(loaded, np.array([0.6, 0.8]))
        self.assertTrue(embeddings.profile_vector_is_current(self.settings))

        payload = json.loads(self.settings.profile_vector_path.read_text())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["model"], "example/model-a")
        self.assertEqual(payload["dimensions"], 2)
        self.assertNotIn("research_profile_text", payload)
        self.assertNotIn("embedding_device", payload)
        self.assertNotIn("vector_threshold", payload)
        self.assertNotIn("private_narrative", self.settings.profile_vector_path.read_text())

    def test_text_and_model_changes_make_cache_stale(self) -> None:
        self.save_current()

        changed_text = deepcopy(self.settings)
        changed_text.research_profile_text += "changed"
        self.assertFalse(embeddings.profile_vector_is_current(changed_text))

        changed_model = deepcopy(self.settings)
        changed_model.filtering.embedding_model = "example/model-b"
        self.assertFalse(embeddings.profile_vector_is_current(changed_model))

    def test_missing_legacy_corrupt_nonfinite_and_wrong_dimensions_are_stale(self) -> None:
        path = self.settings.profile_vector_path
        self.assertFalse(embeddings.profile_vector_is_current(self.settings))

        cases = (
            {"model": "example/model-a", "vector": [1.0, 0.0]},
            {"schema_version": 1, "model": "example/model-a", "vector": [1.0]},
            {
                "schema_version": 1,
                "model": "example/model-a",
                "dimensions": 2,
                "fingerprint": embeddings.profile_fingerprint(self.settings),
                "vector": [float("nan"), 1.0],
            },
            {
                "schema_version": 1,
                "model": "example/model-a",
                "dimensions": 3,
                "fingerprint": embeddings.profile_fingerprint(self.settings),
                "vector": [1.0, 0.0],
            },
            {
                "schema_version": 1,
                "model": "example/model-a",
                "dimensions": 1,
                "fingerprint": embeddings.profile_fingerprint(self.settings),
                "vector": [[1.0]],
            },
        )
        for payload in cases:
            with self.subTest(payload=payload):
                path.write_text(json.dumps(payload))
                self.assertFalse(embeddings.profile_vector_is_current(self.settings))

    def test_path_only_load_keeps_legacy_cache_compatible(self) -> None:
        self.settings.profile_vector_path.write_text(
            json.dumps({"model": "old/model", "vector": [3.0, 4.0]})
        )
        np.testing.assert_allclose(
            embeddings.load_profile_vector(self.settings.profile_vector_path),
            np.array([0.6, 0.8]),
        )
        self.assertFalse(embeddings.profile_vector_is_current(self.settings))

    def test_dimension_mismatch_has_rebuild_guidance(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimension mismatch.*build-profile"):
            embeddings.cosine_to_profile(
                np.ones((2, 3), dtype=float), np.ones(2, dtype=float)
            )

    def test_model_singletons_are_keyed_by_model_and_device(self) -> None:
        constructor_calls: list[tuple[str, str, bool]] = []

        class FakeSentenceTransformer:
            def __init__(
                self,
                model_name: str,
                *,
                device: str,
                local_files_only: bool,
            ) -> None:
                constructor_calls.append((model_name, device, local_files_only))

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        embeddings._models.clear()
        self.addCleanup(embeddings._models.clear)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch.object(
                embeddings,
                "_is_model_cached",
                side_effect=(True, True, False),
            ),
        ):
            first = embeddings.get_model(self.settings)
            self.assertIs(first, embeddings.get_model(self.settings))
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

            other_device = deepcopy(self.settings)
            other_device.filtering.embedding_device = "mps"
            second = embeddings.get_model(other_device)

            other_model = deepcopy(self.settings)
            other_model.filtering.embedding_model = "example/model-b"
            third = embeddings.get_model(other_model)
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

        self.assertIsNot(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(
            constructor_calls,
            [
                ("example/model-a", "cpu", True),
                ("example/model-a", "mps", True),
                ("example/model-b", "cpu", False),
            ],
        )

    def test_sentence_transformers_27_constructor_and_explicit_offline_are_supported(self) -> None:
        calls = []

        class LegacySentenceTransformer:
            # Signature of the supported 2.7 interface: no local_files_only.
            def __init__(self, model_name, *, device):
                calls.append((model_name, device))

        embeddings._models.clear()
        self.addCleanup(embeddings._models.clear)
        with (
            patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=True),
            patch.dict("sys.modules", {"sentence_transformers": SimpleNamespace(
                SentenceTransformer=LegacySentenceTransformer,
            )}),
        ):
            embeddings.get_model(self.settings)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(calls, [("example/model-a", "cpu")])


class IngestCacheProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.path = self.root / "ingest.json"
        self.settings = make_settings(self.root)
        self.paper = make_paper()

    def test_ingest_uses_one_window_for_queries_and_cache_across_midnight(self) -> None:
        original_window = ("2026-09-21", "2026-09-28")
        next_window = ("2026-09-22", "2026-09-29")
        current_window = [original_window]
        requests_seen: list[tuple[str, dict]] = []

        def changing_date_window(settings) -> tuple[str, str]:
            return current_window[0]

        def get_json(session, url, params, settings):
            requests_seen.append((url, params))
            current_window[0] = next_window
            return None

        with (
            patch.object(ingestion, "_date_window", side_effect=changing_date_window),
            patch.object(ingestion, "_get_json", side_effect=get_json),
        ):
            papers = ingestion.ingest(self.settings, cache_path=self.path)

        self.assertEqual(papers, [])
        self.assertEqual(len(requests_seen), 3)
        openalex_filters = [
            params["filter"]
            for url, params in requests_seen
            if url == ingestion.OPENALEX_WORKS
        ]
        self.assertEqual(
            openalex_filters,
            [
                "from_publication_date:2026-09-21,"
                "to_publication_date:2026-09-28,topics.id:T100|T200",
                "from_publication_date:2026-09-21,"
                "to_publication_date:2026-09-28",
            ],
        )
        europe_query = next(
            params["query"]
            for url, params in requests_seen
            if url == ingestion.EUROPE_PMC_SEARCH
        )
        self.assertIn("FIRST_PDATE:[2026-09-21 TO 2026-09-28]", europe_query)

        payload = json.loads(self.path.read_text())
        self.assertEqual(
            payload["query_provenance"]["date_window"],
            {"from": "2026-09-21", "to": "2026-09-28"},
        )
        with (
            patch.object(ingestion, "_date_window", return_value=next_window),
            self.assertLogs(ingestion.log, level="WARNING") as captured,
        ):
            replayed = ingestion.load_ingest_cache(self.path, self.settings)
        self.assertEqual(replayed, [])
        self.assertIn("--from-cache", " ".join(captured.output))

    def test_ingest_without_cache_path_does_not_write(self) -> None:
        with (
            patch.object(ingestion, "fetch_openalex", return_value=[]),
            patch.object(ingestion, "fetch_europe_pmc", return_value=[]),
            patch.object(ingestion, "save_ingest_cache") as save_cache,
        ):
            self.assertEqual(ingestion.ingest(self.settings), [])
        save_cache.assert_not_called()

    def test_captured_provenance_is_checked_without_regenerating_its_date(self) -> None:
        with patch.object(
            ingestion, "_date_window", return_value=("2026-09-21", "2026-09-28")
        ):
            provenance = ingestion._query_provenance(self.settings)

        with patch.object(
            ingestion,
            "_date_window",
            side_effect=AssertionError("must not regenerate the captured date"),
        ):
            ingestion.save_ingest_cache(
                self.path,
                [self.paper],
                self.settings,
                query_provenance=provenance,
            )

        changed = deepcopy(self.settings)
        changed.ingestion.keywords.append("different query")
        with self.assertRaisesRegex(ValueError, "does not match the settings"):
            ingestion.save_ingest_cache(
                self.path,
                [self.paper],
                changed,
                query_provenance=provenance,
            )

    def test_new_cache_roundtrip_records_public_query_provenance_only(self) -> None:
        with patch.object(
            ingestion, "_date_window", return_value=("2026-09-21", "2026-09-28")
        ):
            ingestion.save_ingest_cache(self.path, [self.paper], self.settings)
            loaded = ingestion.load_ingest_cache(self.path, self.settings)

        self.assertEqual(loaded, [self.paper])
        payload = json.loads(self.path.read_text())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["records"], [self.paper.model_dump()])
        self.assertEqual(
            payload["query_provenance"]["selection"],
            {
                "openalex_topics": ["T100", "T200"],
                "keywords": ["override one", "override two"],
            },
        )
        serialized = self.path.read_text()
        for private_value in (
            "private-openalex-key",
            "private-llm-key",
            "Narrative kept verbatim",
            "Display label",
            "not cache metadata",
        ):
            self.assertNotIn(private_value, serialized)

    def test_query_fingerprint_changes_only_for_effective_query_inputs(self) -> None:
        with patch.object(
            ingestion, "_date_window", return_value=("2026-09-21", "2026-09-28")
        ):
            original = ingestion.ingest_query_fingerprint(self.settings)
            self.assertEqual(original, ingestion.ingest_query_fingerprint(self.settings))

            for mutate in (
                lambda settings: settings.ingestion.keywords.append("new query"),
                lambda settings: settings.ingestion.openalex_topics.append("T300"),
                lambda settings: settings.ingestion.europe_pmc.sources.append("PMC"),
                lambda settings: setattr(
                    settings.ingestion, "max_results_per_query", 126
                ),
                lambda settings: setattr(settings.api, "max_pages", 4),
            ):
                changed = deepcopy(self.settings)
                mutate(changed)
                self.assertNotEqual(
                    original, ingestion.ingest_query_fingerprint(changed)
                )

            labels = deepcopy(self.settings)
            labels.ingestion.topic_labels = {"T100": "Entirely new display label"}
            labels.openalex_api_key = "rotated-secret"
            labels.opencode_go_api_key = "another-secret"
            labels.api.request_timeout_seconds = 999
            labels.api.retry_backoff_seconds = 99.0
            labels.private_narrative = "different private text"
            self.assertEqual(original, ingestion.ingest_query_fingerprint(labels))

    def test_changed_query_warns_but_replays_cached_records(self) -> None:
        with patch.object(
            ingestion, "_date_window", return_value=("2026-09-21", "2026-09-28")
        ):
            ingestion.save_ingest_cache(self.path, [self.paper], self.settings)
            changed = deepcopy(self.settings)
            changed.ingestion.keywords.append("new query")
            with self.assertLogs(ingestion.log, level="WARNING") as captured:
                loaded = ingestion.load_ingest_cache(self.path, changed)

        self.assertEqual(loaded, [self.paper])
        self.assertIn("--from-cache", " ".join(captured.output))
        self.assertIn("old result set", " ".join(captured.output))

    def test_legacy_cache_warns_and_replays(self) -> None:
        ingestion.save_ingest_cache(self.path, [self.paper])

        with self.assertLogs(ingestion.log, level="WARNING") as captured:
            loaded = ingestion.load_ingest_cache(self.path, self.settings)

        self.assertEqual(loaded, [self.paper])
        warning = " ".join(captured.output)
        self.assertIn("no query provenance", warning)
        self.assertIn("--from-cache", warning)

    def test_unknown_schema_and_malformed_envelopes_fail_actionably(self) -> None:
        cases = (
            (
                {"schema_version": 99, "records": []},
                "unsupported schema version.*without --from-cache",
            ),
            ({"schema_version": 1, "records": []}, "missing query provenance"),
            (
                {
                    "schema_version": 1,
                    "query_provenance": {},
                    "query_fingerprint": ingestion._provenance_fingerprint({}),
                    "records": [],
                },
                "query provenance has missing or unknown fields",
            ),
            (
                {
                    "schema_version": 1,
                    "query_provenance": {"unexpected": "metadata"},
                    "query_fingerprint": "wrong",
                    "records": [],
                },
                "query provenance has missing or unknown fields",
            ),
        )
        for payload, message in cases:
            with self.subTest(payload=payload):
                self.path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, message):
                    ingestion.load_ingest_cache(self.path, self.settings)


if __name__ == "__main__":
    unittest.main()
