from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import freshlit


class RunOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        root = Path(self.tempdir.name)
        self.settings = SimpleNamespace(
            project_root=root,
            db_path=root / "data" / "cache.db",
            profile_vector_path=root / "data" / "profile.json",
            ingestion=SimpleNamespace(lookback_days=7),
            filtering=SimpleNamespace(
                max_llm_candidates=20,
                embedding_model="mock-embedding-model",
            ),
            llm=SimpleNamespace(model="mock-llm-model"),
        )

        self.loadSettings = MagicMock(return_value=self.settings)
        self.initDb = MagicMock()
        self.buildProfile = MagicMock(return_value=[1.0])
        self.saveProfile = MagicMock()
        self.profileFingerprint = MagicMock(return_value="profile-fingerprint")
        self.profileVectorIsCurrent = MagicMock(return_value=True)
        self.ingest = MagicMock()
        self.loadIngestCache = MagicMock()
        self.saveIngestCache = MagicMock()
        self.preprocess = MagicMock()
        self.filterAndScore = MagicMock()
        self.synthesize = MagicMock()
        self.deliver = MagicMock()
        self.validateDestination = MagicMock()
        self.buildClient = MagicMock()
        self.projectRoot = MagicMock(return_value=root)

        dbModule = self.module("freshlit.utils.db", init_db=self.initDb)
        embeddingsModule = self.module(
            "freshlit.utils.embeddings",
            build_profile_vector=self.buildProfile,
            save_profile_vector=self.saveProfile,
            profile_fingerprint=self.profileFingerprint,
            profile_vector_is_current=self.profileVectorIsCurrent,
        )
        configModule = self.module(
            "freshlit.utils.config",
            _project_root=self.projectRoot,
            load_settings=self.loadSettings,
        )
        llmModule = self.module(
            "freshlit.utils.llm", build_client=self.buildClient
        )
        utilsModule = self.module(
            "freshlit.utils",
            db=dbModule,
            embeddings=embeddingsModule,
            config=configModule,
            llm=llmModule,
        )
        utilsModule.__path__ = []

        ingestionModule = self.module(
            "freshlit.nodes.ingestion",
            ingest=self.ingest,
            load_ingest_cache=self.loadIngestCache,
            save_ingest_cache=self.saveIngestCache,
        )
        filteringModule = self.module(
            "freshlit.nodes.filtering",
            preprocess_and_rank=self.preprocess,
            filter_and_score=self.filterAndScore,
        )
        synthesisModule = self.module(
            "freshlit.nodes.synthesis", synthesize=self.synthesize
        )
        deliveryModule = self.module(
            "freshlit.nodes.delivery",
            deliver=self.deliver,
            validate_destination=self.validateDestination,
        )
        nodesModule = self.module(
            "freshlit.nodes",
            ingestion=ingestionModule,
            filtering=filteringModule,
            synthesis=synthesisModule,
            delivery=deliveryModule,
        )
        nodesModule.__path__ = []

        fakeModules = {
            module.__name__: module
            for module in (
                utilsModule,
                dbModule,
                embeddingsModule,
                configModule,
                llmModule,
                nodesModule,
                ingestionModule,
                filteringModule,
                synthesisModule,
                deliveryModule,
            )
        }
        trackedNames = (*fakeModules, "freshlit.main")
        missing = object()
        previousModules = {
            name: sys.modules.get(name, missing) for name in trackedNames
        }
        previousAttributes = {
            name: getattr(freshlit, name, missing) for name in ("main", "nodes", "utils")
        }
        sys.modules.update(fakeModules)
        sys.modules.pop("freshlit.main", None)

        def restore_imports() -> None:
            for name, previous in previousModules.items():
                if previous is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
            for name, previous in previousAttributes.items():
                if previous is missing:
                    freshlit.__dict__.pop(name, None)
                else:
                    setattr(freshlit, name, previous)

        self.addCleanup(restore_imports)
        self.main = importlib.import_module("freshlit.main")

    @staticmethod
    def module(name: str, **attributes) -> types.ModuleType:
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        return module

    @staticmethod
    def args(**overrides):
        defaults = {
            "lookback": None,
            "max_candidates": None,
            "model": None,
            "provider": None,
            "from_cache": False,
            "limit": None,
            "skip_llm": False,
            "dry_run": False,
            "force": False,
            "output_name": None,
            "reconsider_rejected": False,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_empty_ingestion_does_not_construct_llm_client(self) -> None:
        self.ingest.return_value = []

        self.main._cmd_run(self.args())

        self.loadSettings.assert_called_once_with(
            provider_override=None, require_llm_credentials=True
        )
        self.validateDestination.assert_called_once_with(
            self.settings, filename=None
        )
        self.initDb.assert_called_once_with(self.settings.db_path)
        self.profileVectorIsCurrent.assert_called_once_with(self.settings)
        self.buildProfile.assert_not_called()
        self.ingest.assert_called_once_with(
            self.settings,
            cache_path=self.settings.project_root / "data" / "last_ingest.json",
        )
        self.saveIngestCache.assert_not_called()
        self.loadIngestCache.assert_not_called()
        self.buildClient.assert_not_called()
        self.filterAndScore.assert_not_called()
        self.synthesize.assert_not_called()
        self.deliver.assert_not_called()

    def test_skip_llm_does_not_construct_llm_client(self) -> None:
        paper = SimpleNamespace(source_type="journal", title="Deterministic paper")
        self.ingest.return_value = [paper]
        self.preprocess.return_value = [(paper, 0.75)]

        self.main._cmd_run(self.args(skip_llm=True))

        self.preprocess.assert_called_once_with(
            [paper], self.settings, dry_run=True, reconsider_rejected=False
        )
        self.loadSettings.assert_called_once_with(
            provider_override=None, require_llm_credentials=False
        )
        self.validateDestination.assert_not_called()
        self.initDb.assert_called_once_with(self.settings.db_path)
        self.profileVectorIsCurrent.assert_called_once_with(self.settings)
        self.ingest.assert_called_once_with(
            self.settings,
            cache_path=self.settings.project_root / "data" / "last_ingest.json",
        )
        self.saveIngestCache.assert_not_called()
        self.buildClient.assert_not_called()
        self.filterAndScore.assert_not_called()
        self.synthesize.assert_not_called()
        self.deliver.assert_not_called()

    def test_client_context_exits_before_delivery(self) -> None:
        event_eid = []
        paper = object()
        qualified_paperid = [object()]
        summary = SimpleNamespace(final_score=8.5)
        pulse = object()
        selectedClient = object()
        clientManager = MagicMock()

        self.ingest.return_value = [paper]
        self.filterAndScore.side_effect = lambda *args, **kwargs: (
            event_eid.append("filter") or qualified_paperid
        )
        self.synthesize.side_effect = lambda *args, **kwargs: (
            event_eid.append("synthesize") or ([summary], pulse)
        )
        self.deliver.side_effect = lambda *args, **kwargs: (
            event_eid.append("deliver") or Path(self.tempdir.name) / "digest.md"
        )
        self.buildClient.side_effect = lambda settings: (
            event_eid.append("build") or clientManager
        )
        clientManager.__enter__.side_effect = lambda: (
            event_eid.append("enter") or selectedClient
        )
        clientManager.__exit__.side_effect = lambda *args: (
            event_eid.append("exit") or False
        )

        self.main._cmd_run(self.args())

        self.filterAndScore.assert_called_once_with(
            [paper], self.settings, selectedClient, dry_run=False,
            reconsider_rejected=False,
        )
        self.synthesize.assert_called_once_with(
            selectedClient, self.settings, qualified_paperid
        )
        self.assertEqual(
            event_eid,
            ["build", "enter", "filter", "synthesize", "exit", "deliver"],
        )
        self.buildClient.assert_called_once_with(self.settings)
        clientManager.__exit__.assert_called_once_with(None, None, None)
        self.validateDestination.assert_called_once_with(
            self.settings, filename=None
        )
        self.deliver.assert_called_once_with(
            self.settings,
            [summary],
            pulse,
            papers_analyzed=1,
            force=False,
            dry_run=False,
            filename=None,
        )

    def test_client_context_closes_on_no_qualified_early_return(self) -> None:
        paper = object()
        selectedClient = object()
        clientManager = MagicMock()
        clientManager.__enter__.return_value = selectedClient
        self.ingest.return_value = [paper]
        self.buildClient.return_value = clientManager
        self.filterAndScore.return_value = []

        self.main._cmd_run(self.args())

        self.filterAndScore.assert_called_once_with(
            [paper], self.settings, selectedClient, dry_run=False,
            reconsider_rejected=False,
        )
        self.buildClient.assert_called_once_with(self.settings)
        clientManager.__enter__.assert_called_once_with()
        clientManager.__exit__.assert_called_once_with(None, None, None)
        self.synthesize.assert_not_called()
        self.deliver.assert_not_called()

    def test_provider_override_is_passed_to_load_settings(self) -> None:
        self.ingest.return_value = []

        self.main._cmd_run(self.args(provider="gateway"))

        self.loadSettings.assert_called_once_with(
            provider_override="gateway", require_llm_credentials=True
        )

    def test_invalid_destination_fails_before_pipeline_side_effects(self) -> None:
        self.validateDestination.side_effect = ValueError("unsafe output")

        with self.assertRaisesRegex(ValueError, "unsafe output"):
            self.main._cmd_run(self.args(output_name="digest.md"))

        self.validateDestination.assert_called_once_with(
            self.settings, filename="digest.md"
        )
        self.initDb.assert_not_called()
        self.profileVectorIsCurrent.assert_not_called()
        self.buildProfile.assert_not_called()
        self.profileFingerprint.assert_not_called()
        self.saveProfile.assert_not_called()
        self.ingest.assert_not_called()
        self.loadIngestCache.assert_not_called()
        self.saveIngestCache.assert_not_called()
        self.buildClient.assert_not_called()
        self.preprocess.assert_not_called()
        self.filterAndScore.assert_not_called()
        self.synthesize.assert_not_called()
        self.deliver.assert_not_called()

    def test_dry_run_does_not_preflight_output_destination(self) -> None:
        self.ingest.return_value = []

        self.main._cmd_run(self.args(dry_run=True, output_name="ignored.md"))

        self.validateDestination.assert_not_called()
        self.loadSettings.assert_called_once_with(
            provider_override=None, require_llm_credentials=True
        )
        self.initDb.assert_called_once_with(self.settings.db_path)
        self.ingest.assert_called_once_with(
            self.settings,
            cache_path=self.settings.project_root / "data" / "last_ingest.json",
        )

    def test_stale_profile_vector_is_rebuilt_with_fingerprint(self) -> None:
        self.profileVectorIsCurrent.return_value = False
        self.ingest.return_value = []

        self.main._cmd_run(self.args())

        self.profileVectorIsCurrent.assert_called_once_with(self.settings)
        self.buildProfile.assert_called_once_with(self.settings)
        self.profileFingerprint.assert_called_once_with(self.settings)
        self.saveProfile.assert_called_once_with(
            self.settings.profile_vector_path,
            [1.0],
            self.settings.filtering.embedding_model,
            fingerprint="profile-fingerprint",
        )
        self.ingest.assert_called_once_with(
            self.settings,
            cache_path=self.settings.project_root / "data" / "last_ingest.json",
        )

    def test_current_profile_vector_is_reused(self) -> None:
        self.ingest.return_value = []

        self.main._cmd_run(self.args())

        self.profileVectorIsCurrent.assert_called_once_with(self.settings)
        self.buildProfile.assert_not_called()
        self.profileFingerprint.assert_not_called()
        self.saveProfile.assert_not_called()

    def test_missing_offline_cache_fails_before_writes_or_external_work(self) -> None:
        cachePath = self.settings.project_root / "data" / "last_ingest.json"
        self.assertFalse(cachePath.exists())

        with self.assertRaisesRegex(FileNotFoundError, "No ingestion cache"):
            self.main._cmd_run(self.args(from_cache=True, skip_llm=True))

        self.loadSettings.assert_called_once_with(
            provider_override=None, require_llm_credentials=False
        )
        self.validateDestination.assert_not_called()
        self.initDb.assert_not_called()
        self.profileVectorIsCurrent.assert_not_called()
        self.buildProfile.assert_not_called()
        self.profileFingerprint.assert_not_called()
        self.saveProfile.assert_not_called()
        self.ingest.assert_not_called()
        self.loadIngestCache.assert_not_called()
        self.saveIngestCache.assert_not_called()
        self.buildClient.assert_not_called()
        self.preprocess.assert_not_called()
        self.filterAndScore.assert_not_called()
        self.synthesize.assert_not_called()
        self.deliver.assert_not_called()

    def test_offline_cache_and_reconsider_flag_are_propagated(self) -> None:
        cachePath = self.settings.project_root / "data" / "last_ingest.json"
        cachePath.parent.mkdir(parents=True)
        cachePath.write_text("cached", encoding="utf-8")
        paper = SimpleNamespace(source_type="journal", title="Cached paper")
        self.loadIngestCache.return_value = [paper]
        self.preprocess.return_value = [(paper, 0.8)]

        self.main._cmd_run(
            self.args(
                from_cache=True,
                skip_llm=True,
                reconsider_rejected=True,
            )
        )

        self.loadIngestCache.assert_called_once_with(
            cachePath, settings=self.settings
        )
        self.ingest.assert_not_called()
        self.saveIngestCache.assert_not_called()
        self.preprocess.assert_called_once_with(
            [paper],
            self.settings,
            dry_run=True,
            reconsider_rejected=True,
        )
        self.validateDestination.assert_not_called()
        self.buildClient.assert_not_called()

    def test_online_cache_settings_and_reconsider_flag_are_propagated(self) -> None:
        paper = object()
        selectedClient = object()
        clientManager = MagicMock()
        clientManager.__enter__.return_value = selectedClient
        self.buildClient.return_value = clientManager
        self.ingest.return_value = [paper]
        self.filterAndScore.return_value = []

        self.main._cmd_run(self.args(reconsider_rejected=True))

        self.ingest.assert_called_once_with(
            self.settings,
            cache_path=self.settings.project_root / "data" / "last_ingest.json",
        )
        self.saveIngestCache.assert_not_called()
        self.filterAndScore.assert_called_once_with(
            [paper],
            self.settings,
            selectedClient,
            dry_run=False,
            reconsider_rejected=True,
        )


if __name__ == "__main__":
    unittest.main()
