from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import yaml

from freshlit import main as cli
from freshlit.utils import onboarding, profile


class CLIDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

        rootPatch = patch.object(cli, "_project_root", return_value=self.root)
        loggingPatch = patch.object(cli, "_setup_logging")
        self.projectRoot = rootPatch.start()
        self.setupLogging = loggingPatch.start()
        self.addCleanup(rootPatch.stop)
        self.addCleanup(loggingPatch.stop)

    def invoke(self, *arguments: str) -> str:
        output = io.StringIO()
        with patch.object(sys, "argv", ["freshlit", *arguments]), patch.object(
            sys, "stdout", output
        ):
            cli.main()
        return output.getvalue()

    def test_doctor_errors_exit_but_warnings_do_not(self) -> None:
        with patch.object(cli, "load_settings") as loadSettings, patch.object(
            onboarding,
            "doctor",
            return_value=["WARN: cache is stale", "OK: local checks complete"],
        ) as doctor:
            output = self.invoke("doctor")

        self.assertEqual(
            output, "WARN: cache is stale\nOK: local checks complete\n"
        )
        doctor.assert_called_once_with(self.root)
        loadSettings.assert_not_called()

        with patch.object(cli, "load_settings") as loadSettings, patch.object(
            onboarding,
            "doctor",
            return_value=["WARN: optional cache missing", "ERROR: unsafe output"],
        ) as doctor, self.assertRaises(SystemExit) as raised:
            self.invoke("doctor")

        self.assertEqual(raised.exception.code, 1)
        doctor.assert_called_once_with(self.root)
        loadSettings.assert_not_called()
        self.assertEqual(self.projectRoot.call_count, 2)

    def test_init_propagates_explicit_directory_flags(self) -> None:
        with patch.object(cli, "load_settings") as loadSettings, patch.object(
            onboarding, "initialize", return_value=["initialized"]
        ) as initialize:
            output = self.invoke(
                "init", "--create-output-dir", "--prepare-codex-home"
            )

        self.assertEqual(output, "initialized\n")
        initialize.assert_called_once_with(
            self.root,
            create_output_dir=True,
            prepare_codex_home=True,
        )
        loadSettings.assert_not_called()

    def test_topic_lookup_does_not_load_pipeline_settings(self) -> None:
        topics = [{"id": "T123", "label": "Computational biology"}]
        with patch.object(cli, "load_settings") as loadSettings, patch.object(
            onboarding, "search_topics", return_value=topics
        ) as searchTopics:
            output = self.invoke("topics", "  cell dynamics  ")

        searchTopics.assert_called_once_with("  cell dynamics  ", self.root)
        loadSettings.assert_not_called()
        self.assertEqual(yaml.safe_load(output), {"openalex_topics": topics})

    def test_profile_migration_previews_then_applies(self) -> None:
        migrated = "---\nversion: 1\n---\nProfile body\n"
        migrateProfile = MagicMock(return_value=migrated)
        with patch.object(profile, "migrate_profile", migrateProfile):
            preview = self.invoke("migrate-profile")
            applied = self.invoke("migrate-profile", "--apply")

        self.assertEqual(preview, migrated)
        self.assertEqual(applied, "")
        self.assertEqual(
            migrateProfile.call_args_list,
            [call(self.root, apply=False), call(self.root, apply=True)],
        )

    def test_offline_setup_commands_do_not_require_llm_credentials(self) -> None:
        for command, helperName in (
            ("init-db", "_cmd_init_db"),
            ("build-profile", "_cmd_build_profile"),
        ):
            with self.subTest(command=command):
                settings = object()
                with patch.object(
                    cli, "load_settings", return_value=settings
                ) as loadSettings, patch.object(cli, helperName) as helper:
                    self.invoke(command)

                loadSettings.assert_called_once_with(require_llm_credentials=False)
                helper.assert_called_once_with(settings)

    def test_run_dispatch_preserves_skip_llm_flag(self) -> None:
        with patch.object(cli, "_cmd_run") as runCommand:
            self.invoke("run", "--skip-llm", "--reconsider-rejected")

        runCommand.assert_called_once()
        parsed = runCommand.call_args.args[0]
        self.assertTrue(parsed.skip_llm)
        self.assertTrue(parsed.reconsider_rejected)


if __name__ == "__main__":
    unittest.main()
