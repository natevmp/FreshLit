from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from freshlit.utils.config import FilteringConfig, LLMConfig, load_settings


class LLMConfigTests(unittest.TestCase):
    def test_defaults_and_bounds(self) -> None:
        config = LLMConfig()
        self.assertEqual(config.provider, "codex")
        self.assertEqual(config.model, "gpt-5.6-luna")
        self.assertIsNone(config.codex_home)

        for values in (
            {"timeout_seconds": 0},
            {"workers": 0},
            {"max_retries": -1},
            {"max_retries": 6},
            {"provider": "other"},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                LLMConfig(**values)


class FilteringConfigTests(unittest.TestCase):
    def test_llm_batch_size_default_and_bounds(self) -> None:
        self.assertEqual(FilteringConfig().llm_batch_size, 15)
        for batchSize in (1, 100):
            with self.subTest(batchSize=batchSize):
                self.assertEqual(
                    FilteringConfig(llm_batch_size=batchSize).llm_batch_size,
                    batchSize,
                )

        for batchSize in (0, 101):
            with self.subTest(batchSize=batchSize), self.assertRaises(ValidationError):
                FilteringConfig(llm_batch_size=batchSize)


class LoadSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        config_dir = self.root / "config"
        config_dir.mkdir()
        (config_dir / "research_profile.md").write_text(
            "## Keywords\n- clonal dynamics\n", encoding="utf-8"
        )
        (config_dir / "journal_tiers.json").write_text(
            json.dumps({"1234-5678": 1.5}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_settings(self, provider: str = "codex") -> None:
        (self.root / "config" / "settings.yaml").write_text(
            "obsidian:\n"
            "  vault_path: /yaml/vault\n"
            "llm:\n"
            f"  provider: {provider}\n",
            encoding="utf-8",
        )

    def test_codex_does_not_require_or_retrieve_gateway_key(self) -> None:
        self.write_settings()
        (self.root / ".env").write_text(
            "OPENCODE_GO_API_KEY=dotenv-secret\n", encoding="utf-8"
        )
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings(self.root)
            self.assertIsNone(settings.opencode_go_api_key)
            self.assertNotIn("OPENCODE_GO_API_KEY", os.environ)

    def test_gateway_alone_requires_key(self) -> None:
        self.write_settings("gateway")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENCODE_GO_API_KEY"):
                load_settings(self.root)

            (self.root / ".env").write_text(
                "OPENCODE_GO_API_KEY=gateway-key\n", encoding="utf-8"
            )
            settings = load_settings(self.root)
        self.assertEqual(settings.opencode_go_api_key, "gateway-key")

    def test_cli_and_environment_provider_overrides_are_isolated(self) -> None:
        self.write_settings("gateway")
        codex_home = self.root.parent
        environment = {
            "FRESHLIT_LLM_PROVIDER": "gateway",
            "FRESHLIT_CODEX_HOME": str(codex_home),
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = load_settings(self.root, provider_override="codex")
        self.assertEqual(settings.llm.provider, "codex")
        self.assertEqual(settings.llm.codex_home, codex_home)
        self.assertIsNone(settings.opencode_go_api_key)

    def test_dotenv_openalex_and_vault_values_do_not_mutate_environment(self) -> None:
        self.write_settings()
        (self.root / ".env").write_text(
            "OPENALEX_API_KEY=dotenv-openalex\n"
            "FRESHLIT_VAULT_PATH=/dotenv/vault\n",
            encoding="utf-8",
        )
        environment = {"FRESHLIT_VAULT_PATH": "/process/vault"}
        with patch.dict(os.environ, environment, clear=True):
            settings = load_settings(self.root)
            self.assertNotIn("OPENALEX_API_KEY", os.environ)
        self.assertEqual(settings.openalex_api_key, "dotenv-openalex")
        self.assertEqual(settings.obsidian.vault_path, Path("/process/vault"))
        self.assertEqual(settings.ingestion.keywords, ["clonal dynamics"])
        self.assertEqual(settings.journal_tiers, {"12345678": 1.5})


if __name__ == "__main__":
    unittest.main()
