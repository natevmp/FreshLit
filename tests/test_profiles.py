from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from freshlit.utils.config import load_settings
from freshlit.utils.profile import migrate_profile, parse_research_profile


def modern_profile(
    *,
    topics: str = "  - id: https://openalex.org/T123\n    label: Ecology",
    keywords: str = "- ecological networks",
    description: str = "Quantitative ecology and community dynamics.",
    epmc_enabled: str = "false",
    weights: str = "  - issn: '1234-5678'\n    name: Example Journal\n    weight: 1.2",
    extra: str = "",
) -> str:
    return (
        "---\n"
        "version: 1\n"
        "search:\n"
        "  openalex_topics:\n"
        f"{topics if topics else '    []'}\n"
        "  europe_pmc:\n"
        f"    enabled: {epmc_enabled}\n"
        "    sources: [PPR]\n"
        "venue_weights:\n"
        f"{weights if weights else '  []'}\n"
        f"{extra}"
        "---\n"
        "# Research Profile\n\n"
        "## Keywords\n\n"
        f"{keywords}\n\n"
        "## Research Description\n\n"
        f"{description}\n"
    )


class ProfileParsingTests(unittest.TestCase):
    def test_modern_other_science_profile_normalizes_metadata_and_hides_it_from_body(self) -> None:
        document = modern_profile()
        parsed = parse_research_profile(document)

        self.assertTrue(parsed.is_modern)
        self.assertEqual(parsed.openalex_topics[0].id, "T123")
        self.assertEqual(parsed.keywords, ("ecological networks",))
        self.assertEqual(parsed.venue_weights[0].issn, "12345678")
        self.assertNotIn("openalex_topics", parsed.body)
        self.assertTrue(parsed.body.startswith("# Research Profile"))

    def test_keywords_only_and_topics_only_are_supported(self) -> None:
        keywords_only = parse_research_profile(modern_profile(topics=""))
        topics_only = parse_research_profile(modern_profile(keywords=""))

        self.assertEqual(keywords_only.keywords, ("ecological networks",))
        self.assertEqual(keywords_only.openalex_topics, ())
        self.assertEqual(topics_only.keywords, ())
        self.assertEqual(topics_only.openalex_topics[0].id, "T123")

    def test_template_comments_do_not_make_blank_profile_valid(self) -> None:
        document = modern_profile(
            topics="",
            keywords="<!-- - example keyword -->",
            description="<!-- Describe your research here. -->",
            weights="",
        )
        with self.assertRaisesRegex(ValueError, "Research Description"):
            parse_research_profile(document)

    def test_nonblank_description_still_requires_a_search_selector(self) -> None:
        with self.assertRaisesRegex(ValueError, "no search selectors"):
            parse_research_profile(modern_profile(topics="", keywords="", weights=""))

    def test_rejects_unknown_keys_duplicate_normalized_topics_and_bad_weights(self) -> None:
        cases = (
            (modern_profile(extra="unexpected: true\n"), "Unknown key"),
            (
                modern_profile(
                    topics=(
                        "  - id: T123\n"
                        "  - id: https://openalex.org/T123"
                    )
                ),
                "Duplicate OpenAlex",
            ),
            (
                modern_profile(
                    weights=(
                        "  - issn: '1234-5678'\n"
                        "    weight: 1\n"
                        "  - issn: '12345678'\n"
                        "    weight: 2"
                    )
                ),
                "Duplicate ISSN",
            ),
            (modern_profile(weights="  - issn: '1234-5678'\n    weight: 0"), "positive and finite"),
            (modern_profile(weights="  - issn: '123-5678'\n    weight: 1"), "Malformed ISSN"),
        )
        for document, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                parse_research_profile(document)

    def test_rejects_duplicate_yaml_mapping_keys_at_any_level(self) -> None:
        cases = (
            modern_profile(extra="version: 1\n"),
            modern_profile().replace(
                "    enabled: false\n",
                "    enabled: false\n    enabled: true\n",
            ),
        )
        for document in cases:
            with self.subTest(), self.assertRaisesRegex(ValueError, "duplicate mapping key"):
                parse_research_profile(document)

    def test_rejects_blank_or_comment_only_legacy_profiles(self) -> None:
        for document in ("", "  \n", "<!-- one\n- hidden query\ntwo -->\n"):
            with self.subTest(document=document), self.assertRaisesRegex(
                ValueError, "blank or contains only HTML comments"
            ):
                parse_research_profile(document)


class ProfileSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        (self.root / "config").mkdir()

    def write_settings(
        self,
        *,
        topics: str = "[]",
        keywords: str = "[]",
        epmc_enabled: str | None = None,
        epmc_sources: str | None = None,
        provider: str = "codex",
    ) -> None:
        epmc = ""
        if epmc_enabled is not None or epmc_sources is not None:
            epmc = "  europe_pmc:\n"
            if epmc_enabled is not None:
                epmc += f"    enabled: {epmc_enabled}\n"
            if epmc_sources is not None:
                epmc += f"    sources: {epmc_sources}\n"
        (self.root / "config" / "settings.yaml").write_text(
            "obsidian:\n"
            "  vault_path: /tmp/freshlit-tests\n"
            "ingestion:\n"
            f"  openalex_topics: {topics}\n"
            f"  keywords: {keywords}\n"
            f"{epmc}"
            "llm:\n"
            f"  provider: {provider}\n",
            encoding="utf-8",
        )

    def write_tiers(self, tiers: dict[str, float]) -> None:
        (self.root / "config" / "journal_tiers.json").write_text(
            json.dumps(tiers), encoding="utf-8"
        )

    def test_modern_profile_drives_fields_when_legacy_epmc_is_absent(self) -> None:
        self.write_settings()
        self.write_tiers({})
        document = modern_profile()
        (self.root / "config" / "research_profile.md").write_text(
            document, encoding="utf-8"
        )

        settings = load_settings(self.root)

        self.assertEqual(settings.research_profile_text, parse_research_profile(document).body)
        self.assertEqual(settings.ingestion.openalex_topics, ["T123"])
        self.assertEqual(settings.ingestion.keywords, ["ecological networks"])
        self.assertFalse(settings.ingestion.europe_pmc.enabled)
        self.assertEqual(settings.ingestion.europe_pmc.sources, ["PPR"])
        self.assertEqual(settings.journal_tiers, {"12345678": 1.2})

    def test_matching_legacy_values_can_coexist(self) -> None:
        self.write_settings(
            topics="[https://openalex.org/T123]",
            keywords="[ecological networks]",
            epmc_enabled="false",
            epmc_sources="[PPR]",
        )
        self.write_tiers({"1234-5678": 1.2})
        (self.root / "config" / "research_profile.md").write_text(
            modern_profile(), encoding="utf-8"
        )

        settings = load_settings(self.root)
        self.assertEqual(settings.ingestion.openalex_topics, ["T123"])

    def test_explicit_differing_legacy_epmc_fields_conflict_independently(self) -> None:
        cases = (
            ({"epmc_enabled": "true"}, "Europe PMC enabled"),
            ({"epmc_sources": "[MED]"}, "Europe PMC sources"),
        )
        for settings_overrides, message in cases:
            with self.subTest(message=message):
                self.write_settings(**settings_overrides)
                self.write_tiers({})
                (self.root / "config" / "research_profile.md").write_text(
                    modern_profile(weights=""), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_settings(self.root)

    def test_nonempty_differing_legacy_values_conflict(self) -> None:
        cases = (
            ({"topics": "[T999]"}, {}, "OpenAlex topics"),
            ({"keywords": "[different]"}, {}, "keywords"),
            ({}, {"2041-1723": 2.0}, "venue weights"),
        )
        for settings_overrides, tiers, message in cases:
            with self.subTest(message=message):
                self.write_settings(**settings_overrides)
                self.write_tiers(tiers)
                (self.root / "config" / "research_profile.md").write_text(
                    modern_profile(), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_settings(self.root)


class ProfileMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        config = self.root / "config"
        config.mkdir()
        (config / "settings.yaml").write_text(
            "obsidian:\n"
            "  vault_path: /tmp/freshlit-tests\n"
            "ingestion:\n"
            "  openalex_topics: [https://openalex.org/T123]\n"
            "  keywords: []\n"
            "  europe_pmc:\n"
            "    enabled: true\n"
            "    sources: [PPR]\n"
            "llm:\n"
            "  provider: gateway\n",
            encoding="utf-8",
        )
        (config / "journal_tiers.json").write_text(
            json.dumps({"1234-5678": 1.2}), encoding="utf-8"
        )
        self.body = (
            "# Legacy profile\r\n\r\n"
            "## Keywords\r\n\r\n"
            "- ecological networks\r\n\r\n"
            "## Research Description\r\n\r\n"
            "Quantitative ecology.\r\n"
        )
        with (config / "research_profile.md").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            stream.write(self.body)

    def test_preview_and_apply_preserve_body_and_effective_settings_without_credentials(self) -> None:
        with patch.dict(os.environ, {}, clear=True), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            before = load_settings(self.root, require_llm_credentials=False)
            preview = migrate_profile(self.root)

        self.assertEqual(parse_research_profile(preview).body, self.body)
        self.assertEqual(
            (self.root / "config" / "research_profile.md").read_bytes(),
            self.body.encode("utf-8"),
        )

        with patch.dict(os.environ, {}, clear=True), warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            applied = migrate_profile(self.root, apply=True)
            after = load_settings(self.root, require_llm_credentials=False)

        self.assertEqual(applied, preview)
        self.assertEqual(
            (self.root / "config" / "research_profile.md.bak").read_bytes(),
            self.body.encode("utf-8"),
        )
        self.assertEqual(before.ingestion.keywords, after.ingestion.keywords)
        self.assertEqual(before.ingestion.openalex_topics, after.ingestion.openalex_topics)
        self.assertEqual(before.ingestion.europe_pmc, after.ingestion.europe_pmc)
        self.assertEqual(before.journal_tiers, after.journal_tiers)
        self.assertEqual(after.research_profile_text, self.body)

    def test_ambiguous_keyword_override_blocks_migration(self) -> None:
        settings_path = self.root / "config" / "settings.yaml"
        settings_path.write_text(
            settings_path.read_text(encoding="utf-8").replace(
                "keywords: []", "keywords: [override query]"
            ),
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            ValueError, "ingestion.keywords overrides"
        ):
            migrate_profile(self.root)

    def test_multiline_and_inline_html_comment_keywords_block_migration(self) -> None:
        cases = (
            self.body.replace(
                "- ecological networks\r\n",
                "<!--\r\n- hidden query\r\n-->\r\n- ecological networks\r\n",
            ),
            self.body.replace(
                "- ecological networks\r\n",
                "- ecological networks <!-- legacy annotation -->\r\n",
            ),
        )
        profile_path = self.root / "config" / "research_profile.md"
        for index, body in enumerate(cases):
            with self.subTest(index=index):
                with profile_path.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(body)
                with self.assertRaisesRegex(ValueError, "HTML comment ambiguity"):
                    migrate_profile(self.root)
                self.assertFalse(
                    profile_path.with_name("research_profile.md.bak").exists()
                )

    def test_existing_backup_is_never_replaced(self) -> None:
        backup = self.root / "config" / "research_profile.md.bak"
        backup.write_text("keep me", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            FileExistsError, "backup already exists"
        ):
            migrate_profile(self.root, apply=True)
        self.assertEqual(backup.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(
            (self.root / "config" / "research_profile.md").read_bytes(),
            self.body.encode("utf-8"),
        )

    def test_backup_is_private_even_with_permissive_umask(self) -> None:
        profile_path = self.root / "config" / "research_profile.md"
        profile_path.chmod(0o600)
        previous_umask = os.umask(0)
        try:
            migrate_profile(self.root, apply=True)
        finally:
            os.umask(previous_umask)

        backup_mode = stat.S_IMODE(
            profile_path.with_name("research_profile.md.bak").stat().st_mode
        )
        self.assertEqual(backup_mode, 0o600)

    def test_atomic_replace_failure_preserves_original_and_backup_and_cleans_temp(self) -> None:
        profile_path = self.root / "config" / "research_profile.md"
        with patch("freshlit.utils.profile.os.replace", side_effect=OSError("blocked")):
            with self.assertRaisesRegex(OSError, "blocked"):
                migrate_profile(self.root, apply=True)

        self.assertEqual(profile_path.read_bytes(), self.body.encode("utf-8"))
        self.assertEqual(
            profile_path.with_name("research_profile.md.bak").read_bytes(),
            self.body.encode("utf-8"),
        )
        self.assertEqual(
            list(profile_path.parent.glob(".research_profile.md.*.tmp")), []
        )

    def test_symlink_and_nonregular_profiles_are_rejected_before_backup(self) -> None:
        profile_path = self.root / "config" / "research_profile.md"
        target = self.root / "target-profile.md"
        profile_path.replace(target)
        profile_path.symlink_to(target)

        with self.assertRaisesRegex(ValueError, "non-regular research profile"):
            migrate_profile(self.root, apply=True)
        self.assertEqual(target.read_bytes(), self.body.encode("utf-8"))
        self.assertFalse(profile_path.with_name("research_profile.md.bak").exists())

    def test_existing_modern_profile_is_a_noop(self) -> None:
        document = modern_profile(
            topics="  - id: T123", epmc_enabled="true"
        )
        profile_path = self.root / "config" / "research_profile.md"
        profile_path.write_text(document, encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            result = migrate_profile(self.root, apply=True)

        self.assertEqual(result, document)
        self.assertEqual(profile_path.read_text(encoding="utf-8"), document)
        self.assertFalse(profile_path.with_name("research_profile.md.bak").exists())


if __name__ == "__main__":
    unittest.main()
