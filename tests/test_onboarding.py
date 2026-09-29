from __future__ import annotations

import os
import stat
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

from freshlit.nodes import delivery
from freshlit.utils import onboarding


EMPTY_PROFILE = """---
version: 1
search:
  openalex_topics: []
  europe_pmc:
    enabled: false
    sources: [PPR]
venue_weights: []
---
## Keywords

## Research Description
"""

VALID_PROFILE = """---
version: 1
search:
  openalex_topics: []
  europe_pmc:
    enabled: false
    sources: [PPR]
venue_weights: []
---
## Keywords
- exoplanet atmospheric spectroscopy

## Research Description
Study atmospheric composition in transiting exoplanets.
"""


def _settings(vault: Path, digest_folder: str = "digests") -> SimpleNamespace:
    return SimpleNamespace(
        obsidian=SimpleNamespace(
            vault_path=vault,
            digest_folder=digest_folder,
        )
    )


class CheckoutFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.base = Path(self.temporary_directory.name)
        self.root = self.base / "project"
        (self.root / "config").mkdir(parents=True)
        self.output = self.base / "vault"
        self.codex_home = self.base / "freshlit-codex"
        (self.root / "config" / "settings.yaml").write_text(
            "obsidian:\n"
            f"  vault_path: {self.output}\n"
            "  digest_folder: digests/new\n"
            "llm:\n"
            "  provider: gateway\n"
            f"  codex_home: {self.codex_home}\n",
            encoding="utf-8",
        )
        (self.root / "config" / "research_profile.example.md").write_text(
            EMPTY_PROFILE,
            encoding="utf-8",
        )
        (self.root / ".env.example").write_text(
            "OPENALEX_API_KEY=\nFRESHLIT_VAULT_PATH=\n",
            encoding="utf-8",
        )


class InitializeTests(CheckoutFixture):
    def test_initialize_is_exclusive_private_and_requires_profile_edit(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            messages = onboarding.initialize(self.root)

        profile = self.root / "config" / "research_profile.md"
        environment = self.root / ".env"
        self.assertEqual(profile.read_text(encoding="utf-8"), EMPTY_PROFILE)
        self.assertEqual(
            stat.S_IMODE(profile.stat().st_mode) & 0o077,
            0,
        )
        self.assertEqual(stat.S_IMODE(environment.stat().st_mode) & 0o077, 0)
        self.assertTrue(
            any("Edit config/research_profile.md" in item for item in messages)
        )

        profile.write_text("private profile contents\n", encoding="utf-8")
        environment.write_text("PRIVATE_TOKEN=do-not-disclose\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            second_messages = onboarding.initialize(self.root)

        self.assertEqual(
            profile.read_text(encoding="utf-8"),
            "private profile contents\n",
        )
        self.assertEqual(
            environment.read_text(encoding="utf-8"),
            "PRIVATE_TOKEN=do-not-disclose\n",
        )
        self.assertNotIn("do-not-disclose", "\n".join(second_messages))

    def test_explicit_output_setup_does_not_require_completed_profile(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            messages = onboarding.initialize(self.root, create_output_dir=True)

        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "digests").exists())
        self.assertTrue(any("output root is ready" in item for item in messages))

        diagnostics = onboarding.doctor(self.root)
        self.assertTrue(any(item.startswith("ERROR:") for item in diagnostics))
        self.assertTrue(any("research profile" in item.lower() for item in diagnostics))

    def test_prepare_codex_home_creates_only_fresh_leaf_with_mode_0700(self) -> None:
        (self.root / ".env.example").write_text(
            f"FRESHLIT_CODEX_HOME={self.codex_home}\n",
            encoding="utf-8",
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                Path,
                "chmod",
                side_effect=AssertionError("initialize must not call chmod"),
            ),
        ):
            messages = onboarding.initialize(self.root, prepare_codex_home=True)

        self.assertTrue(self.codex_home.is_dir())
        self.assertEqual(stat.S_IMODE(self.codex_home.stat().st_mode), 0o700)
        expected = (
            "CODEX_HOME="
            f"{onboarding.shlex.quote(str(self.codex_home.resolve()))} codex login"
        )
        self.assertTrue(any(expected in item for item in messages))

        self.codex_home.chmod(0o755)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            RuntimeError, "permissions must be 0700"
        ):
            onboarding.initialize(self.root, prepare_codex_home=True)
        self.assertEqual(stat.S_IMODE(self.codex_home.stat().st_mode), 0o755)


class DoctorTests(CheckoutFixture):
    def test_doctor_is_read_only_and_does_not_load_models_or_use_network(self) -> None:
        (self.root / "config" / "research_profile.md").write_text(
            VALID_PROFILE,
            encoding="utf-8",
        )
        self.output.mkdir()
        before = {
            path.relative_to(self.root): (path.stat().st_mode, path.read_bytes())
            for path in self.root.rglob("*")
            if path.is_file()
        }

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("freshlit.utils.embeddings.get_model") as get_model,
            patch.object(onboarding.requests, "get") as request,
        ):
            messages = onboarding.doctor(self.root)

        after = {
            path.relative_to(self.root): (path.stat().st_mode, path.read_bytes())
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertFalse((self.output / "digests").exists())
        get_model.assert_not_called()
        request.assert_not_called()
        self.assertTrue(any(item.startswith("OK:") for item in messages))
        self.assertTrue(any("OPENCODE_GO_API_KEY" in item for item in messages))
        self.assertTrue(any("rebuilt" in item for item in messages))

    def test_malformed_profile_has_concrete_secret_free_guidance(self) -> None:
        profile_secret = "profile-secret-marker"
        environment_secret = "environment-secret-marker"
        (self.root / "config" / "research_profile.md").write_text(
            "---\n"
            "version: 1\n"
            "search: {}\n"
            "venue_weights: []\n"
            f"{profile_secret}: true\n"
            "---\n"
            "## Keywords\n\n"
            "## Research Description\n"
            "Observe atmospheric chemistry.\n",
            encoding="utf-8",
        )
        (self.root / ".env").write_text(
            f"OPENCODE_GO_API_KEY={environment_secret}\n",
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {"OPENCODE_GO_API_KEY": environment_secret},
            clear=True,
        ):
            messages = onboarding.doctor(self.root)

        rendered = "\n".join(messages)
        self.assertIn("required version, search, and venue_weights", rendered)
        self.assertNotIn(profile_secret, rendered)
        self.assertNotIn(environment_secret, rendered)

    def test_codex_doctor_uses_only_existing_temp_and_read_only_checks(self) -> None:
        from freshlit.utils import embeddings, llm

        (self.root / "config" / "research_profile.md").write_text(
            VALID_PROFILE,
            encoding="utf-8",
        )
        (self.root / "config" / "settings.yaml").write_text(
            "obsidian:\n"
            f"  vault_path: {self.output}\n"
            "  digest_folder: digests/new\n"
            "llm:\n"
            "  provider: codex\n"
            f"  codex_home: {self.codex_home}\n",
            encoding="utf-8",
        )
        self.output.mkdir()
        self.codex_home.mkdir(mode=0o700)
        self.codex_home.chmod(0o700)
        temp_parent = self.base / "existing-temp"
        temp_parent.mkdir()

        def reject_mutation(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("doctor attempted a filesystem mutation")

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {"TMPDIR": str(temp_parent)},
                    clear=True,
                )
            )
            for target, attribute in (
                (tempfile, "gettempdir"),
                (tempfile, "mkdtemp"),
                (Path, "mkdir"),
                (Path, "touch"),
                (Path, "write_text"),
                (Path, "write_bytes"),
                (Path, "unlink"),
                (Path, "chmod"),
                (onboarding.os, "open"),
                (onboarding.os, "mkdir"),
                (onboarding.os, "unlink"),
                (onboarding.os, "remove"),
                (onboarding.os, "replace"),
                (onboarding.os, "rename"),
                (llm, "_outside_project_temp_parent"),
                (llm, "build_client"),
                (llm, "start_codex_runtime"),
                (llm, "require_codex_preflight"),
                (llm, "Codex"),
                (embeddings, "get_model"),
                (onboarding.requests, "get"),
            ):
                stack.enter_context(
                    patch.object(target, attribute, side_effect=reject_mutation)
                )
            validate_home = stack.enter_context(
                patch.object(
                    llm,
                    "validate_codex_home",
                    return_value=self.codex_home.resolve(),
                )
            )
            build_config = stack.enter_context(
                patch.object(llm, "build_codex_config")
            )
            stack.enter_context(
                patch.object(
                    embeddings,
                    "_is_model_cached",
                    return_value=False,
                )
            )
            stack.enter_context(
                patch.object(
                    embeddings,
                    "profile_vector_is_current",
                    return_value=False,
                )
            )
            messages = onboarding.doctor(self.root)

        validate_home.assert_called_once()
        build_config.assert_called_once_with(
            self.codex_home.resolve(),
            temp_parent.resolve(),
        )
        self.assertTrue(any("safe launcher" in item for item in messages))
        self.assertTrue(any("NOT tested" in item for item in messages))


class TopicSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        (self.root / ".env").write_text(
            "OPENALEX_API_KEY=dotenv-secret\n",
            encoding="utf-8",
        )

    def test_search_uses_query_parameter_bearer_header_and_needs_no_profile(self) -> None:
        response = MagicMock()
        response.json.return_value = {
            "results": [
                {
                    "id": "https://openalex.org/T123",
                    "display_name": "Exoplanet atmospheres",
                },
                {"id": "not-a-topic", "display_name": "Invalid"},
                {"id": "T123", "display_name": "Duplicate"},
            ]
        }
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(onboarding.requests, "get", return_value=response) as get,
        ):
            topics = onboarding.search_topics(" exoplanet atmosphere ", self.root)

        self.assertEqual(
            topics,
            [{"id": "T123", "label": "Exoplanet atmospheres"}],
        )
        get.assert_called_once_with(
            onboarding.OPENALEX_TOPIC_AUTOCOMPLETE,
            params={"q": "exoplanet atmosphere"},
            headers={"Authorization": "Bearer dotenv-secret"},
            timeout=onboarding.OPENALEX_TOPIC_TIMEOUT_SECONDS,
        )

    def test_blank_process_key_suppresses_dotenv_key(self) -> None:
        response = MagicMock()
        response.json.return_value = {"results": []}
        with (
            patch.dict(os.environ, {"OPENALEX_API_KEY": "  "}, clear=True),
            patch.object(onboarding.requests, "get", return_value=response) as get,
        ):
            onboarding.search_topics("genomics", self.root)

        self.assertEqual(get.call_args.kwargs["headers"], {})

    def test_network_error_is_sanitized(self) -> None:
        leaked = "dotenv-secret"
        raw_error = requests.ConnectionError(
            f"request failed at https://example.test/?api_key={leaked}"
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(onboarding.requests, "get", side_effect=raw_error),
            self.assertRaises(RuntimeError) as raised,
        ):
            onboarding.search_topics("private query", self.root)

        rendered = str(raised.exception)
        self.assertNotIn(leaked, rendered)
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("private query", rendered)

    def test_http_error_is_sanitized(self) -> None:
        leaked = "dotenv-secret"
        response = MagicMock(status_code=403)
        response.raise_for_status.side_effect = requests.HTTPError(
            f"forbidden https://example.test/?api_key={leaked}&q=private",
            response=response,
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(onboarding.requests, "get", return_value=response),
            self.assertRaises(RuntimeError) as raised,
        ):
            onboarding.search_topics("private query", self.root)

        rendered = str(raised.exception)
        self.assertIn("HTTP status 403", rendered)
        self.assertNotIn(leaked, rendered)
        self.assertNotIn("example.test", rendered)
        self.assertNotIn("private query", rendered)


@unittest.skipUnless(os.name == "posix", "secure delivery requires POSIX")
class DestinationPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.vault = Path(self.temporary_directory.name) / "vault"
        self.vault.mkdir()

    def test_preflight_allows_missing_nested_folders_without_writing(self) -> None:
        settings = _settings(self.vault, "nested/digests")
        before = list(self.vault.iterdir())

        path = delivery.validate_destination(settings, "digest.md")

        self.assertEqual(
            path,
            self.vault.resolve() / "nested" / "digests" / "digest.md",
        )
        self.assertEqual(list(self.vault.iterdir()), before)

    def test_preflight_rejects_missing_vault_and_unsafe_filenames(self) -> None:
        with self.assertRaisesRegex(ValueError, "existing real directory"):
            delivery.validate_destination(
                _settings(self.vault / "missing"),
                "digest.md",
            )

        for filename in ("", ".", "..", "../outside.md", "/tmp/outside.md"):
            with self.subTest(filename=filename), self.assertRaisesRegex(
                ValueError, "single safe basename"
            ):
                delivery.validate_destination(_settings(self.vault), filename)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_preflight_rejects_symlink_ancestors_and_entries_without_mutation(self) -> None:
        external = self.vault.parent / "external"
        external.mkdir()
        digest_directory = self.vault / "digests"
        digest_directory.symlink_to(external, target_is_directory=True)

        with self.assertRaises(ValueError):
            delivery.validate_destination(_settings(self.vault), "digest.md")
        self.assertEqual(list(external.iterdir()), [])

        digest_directory.unlink()
        digest_directory.mkdir()
        target = external / "target.md"
        target.write_text("unchanged\n", encoding="utf-8")
        (digest_directory / "digest.md").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "digest path is not a regular file"):
            delivery.validate_destination(_settings(self.vault), "digest.md")
        self.assertEqual(target.read_text(encoding="utf-8"), "unchanged\n")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_preflight_rejects_symlink_vault(self) -> None:
        real_vault = self.vault.parent / "real-vault"
        real_vault.mkdir()
        linked_vault = self.vault.parent / "linked-vault"
        linked_vault.symlink_to(real_vault, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "existing real directory"):
            delivery.validate_destination(_settings(linked_vault), "digest.md")

        self.assertEqual(list(real_vault.iterdir()), [])

    def test_preflight_rejects_nonregular_target_and_lock(self) -> None:
        digest_directory = self.vault / "digests"
        digest_directory.mkdir()
        (digest_directory / "digest.md").mkdir()
        with self.assertRaisesRegex(ValueError, "digest path is not a regular file"):
            delivery.validate_destination(_settings(self.vault), "digest.md")

        (digest_directory / "digest.md").rmdir()
        (digest_directory / delivery._LOCK_FILENAME).mkdir()
        with self.assertRaisesRegex(ValueError, "lock is not a regular file"):
            delivery.validate_destination(_settings(self.vault), "digest.md")

    def test_preflight_checks_regular_digest_and_lock_access(self) -> None:
        digest_directory = self.vault / "digests"
        digest_directory.mkdir()
        digest = digest_directory / "digest.md"
        lock = digest_directory / delivery._LOCK_FILENAME
        digest.write_text("existing digest\n", encoding="utf-8")
        lock.write_text("", encoding="utf-8")

        def unreadable_digest(path: object, mode: int) -> bool:
            return not (
                Path(path).resolve() == digest.resolve() and mode == os.R_OK
            )

        with (
            patch.object(delivery.os, "access", side_effect=unreadable_digest),
            self.assertRaisesRegex(PermissionError, "digest must be readable"),
        ):
            delivery.validate_destination(_settings(self.vault), digest.name)

        def unwritable_lock(path: object, mode: int) -> bool:
            return not (
                Path(path).resolve() == lock.resolve()
                and mode == (os.R_OK | os.W_OK)
            )

        with (
            patch.object(delivery.os, "access", side_effect=unwritable_lock),
            self.assertRaisesRegex(
                PermissionError,
                "lock must be readable and writable",
            ),
        ):
            delivery.validate_destination(_settings(self.vault), digest.name)

        self.assertEqual(digest.read_text(encoding="utf-8"), "existing digest\n")
        self.assertEqual(lock.read_text(encoding="utf-8"), "")

    def test_preflight_requires_writable_traversable_existing_parent(self) -> None:
        with patch.object(delivery.os, "access", return_value=False):
            with self.assertRaisesRegex(
                PermissionError,
                "writable and traversable",
            ):
                delivery.validate_destination(
                    _settings(self.vault, "missing/digests"),
                    "digest.md",
                )

        self.assertEqual(list(self.vault.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
