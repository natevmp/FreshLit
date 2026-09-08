from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from freshlit.nodes import delivery


def _summary(**overrides: object) -> SimpleNamespace:
    values = {
        "paper_id": "paper-1",
        "title": "A useful paper",
        "authors_formatted": "A. Author et al.",
        "venue_and_year": "Journal (2026)",
        "doi_url": "https://doi.org/10.1/example",
        "doi": "10.1/example",
        "final_score": 9.2,
        "core_question": "What is measured?",
        "framework_and_method": "A cohort model.",
        "key_finding": "The effect was reproducible.",
        "code_data_link": "None stated",
        "relevance_rationale": "It is directly relevant.",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class DeliveryFrontmatterTests(unittest.TestCase):
    def test_frontmatter_records_model_and_provider(self) -> None:
        markdown = delivery.render_markdown(
            date(2026, 9, 7),
            [],
            [],
            papers_analyzed=4,
            llm_model="gpt-5.6-luna",
            llm_provider="codex",
        )

        self.assertIn('\nllm_model: "gpt-5.6-luna"\n', markdown)
        self.assertIn('\nllm_provider: "codex"\n', markdown)

    def test_existing_callers_keep_blank_provenance_defaults(self) -> None:
        markdown = delivery.render_markdown(date(2026, 9, 7), [], [], 0)
        positional = delivery.render_markdown(
            date(2026, 9, 7), [], [], 0, "legacy-model"
        )

        self.assertIn('\nllm_model: ""\n', markdown)
        self.assertIn('\nllm_provider: ""\n', markdown)
        self.assertIn('\nllm_model: "legacy-model"\n', positional)
        self.assertIn('\nllm_provider: ""\n', positional)

    def test_deliver_passes_configured_provenance(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(
                    vault_path=Path(temporaryDirectory), digest_folder=""
                ),
                llm=SimpleNamespace(model="configured-model", provider="gateway"),
            )

            with patch.object(
                delivery, "render_markdown", return_value="digest"
            ) as render:
                delivery.deliver(settings, [], [], 0, dry_run=True)

        self.assertEqual(render.call_args.kwargs["llm_model"], "configured-model")
        self.assertEqual(render.call_args.kwargs["llm_provider"], "gateway")

    def test_frontmatter_values_cannot_inject_new_keys_or_delimiters(self) -> None:
        markdown = delivery.render_markdown(
            date(2026, 9, 7),
            [],
            [],
            0,
            llm_model='model\n---\nowned: true "quoted"',
            llm_provider="provider\r\ntags:\n  - owned",
        )

        frontmatterLines = markdown.splitlines()
        frontmatter = "\n".join(frontmatterLines[1 : frontmatterLines.index("---", 1)])
        self.assertIn('llm_model: "model --- owned: true \\"quoted\\""', frontmatter)
        self.assertIn('llm_provider: "provider tags: - owned"', frontmatter)
        self.assertNotIn("\nowned:", frontmatter)
        self.assertNotIn("\ntags:\n  - owned", frontmatter)


class DeliverySafetyTests(unittest.TestCase):
    def test_paper_fields_render_as_single_line_literal_text(self) -> None:
        payload = '# Injected\n- item <img src=x> ![alt](https://evil.test) [go](x)'
        summary = _summary(
            title=payload,
            authors_formatted=payload,
            venue_and_year=payload,
            doi=payload,
            core_question=payload,
            framework_and_method=payload,
            key_finding=payload,
            code_data_link=payload,
            relevance_rationale=payload,
        )

        markdown = delivery.render_markdown(date(2026, 9, 7), [summary], [], 1)

        self.assertEqual(markdown.count("\n### "), 1)
        self.assertNotIn("\n# Injected", markdown)
        self.assertNotIn("\n- item", markdown)
        self.assertNotIn("<img", markdown)
        self.assertNotIn("![alt]", markdown)
        self.assertNotIn("[go](x)", markdown)
        self.assertIn(r"\# Injected \- item &lt;img src\=x&gt;", markdown)

    def test_pulse_items_are_sanitized_and_list_structure_is_generated(self) -> None:
        markdown = delivery.render_markdown(
            date(2026, 9, 7),
            [],
            ["- Normal trend", "# Owned\n- second <script>x</script> ![x](y)"],
            0,
        )

        pulseSection = markdown.split("## Executive Pulse\n", 1)[1].split("\n\n---", 1)[0]
        self.assertEqual(len(pulseSection.splitlines()), 2)
        self.assertIn("- Normal trend", pulseSection)
        self.assertNotIn("\n# Owned", pulseSection)
        self.assertNotIn("<script>", pulseSection)
        self.assertNotIn("![x](y)", pulseSection)

    def test_safe_bold_pulse_label_is_preserved(self) -> None:
        markdown = delivery.render_markdown(
            date(2026, 9, 7),
            [],
            ["- **Shared methods:** Models overlap <script>bad</script>."],
            0,
        )

        self.assertIn("- **Shared methods:** Models overlap", markdown)
        self.assertNotIn("<script>", markdown)

    def test_url_destinations_are_parsed_and_markdown_safe(self) -> None:
        summary = _summary(
            doi_url="https://example.test/a path/(draft)<final>",
            code_data_link="https://code.test/repo name)/README.md",
        )

        markdown = delivery.render_markdown(date(2026, 9, 7), [summary], [], 1)

        self.assertIn(
            "(https://example.test/a%20path/%28draft%29%3Cfinal%3E)", markdown
        )
        self.assertIn(
            "[Repository](https://code.test/repo%20name%29/README.md)", markdown
        )

    def test_invalid_or_credential_bearing_destinations_become_hash(self) -> None:
        self.assertEqual(delivery._safe_url("javascript:alert(1)"), "#")
        self.assertEqual(delivery._safe_url("ftp://example.test/file"), "#")
        self.assertEqual(delivery._safe_url("https://user:pass@example.test/file"), "#")
        self.assertEqual(delivery._safe_url("https://example.test/a\nfile"), "#")
        self.assertEqual(delivery._safe_url("https://" + "a" * 2050), "#")

        markdown = delivery.render_markdown(
            date(2026, 9, 7),
            [
                _summary(
                    doi_url="ftp://example.test/paper",
                    code_data_link="https://user@example.test/repository",
                )
            ],
            [],
            1,
        )
        self.assertIn("### [A useful paper](#)", markdown)
        self.assertIn("[Repository](#)", markdown)

    def test_malicious_non_url_code_data_fallback_is_literal(self) -> None:
        markdown = delivery.render_markdown(
            date(2026, 9, 7),
            [_summary(code_data_link="![steal](javascript:owned)\n## heading")],
            [],
            1,
        )

        self.assertNotIn("![steal]", markdown)
        self.assertNotIn("\n## heading", markdown)
        self.assertIn(r"\!\[steal\]\(javascript\:owned\) \#\# heading", markdown)


class DeliveryRerunTests(unittest.TestCase):
    def test_regular_file_rerun_preserves_content_and_records_provenance(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(
                    vault_path=Path(temporaryDirectory), digest_folder="digests"
                ),
                llm=SimpleNamespace(
                    model="new-model\n## forged", provider="gateway<script>bad</script>"
                ),
            )
            path = delivery._target_path(settings, date.today()).with_name("digest.md")
            path.parent.mkdir(parents=True)
            original = "original digest\n"
            path.write_text(original)

            result = delivery.deliver(
                settings, [_summary(title="New paper")], [], 1, filename=path.name
            )
            content = path.read_text()

        self.assertEqual(result, path)
        self.assertTrue(content.startswith(original))
        self.assertIn("*Re-run ", content)
        self.assertIn("provider: gateway&lt;script&gt;bad&lt;\\/script&gt;", content)
        self.assertIn("model: new\\-model \\#\\# forged", content)
        self.assertIn("### [New paper]", content)
        self.assertNotIn("\n## forged", content)
        self.assertNotIn("<script>", content)

    def test_filename_must_be_a_single_safe_basename(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(
                    vault_path=Path(temporaryDirectory), digest_folder="digests"
                ),
                llm=SimpleNamespace(model="model", provider="provider"),
            )

            for filename in ("", ".", "..", "../outside.md", "/tmp/outside.md"):
                with self.subTest(filename=filename):
                    with self.assertRaisesRegex(ValueError, "single safe basename"):
                        delivery.deliver(settings, [], [], 0, filename=filename)

    def test_missing_dir_fd_mkdir_fails_closed(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            supportedDirectoryFd = os.supports_dir_fd - {os.mkdir}

            with patch.object(
                delivery.os, "supports_dir_fd", supportedDirectoryFd
            ):
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    delivery.deliver(settings, [], [], 0, filename="digest.md")

            self.assertFalse((root / "digests").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_delivery_lock_symlink_is_rejected_without_following_it(self) -> None:
        with (
            TemporaryDirectory() as temporaryDirectory,
            TemporaryDirectory() as externalTemporaryDirectory,
        ):
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            digestDirectory = root / "digests"
            digestDirectory.mkdir()
            externalPath = Path(externalTemporaryDirectory) / "external.lock"
            externalPath.write_text("do not change\n")
            (digestDirectory / delivery._LOCK_FILENAME).symlink_to(externalPath)

            with self.assertRaisesRegex(ValueError, "lock is not a regular file"):
                delivery.deliver(settings, [], [], 0, filename="digest.md")

            self.assertEqual(externalPath.read_text(), "do not change\n")
            self.assertFalse((digestDirectory / "digest.md").exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_intermediate_swap_cannot_create_entries_outside_vault(self) -> None:
        with (
            TemporaryDirectory() as temporaryDirectory,
            TemporaryDirectory() as externalTemporaryDirectory,
        ):
            root = Path(temporaryDirectory)
            collectionDirectory = root / "collection"
            collectionDirectory.mkdir()
            parkedDirectory = root / "parked-collection"
            externalDirectory = Path(externalTemporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(
                    vault_path=root, digest_folder="collection/digests"
                ),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            openDigestDirectory = delivery._opened_digest_directory

            @contextmanager
            def swap_then_open(vaultPath: Path, path: Path):
                collectionDirectory.rename(parkedDirectory)
                collectionDirectory.symlink_to(
                    externalDirectory, target_is_directory=True
                )
                with openDigestDirectory(vaultPath, path) as openedDirectory:
                    yield openedDirectory

            with patch.object(
                delivery, "_opened_digest_directory", swap_then_open
            ):
                with self.assertRaisesRegex(
                    ValueError, "component is not a real directory"
                ):
                    delivery.deliver(settings, [], [], 0, filename="digest.md")

            self.assertFalse((externalDirectory / "digests").exists())
            self.assertEqual(list(externalDirectory.iterdir()), [])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_existing_symlink_is_rejected_without_changing_target(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            path = root / "digests" / "digest.md"
            path.parent.mkdir(parents=True)
            target = root / "symlink-target.md"
            target.write_text("do not change\n")
            path.symlink_to(target)

            with self.assertRaisesRegex(ValueError, "not a regular file"):
                delivery.deliver(settings, [_summary()], [], 1, filename=path.name)

            self.assertTrue(path.is_symlink())
            self.assertEqual(target.read_text(), "do not change\n")

    def test_existing_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            path = root / "digests" / "digest.md"
            path.mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "not a regular file"):
                delivery.deliver(settings, [], [], 0, filename=path.name)

            self.assertTrue(path.is_dir())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs are not supported")
    def test_existing_fifo_is_rejected(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            path = root / "digests" / "digest.md"
            path.parent.mkdir(parents=True)
            os.mkfifo(path)

            with self.assertRaisesRegex(ValueError, "not a regular file"):
                delivery.deliver(settings, [], [], 0, filename=path.name)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_symlink_swap_after_read_replaces_entry_not_target(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            path = root / "digests" / "digest.md"
            path.parent.mkdir(parents=True)
            path.write_text("original\n")
            target = root / "race-target.md"
            target.write_text("do not change\n")
            atomicWrite = delivery._atomic_write

            def swap_then_write(
                directoryFd: int, filename: str, content: str
            ) -> None:
                path.unlink()
                path.symlink_to(target)
                atomicWrite(directoryFd, filename, content)

            with patch.object(delivery, "_atomic_write", side_effect=swap_then_write):
                delivery.deliver(settings, [_summary()], [], 1, filename=path.name)

            self.assertFalse(path.is_symlink())
            self.assertTrue(path.read_text().startswith("original\n"))
            self.assertEqual(target.read_text(), "do not change\n")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_parent_swap_does_not_redirect_write_outside_digest_directory(self) -> None:
        with (
            TemporaryDirectory() as temporaryDirectory,
            TemporaryDirectory() as externalTemporaryDirectory,
        ):
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            path = root / "digests" / "digest.md"
            path.parent.mkdir(parents=True)
            path.write_text("original\n")
            externalDirectory = Path(externalTemporaryDirectory)
            externalPath = externalDirectory / path.name
            externalPath.write_text("do not change\n")
            parkedDirectory = root / "parked-digests"
            atomicWrite = delivery._atomic_write

            def swap_parent_then_write(
                directoryFd: int, filename: str, content: str
            ) -> None:
                path.parent.rename(parkedDirectory)
                path.parent.symlink_to(externalDirectory, target_is_directory=True)
                atomicWrite(directoryFd, filename, content)

            with patch.object(
                delivery, "_atomic_write", side_effect=swap_parent_then_write
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "digest directory changed during delivery"
                ):
                    delivery.deliver(settings, [_summary()], [], 1, filename=path.name)

            self.assertEqual(externalPath.read_text(), "do not change\n")
            self.assertFalse((externalDirectory / ".freshlit-delivery.lock").exists())

    def test_concurrent_reruns_preserve_both_paper_blocks(self) -> None:
        with TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            settings = SimpleNamespace(
                obsidian=SimpleNamespace(vault_path=root, digest_folder="digests"),
                llm=SimpleNamespace(model="model", provider="provider"),
            )
            path = root / "digests" / "digest.md"
            path.parent.mkdir(parents=True)
            path.write_text("original\n")
            firstReadStarted = threading.Event()
            releaseFirstRead = threading.Event()
            secondLockAttempted = threading.Event()
            attemptsGuard = threading.Lock()
            lockAttempts = 0
            readCalls = 0
            readExisting = delivery._read_existing_digest
            inProcessLock = delivery._in_process_lock

            def coordinated_read(directoryFd: int, filename: str) -> str | None:
                nonlocal readCalls
                previousContent = readExisting(directoryFd, filename)
                with attemptsGuard:
                    readCalls += 1
                    isFirstRead = readCalls == 1
                if isFirstRead:
                    firstReadStarted.set()
                    if not releaseFirstRead.wait(timeout=5):
                        raise TimeoutError("test did not release the first digest read")
                return previousContent

            @contextmanager
            def coordinated_lock(key: tuple[int, int, str]):
                nonlocal lockAttempts
                with attemptsGuard:
                    lockAttempts += 1
                    if lockAttempts == 2:
                        secondLockAttempted.set()
                with inProcessLock(key):
                    yield

            def rerun(title: str) -> None:
                delivery.deliver(
                    settings, [_summary(title=title)], [], 1, filename=path.name
                )

            with (
                patch.object(
                    delivery, "_read_existing_digest", side_effect=coordinated_read
                ),
                patch.object(delivery, "_in_process_lock", coordinated_lock),
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    firstFuture = executor.submit(rerun, "Concurrent paper one")
                    self.assertTrue(firstReadStarted.wait(timeout=5))
                    secondFuture = executor.submit(rerun, "Concurrent paper two")
                    try:
                        self.assertTrue(secondLockAttempted.wait(timeout=5))
                    finally:
                        releaseFirstRead.set()
                    firstFuture.result(timeout=5)
                    secondFuture.result(timeout=5)

            content = path.read_text()
            self.assertEqual(content.count("### [Concurrent paper one]"), 1)
            self.assertEqual(content.count("### [Concurrent paper two]"), 1)


if __name__ == "__main__":
    unittest.main()
