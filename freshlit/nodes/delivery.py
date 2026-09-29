"""Node 4: Obsidian delivery — render the weekly digest and write to the vault."""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import secrets
import stat
import threading
import unicodedata
from contextlib import ExitStack, contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit, urlunsplit

try:
    import fcntl
except ImportError:  # pragma: no cover - delivery explicitly fails closed below.
    fcntl = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..utils.config import Settings
    from .synthesis import ProcessedPaperSummary

log = logging.getLogger(__name__)

TOP_RANKED_COUNT = 3
MAX_URL_LENGTH = 2048
_LOCK_FILENAME = ".freshlit-delivery.lock"
_MARKDOWN_PUNCTUATION = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
_PULSE_MARKER = re.compile(r"^(?:[-+*]|\d+[.)])\s+")
_PULSE_BOLD_LABEL = re.compile(r"^\*\*([^*]{1,100}?):\*\*\s*(.*)$")
_processLocksGuard = threading.Lock()
_processLocks: dict[tuple[int, int, str], tuple[threading.Lock, int]] = {}


def _normalize_text(text: str) -> str:
    """Return untrusted text as a single printable line."""
    normalized = unicodedata.normalize("NFKC", text)
    character_cid: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category.startswith("C"):
            if character.isspace():
                character_cid.append(" ")
            continue
        character_cid.append(" " if character.isspace() else character)
    return " ".join("".join(character_cid).split())


def _safe_text(text: str) -> str:
    """Render untrusted text literally rather than as Markdown."""
    escaped = _MARKDOWN_PUNCTUATION.sub(r"\\\1", _normalize_text(text))
    return escaped.replace(r"\<", "&lt;").replace(r"\>", "&gt;")


def _yaml_string(text: str) -> str:
    """Serialize a normalized string as a YAML-compatible JSON scalar."""
    return json.dumps(_normalize_text(text), ensure_ascii=False)


def _render_pulse_item(text: str) -> str:
    """Render one generated pulse item while preserving a safe bold label."""
    normalized = _PULSE_MARKER.sub("", _normalize_text(text), count=1)
    match = _PULSE_BOLD_LABEL.fullmatch(normalized)
    if match is None:
        return f"- {_safe_text(normalized)}"
    label, body = match.groups()
    renderedBody = f" {_safe_text(body)}" if body else ""
    return f"- **{_safe_text(label)}:**{renderedBody}"


def _safe_url(url: str) -> str:
    """Return a Markdown-safe web URL, or ``#`` when it is not acceptable."""
    if not url or len(url) > MAX_URL_LENGTH:
        return "#"
    if any(unicodedata.category(character).startswith("C") for character in url):
        return "#"

    normalized = unicodedata.normalize("NFKC", url)
    if len(normalized) > MAX_URL_LENGTH or normalized != normalized.strip():
        return "#"
    try:
        parsed = urlsplit(normalized)
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError):
        return "#"
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or any(character.isspace() for character in hostname)
        or any(character in "/\\?#@<>[]" for character in hostname)
    ):
        return "#"

    try:
        asciiHost = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return "#"
    if any(not (character.isalnum() or character in ".:-") for character in asciiHost):
        return "#"
    netloc = f"[{asciiHost}]" if ":" in asciiHost else asciiHost
    if port is not None:
        netloc = f"{netloc}:{port}"
    rebuilt = urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment)
    )
    # Parentheses, spaces, angle brackets, and backslashes are deliberately not safe.
    return quote(rebuilt, safe=":/?#@!$&'*+,;=-._~%[]")


def _paper_block(s: ProcessedPaperSummary) -> str:
    code = s.code_data_link
    code_md = (
        f"[Repository]({_safe_url(code)})"
        if re.match(r"^\s*https?://", code, flags=re.IGNORECASE)
        else _safe_text(code if code.strip() else "None stated")
    )
    return (
        f"### [{_safe_text(s.title)}]({_safe_url(s.doi_url)})\n"
        f"- **Score:** `{s.final_score:.1f}/10` | **Venue:** {_safe_text(s.venue_and_year)}"
        f" | **Authors:** {_safe_text(s.authors_formatted)}\n"
        f"- **DOI:** {_safe_text(s.doi or 'n/a')}\n"
        f"- **Core Question:** {_safe_text(s.core_question)}\n"
        f"- **Framework & Method:** {_safe_text(s.framework_and_method)}\n"
        f"- **Key Finding:** {_safe_text(s.key_finding)}\n"
        f"- **Code & Data:** {code_md}\n"
        f"- **Why It Matters:** {_safe_text(s.relevance_rationale)}\n"
    )


def render_markdown(
    today: date,
    summaries: list[ProcessedPaperSummary],
    pulse: list[str],
    papers_analyzed: int,
    llm_model: str = "",
    llm_provider: str = "",
) -> str:
    iso_year, iso_week, _ = today.isocalendar()
    top = summaries[:TOP_RANKED_COUNT]
    secondary = summaries[TOP_RANKED_COUNT:]
    top_score = summaries[0].final_score if summaries else 0.0

    lines = [
        "---",
        f"date: {today.isoformat()}",
        "type: literature-digest",
        "tags:",
        "  - literature-digest",
        "  - automated",
        f"papers_analyzed: {papers_analyzed}",
        f"papers_selected: {len(summaries)}",
        f"top_score: {top_score:.1f}",
        f"llm_model: {_yaml_string(llm_model)}",
        f"llm_provider: {_yaml_string(llm_provider)}",
        "---",
        "",
        f"# 🔬 Literature Digest: Week {iso_week:02d}, {iso_year}",
        "",
        "## Executive Pulse",
    ]
    if pulse:
        for item in pulse:
            lines.append(_render_pulse_item(item))
    else:
        lines.append("- No field pulse generated.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🌟 Top Ranked Papers")
    lines.append("")
    for s in top:
        lines.append(_paper_block(s))
    if secondary:
        lines.append("---")
        lines.append("")
        lines.append("## 📚 Secondary Selections")
        lines.append("")
        for s in secondary:
            lines.append(_paper_block(s))
    return "\n".join(lines).rstrip() + "\n"


def _target_path(settings: Settings, today: date) -> Path:
    vault = settings.obsidian.vault_path.expanduser().resolve()
    target_dir = (vault / settings.obsidian.digest_folder).resolve()
    if not target_dir.is_relative_to(vault):
        raise ValueError(
            f"digest_folder escapes vault_path: {settings.obsidian.digest_folder}"
        )
    iso_year, iso_week, _ = today.isocalendar()
    return target_dir / f"{iso_year}-W{iso_week:02d}.md"


def _require_secure_primitives() -> None:
    """Fail closed unless directory-anchored POSIX operations are available."""
    supportsDirectoryFd = getattr(os, "supports_dir_fd", set())
    supportsNoFollow = getattr(os, "supports_follow_symlinks", set())
    requiredDirectoryFd = tuple(
        getattr(os, name, None)
        for name in ("mkdir", "open", "stat", "unlink", "rename")
    )
    requiredFunctions = tuple(
        getattr(os, name, None)
        for name in ("close", "fdopen", "fstat", "lstat", "replace")
    )
    if (
        os.name != "posix"
        or fcntl is None
        or not callable(getattr(fcntl, "flock", None))
        or not getattr(os, "O_DIRECTORY", 0)
        or not getattr(os, "O_NOFOLLOW", 0)
        or any(
            not callable(operation) or operation not in supportsDirectoryFd
            for operation in requiredDirectoryFd
        )
        or any(not callable(operation) for operation in requiredFunctions)
        or getattr(os, "stat", None) not in supportsNoFollow
        or not callable(getattr(os.path, "samestat", None))
    ):
        raise RuntimeError("secure digest delivery is unsupported on this platform")


def _validate_filename(filename: str) -> None:
    """Require a single, non-reserved POSIX basename."""
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", "..", _LOCK_FILENAME}
        or "\0" in filename
        or "/" in filename
        or (os.altsep is not None and os.altsep in filename)
        or Path(filename).name != filename
    ):
        raise ValueError("filename must be a single safe basename")


def validate_destination(settings: Settings, filename: str | None = None) -> Path:
    """Read-only preflight for the configured digest destination.

    This provides useful failures before expensive pipeline work, but is not a
    TOCTOU-safe authorization to write. :func:`deliver` remains the final
    authority and repeats its descriptor-anchored checks while writing.
    """

    _require_secure_primitives()
    if filename is not None:
        _validate_filename(filename)
    configuredVault = settings.obsidian.vault_path.expanduser()
    try:
        configuredStat = os.lstat(configuredVault)
    except OSError as error:
        raise ValueError("vault path must be an existing real directory") from error
    if not stat.S_ISDIR(configuredStat.st_mode):
        raise ValueError("vault path must be an existing real directory")

    try:
        vaultPath = configuredVault.resolve(strict=True)
    except OSError as error:
        raise ValueError("vault path must be an existing real directory") from error

    path = _target_path(settings, datetime.now(timezone.utc).date())
    if filename is not None:
        path = path.with_name(filename)
    _validate_filename(path.name)

    # Inspect the configured route without following any digest-folder symlink.
    # abspath normalizes ``.`` and ``..`` but deliberately does not resolve links.
    configuredDirectory = Path(
        os.path.abspath(vaultPath / settings.obsidian.digest_folder)
    )
    try:
        directoryParts = configuredDirectory.relative_to(vaultPath).parts
        path.relative_to(vaultPath)
    except ValueError as error:
        raise ValueError("digest destination escapes the configured vault") from error

    nearestExistingParent = vaultPath
    current = vaultPath
    missingComponent = False
    for component in directoryParts:
        current = current / component
        if missingComponent:
            continue
        try:
            currentStat = os.lstat(current)
        except FileNotFoundError:
            missingComponent = True
            continue
        except OSError as error:
            raise ValueError(
                "digest directory ancestor cannot be inspected"
            ) from error
        if not stat.S_ISDIR(currentStat.st_mode):
            raise ValueError(
                "digest directory ancestor is not a real directory"
            )
        nearestExistingParent = current

    if not os.access(nearestExistingParent, os.W_OK | os.X_OK):
        raise PermissionError(
            "nearest existing digest parent must be writable and traversable"
        )

    if not missingComponent:
        for entry, message, accessMode, accessMessage in (
            (
                configuredDirectory / path.name,
                "digest path is not a regular file",
                os.R_OK,
                "existing digest must be readable for safe rerun delivery",
            ),
            (
                configuredDirectory / _LOCK_FILENAME,
                "delivery lock is not a regular file",
                os.R_OK | os.W_OK,
                "existing delivery lock must be readable and writable",
            ),
        ):
            try:
                entryStat = os.lstat(entry)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise ValueError(
                    f"delivery entry cannot be inspected: {entry.name}"
                ) from error
            if not stat.S_ISREG(entryStat.st_mode):
                raise ValueError(message)
            if not os.access(entry, accessMode):
                raise PermissionError(accessMessage)

    return path


def _directory_changed_error() -> RuntimeError:
    return RuntimeError("digest directory changed during delivery")


def _check_directory_identity(path: Path, expectedStat: os.stat_result) -> None:
    """Detect path drift without using the path for delivery operations."""
    try:
        currentStat = os.lstat(path)
    except OSError as error:
        raise _directory_changed_error() from error
    if not stat.S_ISDIR(currentStat.st_mode) or not os.path.samestat(
        expectedStat, currentStat
    ):
        raise _directory_changed_error()


@contextmanager
def _opened_digest_directory(vaultPath: Path, path: Path):
    """Open the vault and securely walk to the digest directory beneath it."""
    _require_secure_primitives()
    try:
        directoryParts = path.relative_to(vaultPath).parts
    except ValueError as error:
        raise ValueError("digest directory escapes the configured vault") from error

    try:
        expectedStat = os.lstat(vaultPath)
    except OSError as error:
        raise ValueError("vault path is not a real directory") from error
    if not stat.S_ISDIR(expectedStat.st_mode):
        raise ValueError("vault path is not a real directory")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        directoryFd = os.open(vaultPath, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("vault path is not a real directory") from error
        raise

    with ExitStack() as descriptorStack:
        descriptorStack.callback(os.close, directoryFd)
        openedStat = os.fstat(directoryFd)
        if not stat.S_ISDIR(openedStat.st_mode) or not os.path.samestat(
            expectedStat, openedStat
        ):
            raise _directory_changed_error()

        for component in directoryParts:
            try:
                os.mkdir(component, dir_fd=directoryFd)
            except FileExistsError:
                pass
            try:
                expectedStat = os.stat(
                    component, dir_fd=directoryFd, follow_symlinks=False
                )
            except OSError as error:
                raise ValueError(
                    "digest directory component is not a real directory"
                ) from error
            if not stat.S_ISDIR(expectedStat.st_mode):
                raise ValueError(
                    "digest directory component is not a real directory"
                )

            try:
                childFd = os.open(component, flags, dir_fd=directoryFd)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ValueError(
                        "digest directory component is not a real directory"
                    ) from error
                raise
            descriptorStack.callback(os.close, childFd)
            openedStat = os.fstat(childFd)
            if not stat.S_ISDIR(openedStat.st_mode) or not os.path.samestat(
                expectedStat, openedStat
            ):
                raise _directory_changed_error()
            directoryFd = childFd

        yield directoryFd, openedStat


@contextmanager
def _in_process_lock(key: tuple[int, int, str]):
    """Serialize deliveries targeting the same directory entry in this process."""
    with _processLocksGuard:
        entry = _processLocks.get(key)
        if entry is None:
            localLock = threading.Lock()
            users = 1
        else:
            localLock, users = entry
            users += 1
        _processLocks[key] = (localLock, users)

    acquired = False
    try:
        localLock.acquire()
        acquired = True
        yield
    finally:
        try:
            if acquired:
                localLock.release()
        finally:
            with _processLocksGuard:
                _, users = _processLocks[key]
                if users == 1:
                    del _processLocks[key]
                else:
                    _processLocks[key] = (localLock, users - 1)


@contextmanager
def _advisory_lock(directoryFd: int):
    """Hold a regular, content-free lock entry in the anchored directory."""
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        lockFd = os.open(_LOCK_FILENAME, flags, 0o600, dir_fd=directoryFd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EISDIR, errno.ENXIO, errno.ENOTDIR}:
            raise ValueError("delivery lock is not a regular file") from error
        raise

    locked = False
    try:
        openedStat = os.fstat(lockFd)
        if not stat.S_ISREG(openedStat.st_mode):
            raise ValueError("delivery lock is not a regular file")
        try:
            currentStat = os.stat(
                _LOCK_FILENAME, dir_fd=directoryFd, follow_symlinks=False
            )
        except OSError as error:
            raise ValueError("delivery lock changed while opening") from error
        if not stat.S_ISREG(currentStat.st_mode) or not os.path.samestat(
            openedStat, currentStat
        ):
            raise ValueError("delivery lock changed while opening")

        fcntl.flock(lockFd, fcntl.LOCK_EX)
        locked = True
        try:
            currentStat = os.stat(
                _LOCK_FILENAME, dir_fd=directoryFd, follow_symlinks=False
            )
        except OSError as error:
            raise ValueError("delivery lock changed while locking") from error
        if not stat.S_ISREG(currentStat.st_mode) or not os.path.samestat(
            openedStat, currentStat
        ):
            raise ValueError("delivery lock changed while locking")
        yield
    finally:
        try:
            if locked:
                fcntl.flock(lockFd, fcntl.LOCK_UN)
        finally:
            os.close(lockFd)


def _atomic_write(directoryFd: int, filename: str, content: str) -> None:
    """Replace a digest using only entries relative to its anchored directory."""
    temporaryName = ""
    temporaryFd = -1
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        for _attempt in range(128):
            candidate = f".freshlit-{secrets.token_hex(16)}.tmp"
            try:
                temporaryFd = os.open(
                    candidate, flags, 0o600, dir_fd=directoryFd
                )
            except FileExistsError:
                continue
            temporaryName = candidate
            break
        else:
            raise FileExistsError("could not create a unique digest temporary file")

        temporaryStat = os.fstat(temporaryFd)
        if not stat.S_ISREG(temporaryStat.st_mode):
            raise RuntimeError("digest temporary entry is not a regular file")
        with os.fdopen(temporaryFd, "w", encoding="utf-8") as temporaryFile:
            temporaryFd = -1
            temporaryFile.write(content)
        currentStat = os.stat(
            temporaryName, dir_fd=directoryFd, follow_symlinks=False
        )
        if not stat.S_ISREG(currentStat.st_mode) or not os.path.samestat(
            temporaryStat, currentStat
        ):
            raise RuntimeError("digest temporary entry changed before replacement")
        os.replace(
            temporaryName,
            filename,
            src_dir_fd=directoryFd,
            dst_dir_fd=directoryFd,
        )
        try:
            replacedStat = os.stat(
                filename, dir_fd=directoryFd, follow_symlinks=False
            )
        except OSError as error:
            raise RuntimeError("digest entry changed during replacement") from error
        if not stat.S_ISREG(replacedStat.st_mode) or not os.path.samestat(
            temporaryStat, replacedStat
        ):
            raise RuntimeError("digest entry changed during replacement")
        temporaryName = ""
    finally:
        if temporaryFd >= 0:
            os.close(temporaryFd)
        if temporaryName:
            try:
                os.unlink(temporaryName, dir_fd=directoryFd)
            except FileNotFoundError:
                pass


def _read_existing_digest(directoryFd: int, filename: str) -> str | None:
    """Read an existing regular digest without following its final symlink."""
    try:
        expectedStat = os.stat(filename, dir_fd=directoryFd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(expectedStat.st_mode):
        raise ValueError("digest path is not a regular file")

    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(filename, flags, dir_fd=directoryFd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EISDIR, errno.ENXIO, errno.ENOTDIR}:
            raise ValueError("digest path is not a regular file") from error
        raise

    try:
        openedStat = os.fstat(fd)
        if not stat.S_ISREG(openedStat.st_mode):
            raise ValueError("digest path is not a regular file")
        if not os.path.samestat(expectedStat, openedStat):
            raise ValueError("digest path changed while opening it")
        with os.fdopen(fd, "r", encoding="utf-8") as fh:
            fd = -1
            return fh.read()
    finally:
        if fd >= 0:
            os.close(fd)


def deliver(
    settings: Settings,
    summaries: list[ProcessedPaperSummary],
    pulse: list[str],
    papers_analyzed: int,
    force: bool = False,
    dry_run: bool = False,
    filename: str | None = None,
) -> Path | None:
    today = datetime.now(timezone.utc).date()
    vaultPath = settings.obsidian.vault_path.expanduser().resolve()
    path = _target_path(settings, today)
    if filename is not None:
        _validate_filename(filename)
        path = path.with_name(filename)
    _validate_filename(path.name)
    content = render_markdown(
        today,
        summaries,
        pulse,
        papers_analyzed,
        llm_model=settings.llm.model,
        llm_provider=settings.llm.provider,
    )

    if dry_run:
        log.info("[dry-run] would write digest to %s", path)
        return path

    with _opened_digest_directory(vaultPath, path.parent) as (
        directoryFd,
        directoryStat,
    ):
        lockKey = (directoryStat.st_dev, directoryStat.st_ino, path.name)
        with _in_process_lock(lockKey), _advisory_lock(directoryFd):
            _check_directory_identity(path.parent, directoryStat)
            previousContent = _read_existing_digest(directoryFd, path.name)
            if previousContent is not None and not force:
                stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                provider = _safe_text(settings.llm.provider) or "unspecified"
                model = _safe_text(settings.llm.model) or "unspecified"
                rerunContent = (
                    f"\n---\n\n*Re-run {stamp} \u2014 provider: {provider}; "
                    f"model: {model}; merged {len(summaries)} papers.*\n\n"
                    + "\n".join(_paper_block(s) for s in summaries)
                )
                outputContent = previousContent + rerunContent
                merged = True
            else:
                outputContent = content
                merged = False

            _check_directory_identity(path.parent, directoryStat)
            try:
                _atomic_write(directoryFd, path.name, outputContent)
            except BaseException as error:
                try:
                    _check_directory_identity(path.parent, directoryStat)
                except RuntimeError as changedError:
                    raise changedError from error
                raise
            _check_directory_identity(path.parent, directoryStat)

    if merged:
        log.warning("Digest %s already existed; merged new content", path)
    else:
        log.info("Wrote digest: %s", path)
    return path
