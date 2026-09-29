"""Local onboarding, diagnostics, and explicit OpenAlex topic lookup helpers."""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
import yaml
from dotenv import dotenv_values

from .config import ObsidianConfig, load_settings
from .profile import normalize_topic_id, parse_research_profile


OPENALEX_TOPIC_AUTOCOMPLETE = "https://api.openalex.org/autocomplete/topics"
OPENALEX_TOPIC_TIMEOUT_SECONDS = 30


def _is_regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def _copy_exclusive(source: Path, destination: Path) -> bool:
    """Copy a template to a private new file, returning false if it exists."""

    try:
        content = source.read_bytes()
    except OSError:
        raise RuntimeError("Required onboarding template cannot be read") from None

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError:
        return False
    except OSError:
        raise RuntimeError("Onboarding file could not be created safely") from None

    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return True


def _dotenv_environment_value(
    name: str, dotenv: dict[str, str | None]
) -> str:
    """Match the central loader's environment-over-dotenv blank semantics."""

    value = os.environ.get(name)
    if value is None:
        value = dotenv.get(name)
    return value.strip() if isinstance(value, str) else ""


def _load_raw_settings(settings_path: Path) -> dict[str, Any]:
    try:
        with settings_path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        raise RuntimeError("config/settings.yaml cannot be read as YAML") from None
    if not isinstance(loaded, dict):
        raise RuntimeError("config/settings.yaml must contain a mapping")
    return loaded


def _nested_setting(raw: dict[str, Any], section: str, name: str) -> str:
    section_value = raw.get(section)
    if not isinstance(section_value, dict):
        return ""
    value = section_value.get(name)
    if isinstance(value, (str, Path)):
        return str(value).strip()
    return ""


def _configured_path(
    raw: dict[str, Any],
    dotenv: dict[str, str | None],
    *,
    environment_name: str,
    section: str,
    setting_name: str,
    description: str,
) -> Path:
    value = _dotenv_environment_value(environment_name, dotenv)
    if not value:
        value = _nested_setting(raw, section, setting_name)
    if not value:
        raise RuntimeError(f"No {description} is configured")
    return Path(value).expanduser()


def initialize(
    project_root: Path,
    *,
    create_output_dir: bool = False,
    prepare_codex_home: bool = False,
) -> list[str]:
    """Initialize only missing user files and explicitly requested directories."""

    root = Path(project_root)
    settings_path = root / "config" / "settings.yaml"
    profile_template = root / "config" / "research_profile.example.md"
    environment_template = root / ".env.example"
    profile_path = root / "config" / "research_profile.md"
    environment_path = root / ".env"

    if not _is_regular_file(settings_path):
        raise FileNotFoundError(
            "Existing FreshLit checkout is missing config/settings.yaml"
        )
    if not _is_regular_file(profile_template):
        raise FileNotFoundError(
            "Existing FreshLit checkout is missing the research profile template"
        )
    if not environment_path.exists() and not _is_regular_file(environment_template):
        raise FileNotFoundError(
            "Existing FreshLit checkout is missing the environment template"
        )

    messages: list[str] = []
    if _copy_exclusive(profile_template, profile_path):
        messages.append("Created config/research_profile.md from the example template.")
    else:
        messages.append(
            "Kept existing config/research_profile.md; it was not overwritten."
        )
    messages.append(
        "Edit config/research_profile.md before running FreshLit: add your own "
        "search selectors and research description."
    )

    if environment_path.exists():
        messages.append("Kept existing .env; it was not overwritten.")
    elif _copy_exclusive(environment_template, environment_path):
        messages.append("Created .env from .env.example.")
    else:
        messages.append("Kept existing .env; it was not overwritten.")

    if not create_output_dir and not prepare_codex_home:
        return messages

    raw = _load_raw_settings(settings_path)
    dotenv = dotenv_values(environment_path)

    if create_output_dir:
        output_root = _configured_path(
            raw,
            dotenv,
            environment_name="FRESHLIT_VAULT_PATH",
            section="obsidian",
            setting_name="vault_path",
            description="output root",
        )
        try:
            output_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            raise RuntimeError(
                "Configured output root could not be created as a directory"
            ) from None

        # Reuse delivery's nonwriting safety checks after creation. It permits
        # missing nested digest folders, which the secure writer creates later.
        from ..nodes.delivery import validate_destination

        obsidian_raw = raw.get("obsidian")
        digest_folder = (
            obsidian_raw.get("digest_folder", "")
            if isinstance(obsidian_raw, dict)
            else ""
        )
        try:
            obsidian = ObsidianConfig(
                vault_path=output_root,
                digest_folder=digest_folder,
            )
        except Exception:
            raise RuntimeError("Configured output destination is invalid") from None
        try:
            validate_destination(SimpleNamespace(obsidian=obsidian))
        except Exception:
            raise RuntimeError(
                "Configured output destination failed local safety checks"
            ) from None
        messages.append("Configured output root is ready.")

    if prepare_codex_home:
        configured_home = _configured_path(
            raw,
            dotenv,
            environment_name="FRESHLIT_CODEX_HOME",
            section="llm",
            setting_name="codex_home",
            description="dedicated Codex home",
        )
        unresolved_home = configured_home.resolve(strict=False)
        resolved_root = root.expanduser().resolve()
        shared_home = (Path.home() / ".codex").resolve()
        if (
            unresolved_home == shared_home
            or unresolved_home.is_relative_to(resolved_root)
            or resolved_root.is_relative_to(unresolved_home)
        ):
            raise RuntimeError(
                "Dedicated Codex home must be outside the project and must not "
                "use ~/.codex"
            )

        created = False
        try:
            configured_home.mkdir(parents=True, mode=0o700, exist_ok=False)
            created = True
        except FileExistsError:
            pass
        except OSError:
            raise RuntimeError("Dedicated Codex home could not be created") from None

        # Importing the LLM stack is intentionally deferred until this explicit
        # setup operation needs its existing security validator.
        from .llm import validate_codex_home

        validated_home = validate_codex_home(configured_home, root)
        messages.append(
            "Created dedicated Codex home with mode 0700."
            if created
            else "Existing dedicated Codex home passed local safety checks."
        )
        login_command = f"CODEX_HOME={shlex.quote(str(validated_home))} codex login"
        messages.append(f"Authenticate it when needed with: {login_command}")

    return messages


def _template_diagnostics(root: Path) -> list[str]:
    checks = (
        (root / "config" / "settings.yaml", "config/settings.yaml"),
        (
            root / "config" / "research_profile.example.md",
            "config/research_profile.example.md",
        ),
        (root / ".env.example", ".env.example"),
    )
    messages: list[str] = []
    for path, label in checks:
        if _is_regular_file(path):
            messages.append(f"OK: Required checkout file {label} is present.")
        else:
            messages.append(
                f"ERROR: Required checkout file {label} is missing or not a regular file; "
                "restore it from the FreshLit checkout."
            )
    return messages


def _profile_error_guidance(error: ValueError) -> str:
    """Return actionable profile guidance without reflecting untrusted values."""

    reason = str(error).casefold()
    if "no closing" in reason:
        guidance = "close the YAML front matter with a second `---` line"
    elif "research description" in reason:
        guidance = (
            "add a nonblank Research Description and at least one user-chosen "
            "keyword or OpenAlex topic ID"
        )
    elif "keywords" in reason:
        guidance = "add a `## Keywords` section and a user-chosen search selector"
    elif "search selector" in reason:
        guidance = "add a user-chosen keyword or canonical OpenAlex topic ID"
    elif "topic" in reason:
        guidance = "use canonical OpenAlex topic IDs such as T123"
    elif "issn" in reason or "venue" in reason or "weight" in reason:
        guidance = "check venue_weights ISSNs and positive numeric weights"
    elif "yaml" in reason:
        guidance = "repair the YAML syntax and required front-matter structure"
    else:
        guidance = (
            "use only the required version, search, and venue_weights front-matter "
            "keys and complete the profile sections"
        )
    return f"ERROR: Research profile is invalid; {guidance}."


def _existing_outside_project_temp_parent(project_root: Path) -> Path:
    """Select an existing usable temp directory using read-only filesystem checks."""

    root = project_root.expanduser().resolve()
    configured = [
        os.environ.get(name, "").strip()
        for name in ("TMPDIR", "TMP", "TEMP")
    ]
    candidates = [Path(value).expanduser() for value in configured if value]
    if os.name == "posix":
        candidates.append(Path("/tmp"))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            candidateStat = os.lstat(resolved)
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if (
            stat.S_ISDIR(candidateStat.st_mode)
            and not resolved.is_relative_to(root)
            and os.access(resolved, os.W_OK | os.X_OK)
        ):
            return resolved
    raise RuntimeError("No existing writable temporary directory is available")


def doctor(project_root: Path) -> list[str]:
    """Run local, read-only diagnostics without clients, auth reads, or downloads."""

    root = Path(project_root)
    messages = _template_diagnostics(root)

    profile_path = root / "config" / "research_profile.md"
    if not _is_regular_file(profile_path):
        messages.append(
            "ERROR: config/research_profile.md is missing or not a regular file; "
            "run `freshlit init`, then edit the profile."
        )
        return messages

    try:
        profile_text = profile_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        messages.append(
            "ERROR: config/research_profile.md must be readable UTF-8 text; check "
            "its encoding and permissions."
        )
        return messages
    try:
        parsed_profile = parse_research_profile(profile_text)
    except ValueError as error:
        messages.append(_profile_error_guidance(error))
        return messages

    try:
        settings = load_settings(root, require_llm_credentials=False)
    except Exception:
        if not parsed_profile.keywords and not parsed_profile.openalex_topics:
            messages.append(
                "ERROR: No research search selectors are effective; add a user-chosen "
                "keyword or canonical OpenAlex topic ID, then rerun doctor."
            )
        else:
            messages.append(
                "ERROR: Local settings conflict with or cannot apply the research "
                "profile; check config/settings.yaml and profile front matter, then "
                "rerun doctor."
            )
        return messages
    messages.append("OK: Settings and research profile loaded successfully.")
    messages.append(
        f"OK: Explicit search choices: {len(settings.ingestion.keywords)} keywords, "
        f"{len(settings.ingestion.openalex_topics)} OpenAlex topics; "
        f"Europe PMC {'enabled' if settings.ingestion.europe_pmc.enabled else 'disabled'}; "
        f"{len(settings.journal_tiers)} venue-weight overrides."
    )

    try:
        ZoneInfo(settings.ingestion.timezone)
    except (ZoneInfoNotFoundError, ValueError):
        messages.append(
            "ERROR: The configured ingestion timezone is unavailable; use a valid "
            "IANA timezone such as UTC."
        )
    else:
        messages.append("OK: The configured ingestion timezone is available.")

    from ..nodes import delivery

    try:
        delivery._require_secure_primitives()
    except RuntimeError:
        messages.append(
            "ERROR: Secure POSIX delivery primitives are unavailable on this platform."
        )
    else:
        messages.append("OK: Secure POSIX delivery primitives are available.")

    try:
        delivery.validate_destination(settings)
    except Exception:
        messages.append(
            "ERROR: Digest destination is not ready or safe; verify the configured "
            "vault exists, is writable, and contains no symlink/non-file entries."
        )
    else:
        messages.append("OK: Digest destination passed read-only preflight checks.")

    from . import embeddings

    if embeddings._is_model_cached(settings.filtering.embedding_model):
        messages.append("OK: The configured embedding model is cached locally.")
    else:
        messages.append(
            "WARN: The configured embedding model is not cached locally; doctor did "
            "not download it, and a later profile build may need a download."
        )

    if embeddings.profile_vector_is_current(settings):
        messages.append("OK: The local research profile vector is current.")
    else:
        messages.append(
            "WARN: The local research profile vector is missing or stale; it will be "
            "rebuilt for the current profile before use."
        )

    if settings.llm.provider == "gateway":
        if settings.opencode_go_api_key:
            messages.append("OK: Gateway credentials are configured for full runs.")
        else:
            messages.append(
                "WARN: OPENCODE_GO_API_KEY is missing; offline checks remain useful, "
                "but full gateway runs require the key in .env or the environment."
            )
        return messages

    # These imports are intentionally provider-gated. No runtime is launched and
    # no Codex authentication file is inspected.
    try:
        from .llm import (
            build_codex_config,
            validate_codex_home,
        )
    except ImportError:
        messages.append(
            "ERROR: Codex runtime dependencies are unavailable; reinstall the "
            "project environment before a full run."
        )
        messages.append(
            "WARN: Codex account login and requested model availability were NOT "
            "tested; doctor does not launch Codex or read authentication state."
        )
        return messages

    try:
        codex_home = validate_codex_home(
            settings.llm.codex_home,
            settings.project_root,
        )
    except RuntimeError:
        messages.append(
            "ERROR: Dedicated Codex home is missing or unsafe; configure an existing "
            "owner-only mode-0700 directory outside this project and ~/.codex."
        )
    else:
        messages.append("OK: Dedicated Codex home passed local safety checks.")
        try:
            temp_parent = _existing_outside_project_temp_parent(
                settings.project_root
            )
            build_codex_config(codex_home, temp_parent)
        except (OSError, RuntimeError):
            messages.append(
                "ERROR: The bundled Codex executable or safe local launcher "
                "configuration is unavailable."
            )
        else:
            messages.append(
                "OK: Bundled Codex executable and safe launcher configuration "
                "are available."
            )

    messages.append(
        "WARN: Codex account login and requested model availability were NOT tested; "
        "doctor does not launch Codex or read authentication state."
    )
    return messages


def search_topics(query: str, project_root: Path) -> list[dict[str, str]]:
    """Search OpenAlex topic autocomplete for an explicit user-supplied query."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("OpenAlex topic query must be nonblank")

    root = Path(project_root)
    dotenv = dotenv_values(root / ".env")
    api_key = _dotenv_environment_value("OPENALEX_API_KEY", dotenv)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        response = requests.get(
            OPENALEX_TOPIC_AUTOCOMPLETE,
            params={"q": query.strip()},
            headers=headers,
            timeout=OPENALEX_TOPIC_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.HTTPError as error:
        status_code = getattr(error.response, "status_code", None)
        status = str(status_code) if isinstance(status_code, int) else "unknown"
        raise RuntimeError(
            f"OpenAlex topic search failed with HTTP status {status}"
        ) from None
    except requests.RequestException:
        raise RuntimeError(
            "OpenAlex topic search failed because the service could not be reached"
        ) from None

    try:
        payload = response.json()
    except (ValueError, TypeError):
        raise RuntimeError("OpenAlex topic search returned invalid JSON") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RuntimeError("OpenAlex topic search returned an invalid response")

    topics: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in payload["results"]:
        if not isinstance(result, dict):
            continue
        label = result.get("display_name")
        try:
            topic_id = normalize_topic_id(result.get("id"))
        except ValueError:
            continue
        if not isinstance(label, str) or not label.strip() or topic_id in seen:
            continue
        seen.add(topic_id)
        topics.append({"id": topic_id, "label": label.strip()})
    return topics
