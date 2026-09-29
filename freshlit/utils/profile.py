"""Parsing and migration helpers for user-authored research profiles."""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


_TOPIC_ID = re.compile(r"T[1-9][0-9]*")
_TOPIC_URI = re.compile(r"https://openalex\.org/(T[1-9][0-9]*)")
_ISSN = re.compile(r"[0-9]{4}-?[0-9]{3}[0-9Xx]")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """A profile-local safe loader which rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class Topic:
    id: str
    label: str | None = None


@dataclass(frozen=True)
class EuropePMCSearch:
    enabled: bool
    sources: tuple[str, ...]


@dataclass(frozen=True)
class VenueWeight:
    issn: str
    name: str | None
    weight: float


@dataclass(frozen=True)
class ResearchProfile:
    """A parsed profile whose body is safe to send to embeddings and LLMs."""

    body: str
    keywords: tuple[str, ...]
    version: int | None = None
    openalex_topics: tuple[Topic, ...] = ()
    europe_pmc: EuropePMCSearch | None = None
    venue_weights: tuple[VenueWeight, ...] = ()

    @property
    def is_modern(self) -> bool:
        return self.version is not None


def extract_keywords(profile_md: str) -> list[str]:
    """Pull the ``- `` bullet list under the ``## Keywords`` heading."""
    keywords: list[str] = []
    in_section = False
    for line in profile_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lstrip("# ").strip().lower() == "keywords"
            continue
        if in_section and stripped.startswith("- "):
            keyword = stripped[2:].strip()
            if keyword:
                keywords.append(keyword)
    return keywords


def normalize_topic_id(value: str) -> str:
    """Normalize an OpenAlex short topic ID or canonical topic URI."""
    if not isinstance(value, str):
        raise ValueError("OpenAlex topic IDs must be strings such as T123")
    candidate = value.strip()
    short_match = _TOPIC_ID.fullmatch(candidate)
    if short_match:
        return candidate
    uri_match = _TOPIC_URI.fullmatch(candidate)
    if uri_match:
        return uri_match.group(1)
    raise ValueError(
        f"Malformed OpenAlex topic ID {value!r}; use T followed by digits "
        "or https://openalex.org/T followed by digits"
    )


def normalize_issn(value: str) -> str:
    """Validate ISSN syntax and return the downstream eight-character form."""
    if not isinstance(value, str) or not _ISSN.fullmatch(value.strip()):
        raise ValueError(f"Malformed ISSN {value!r}; use the form 1234-5678")
    return value.strip().replace("-", "").upper()


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Profile front matter {location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"Profile front matter {location} keys must be strings")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    location: str,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> None:
    extra = set(value) - required - optional
    missing = required - set(value)
    if extra:
        raise ValueError(
            f"Unknown key(s) in profile front matter {location}: {', '.join(sorted(extra))}"
        )
    if missing:
        raise ValueError(
            f"Missing key(s) in profile front matter {location}: {', '.join(sorted(missing))}"
        )


def _nonblank_optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Profile front matter {location} must be a nonblank string")
    return value.strip()


def _split_front_matter(text: str) -> tuple[str | None, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None, text
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    raise ValueError("Research profile starts YAML front matter but has no closing '---'")


def _section_body(markdown: str, heading: str) -> str | None:
    wanted = heading.casefold()
    lines = markdown.splitlines(keepends=True)
    start: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## "):
            title = stripped.lstrip("# ").strip().casefold()
            if start is not None:
                return "".join(lines[start:index])
            if title == wanted:
                start = index + 1
    return None if start is None else "".join(lines[start:])


def parse_research_profile(text: str) -> ResearchProfile:
    """Parse a legacy Markdown profile or a strict version-1 modern profile."""
    front_matter_text, body = _split_front_matter(text)
    if front_matter_text is None:
        uncommented_body = _HTML_COMMENT.sub("", body)
        if not uncommented_body.strip():
            raise ValueError(
                "Legacy research profile is blank or contains only HTML comments; "
                "add your user-authored research profile content"
            )
        return ResearchProfile(body=body, keywords=tuple(extract_keywords(body)))

    try:
        loaded = yaml.load(front_matter_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid research profile YAML front matter: {exc}") from exc
    root = _require_mapping(loaded, "root")
    _require_exact_keys(root, "root", {"version", "search", "venue_weights"})
    if type(root["version"]) is not int or root["version"] != 1:
        raise ValueError("Profile front matter version must be exactly 1")

    search = _require_mapping(root["search"], "search")
    _require_exact_keys(search, "search", {"openalex_topics", "europe_pmc"})

    raw_topics = search["openalex_topics"]
    if not isinstance(raw_topics, list):
        raise ValueError("Profile front matter search.openalex_topics must be a list")
    topics: list[Topic] = []
    topic_ids: set[str] = set()
    for index, raw_topic in enumerate(raw_topics):
        location = f"search.openalex_topics[{index}]"
        topic_data = _require_mapping(raw_topic, location)
        _require_exact_keys(topic_data, location, {"id"}, {"label"})
        topic_id = normalize_topic_id(topic_data["id"])
        if topic_id in topic_ids:
            raise ValueError(f"Duplicate OpenAlex topic ID in profile: {topic_id}")
        topic_ids.add(topic_id)
        label = _nonblank_optional_string(
            topic_data.get("label"), f"{location}.label"
        )
        topics.append(Topic(topic_id, label))

    epmc_data = _require_mapping(search["europe_pmc"], "search.europe_pmc")
    _require_exact_keys(epmc_data, "search.europe_pmc", {"enabled", "sources"})
    if type(epmc_data["enabled"]) is not bool:
        raise ValueError("Profile front matter search.europe_pmc.enabled must be true or false")
    raw_sources = epmc_data["sources"]
    if not isinstance(raw_sources, list):
        raise ValueError("Profile front matter search.europe_pmc.sources must be a list")
    sources: list[str] = []
    for source in raw_sources:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Europe PMC sources must be nonblank strings")
        normalized_source = source.strip()
        if normalized_source in sources:
            raise ValueError(f"Duplicate Europe PMC source in profile: {normalized_source}")
        sources.append(normalized_source)
    if epmc_data["enabled"] and not sources:
        raise ValueError(
            "Profile front matter search.europe_pmc.sources cannot be empty when "
            "Europe PMC is enabled"
        )

    raw_weights = root["venue_weights"]
    if not isinstance(raw_weights, list):
        raise ValueError("Profile front matter venue_weights must be a list")
    venue_weights: list[VenueWeight] = []
    weighted_issns: set[str] = set()
    for index, raw_weight in enumerate(raw_weights):
        location = f"venue_weights[{index}]"
        weight_data = _require_mapping(raw_weight, location)
        _require_exact_keys(weight_data, location, {"issn", "weight"}, {"name"})
        issn = normalize_issn(weight_data["issn"])
        if issn in weighted_issns:
            raise ValueError(f"Duplicate ISSN in profile: {issn}")
        weighted_issns.add(issn)
        weight = weight_data["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"Profile front matter {location}.weight must be a number")
        normalized_weight = float(weight)
        if not math.isfinite(normalized_weight) or normalized_weight <= 0:
            raise ValueError(
                f"Profile front matter {location}.weight must be positive and finite"
            )
        venue_weights.append(
            VenueWeight(
                issn,
                _nonblank_optional_string(weight_data.get("name"), f"{location}.name"),
                normalized_weight,
            )
        )

    uncommented_body = _HTML_COMMENT.sub("", body)
    keywords_section = _section_body(uncommented_body, "Keywords")
    description = _section_body(uncommented_body, "Research Description")
    if keywords_section is None:
        raise ValueError("Modern research profile must contain a '## Keywords' section")
    if description is None or not any(character.isalnum() for character in description):
        raise ValueError(
            "Modern research profile needs a nonblank '## Research Description'; "
            "replace the template instructions with your own research description"
        )
    keywords = tuple(extract_keywords(uncommented_body))
    if not keywords and not topics:
        raise ValueError(
            "Research profile has no search selectors; add at least one user-chosen "
            "Keywords bullet or OpenAlex topic ID"
        )

    return ResearchProfile(
        body=body,
        keywords=keywords,
        version=1,
        openalex_topics=tuple(topics),
        europe_pmc=EuropePMCSearch(epmc_data["enabled"], tuple(sources)),
        venue_weights=tuple(venue_weights),
    )


def _read_regular_profile(path: Path) -> str:
    """Read a regular profile without following a symlink."""
    try:
        initial_stat = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing research profile: {path}. Run `freshlit init` to create it."
        ) from exc
    if not stat.S_ISREG(initial_stat.st_mode):
        raise ValueError(
            f"Refusing to migrate non-regular research profile: {path}"
        )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(
                f"Refusing to migrate non-regular research profile: {path}"
            )
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            initial_stat.st_dev,
            initial_stat.st_ino,
        ):
            raise RuntimeError(f"Research profile changed while opening it: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8", newline="") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _format_issn(issn: str) -> str:
    return f"{issn[:4]}-{issn[4:]}"


def _modern_document(settings: Any, legacy_body: str) -> str:
    metadata = {
        "version": 1,
        "search": {
            "openalex_topics": [
                {"id": normalize_topic_id(topic)}
                for topic in settings.ingestion.openalex_topics
            ],
            "europe_pmc": {
                "enabled": settings.ingestion.europe_pmc.enabled,
                "sources": list(settings.ingestion.europe_pmc.sources),
            },
        },
        "venue_weights": [
            {"issn": _format_issn(issn), "weight": weight}
            for issn, weight in settings.journal_tiers.items()
        ],
    }
    rendered = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
    return f"---\n{rendered}---\n{legacy_body}"


@dataclass
class _MigrationSettings:
    ingestion: Any
    journal_tiers: dict[str, float]


def _load_migration_settings(root: Path, legacy_body: str) -> _MigrationSettings:
    """Resolve query inputs without reading dotenv or any authentication data."""
    from .config import IngestionConfig

    settings_path = root / "config" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Missing config file: {settings_path}")
    with settings_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    ingestion = IngestionConfig(**(raw.get("ingestion") or {}))
    ingestion.openalex_topics = [
        normalize_topic_id(topic) for topic in ingestion.openalex_topics
    ]
    if not ingestion.keywords:
        ingestion.keywords = extract_keywords(legacy_body)

    journal_tiers: dict[str, float] = {}
    tiers_path = root / "config" / "journal_tiers.json"
    if tiers_path.exists():
        with tiers_path.open("r", encoding="utf-8") as stream:
            raw_tiers = json.load(stream)
        journal_tiers = {
            key.replace("-", "").upper(): float(weight)
            for key, weight in raw_tiers.items()
            if not key.startswith("_")
        }
    return _MigrationSettings(ingestion=ingestion, journal_tiers=journal_tiers)


def _assert_equivalent_migration(
    validated: ResearchProfile, settings: _MigrationSettings
) -> None:
    mismatches: list[str] = []
    if list(validated.keywords) != settings.ingestion.keywords:
        mismatches.append("keywords")
    if [topic.id for topic in validated.openalex_topics] != settings.ingestion.openalex_topics:
        mismatches.append("OpenAlex topics")
    if validated.europe_pmc is None or (
        validated.europe_pmc.enabled != settings.ingestion.europe_pmc.enabled
        or list(validated.europe_pmc.sources) != settings.ingestion.europe_pmc.sources
    ):
        mismatches.append("Europe PMC settings")
    validated_weights = {
        venue_weight.issn: venue_weight.weight
        for venue_weight in validated.venue_weights
    }
    if validated_weights != settings.journal_tiers:
        mismatches.append("venue weights")
    if mismatches:
        raise ValueError(
            "Cannot safely migrate because modern profile validation would change "
            f"effective {', '.join(mismatches)}. Resolve settings overrides and "
            "remove Markdown/HTML comment ambiguity before retrying."
        )


def _create_private_backup(backup_path: Path, original: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(backup_path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to migrate because backup already exists: {backup_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as backup:
            descriptor = -1
            backup.write(original)
            backup.flush()
            os.fsync(backup.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        backup_path.unlink(missing_ok=True)
        raise


def _atomic_replace_profile(profile_path: Path, migrated: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=profile_path.parent,
        prefix=f".{profile_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            stream.write(migrated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, profile_path)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise


def migrate_profile(project_root: Path, *, apply: bool = False) -> str:
    """Preview or apply migration of a legacy profile to version-1 front matter.

    Applying creates ``research_profile.md.bak`` exclusively before replacing the
    profile. Existing modern profiles are returned unchanged and never rewritten.
    """
    root = Path(project_root)
    profile_path = root / "config" / "research_profile.md"
    original = _read_regular_profile(profile_path)
    parsed = parse_research_profile(original)
    if parsed.is_modern:
        return original

    settings = _load_migration_settings(root, parsed.body)

    markdown_keywords = extract_keywords(parsed.body)
    if settings.ingestion.keywords != markdown_keywords:
        raise ValueError(
            "Cannot migrate because ingestion.keywords overrides the Markdown Keywords "
            "section. Move the intended user-chosen keywords into research_profile.md "
            "and remove the settings.yaml keyword override first."
        )

    migrated = _modern_document(settings, parsed.body)
    validated = parse_research_profile(migrated)
    if validated.body != original:
        raise RuntimeError("Generated profile did not preserve the legacy Markdown body")
    _assert_equivalent_migration(validated, settings)
    if not apply:
        return migrated

    backup_path = profile_path.with_name("research_profile.md.bak")
    _create_private_backup(backup_path, original)
    _atomic_replace_profile(profile_path, migrated)
    return migrated
