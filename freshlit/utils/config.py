"""Central configuration loader for FreshLit.

Loads .env, validates config/settings.yaml and config/journal_tiers.json,
and exposes a single Settings object. No node reads config files directly.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, PositiveInt

from .profile import extract_keywords, normalize_topic_id, parse_research_profile


class ObsidianConfig(BaseModel):
    vault_path: Path
    digest_folder: str = ""


class EuropePMCConfig(BaseModel):
    # Preserve old profiles' default; modern profiles choose sources explicitly.
    enabled: bool = True
    sources: list[str] = Field(default_factory=lambda: ["PPR"])


class IngestionConfig(BaseModel):
    lookback_days: int = 7
    timezone: str = "UTC"
    max_results_per_query: int = 200
    openalex_topics: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    europe_pmc: EuropePMCConfig = Field(default_factory=EuropePMCConfig)


class APIConfig(BaseModel):
    request_timeout_seconds: int = 30
    max_pages: int = 50
    retry_backoff_seconds: float = 2.0


class FilteringConfig(BaseModel):
    dedup_title_similarity: float = 0.92
    vector_threshold: float = 0.65
    max_llm_candidates: int = 40
    llm_batch_size: int = Field(default=15, ge=1, le=100)
    score_cutoff: float = 7.0
    default_journal_weight: float = 1.0
    embedding_model: str = "allenai/specter2_base"
    embedding_device: str = "cpu"


class LLMConfig(BaseModel):
    provider: Literal["codex", "gateway"] = "codex"
    base_url: str = "https://opencode.ai/zen/go/v1"
    model: str = "gpt-5.6-luna"
    timeout_seconds: PositiveInt = 120
    workers: PositiveInt = 3
    max_retries: int = Field(default=2, ge=0, le=5)
    codex_home: Path | None = None


class Settings(BaseModel):
    """Fully-resolved runtime settings."""

    model_config = {"arbitrary_types_allowed": True}

    project_root: Path
    obsidian: ObsidianConfig
    ingestion: IngestionConfig
    api: APIConfig
    filtering: FilteringConfig
    llm: LLMConfig
    opencode_go_api_key: str | None = None
    openalex_api_key: str | None
    journal_tiers: dict[str, float]
    research_profile_text: str

    @property
    def db_path(self) -> Path:
        return self.project_root / "data" / "cache.db"

    @property
    def profile_vector_path(self) -> Path:
        return self.project_root / "data" / "profile_exemplars.json"


def _project_root() -> Path:
    # freshlit/utils/config.py -> project root is two levels up
    return Path(__file__).resolve().parents[2]


def _extract_keywords(profile_md: str) -> list[str]:
    """Compatibility wrapper used by the profile-vector builder."""
    return extract_keywords(profile_md)


def _read_utf8(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _modern_conflict(name: str, legacy_file: str) -> ValueError:
    return ValueError(
        f"Modern research profile {name} conflicts with explicit legacy values in "
        f"{legacy_file}. Remove the legacy values or make them match the profile."
    )


def load_settings(
    project_root: Path | None = None,
    provider_override: Literal["codex", "gateway"] | None = None,
    require_llm_credentials: bool = True,
) -> Settings:
    root = Path(project_root) if project_root else _project_root()

    # Read dotenv values without leaking project configuration into the process.
    dotenv = dotenv_values(root / ".env")

    def env_value(name: str) -> str:
        value = os.environ.get(name)
        if value is None:
            value = dotenv.get(name)
        return value.strip() if isinstance(value, str) else ""

    settings_file = root / "config" / "settings.yaml"
    if not settings_file.exists():
        raise FileNotFoundError(f"Missing config file: {settings_file}")
    with settings_file.open() as fh:
        raw = yaml.safe_load(fh) or {}

    tiers_file = root / "config" / "journal_tiers.json"
    journal_tiers: dict[str, float] = {}
    if tiers_file.exists():
        with tiers_file.open() as fh:
            raw_tiers = json.load(fh)
        journal_tiers = {
            k.replace("-", "").upper(): float(v)
            for k, v in raw_tiers.items()
            if not k.startswith("_")
        }

    profile_file = root / "config" / "research_profile.md"
    if not profile_file.exists():
        raise FileNotFoundError(
            f"Missing research profile: {profile_file}. Run `freshlit init` to create it."
        )
    profile = parse_research_profile(_read_utf8(profile_file))

    ingestion_raw = raw.get("ingestion") or {}
    ingestion = IngestionConfig(**ingestion_raw)
    if profile.is_modern:
        modern_topics = [topic.id for topic in profile.openalex_topics]
        if ingestion.openalex_topics:
            try:
                legacy_topics = [normalize_topic_id(topic) for topic in ingestion.openalex_topics]
            except ValueError as exc:
                raise _modern_conflict("OpenAlex topics", "config/settings.yaml") from exc
            if legacy_topics != modern_topics:
                raise _modern_conflict("OpenAlex topics", "config/settings.yaml")
        modern_keywords = list(profile.keywords)
        if ingestion.keywords and ingestion.keywords != modern_keywords:
            raise _modern_conflict("keywords", "config/settings.yaml")
        modern_tiers = {weight.issn: weight.weight for weight in profile.venue_weights}
        if journal_tiers and journal_tiers != modern_tiers:
            raise _modern_conflict("venue weights", "config/journal_tiers.json")
        legacy_epmc = ingestion_raw.get("europe_pmc") or {}
        if (
            "enabled" in legacy_epmc
            and ingestion.europe_pmc.enabled != profile.europe_pmc.enabled
        ):
            raise _modern_conflict("Europe PMC enabled setting", "config/settings.yaml")
        if (
            "sources" in legacy_epmc
            and ingestion.europe_pmc.sources != list(profile.europe_pmc.sources)
        ):
            raise _modern_conflict("Europe PMC sources", "config/settings.yaml")

        ingestion.openalex_topics = modern_topics
        ingestion.keywords = modern_keywords
        ingestion.europe_pmc = EuropePMCConfig(
            enabled=profile.europe_pmc.enabled,
            sources=list(profile.europe_pmc.sources),
        )
        journal_tiers = modern_tiers
    else:
        warnings.warn(
            "Legacy research_profile.md without YAML front matter is supported for now; "
            "run `freshlit migrate-profile` to preview migration.",
            FutureWarning,
            stacklevel=2,
        )
        ingestion.openalex_topics = [
            normalize_topic_id(topic) for topic in ingestion.openalex_topics
        ]
        if not ingestion.keywords:
            ingestion.keywords = _extract_keywords(profile.body)

    if not ingestion.keywords and not ingestion.openalex_topics:
        raise ValueError(
            "No research search selectors are configured. Add at least one "
            "user-chosen Keywords bullet or OpenAlex topic ID; FreshLit will not "
            "generate queries from the research description."
        )

    llm_raw = dict(raw.get("llm") or {})
    env_provider = env_value("FRESHLIT_LLM_PROVIDER")
    if env_provider:
        llm_raw["provider"] = env_provider
    if provider_override is not None:
        llm_raw["provider"] = provider_override
    codex_home = env_value("FRESHLIT_CODEX_HOME")
    if codex_home:
        llm_raw["codex_home"] = codex_home
    llm = LLMConfig(**llm_raw)

    opencode_key: str | None = None
    if llm.provider == "gateway":
        opencode_key = env_value("OPENCODE_GO_API_KEY") or None
        if require_llm_credentials and not opencode_key:
            raise RuntimeError(
                "OPENCODE_GO_API_KEY is missing or blank. Add it to .env "
                "(see .env.example)."
            )
    openalex_key = env_value("OPENALEX_API_KEY") or None

    obsidian_raw = raw.get("obsidian") or {}
    vault_path = env_value("FRESHLIT_VAULT_PATH")
    if not vault_path:
        vault_path = obsidian_raw.get("vault_path")
    obsidian = ObsidianConfig(
        vault_path=vault_path, digest_folder=obsidian_raw.get("digest_folder", "")
    )

    return Settings(
        project_root=root,
        obsidian=obsidian,
        ingestion=ingestion,
        api=APIConfig(**(raw.get("api") or {})),
        filtering=FilteringConfig(**(raw.get("filtering") or {})),
        llm=llm,
        opencode_go_api_key=opencode_key,
        openalex_api_key=openalex_key,
        journal_tiers=journal_tiers,
        research_profile_text=profile.body,
    )
