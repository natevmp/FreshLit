"""Central configuration loader for FreshLit.

Loads .env, validates config/settings.yaml and config/journal_tiers.json,
and exposes a single Settings object. No node reads config files directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ObsidianConfig(BaseModel):
    vault_path: Path
    digest_folder: str = ""


class EuropePMCConfig(BaseModel):
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
    llm_batch_size: int = 15  # papers scored per single LLM request
    score_cutoff: float = 7.0
    default_journal_weight: float = 1.0
    embedding_model: str = "allenai/specter2"
    embedding_device: str = "cpu"


class LLMConfig(BaseModel):
    base_url: str = "https://opencode.ai/zen/go/v1"
    model: str = "qwen3.5-plus"
    timeout_seconds: int = 120  # thinking-mode responses can exceed 30s
    workers: int = 3  # parallel requests; keep low to avoid gateway throttling


class Settings(BaseModel):
    """Fully-resolved runtime settings."""

    model_config = {"arbitrary_types_allowed": True}

    project_root: Path
    obsidian: ObsidianConfig
    ingestion: IngestionConfig
    api: APIConfig
    filtering: FilteringConfig
    llm: LLMConfig
    opencode_go_api_key: str
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
    """Pull the '- ' bullet list under the '## Keywords' heading."""
    keywords: list[str] = []
    in_section = False
    for line in profile_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped.lstrip("# ").strip().lower() == "keywords"
            continue
        if in_section and stripped.startswith("- "):
            kw = stripped[2:].strip()
            if kw:
                keywords.append(kw)
    return keywords


def load_settings(project_root: Path | None = None) -> Settings:
    root = Path(project_root) if project_root else _project_root()

    load_dotenv(root / ".env")

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
            k.replace("-", ""): float(v)
            for k, v in raw_tiers.items()
            if not k.startswith("_")
        }

    profile_file = root / "config" / "research_profile.md"
    if not profile_file.exists():
        profile_file = root / "config" / "research_profile.example.md"
    if not profile_file.exists():
        raise FileNotFoundError(
            f"Missing research profile: {root / 'config' / 'research_profile.md'}"
            " (and no example fallback)"
        )
    profile_text = profile_file.read_text()

    ingestion = IngestionConfig(**(raw.get("ingestion") or {}))
    if not ingestion.keywords:
        ingestion.keywords = _extract_keywords(profile_text)

    opencode_key = os.environ.get("OPENCODE_GO_API_KEY", "").strip()
    if not opencode_key:
        raise RuntimeError(
            "OPENCODE_GO_API_KEY is missing or blank. Add it to .env "
            "(see .env.example)."
        )
    openalex_key = os.environ.get("OPENALEX_API_KEY", "").strip() or None

    obsidian_raw = raw.get("obsidian") or {}
    vault_path = os.environ.get("FRESHLIT_VAULT_PATH", "").strip()
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
        llm=LLMConfig(**(raw.get("llm") or {})),
        opencode_go_api_key=opencode_key,
        openalex_api_key=openalex_key,
        journal_tiers=journal_tiers,
        research_profile_text=profile_text,
    )
