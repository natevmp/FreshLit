"""Node 1: Ingestion engine (OpenAlex + Europe PMC -> RawPaper list)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from pydantic import BaseModel

from ..utils.config import Settings

log = logging.getLogger(__name__)

OPENALEX_WORKS = "https://api.openalex.org/works"
EUROPE_PMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
INGEST_CACHE_SCHEMA_VERSION = 1
QUERY_PROVENANCE_VERSION = 1


class RawPaper(BaseModel):
    id: str  # OpenAlex ID or source URI
    doi: Optional[str] = None
    title: str
    authors: list[str]
    publication_date: str
    venue_name: Optional[str] = None
    venue_issn: Optional[str] = None
    venue_impact_factor: Optional[float] = 0.0
    abstract: str
    pdf_url: Optional[str] = None
    source_type: str  # 'journal' or 'preprint'


def _date_window(settings: Settings) -> tuple[str, str]:
    tz = ZoneInfo(settings.ingestion.timezone)
    end = datetime.now(tz).date()
    start = end - timedelta(days=settings.ingestion.lookback_days)
    return start.isoformat(), end.isoformat()


def _get_json(
    session: requests.Session,
    url: str,
    params: dict,
    settings: Settings,
    max_retries: int = 4,
) -> dict | None:
    """GET with timeout, exponential backoff on 429/5xx, honoring Retry-After."""
    delay = settings.api.retry_backoff_seconds
    max_wait = 120.0  # never park the pipeline for hours on a long Retry-After
    for attempt in range(max_retries):
        try:
            resp = session.get(
                url, params=params, timeout=settings.api.request_timeout_seconds
            )
            if resp.status_code in (403, 429) or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay * (2**attempt)
                if wait > max_wait:
                    log.error(
                        "HTTP %s from %s demands %.0fs wait (quota exhausted?); "
                        "skipping this query", resp.status_code, url, wait,
                    )
                    return None
                log.warning(
                    "HTTP %s from %s; retrying in %.1fs", resp.status_code, url, wait
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Request error (%s); retrying in %.1fs", exc, delay)
            time.sleep(delay * (2**attempt))
    log.error("Giving up on %s after %d attempts", url, max_retries)
    return None


# ---------------------------------------------------------------- OpenAlex


def _abstract_from_inverted_index(inverted: dict | None) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return " ".join(word for _, word in positions)


def _openalex_to_paper(work: dict) -> RawPaper | None:
    try:
        title = (work.get("title") or "").strip()
        if not title:
            return None
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in work.get("authorships", [])
        ]
        authors = [a for a in authors if a]
        loc = work.get("primary_location") or {}
        source = loc.get("source") or {}
        issns = source.get("issn") or []
        venue_issn = source.get("issn_l") or (issns[0] if issns else None)
        if venue_issn:
            venue_issn = venue_issn.replace("-", "")
        source_type = (
            "preprint" if (source.get("type") or "") == "preprint" else "journal"
        )
        best_oa = work.get("best_oa_location") or {}
        pdf_url = best_oa.get("pdf_url") or (work.get("open_access") or {}).get(
            "oa_url"
        )
        return RawPaper(
            id=work.get("id", ""),
            doi=work.get("doi"),
            title=title,
            authors=authors,
            publication_date=work.get("publication_date", ""),
            venue_name=source.get("display_name"),
            venue_issn=venue_issn,
            abstract=_abstract_from_inverted_index(
                work.get("abstract_inverted_index")
            ),
            pdf_url=pdf_url,
            source_type=source_type,
        )
    except Exception as exc:  # per-record isolation
        log.warning("Skipping malformed OpenAlex record: %s", exc)
        return None


def _fetch_openalex_filter(
    session: requests.Session, settings: Settings, extra_filter: str | None,
    search: str | None, date_from: str, date_to: str,
) -> list[RawPaper]:
    papers: list[RawPaper] = []
    cursor = "*"
    cap = settings.ingestion.max_results_per_query
    base_filter = f"from_publication_date:{date_from},to_publication_date:{date_to}"
    if extra_filter:
        base_filter = f"{base_filter},{extra_filter}"
    for _ in range(settings.api.max_pages):
        params: dict = {
            "filter": base_filter,
            "per-page": "200",
            "cursor": cursor,
            "sort": "relevance_score:desc" if search else "publication_date:desc",
        }
        if search:
            params["search"] = search
        if settings.openalex_api_key:
            params["api_key"] = settings.openalex_api_key
        data = _get_json(session, OPENALEX_WORKS, params, settings)
        if not data:
            break
        for work in data.get("results", []):
            paper = _openalex_to_paper(work)
            if paper:
                papers.append(paper)
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor or not data.get("results") or len(papers) >= cap:
            break
    return papers[:cap]


def _chunk_keywords(keywords: list[str], budget: int = 3000) -> list[str]:
    """Group keywords into OR-combined search strings under the URL length budget."""
    searches: list[str] = []
    current = ""
    for kw in keywords:
        term = f'"{kw}"'
        candidate = f"{current} OR {term}" if current else term
        if current and len(candidate) > budget:
            searches.append(current)
            current = term
        else:
            current = candidate
    if current:
        searches.append(current)
    return searches


def fetch_openalex(
    settings: Settings,
    *,
    date_window: tuple[str, str] | None = None,
) -> list[RawPaper]:
    session = requests.Session()
    date_from, date_to = date_window or _date_window(settings)
    papers: list[RawPaper] = []

    topics = settings.ingestion.openalex_topics
    if topics:
        topic_filter = "topics.id:" + "|".join(topics)
        got = _fetch_openalex_filter(
            session, settings, topic_filter, None, date_from, date_to
        )
        log.info("OpenAlex topic query returned %d works", len(got))
        papers.extend(got)

    for search in _chunk_keywords(settings.ingestion.keywords):
        got = _fetch_openalex_filter(
            session, settings, None, search, date_from, date_to
        )
        log.info("OpenAlex search %r returned %d works", search, len(got))
        papers.extend(got)

    return papers


# -------------------------------------------------------------- Europe PMC


def _epmc_to_paper(item: dict) -> RawPaper | None:
    try:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        authors = [
            a.strip() for a in (item.get("authorString") or "").split(",") if a.strip()
        ]
        return RawPaper(
            id=f"epmc:{item.get('id', '')}",
            doi=item.get("doi"),
            title=title,
            authors=authors,
            publication_date=item.get("firstPublicationDate")
            or item.get("firstIndexDate")
            or "",
            venue_name=item.get("journalTitle") or "preprint",
            venue_issn=(item.get("journalIssn") or "").replace("-", "") or None,
            abstract=item.get("abstractText") or "",
            pdf_url=None,
            source_type="preprint",
        )
    except Exception as exc:
        log.warning("Skipping malformed Europe PMC record: %s", exc)
        return None


def fetch_europe_pmc(
    settings: Settings,
    *,
    date_window: tuple[str, str] | None = None,
) -> list[RawPaper]:
    if not settings.ingestion.europe_pmc.enabled:
        return []
    session = requests.Session()
    date_from, date_to = date_window or _date_window(settings)
    srcs = " OR ".join(
        f'SRC:"{s}"' for s in settings.ingestion.europe_pmc.sources
    )
    kws = " OR ".join(
        f'TITLE_ABS:"{kw}"' for kw in settings.ingestion.keywords
    )
    if not kws:
        return []
    query = f"({srcs}) AND (FIRST_PDATE:[{date_from} TO {date_to}]) AND ({kws})"

    papers: list[RawPaper] = []
    cursor = "*"
    cap = settings.ingestion.max_results_per_query
    for _ in range(settings.api.max_pages):
        params = {
            "query": query,
            "format": "json",
            "pageSize": "200",
            "resultType": "core",
            "cursorMark": cursor,
        }
        data = _get_json(session, EUROPE_PMC_SEARCH, params, settings)
        if not data:
            break
        results = (data.get("resultList") or {}).get("result", [])
        for item in results:
            paper = _epmc_to_paper(item)
            if paper:
                papers.append(paper)
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor or not results:
            break
        if len(papers) >= cap:
            break
        cursor = next_cursor
    papers = papers[:cap]
    log.info("Europe PMC query returned %d preprints", len(papers))
    return papers


# ------------------------------------------------------------------ driver


def ingest(
    settings: Settings,
    *,
    cache_path: Path | None = None,
) -> list[RawPaper]:
    """Fetch and deduplicate, optionally caching with the exact query window."""
    provenance = _query_provenance(settings)
    date_window = (
        provenance["date_window"]["from"],
        provenance["date_window"]["to"],
    )
    papers = fetch_openalex(settings, date_window=date_window) + fetch_europe_pmc(
        settings, date_window=date_window
    )
    seen: set[str] = set()
    unique: list[RawPaper] = []
    for p in papers:
        if p.id in seen:
            continue
        seen.add(p.id)
        unique.append(p)
    log.info("Ingestion complete: %d unique papers", len(unique))
    if cache_path is not None:
        save_ingest_cache(
            cache_path,
            unique,
            settings,
            query_provenance=provenance,
        )
    return unique


def _query_provenance(
    settings: Settings,
    *,
    date_window: tuple[str, str] | None = None,
) -> dict:
    """Describe only settings that determine the effective API result set."""
    date_from, date_to = date_window or _date_window(settings)
    return {
        "version": QUERY_PROVENANCE_VERSION,
        "selection": {
            "openalex_topics": list(settings.ingestion.openalex_topics),
            "keywords": list(settings.ingestion.keywords),
        },
        "date_window": {"from": date_from, "to": date_to},
        "europe_pmc": {
            "enabled": settings.ingestion.europe_pmc.enabled,
            "sources": list(settings.ingestion.europe_pmc.sources),
        },
        "limits": {
            "max_results_per_query": settings.ingestion.max_results_per_query,
            "max_pages": settings.api.max_pages,
            "openalex_page_size": 200,
            "europe_pmc_page_size": 200,
            "keyword_chunk_budget": 3000,
        },
    }


def _provenance_fingerprint(provenance: dict) -> str:
    encoded = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_query_provenance(path: Path, provenance: dict) -> None:
    def malformed(detail: str) -> None:
        raise ValueError(
            f"Malformed ingest cache {path}: query provenance {detail}; remove "
            "it or rerun without --from-cache."
        )

    if set(provenance) != {
        "version",
        "selection",
        "date_window",
        "europe_pmc",
        "limits",
    }:
        malformed("has missing or unknown fields")
    if (
        not isinstance(provenance["version"], int)
        or isinstance(provenance["version"], bool)
        or provenance["version"] != QUERY_PROVENANCE_VERSION
    ):
        malformed(f"has unsupported version {provenance['version']!r}")

    selection = provenance["selection"]
    if not isinstance(selection, dict) or set(selection) != {
        "openalex_topics",
        "keywords",
    }:
        malformed("has an invalid selection")
    for field in ("openalex_topics", "keywords"):
        if not isinstance(selection[field], list) or not all(
            isinstance(value, str) for value in selection[field]
        ):
            malformed(f"selection.{field} must be a string array")

    date_window = provenance["date_window"]
    if (
        not isinstance(date_window, dict)
        or set(date_window) != {"from", "to"}
        or not all(isinstance(date_window[field], str) for field in ("from", "to"))
    ):
        malformed("has an invalid date window")

    europe_pmc = provenance["europe_pmc"]
    if (
        not isinstance(europe_pmc, dict)
        or set(europe_pmc) != {"enabled", "sources"}
        or not isinstance(europe_pmc["enabled"], bool)
        or not isinstance(europe_pmc["sources"], list)
        or not all(isinstance(value, str) for value in europe_pmc["sources"])
    ):
        malformed("has invalid Europe PMC settings")

    limits = provenance["limits"]
    expected_limit_fields = {
        "max_results_per_query",
        "max_pages",
        "openalex_page_size",
        "europe_pmc_page_size",
        "keyword_chunk_budget",
    }
    if not isinstance(limits, dict) or set(limits) != expected_limit_fields:
        malformed("has invalid query limits")
    if not all(
        isinstance(limits[field], int) and not isinstance(limits[field], bool)
        for field in expected_limit_fields
    ):
        malformed("query limits must be integers")
    if (
        limits["openalex_page_size"] != 200
        or limits["europe_pmc_page_size"] != 200
        or limits["keyword_chunk_budget"] != 3000
    ):
        malformed("does not match this query implementation")


def ingest_query_fingerprint(settings: Settings) -> str:
    """Return a deterministic fingerprint of the effective ingestion queries."""
    return _provenance_fingerprint(_query_provenance(settings))


def save_ingest_cache(
    path: Path,
    papers: list[RawPaper],
    settings: Settings | None = None,
    *,
    query_provenance: dict | None = None,
) -> None:
    """Save papers, including query provenance when settings are available."""
    records = [paper.model_dump() for paper in papers]
    if settings is None:
        if query_provenance is not None:
            raise ValueError("query_provenance requires settings")
        payload: list[dict] | dict = records
    else:
        if query_provenance is None:
            provenance = _query_provenance(settings)
        else:
            _validate_query_provenance(path, query_provenance)
            captured_window = (
                query_provenance["date_window"]["from"],
                query_provenance["date_window"]["to"],
            )
            expected = _query_provenance(settings, date_window=captured_window)
            if query_provenance != expected:
                raise ValueError(
                    "Query provenance does not match the settings used for "
                    "ingestion; refusing to write an inconsistent cache envelope."
                )
            provenance = query_provenance
        payload = {
            "schema_version": INGEST_CACHE_SCHEMA_VERSION,
            "query_provenance": provenance,
            "query_fingerprint": _provenance_fingerprint(provenance),
            "records": records,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _validate_cached_records(path: Path, records: object) -> list[RawPaper]:
    if not isinstance(records, list):
        raise ValueError(
            f"Malformed ingest cache {path}: 'records' must be a JSON array"
        )
    papers: list[RawPaper] = []
    for index, record in enumerate(records):
        try:
            papers.append(RawPaper.model_validate(record))
        except Exception as exc:
            raise ValueError(
                f"Malformed ingest cache {path}: invalid record at index {index}: "
                f"{exc}"
            ) from exc
    return papers


def load_ingest_cache(
    path: Path, settings: Settings | None = None
) -> list[RawPaper]:
    """Load a cache; explicit offline replay is allowed despite stale provenance."""
    try:
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read ingest cache {path}: {exc}") from exc

    if isinstance(payload, list):
        log.warning(
            "Legacy ingest cache %s has no query provenance; --from-cache will "
            "replay its original, unknown selection and date window.",
            path,
        )
        return _validate_cached_records(path, payload)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Malformed ingest cache {path}: expected a versioned object or "
            "legacy record array"
        )

    schema_version = payload.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != INGEST_CACHE_SCHEMA_VERSION
    ):
        if schema_version is None:
            detail = "missing 'schema_version'"
        else:
            detail = f"unsupported schema version {schema_version!r}"
        raise ValueError(
            f"Cannot load ingest cache {path}: {detail}; remove it or rerun "
            "without --from-cache."
        )
    provenance = payload.get("query_provenance")
    stored_fingerprint = payload.get("query_fingerprint")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"Malformed ingest cache {path}: missing query provenance"
        )
    _validate_query_provenance(path, provenance)
    if not isinstance(stored_fingerprint, str) or not stored_fingerprint:
        raise ValueError(
            f"Malformed ingest cache {path}: missing query fingerprint"
        )
    if _provenance_fingerprint(provenance) != stored_fingerprint:
        raise ValueError(
            f"Malformed ingest cache {path}: query provenance fingerprint "
            "does not match its metadata"
        )

    papers = _validate_cached_records(path, payload.get("records"))
    if settings is not None and stored_fingerprint != ingest_query_fingerprint(settings):
        log.warning(
            "Ingest cache provenance differs from the current query selection, "
            "date window, or query limits; --from-cache is explicitly replaying "
            "the old result set and will not refresh it."
        )
    return papers
