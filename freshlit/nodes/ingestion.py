"""Node 1: Ingestion engine (OpenAlex + Europe PMC -> RawPaper list)."""

from __future__ import annotations

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


def fetch_openalex(settings: Settings) -> list[RawPaper]:
    session = requests.Session()
    date_from, date_to = _date_window(settings)
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


def fetch_europe_pmc(settings: Settings) -> list[RawPaper]:
    if not settings.ingestion.europe_pmc.enabled:
        return []
    session = requests.Session()
    date_from, date_to = _date_window(settings)
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


def ingest(settings: Settings) -> list[RawPaper]:
    """Fetch from all sources; dedup identical IDs within the batch."""
    papers = fetch_openalex(settings) + fetch_europe_pmc(settings)
    seen: set[str] = set()
    unique: list[RawPaper] = []
    for p in papers:
        if p.id in seen:
            continue
        seen.add(p.id)
        unique.append(p)
    log.info("Ingestion complete: %d unique papers", len(unique))
    return unique


def save_ingest_cache(path: Path, papers: list[RawPaper]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([p.model_dump() for p in papers]))


def load_ingest_cache(path: Path) -> list[RawPaper]:
    with path.open() as fh:
        return [RawPaper.model_validate(d) for d in json.load(fh)]
