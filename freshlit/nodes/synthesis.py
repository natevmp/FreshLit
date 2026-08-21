"""Node 3: LLM synthesis — structured per-paper summaries + field pulse."""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from ..utils.config import Settings
from ..utils.llm import chat_structured
from .filtering import ScoredPaper

log = logging.getLogger(__name__)


class PaperExtraction(BaseModel):
    core_question: str = Field(..., description="1 sentence on central question")
    framework_and_method: str = Field(
        ..., description="Specific mathematical/computational/empirical method"
    )
    key_finding: str = Field(
        ..., description="Main result including quantitative metrics"
    )
    code_data_link: str = Field(
        default="None stated", description="GitHub or dataset URL if present"
    )


class PaperExtractionBatch(BaseModel):
    papers: list[PaperExtraction]


class ProcessedPaperSummary(BaseModel):
    paper_id: str
    title: str
    authors_formatted: str
    venue_and_year: str
    doi_url: str
    doi: str
    final_score: float
    core_question: str
    framework_and_method: str
    key_finding: str
    code_data_link: str = "None stated"
    relevance_rationale: str


class FieldPulse(BaseModel):
    trends: list[str] = Field(
        ..., description="Exactly 3 bullet points on macro trends/overlapping methods"
    )


EXTRACTION_SYSTEM = """You extract structured summaries of scientific papers for a \
mathematical/computational biologist working on somatic evolution and clonal \
dynamics. Be specific: name the actual mathematical/computational method, and \
include quantitative results when the abstract gives them. If no code/data link \
is visible in the abstract, use "None stated"."""

BATCH_EXTRACTION_SYSTEM = EXTRACTION_SYSTEM + """

Summarize EVERY paper listed in the user message. Return an object with a
"papers" array containing one summary per paper, in the SAME ORDER as listed."""


def _authors_formatted(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    return f"{authors[0]} et al."


def _doi_url(paper) -> str:
    if paper.doi:
        d = paper.doi.strip()
        d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
        return f"https://doi.org/{d}"
    return paper.id


def _raw_doi(paper) -> str:
    if paper.doi:
        d = paper.doi.strip()
        d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.IGNORECASE)
        return d
    return ""


def _to_summary(scored: ScoredPaper, extraction: PaperExtraction) -> ProcessedPaperSummary:
    paper = scored.raw_paper
    year = (paper.publication_date or "")[:4] or "n.d."
    venue = paper.venue_name or "preprint"
    return ProcessedPaperSummary(
        paper_id=paper.id,
        title=paper.title,
        authors_formatted=_authors_formatted(paper.authors),
        venue_and_year=f"{venue} ({year})",
        doi_url=_doi_url(paper),
        doi=_raw_doi(paper),
        final_score=scored.final_score,
        core_question=extraction.core_question,
        framework_and_method=extraction.framework_and_method,
        key_finding=extraction.key_finding,
        code_data_link=extraction.code_data_link,
        relevance_rationale=scored.llm_eval.fit_rationale,
    )


def summarize_paper(
    client, settings: Settings, scored: ScoredPaper
) -> ProcessedPaperSummary | None:
    paper = scored.raw_paper
    try:
        extraction = chat_structured(
            client,
            settings,
            PaperExtraction,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Title: {paper.title}\n"
                        f"Venue: {paper.venue_name or 'unknown'}\n"
                        f"Abstract: {paper.abstract[:6000]}"
                    ),
                },
            ],
        )
    except Exception as exc:
        log.warning("Synthesis failed for %s: %s", paper.id, exc)
        return None
    return _to_summary(scored, extraction)


def _summarize_chunk(
    client, settings: Settings, chunk: list[ScoredPaper]
) -> list[ProcessedPaperSummary | None]:
    """Summarize one chunk in a single request; fall back to per-paper on error."""
    try:
        items = "\n\n".join(
            f"PAPER {idx}:\nTitle: {s.raw_paper.title}\n"
            f"Venue: {s.raw_paper.venue_name or 'unknown'}\n"
            f"Abstract: {s.raw_paper.abstract[:2500]}"
            for idx, s in enumerate(chunk, 1)
        )
        batch = chat_structured(
            client,
            settings,
            PaperExtractionBatch,
            messages=[
                {"role": "system", "content": BATCH_EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": f"Summarize these {len(chunk)} papers:\n\n{items}",
                },
            ],
        )
        exts = batch.papers
        return [
            _to_summary(chunk[idx], exts[idx]) if idx < len(exts) else None
            for idx in range(len(chunk))
        ]
    except Exception as exc:
        log.warning("Batch synthesis failed (%s); falling back per-paper", exc)
        return [summarize_paper(client, settings, s) for s in chunk]


def _summarize_batch(
    client, settings: Settings, scored_list: list[ScoredPaper]
) -> list[ProcessedPaperSummary | None]:
    """Summarize papers in chunks, run concurrently at `llm.workers`."""
    from concurrent.futures import ThreadPoolExecutor

    batch_size = max(1, settings.filtering.llm_batch_size)
    chunks = [
        scored_list[i : i + batch_size]
        for i in range(0, len(scored_list), batch_size)
    ]
    results: list[ProcessedPaperSummary | None] = []
    workers = max(1, min(settings.llm.workers, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk_result in pool.map(
            lambda c: _summarize_chunk(client, settings, c), chunks
        ):
            results.extend(chunk_result)
    return results


PULSE_SYSTEM = """You write an 'Executive Pulse' for a weekly literature digest. Given the \
structured summaries of this week's selected papers, produce exactly 3 bullet \
points characterizing macro trends or overlapping methodologies in the batch. \
Each bullet must start with a bold trend label like '- **Trend label:** ...'."""


def generate_field_pulse(
    client, settings: Settings, summaries: list[ProcessedPaperSummary]
) -> list[str]:
    if not summaries:
        return []
    digest = "\n\n".join(
        f"- {s.title}: {s.core_question} | {s.framework_and_method} | "
        f"{s.key_finding}"
        for s in summaries
    )
    try:
        pulse = chat_structured(
            client,
            settings,
            FieldPulse,
            messages=[
                {"role": "system", "content": PULSE_SYSTEM},
                {"role": "user", "content": f"This week's papers:\n{digest}"},
            ],
        )
        return pulse.trends[:3]
    except Exception as exc:
        log.warning("Field pulse generation failed: %s", exc)
        return []


def synthesize(
    client, settings: Settings, qualified: list[ScoredPaper]
) -> tuple[list[ProcessedPaperSummary], list[str]]:
    summaries = [
        s for s in _summarize_batch(client, settings, qualified) if s is not None
    ]
    log.info("Synthesis complete: %d summaries", len(summaries))
    pulse = generate_field_pulse(client, settings, summaries)
    return summaries, pulse
