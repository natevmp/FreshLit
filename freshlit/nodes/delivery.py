"""Node 4: Obsidian delivery — render the weekly digest and write to the vault."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from ..utils.config import Settings
from .synthesis import ProcessedPaperSummary

log = logging.getLogger(__name__)

TOP_RANKED_COUNT = 3


def _escape_md_link(text: str) -> str:
    return text.replace("[", "\\[").replace("]", "\\]")


def _safe_url(url: str) -> str:
    if re.match(r"^https?://", url):
        return url
    return "#"


def _paper_block(s: ProcessedPaperSummary) -> str:
    code = s.code_data_link.strip()
    code_md = (
        f"[Repository]({_safe_url(code)})"
        if re.match(r"^https?://", code)
        else code or "None stated"
    )
    return (
        f"### [{_escape_md_link(s.title)}]({_safe_url(s.doi_url)})\n"
        f"- **Score:** `{s.final_score:.1f}/10` | **Venue:** {s.venue_and_year}"
        f" | **Authors:** {s.authors_formatted}\n"
        f"- **DOI:** `{s.doi or 'n/a'}`\n"
        f"- **Core Question:** {s.core_question}\n"
        f"- **Framework & Method:** {s.framework_and_method}\n"
        f"- **Key Finding:** {s.key_finding}\n"
        f"- **Code & Data:** {code_md}\n"
        f"- **Why It Matters:** {s.relevance_rationale}\n"
    )


def render_markdown(
    today: date,
    summaries: list[ProcessedPaperSummary],
    pulse: list[str],
    papers_analyzed: int,
    llm_model: str = "",
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
        f"llm_model: {llm_model}",
        "---",
        "",
        f"# 🔬 Literature Digest: Week {iso_week:02d}, {iso_year}",
        "",
        "## Executive Pulse",
    ]
    lines.extend(pulse if pulse else ["- No field pulse generated."])
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


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


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
    path = _target_path(settings, today)
    if filename:
        path = path.with_name(filename)
    content = render_markdown(
        today, summaries, pulse, papers_analyzed, llm_model=settings.llm.model
    )

    if dry_run:
        log.info("[dry-run] would write digest to %s", path)
        return path

    if path.exists() and not force:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with path.open("a") as fh:
            fh.write(f"\n---\n\n*Re-run {stamp}: merged {len(summaries)} papers.*\n\n")
            fh.write("\n".join(_paper_block(s) for s in summaries))
        log.warning("Digest %s already existed; appended new content", path)
        return path

    _atomic_write(path, content)
    log.info("Wrote digest: %s", path)
    return path
