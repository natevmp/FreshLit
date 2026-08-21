"""SQLite wrapper for FreshLit state persistence (data/cache.db)."""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_papers (
    paper_id TEXT PRIMARY KEY,
    doi TEXT,
    normalized_title TEXT,
    first_author TEXT,
    processed_date TEXT,
    disposition TEXT, -- 'dropped_dedup', 'dropped_vector', 'dropped_llm', 'passed'
    llm_score INTEGER,
    final_score REAL
);

CREATE INDEX IF NOT EXISTS idx_doi ON processed_papers(doi);
CREATE INDEX IF NOT EXISTS idx_norm_title ON processed_papers(normalized_title);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


def normalize_doi(doi: str | None) -> str | None:
    """Lowercase, strip https://doi.org/ prefix and surrounding whitespace."""
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = d.rstrip(".;,")
    return d or None


def normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (stop-words kept simple)."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def first_author_surname(authors: list[str]) -> str:
    """Best-effort surname of the first author.

    Handles 'Last, First', 'First Last', and 'Last Initial(s)' formats
    (e.g. 'Parks M' or 'Smith JW' from Europe PMC).
    """
    if not authors:
        return ""
    first = authors[0].strip()
    if not first:
        return ""
    if "," in first:
        return first.split(",", 1)[0].strip().lower()
    parts = first.split()
    if len(parts) >= 2 and _is_initials(parts[-1]):
        return " ".join(parts[:-1]).lower()
    return parts[-1].lower()


def _is_initials(token: str) -> bool:
    return 1 <= len(token) <= 3 and token.isalpha() and token.isupper()


@contextmanager
def connect(db_path: Path):
    """Context-managed connection with implicit transaction (commits on exit)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def paper_seen(conn: sqlite3.Connection, paper_id: str, doi: str | None) -> bool:
    """Exact dedup check: primary ID, or DOI with NULL-safe fallback."""
    row = conn.execute(
        "SELECT 1 FROM processed_papers WHERE paper_id = ?", (paper_id,)
    ).fetchone()
    if row:
        return True
    if doi:
        row = conn.execute(
            "SELECT 1 FROM processed_papers WHERE doi = ?", (normalize_doi(doi),)
        ).fetchone()
        if row:
            return True
    return False


def fetch_known_titles(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (normalized_title, first_author) history for fuzzy dedup."""
    rows = conn.execute(
        "SELECT normalized_title, first_author FROM processed_papers"
    ).fetchall()
    return [(r["normalized_title"] or "", r["first_author"] or "") for r in rows]


def record_disposition(
    conn: sqlite3.Connection,
    paper_id: str,
    doi: str | None,
    title: str,
    authors: list[str],
    disposition: str,
    llm_score: int | None = None,
    final_score: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO processed_papers
            (paper_id, doi, normalized_title, first_author, processed_date,
             disposition, llm_score, final_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            normalize_doi(doi),
            normalize_title(title),
            first_author_surname(authors),
            datetime.now(timezone.utc).isoformat(),
            disposition,
            llm_score,
            final_score,
        ),
    )
