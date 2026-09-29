"""SQLite wrapper for FreshLit state persistence (data/cache.db)."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing, contextmanager
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

CREATE TABLE IF NOT EXISTS history_contexts (
    fingerprint TEXT PRIMARY KEY
);
"""

LEGACY_HISTORY_CONTEXT = "legacy/unknown"


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


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


def paper_seen(
    conn: sqlite3.Connection,
    paper_id: str,
    doi: str | None,
    reconsider_rejected: bool = False,
) -> bool:
    """Check ID/DOI history, optionally treating only passed papers as seen."""
    disposition_filter = " AND disposition = 'passed'" if reconsider_rejected else ""
    row = conn.execute(
        f"SELECT 1 FROM processed_papers WHERE paper_id = ?{disposition_filter}",
        (paper_id,),
    ).fetchone()
    if row:
        return True
    if doi:
        row = conn.execute(
            f"SELECT 1 FROM processed_papers WHERE doi = ?{disposition_filter}",
            (normalize_doi(doi),),
        ).fetchone()
        if row:
            return True
    return False


def fetch_known_titles(
    conn: sqlite3.Connection, reconsider_rejected: bool = False
) -> list[tuple[str, str]]:
    """Return fuzzy-dedup history, optionally limited to passed papers."""
    disposition_filter = " WHERE disposition = 'passed'" if reconsider_rejected else ""
    rows = conn.execute(
        "SELECT normalized_title, first_author FROM processed_papers"
        + disposition_filter
    ).fetchall()
    return [(r["normalized_title"] or "", r["first_author"] or "") for r in rows]


def has_processed_history(conn: sqlite3.Connection) -> bool:
    """Return whether any paper disposition history exists."""
    return conn.execute("SELECT 1 FROM processed_papers LIMIT 1").fetchone() is not None


def fetch_history_contexts(conn: sqlite3.Connection) -> set[str]:
    """Return every recorded selection-context fingerprint."""
    rows = conn.execute("SELECT fingerprint FROM history_contexts").fetchall()
    return {row["fingerprint"] for row in rows}


def record_history_context(conn: sqlite3.Connection, fingerprint: str) -> None:
    """Add a selection context without replacing prior history."""
    conn.execute(
        "INSERT OR IGNORE INTO history_contexts (fingerprint) VALUES (?)",
        (fingerprint,),
    )


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
        INSERT INTO processed_papers
            (paper_id, doi, normalized_title, first_author, processed_date,
             disposition, llm_score, final_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            doi = excluded.doi,
            normalized_title = excluded.normalized_title,
            first_author = excluded.first_author,
            processed_date = excluded.processed_date,
            disposition = excluded.disposition,
            llm_score = excluded.llm_score,
            final_score = excluded.final_score
        WHERE processed_papers.disposition <> 'passed'
           OR processed_papers.disposition IS NULL
           OR excluded.disposition = 'passed'
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
