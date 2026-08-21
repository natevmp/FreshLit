"""Local SPECTER2 embeddings (lazy singleton) for the vector-similarity filter."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

_model = None


def _is_model_cached(model_name: str) -> bool:
    """Check whether a Hugging Face model is already present locally.

    Uses the default Hugging Face cache layout and respects HF_HOME/HF_HUB_CACHE.
    This check is done before importing sentence_transformers so HF_HUB_OFFLINE
    can be set before any Hugging Face library reads it at import time.
    """
    hub_root = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub"
    )
    model_dir = Path(hub_root).expanduser() / (
        "models--" + model_name.replace("/", "--")
    )
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    for weight_file in ("model.safetensors", "pytorch_model.bin"):
        if any(snapshots.glob(f"*/{weight_file}")):
            return True
    return False


def get_model(settings: Settings):
    """Lazy-load the sentence-transformers model once per process."""
    global _model
    if _model is None:
        model_name = settings.filtering.embedding_model
        if _is_model_cached(model_name):
            os.environ["HF_HUB_OFFLINE"] = "1"
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s ...", model_name)
        _model = SentenceTransformer(
            model_name,
            device=settings.filtering.embedding_device,
        )
    return _model


def embed_texts(settings: Settings, texts: list[str]) -> np.ndarray:
    """Embed texts, L2-normalized so dot product == cosine similarity."""
    model = get_model(settings)
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def cosine_to_profile(vecs: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row against the (normalized) profile vector."""
    return vecs @ profile


def build_profile_vector(settings: Settings) -> np.ndarray:
    """Embed the research-profile description and cache the normalized vector."""
    from .config import _extract_keywords  # local import to keep config simple

    profile_text = settings.research_profile_text
    keywords = _extract_keywords(profile_text)
    corpus = [profile_text, " | ".join(keywords)] if keywords else [profile_text]
    vecs = embed_texts(settings, corpus)
    profile = vecs.mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile


def save_profile_vector(path: Path, profile: np.ndarray, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_name, "vector": profile.tolist()}
    path.write_text(json.dumps(payload))


def load_profile_vector(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    vec = np.asarray(payload["vector"], dtype=np.float64)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec
