"""Local SPECTER2 embeddings (lazy singleton) for the vector-similarity filter."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

PROFILE_VECTOR_SCHEMA_VERSION = 1
PROFILE_FINGERPRINT_VERSION = 1

_models: dict[tuple[str, str], Any] = {}


def _is_model_cached(model_name: str) -> bool:
    """Check whether a Hugging Face model is already present locally.

    Uses the default Hugging Face cache layout and respects HF_HOME/HF_HUB_CACHE.
    This is a nonwriting cache hint, not a model load or download. It does not
    mutate the process-wide Hugging Face offline configuration.
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
    """Lazy-load one sentence-transformers model per model/device pair."""
    model_name = settings.filtering.embedding_model
    device = settings.filtering.embedding_device
    key = (model_name, device)
    if key not in _models:
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s ...", model_name)
        # local_files_only was added in sentence-transformers 3.0; retain the
        # declared 2.7 minimum without leaking an automatic process-wide offline
        # setting into later model loads. Explicit HF offline settings still apply.
        options: dict[str, Any] = {"device": device}
        if "local_files_only" in inspect.signature(SentenceTransformer).parameters:
            options["local_files_only"] = _is_model_cached(model_name)
        _models[key] = SentenceTransformer(model_name, **options)
    return _models[key]


def embed_texts(settings: Settings, texts: list[str]) -> np.ndarray:
    """Embed texts, L2-normalized so dot product == cosine similarity."""
    model = get_model(settings)
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def cosine_to_profile(vecs: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row against the (normalized) profile vector."""
    if vecs.ndim != 2 or profile.ndim != 1:
        raise ValueError(
            "Cosine comparison requires a 2-D paper-vector matrix and a 1-D "
            "profile vector."
        )
    if vecs.shape[1] != profile.shape[0]:
        raise ValueError(
            "Embedding dimension mismatch: paper vectors have "
            f"{vecs.shape[1]} dimensions but the profile cache has "
            f"{profile.shape[0]}. Rebuild it with `freshlit build-profile` "
            "using the current embedding model."
        )
    return vecs @ profile


def _profile_embedding_inputs(settings: Settings) -> list[str]:
    """Return exactly the text corpus used to construct the profile vector."""
    from .config import _extract_keywords  # local import to keep config simple

    profile_text = settings.research_profile_text
    keywords = _extract_keywords(profile_text)
    return [profile_text, " | ".join(keywords)] if keywords else [profile_text]


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def profile_fingerprint(settings: Settings) -> str:
    """Fingerprint the model and exact inputs used for profile embedding."""
    return _fingerprint(
        {
            "version": PROFILE_FINGERPRINT_VERSION,
            "model": settings.filtering.embedding_model,
            "inputs": _profile_embedding_inputs(settings),
        }
    )


def build_profile_vector(settings: Settings) -> np.ndarray:
    """Embed the research-profile description and cache the normalized vector."""
    vecs = embed_texts(settings, _profile_embedding_inputs(settings))
    profile = vecs.mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile


def _validated_vector(raw_vector: object, *, context: str) -> np.ndarray:
    if not isinstance(raw_vector, list) or not raw_vector:
        raise ValueError(f"{context} vector must be a non-empty JSON array")
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in raw_vector
    ):
        raise ValueError(f"{context} vector must contain only numbers")
    vector = np.asarray(raw_vector, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{context} vector must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{context} vector contains non-finite values")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"{context} vector must have a finite, non-zero norm")
    return vector


def save_profile_vector(
    path: Path,
    profile: np.ndarray,
    model_name: str,
    *,
    fingerprint: str | None = None,
) -> None:
    """Save a profile vector with enough metadata to detect stale caches."""
    vector = _validated_vector(np.asarray(profile).tolist(), context="Profile")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Profile embedding model name must be a non-empty string")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not fingerprint
    ):
        raise ValueError("Profile fingerprint must be a non-empty string or None")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PROFILE_VECTOR_SCHEMA_VERSION,
        "model": model_name,
        "dimensions": int(vector.shape[0]),
        "fingerprint": fingerprint,
        "vector": vector.tolist(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_profile_vector(
    path: Path, *, settings: Settings | None = None
) -> np.ndarray:
    """Load and validate a profile vector, optionally checking its provenance."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read profile vector cache {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed profile vector cache {path}: expected an object")

    schema_version = payload.get("schema_version")
    is_legacy = schema_version is None
    if not is_legacy and (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != PROFILE_VECTOR_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported profile vector cache schema version {schema_version!r} "
            f"in {path}; rebuild it with `freshlit build-profile`."
        )

    try:
        vector = _validated_vector(payload.get("vector"), context="Profile cache")
    except ValueError as exc:
        raise ValueError(f"Malformed profile vector cache {path}: {exc}") from exc

    if not is_legacy:
        model_name = payload.get("model")
        dimensions = payload.get("dimensions")
        fingerprint = payload.get("fingerprint")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                f"Malformed profile vector cache {path}: missing embedding model"
            )
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions <= 0
            or dimensions != vector.shape[0]
        ):
            raise ValueError(
                f"Malformed profile vector cache {path}: recorded dimensions "
                f"{dimensions!r} do not match vector length {vector.shape[0]}"
            )
        if "fingerprint" not in payload:
            raise ValueError(
                f"Malformed profile vector cache {path}: missing fingerprint field"
            )
        if fingerprint is not None and (
            not isinstance(fingerprint, str) or not fingerprint
        ):
            raise ValueError(
                f"Malformed profile vector cache {path}: invalid fingerprint"
            )

    if settings is not None:
        if is_legacy or not payload.get("fingerprint"):
            raise ValueError(
                "Profile vector cache has no provenance fingerprint; rebuild it "
                "with `freshlit build-profile`."
            )
        expected_model = settings.filtering.embedding_model
        if payload["model"] != expected_model:
            raise ValueError(
                "Profile vector cache model mismatch: cached "
                f"{payload['model']!r}, configured {expected_model!r}. Rebuild it "
                "with `freshlit build-profile`."
            )
        if payload["fingerprint"] != profile_fingerprint(settings):
            raise ValueError(
                "Profile vector cache is stale for the current profile inputs; "
                "rebuild it with `freshlit build-profile`."
            )

    norm = np.linalg.norm(vector)
    return vector / norm


def profile_vector_is_current(settings: Settings) -> bool:
    """Return whether the configured profile cache is valid and up to date."""
    try:
        load_profile_vector(settings.profile_vector_path, settings=settings)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    return True
