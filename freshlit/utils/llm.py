"""opencode-go (OpenAI-compatible) client + instructor wrapper.

The gateway's structured-output support varies by model, so `chat_structured`
auto-negotiates: try JSON mode first (the gateway serves thinking-mode models
that reject tool_choice), then TOOLS, then MD_JSON, and cache whichever mode
works for the rest of the run.
"""

from __future__ import annotations

import logging
import threading
from typing import TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

from .config import Settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_working_mode: instructor.Mode | None = None
_mode_lock = threading.Lock()


def build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        base_url=settings.llm.base_url,
        api_key=settings.opencode_go_api_key,
        timeout=settings.llm.timeout_seconds,
        max_retries=3,
    )


def chat_structured(
    client: OpenAI,
    settings: Settings,
    response_model: type[T],
    messages: list[dict[str, str]],
) -> T:
    """Call the chat API with schema-enforced output, negotiating the mode."""
    global _working_mode
    with _mode_lock:
        modes = (
            [_working_mode]
            if _working_mode is not None
            else [
                instructor.Mode.JSON,
                instructor.Mode.TOOLS,
                instructor.Mode.MD_JSON,
            ]
        )
    last_exc: Exception | None = None
    for mode in modes:
        try:
            iclient = instructor.from_openai(client, mode=mode)
            result = iclient.chat.completions.create(
                model=settings.llm.model,
                response_model=response_model,
                messages=messages,
                max_retries=2,
            )
            if _working_mode is None:
                with _mode_lock:
                    _working_mode = mode
                log.debug("instructor mode locked: %s", mode)
            return result
        except Exception as exc:  # try next mode
            last_exc = exc
            log.debug("instructor mode %s failed: %s", mode, exc)
    raise RuntimeError(
        f"All instructor modes failed for model {settings.llm.model}"
    ) from last_exc
