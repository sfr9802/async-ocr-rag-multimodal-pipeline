"""Shared helpers for the Phase 9 dataset synthesis scripts.

Three primitives used everywhere:

  * ``RateLimiter`` — minimum-interval sleep-based limiter. Simple enough
    to audit, sufficient for single-process generation (we never run
    multiple producers against the same provider simultaneously).

  * ``ResumableJsonlWriter`` — append-only JSONL writer that knows how
    to skip entries whose unique key is already present in the file.
    Every long-running generator (corpus, queries, OCR pages, charts)
    uses this so a crash halfway through doesn't mean restarting from
    scratch. Key extraction is caller-supplied to keep the writer
    agnostic to the row shape.

  * ``GenerationLog`` — append-only JSONL audit trail of every LLM
    call. Fields: timestamp, script, provider, model, prompt_tokens,
    completion_tokens, latency_ms, seed, status, note. Provenance is
    the whole point — all of Phase 9 must be reproducible from this
    log plus the script arguments.

Two provider factories:

  * ``load_anthropic_client()`` — reads ``ANTHROPIC_API_KEY`` or
    ``AIPIPELINE_WORKER_ANTHROPIC_API_KEY`` (Phase 4 worker env var).
    Returns a ready ``anthropic.Anthropic`` client.

  * ``ollama_chat_json()`` — thin wrapper around ``POST /api/chat`` with
    ``format=json``. Mirrors ``OllamaChatProvider.chat_json`` but lives
    here so the dataset scripts don't pull in the full worker runtime
    (Pydantic settings, capability registry, etc). Honors the same env
    vars.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class RateLimiter:
    """Minimum-interval limiter: no more than ``1 / rate_per_sec`` calls/sec.

    ``rate_per_sec <= 0`` disables the limiter. Thread-unsafe on purpose —
    the dataset scripts are single-process.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._last_call_at: float = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0.0:
            return
        elapsed = time.monotonic() - self._last_call_at
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call_at = time.monotonic()


# ---------------------------------------------------------------------------
# Resumable JSONL writing
# ---------------------------------------------------------------------------


class ResumableJsonlWriter:
    """Append-only JSONL writer with a caller-supplied dedup key.

    Reads existing rows on construction, extracts their keys via
    ``key_fn``, and exposes ``has(key)`` so callers can skip re-generating
    an already-written row. ``append(row)`` flushes immediately — we
    prefer durability over throughput; the generators make one expensive
    LLM call per row and a crash just after that call must not lose the
    result.
    """

    def __init__(
        self,
        path: Path,
        *,
        key_fn: Callable[[Dict[str, Any]], str],
    ) -> None:
        self._path = path
        self._key_fn = key_fn
        self._keys: set[str] = set()
        if path.exists():
            for row in _read_jsonl(path):
                try:
                    self._keys.add(key_fn(row))
                except Exception:  # noqa: BLE001
                    continue
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def existing_keys(self) -> set[str]:
        return set(self._keys)

    def has(self, key: str) -> bool:
        return key in self._keys

    def append(self, row: Dict[str, Any]) -> None:
        key = self._key_fn(row)
        if key in self._keys:
            return
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")
            fp.flush()
        self._keys.add(key)

    def drop(self, key: str) -> bool:
        """Remove every row whose key equals ``key`` by rewriting the file.

        O(n) in the number of rows; only used for rare corrections (e.g.
        the diversity-guard regenerating a duplicate doc). Returns True
        if a row was actually removed.
        """
        if key not in self._keys:
            return False
        kept: list[Dict[str, Any]] = []
        if self._path.exists():
            for row in _read_jsonl(self._path):
                try:
                    if self._key_fn(row) == key:
                        continue
                except Exception:  # noqa: BLE001
                    pass
                kept.append(row)
        with self._path.open("w", encoding="utf-8") as fp:
            for row in kept:
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
        self._keys.discard(key)
        return True


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as ex:
                log.warning("Skipping malformed JSONL line in %s: %s", path.name, ex)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Eager-load a JSONL file into a list of dicts. Comment lines are skipped."""
    return list(_read_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], *, header: Optional[str] = None) -> int:
    """Overwrite ``path`` with ``rows`` (and optional leading ``#`` header)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        if header:
            for line in header.splitlines():
                fp.write("# " + line + "\n")
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Generation log (provenance audit trail)
# ---------------------------------------------------------------------------


@dataclass
class GenerationLog:
    """Append-only JSONL audit of every LLM call.

    Every generator script creates one and records an entry per model
    invocation (success or failure). The log lives next to the dataset
    it describes so the provenance stays with the data.
    """

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        *,
        script: str,
        provider: str,
        model: str,
        status: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        seed: Optional[int] = None,
        note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "ts": time.time(),
            "script": script,
            "provider": provider,
            "model": model,
            "status": status,
        }
        if prompt_tokens is not None:
            entry["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            entry["completion_tokens"] = int(completion_tokens)
        if latency_ms is not None:
            entry["latency_ms"] = round(float(latency_ms), 2)
        if seed is not None:
            entry["seed"] = int(seed)
        if note:
            entry["note"] = note
        if extra:
            entry["extra"] = extra
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False))
            fp.write("\n")


@contextmanager
def log_call(
    log_: GenerationLog,
    *,
    script: str,
    provider: str,
    model: str,
    seed: Optional[int] = None,
    note: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Context manager that records success/failure + latency automatically.

    Usage::

        with log_call(gen_log, script="build_corpus", provider="claude",
                       model="claude-sonnet-4-6", seed=seed) as slot:
            resp = client.messages.create(...)
            slot["prompt_tokens"] = resp.usage.input_tokens
            slot["completion_tokens"] = resp.usage.output_tokens
    """
    slot: Dict[str, Any] = {}
    started_at = time.perf_counter()
    try:
        yield slot
    except Exception as ex:  # noqa: BLE001
        log_.log(
            script=script,
            provider=provider,
            model=model,
            status=f"error:{type(ex).__name__}",
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            seed=seed,
            note=note or str(ex)[:200],
        )
        raise
    else:
        log_.log(
            script=script,
            provider=provider,
            model=model,
            status="ok",
            prompt_tokens=slot.get("prompt_tokens"),
            completion_tokens=slot.get("completion_tokens"),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
            seed=seed,
            note=note,
        )


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------


def load_anthropic_client(
    *,
    api_key: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> Any:
    """Construct an ``anthropic.Anthropic`` client.

    API key resolution order:
      1. explicit ``api_key`` argument
      2. ``ANTHROPIC_API_KEY`` env var
      3. ``AIPIPELINE_WORKER_ANTHROPIC_API_KEY`` (Phase 4 worker env var)
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
        "AIPIPELINE_WORKER_ANTHROPIC_API_KEY"
    )
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY or "
            "AIPIPELINE_WORKER_ANTHROPIC_API_KEY in the environment."
        )
    try:
        import anthropic
    except ImportError as ex:  # pragma: no cover
        raise RuntimeError(
            "anthropic SDK not installed. Run `pip install -r requirements.txt`."
        ) from ex
    return anthropic.Anthropic(api_key=key, timeout=timeout_seconds)


def claude_json_call(
    client: Any,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """One Claude ``messages.create`` call returning a parsed JSON dict.

    The caller is responsible for tailoring ``system`` / ``user`` such
    that the model emits a JSON object. We strip ```` ```json ```` code
    fences when present so the generator prompts can stay readable.
    Returns the parsed dict alongside a ``_usage`` key carrying the
    prompt/completion token counts from the Anthropic response.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw_text = ""
    for block in response.content or []:
        if hasattr(block, "text"):
            raw_text += block.text
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = _strip_code_fence(raw_text)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as ex:
        raise ClaudeResponseError(
            f"Claude returned non-JSON content: {ex}; body: {raw_text[:200]}"
        ) from ex
    if not isinstance(parsed, dict):
        raise ClaudeResponseError(
            f"Claude JSON response must be an object; got {type(parsed).__name__}."
        )
    usage = getattr(response, "usage", None)
    parsed["_usage"] = {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }
    return parsed


class ClaudeResponseError(Exception):
    """Raised when Claude returns content the caller can't parse as JSON."""


def _strip_code_fence(text: str) -> str:
    """Strip a leading ```json / trailing ``` fence pair if present."""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gemma4:e2b"
    timeout_s: float = 60.0
    keep_alive: str = "30m"


def ollama_config_from_env() -> OllamaConfig:
    return OllamaConfig(
        base_url=os.environ.get("AIPIPELINE_WORKER_OLLAMA_URL", "http://localhost:11434"),
        model=os.environ.get("AIPIPELINE_WORKER_OLLAMA_MODEL", "gemma4:e2b"),
        timeout_s=float(os.environ.get("AIPIPELINE_WORKER_OLLAMA_TIMEOUT_S", "60")),
        keep_alive=os.environ.get("AIPIPELINE_WORKER_OLLAMA_KEEP_ALIVE", "30m"),
    )


def ollama_chat_json(
    cfg: OllamaConfig,
    *,
    system: str,
    user: str,
    schema_hint: str,
    max_tokens: int = 1024,
    temperature: float = 0.4,
) -> Dict[str, Any]:
    """Call Ollama ``/api/chat`` with ``format=json`` and return a dict."""
    import httpx

    messages = [
        {"role": "system", "content": system + "\n\nRespond ONLY with JSON: " + schema_hint},
        {"role": "user", "content": user},
    ]
    payload = {
        "model": cfg.model,
        "messages": messages,
        "format": "json",
        "stream": False,
        "keep_alive": cfg.keep_alive,
        "options": {"temperature": float(temperature), "num_predict": int(max_tokens)},
    }
    url = f"{cfg.base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=cfg.timeout_s) as client:
        resp = client.post(url, json=payload)
    resp.raise_for_status()
    body = resp.json()
    content = (body.get("message") or {}).get("content") or ""
    content = content.strip()
    if not content:
        raise RuntimeError("Ollama returned empty content.")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"Ollama returned non-JSON content: {ex}; body: {content[:200]}") from ex
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Ollama JSON response must be an object; got {type(parsed).__name__}.")
    parsed["_usage"] = {
        "input_tokens": int(body.get("prompt_eval_count") or 0),
        "output_tokens": int(body.get("eval_count") or 0),
    }
    return parsed


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool = False) -> None:
    """Standard logging setup for the dataset scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def stable_seed(*parts: Any) -> int:
    """Produce a deterministic 32-bit seed from positional parts.

    Used so every doc_id / query pair gets the same seed across runs —
    lets the generators be genuinely reproducible even when the caller
    resumes partway through.
    """
    import hashlib

    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int.from_bytes(digest, "big")


__all__ = [
    "ClaudeResponseError",
    "GenerationLog",
    "OllamaConfig",
    "RateLimiter",
    "ResumableJsonlWriter",
    "claude_json_call",
    "configure_logging",
    "load_anthropic_client",
    "log_call",
    "ollama_chat_json",
    "ollama_config_from_env",
    "read_jsonl",
    "stable_seed",
    "write_jsonl",
]
