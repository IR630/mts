"""FastAPI application — /generate + session API + /metrics + static UI."""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import GenerateResult, generate as agent_generate
from app.clarifier import analyze as clarifier_analyze
from app.knowledge import infer_schema
from app.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

_client: OllamaClient | None = None

# ---------------------------------------------------------------------------
# In-memory session store (lost on restart — fine for a single-host demo)
# ---------------------------------------------------------------------------

TurnRole = Literal["user", "assistant", "clarification", "clarification_answer"]


@dataclass
class SessionTurn:
    role: TurnRole
    content: str
    # Optional telemetry (only populated on assistant turns).
    code: str | None = None
    retries: int | None = None
    examples_used: list[str] | None = None
    status: str | None = None
    llm_ms: int | None = None
    validation_ms: int | None = None


@dataclass
class Session:
    id: str
    turns: list[SessionTurn] = field(default_factory=list)
    pending_clarification: str | None = None  # stores the classifier question


_SESSIONS: dict[str, Session] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = OllamaClient(OLLAMA_HOST, OLLAMA_MODEL)
    log.info("Ollama client ready: host=%s model=%s", OLLAMA_HOST, OLLAMA_MODEL)
    if not await _client.check_health():
        log.warning(
            "Model %s not available on %s — ensure entrypoint.sh pulled it",
            OLLAMA_MODEL,
            OLLAMA_HOST,
        )
    yield
    await _client.close()
    _client = None


app = FastAPI(title="LocalScript API", version="2.0.0", lifespan=lifespan)

# Serve the minimal chat UI at /ui. The directory is created on demand when
# the UI file is first written; StaticFiles is permissive if the dir is
# empty at import time.
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR, html=True), name="ui")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    # Optional: send a sample `wf.vars` payload to ground generation in the
    # real data shape.  The agent infers available paths and feeds them into
    # the system prompt so the model stops hallucinating field names.
    sample_wf_vars: dict | None = None


class GenerateResponse(BaseModel):
    code: str
    retries: int = 0
    examples_used: list[str] = Field(default_factory=list)
    status: str = "ok"
    llm_ms: int = 0
    validation_ms: int = 0
    plan: str | None = None
    last_error: str | None = None


class SessionStartResponse(BaseModel):
    session_id: str


class SessionMessageRequest(BaseModel):
    prompt: str
    # If the user is answering a pending clarification, set this to True.
    is_clarification_answer: bool = False
    # Optional sample payload for schema-grounded generation.
    sample_wf_vars: dict | None = None


class SessionMessageResponse(BaseModel):
    kind: Literal["code", "clarification"]
    # Populated when kind == "code".
    code: str | None = None
    retries: int | None = None
    examples_used: list[str] | None = None
    status: str | None = None
    llm_ms: int | None = None
    validation_ms: int | None = None
    plan: str | None = None
    last_error: str | None = None
    # Populated when kind == "clarification".
    question: str | None = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    turns: list[dict]
    pending_clarification: str | None = None


class MetricsResponse(BaseModel):
    vram_used_mb: int | None
    vram_total_mb: int | None
    vram_peak_mb: int | None
    vram_compliant: bool | None
    model: str
    sessions_open: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_client() -> OllamaClient:
    if _client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return _client


def _result_to_response(result: GenerateResult) -> GenerateResponse:
    return GenerateResponse(
        code=result.code,
        retries=result.retries,
        examples_used=result.examples_used,
        status=result.status,
        llm_ms=result.llm_ms,
        validation_ms=result.validation_ms,
        plan=result.plan,
        last_error=result.last_error,
    )


def _nvidia_smi_vram() -> tuple[int | None, int | None]:
    """Return (used_mb, total_mb) or (None, None) if nvidia-smi is missing."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None
    if proc.returncode != 0:
        return None, None
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    try:
        used_s, total_s = (p.strip() for p in line.split(","))
        return int(used_s), int(total_s)
    except (ValueError, IndexError):
        return None, None


# Track peak across the lifetime of the process — /metrics callers see the
# highest VRAM reading we've observed, which is what the hackathon judges
# actually measure.
_vram_peak_mb: int = 0


def _update_peak(used: int | None) -> None:
    global _vram_peak_mb
    if used is not None and used > _vram_peak_mb:
        _vram_peak_mb = used


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Stateless one-shot code generation.

    Preserves the legacy contract — tests POST {"prompt": "..."} and read
    `code` out of the response. The extra telemetry fields are additive
    and default to sensible zeros, so any client that only reads `code`
    continues to work.

    Clarification is NOT applied here; use /session/* for multi-turn
    dialogue with clarification questions.
    """
    client = _require_client()
    schema_info = infer_schema(req.sample_wf_vars) if req.sample_wf_vars else None
    try:
        result = await agent_generate(
            req.prompt, client, schema_info=schema_info,
        )
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
    used, _ = _nvidia_smi_vram()
    _update_peak(used)
    return _result_to_response(result)


@app.post("/session/start", response_model=SessionStartResponse)
async def session_start() -> SessionStartResponse:
    sid = uuid.uuid4().hex[:12]
    _SESSIONS[sid] = Session(id=sid)
    log.info("Session started: %s", sid)
    return SessionStartResponse(session_id=sid)


@app.post("/session/{session_id}/message", response_model=SessionMessageResponse)
async def session_message(
    session_id: str, req: SessionMessageRequest
) -> SessionMessageResponse:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    client = _require_client()
    schema_info = infer_schema(req.sample_wf_vars) if req.sample_wf_vars else None

    # Case A: user is answering a pending clarification. Splice the
    # clarification Q+A into the prompt and skip the classifier this turn.
    if req.is_clarification_answer and session.pending_clarification:
        original_prompt = _last_user_task(session)
        combined_prompt = (
            f"{original_prompt}\n\nClarification question: "
            f"{session.pending_clarification}\nUser answer: {req.prompt}"
        )
        session.turns.append(
            SessionTurn(role="clarification_answer", content=req.prompt)
        )
        session.pending_clarification = None
        return await _run_code_generation(
            session, combined_prompt, client, schema_info=schema_info,
        )

    # Case B: multi-turn refinement — the session already has generated
    # code and the user is asking for an improvement / change.
    previous_code = _last_assistant_code(session)
    if previous_code:
        session.turns.append(SessionTurn(role="user", content=req.prompt))
        return await _run_code_generation(
            session, req.prompt, client,
            previous_code=previous_code, schema_info=schema_info,
        )

    # Case C: brand-new user task. Run the clarification classifier first.
    session.turns.append(SessionTurn(role="user", content=req.prompt))
    verdict = await clarifier_analyze(req.prompt, client)
    if verdict.needs_clarification and verdict.question:
        session.pending_clarification = verdict.question
        session.turns.append(
            SessionTurn(role="clarification", content=verdict.question)
        )
        return SessionMessageResponse(
            kind="clarification",
            question=verdict.question,
        )

    # Case D: clear task, go straight to generation.
    return await _run_code_generation(
        session, req.prompt, client, schema_info=schema_info,
    )


def _last_user_task(session: Session) -> str:
    """Walk backwards to find the most recent raw user task (not an answer)."""
    for turn in reversed(session.turns):
        if turn.role == "user":
            return turn.content
    return ""


def _last_assistant_code(session: Session) -> str | None:
    """Return the most recent assistant code, or None if no code yet."""
    for turn in reversed(session.turns):
        if turn.role == "assistant" and turn.code:
            return turn.code
    return None


async def _run_code_generation(
    session: Session,
    prompt: str,
    client: OllamaClient,
    *,
    previous_code: str | None = None,
    schema_info: str | None = None,
) -> SessionMessageResponse:
    try:
        result = await agent_generate(
            prompt, client,
            previous_code=previous_code,
            schema_info=schema_info,
        )
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
    used, _ = _nvidia_smi_vram()
    _update_peak(used)

    session.turns.append(
        SessionTurn(
            role="assistant",
            content=result.code,
            code=result.code,
            retries=result.retries,
            examples_used=result.examples_used,
            status=result.status,
            llm_ms=result.llm_ms,
            validation_ms=result.validation_ms,
        )
    )
    return SessionMessageResponse(
        kind="code",
        code=result.code,
        retries=result.retries,
        examples_used=result.examples_used,
        status=result.status,
        llm_ms=result.llm_ms,
        validation_ms=result.validation_ms,
        plan=result.plan,
        last_error=result.last_error,
    )


@app.get("/session/{session_id}", response_model=SessionHistoryResponse)
async def session_history(session_id: str) -> SessionHistoryResponse:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    return SessionHistoryResponse(
        session_id=session.id,
        turns=[asdict(t) for t in session.turns],
        pending_clarification=session.pending_clarification,
    )


@app.delete("/session/{session_id}")
async def session_delete(session_id: str) -> dict:
    if session_id not in _SESSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown session: {session_id}")
    del _SESSIONS[session_id]
    return {"deleted": session_id}


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    used, total = _nvidia_smi_vram()
    _update_peak(used)
    compliant: bool | None = None
    if total is not None:
        compliant = _vram_peak_mb < 8 * 1024
    return MetricsResponse(
        vram_used_mb=used,
        vram_total_mb=total,
        vram_peak_mb=_vram_peak_mb if _vram_peak_mb > 0 else None,
        vram_compliant=compliant,
        model=OLLAMA_MODEL,
        sessions_open=len(_SESSIONS),
    )


@app.get("/health")
async def health():
    ok = _client is not None and await _client.check_health()
    if not ok:
        raise HTTPException(status_code=503, detail="Ollama not available")
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Redirect-ish landing page pointing at /ui."""
    return HTMLResponse(
        '<html><body style="font-family:sans-serif;padding:40px">'
        '<h1>LocalScript API</h1>'
        '<p>Interactive chat: <a href="/ui">/ui</a></p>'
        '<p>API docs: <a href="/docs">/docs</a></p>'
        '<p>Metrics: <a href="/metrics">/metrics</a></p>'
        '</body></html>'
    )
