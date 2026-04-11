"""FastAPI application — POST /generate endpoint."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agent import generate as agent_generate
from app.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

_client: OllamaClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = OllamaClient(OLLAMA_HOST, OLLAMA_MODEL)
    log.info("Ollama client ready: host=%s model=%s", OLLAMA_HOST, OLLAMA_MODEL)

    # Model pulling is handled exclusively by entrypoint.sh before uvicorn
    # starts. We only verify availability here and log a warning if missing.
    if not await _client.check_health():
        log.warning(
            "Model %s not available on %s — ensure entrypoint.sh pulled it",
            OLLAMA_MODEL,
            OLLAMA_HOST,
        )

    yield

    await _client.close()
    _client = None


app = FastAPI(
    title="LocalScript API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Pydantic models ----------

class GenerateRequest(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    code: str


# ---------- Routes ----------

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _client is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        code = await agent_generate(req.prompt, _client)
    except Exception as exc:
        log.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
    return GenerateResponse(code=code)


@app.get("/health")
async def health():
    ok = _client is not None and await _client.check_health()
    if not ok:
        raise HTTPException(status_code=503, detail="Ollama not available")
    return {"status": "ok"}
