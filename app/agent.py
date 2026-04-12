"""Agentic loop: plan → generate Lua code → validate → self-correct.

The agent owns the full pipeline:
  1. **Planning** — a lightweight LLM call that outlines the approach
     before code generation (chain-of-thought for 7B models).
  2. Few-shot retrieval via BM25.
  3. System-prompt assembly (with token-budget guard).
  4. LLM code-generation call (low temperature first, higher on retries).
  5. Code extraction from the raw model response.
  6. Two-stage validation: `luac -p` (syntax) then `lua` sandbox (runtime).
  7. Conversation-history self-correction with the error log fed back as
     the next user message.

Returns a rich `GenerateResult` so the HTTP layer can expose telemetry
(retry count, examples used, plan, validation stage, timings) alongside
the plain `code` field required by the legacy /generate contract.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal

from app.knowledge import select_examples, build_system_prompt, build_user_prompt
from app.lua_validator import extract_lua_code, validate_lua
from app.runtime_validator import validate_runtime
from app.ollama_client import Message, OllamaClient

log = logging.getLogger(__name__)

MAX_RETRIES = 2
INITIAL_TEMPERATURE = 0.1
RETRY_TEMPERATURE = 0.5

ValidationStatus = Literal[
    "ok",                # passed everything on first try
    "ok_after_retry",    # passed after one or more retries
    "exhausted",         # all retries used, returning best-effort code
]

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are a Lua coding planner for a LowCode automation platform.
Given a task description, write a brief plan (2-4 bullet points):
- Which variables/data to read (wf.vars.*, wf.initVariables.*)
- What operations to perform (loop, filter, transform, etc.)
- What the return value should be
- Any edge cases to handle (nil, empty array, type mismatch)

Rules:
- Be concise — no more than 4 lines.
- Do NOT write any Lua code — only describe the approach.
- Write the plan in English regardless of the task language.
- If the task is simple (one-liner), write a one-line plan.\
"""


async def _plan(prompt: str, client: OllamaClient) -> str:
    """Run a lightweight planning LLM call.

    Returns a short English-language plan (2-4 bullet points).
    Uses low temperature and limited num_predict to keep it cheap.
    """
    messages: list[Message] = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {prompt}"},
    ]
    try:
        raw = await client.chat(
            messages,
            temperature=0.0,
            extra_options={"num_predict": 120},
        )
        plan = raw.strip()
        log.info("Plan: %s", plan[:200])
        return plan
    except Exception as exc:
        log.warning("Planner failed (%s) — proceeding without plan", exc)
        return ""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GenerateResult:
    code: str
    retries: int
    examples_used: list[str]
    status: ValidationStatus
    llm_ms: int
    validation_ms: int
    plan: str | None = None
    last_error: str | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate(
    prompt: str,
    client: OllamaClient,
    *,
    previous_code: str | None = None,
    schema_info: str | None = None,
) -> GenerateResult:
    """Generate valid Lua code for a natural-language prompt.

    Pipeline:
      1. Plan — lightweight LLM call to outline the approach.
      2. Retrieve top-2 few-shot examples (BM25).
      3. Build the initial system/user conversation (with optional plan,
         previous_code for refinement, and schema_info for grounding).
      4. LLM code call at T=0.1 (deterministic).
      5. Extract code and run two-stage validation.
      6. On failure, append the broken reply + a corrective user message
         to `history`, raise the temperature to 0.5, retry.
      7. After MAX_RETRIES, return the best-effort code with status='exhausted'.

    Parameters
    ----------
    previous_code : optional code from a prior turn — enables multi-turn
        refinement ("add nil check", "rename variable", etc.).
    schema_info : optional data-path description inferred from a sample
        wf.vars payload — grounds generation in the real data shape.
    """
    llm_ms_total = 0

    # --- Step 1: Plan ---
    t0 = time.monotonic()
    plan = await _plan(prompt, client)
    llm_ms_total += int((time.monotonic() - t0) * 1000)

    # --- Step 2: Retrieve examples ---
    examples = select_examples(prompt, k=2)
    example_titles = [ex.title for ex in examples]
    log.info("Selected examples: %s", example_titles)

    # --- Step 3: Build conversation ---
    system_prompt = build_system_prompt(examples)
    user_prompt = build_user_prompt(
        prompt,
        plan=plan or None,
        previous_code=previous_code,
        schema_info=schema_info,
    )
    history: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    val_ms_total = 0

    # --- Attempt 0 ---
    t0 = time.monotonic()
    raw = await client.chat(history, temperature=INITIAL_TEMPERATURE)
    llm_ms_total += int((time.monotonic() - t0) * 1000)
    code = extract_lua_code(raw)

    t0 = time.monotonic()
    stage, error = _two_stage_validate(code)
    val_ms_total += int((time.monotonic() - t0) * 1000)

    if stage == "ok":
        log.info("Code valid on first attempt.")
        return GenerateResult(
            code=code,
            retries=0,
            examples_used=example_titles,
            status="ok",
            llm_ms=llm_ms_total,
            validation_ms=val_ms_total,
            plan=plan or None,
        )

    # --- Retry loop ---
    for attempt in range(1, MAX_RETRIES + 1):
        log.warning(
            "Validation failed (attempt %d/%d, stage=%s): %s",
            attempt, MAX_RETRIES, stage, error,
        )
        history.append({"role": "assistant", "content": raw})
        history.append({
            "role": "user",
            "content": _build_corrective_message(stage, error or ""),
        })

        t0 = time.monotonic()
        raw = await client.chat(history, temperature=RETRY_TEMPERATURE)
        llm_ms_total += int((time.monotonic() - t0) * 1000)
        code = extract_lua_code(raw)

        t0 = time.monotonic()
        stage, error = _two_stage_validate(code)
        val_ms_total += int((time.monotonic() - t0) * 1000)

        if stage == "ok":
            log.info("Code valid after retry %d.", attempt)
            return GenerateResult(
                code=code,
                retries=attempt,
                examples_used=example_titles,
                status="ok_after_retry",
                llm_ms=llm_ms_total,
                validation_ms=val_ms_total,
                plan=plan or None,
                last_error=None,
            )

    log.error("Exhausted %d retries. Returning best-effort code.", MAX_RETRIES)
    return GenerateResult(
        code=code,
        retries=MAX_RETRIES,
        examples_used=example_titles,
        status="exhausted",
        llm_ms=llm_ms_total,
        validation_ms=val_ms_total,
        plan=plan or None,
        last_error=error,
    )


def _two_stage_validate(code: str) -> tuple[Literal["ok", "syntax", "runtime"], str | None]:
    """Run luac -p first, then the runtime mock. Return which stage failed."""
    syntax_ok, syntax_err = validate_lua(code)
    if not syntax_ok:
        return "syntax", syntax_err
    runtime_ok, runtime_err = validate_runtime(code)
    if not runtime_ok:
        return "runtime", runtime_err
    return "ok", None


def _build_corrective_message(stage: str, error: str) -> str:
    """Construct the corrective user message fed back to the model.

    English matches the system prompt's language for consistent
    instruction-following on qwen2.5-coder.
    """
    if stage == "syntax":
        header = "Your previous code failed `luac -p` with this syntax error:"
    else:  # runtime
        header = (
            "Your previous code passed `luac -p` but raised a RUNTIME error "
            "when executed against a mock `wf.vars` sandbox:"
        )
    return (
        f"{header}\n{error}\n\n"
        "Return a corrected version in the same lua{ ... }lua format. "
        "No prelude, no markdown, no explanation — only the fixed code."
    )
