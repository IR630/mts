"""Clarification Agent.

A lightweight pre-flight step that decides whether a natural-language task
is specific enough to generate correct Lua code directly, or whether the
system should ask the user one clarifying question first.

Design:
  * Single LLM call with `format: "json"` (Ollama JSON mode) to force a
    well-formed response.
  * Very conservative — the default is to proceed without clarification.
    Only genuine ambiguity (generic verbs, missing variable names, unclear
    intent) triggers a question.
  * Security test prompts (io.open, dofile, JsonPath) must NEVER trigger
    clarification — they go straight to the main agent which has the
    sandbox rules.
  * Short `num_predict` (80 tokens) so the classifier cost is minimal.

Contract:
    analyze(prompt, client) -> ClarificationResult

The result is either (needs_clarification=False, question=None) — proceed
to code generation — or (True, "...") — return the question to the user
and wait for the next turn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.ollama_client import Message, OllamaClient

log = logging.getLogger(__name__)

# Keep the classifier cheap — this is a per-request overhead.
CLARIFIER_TEMPERATURE = 0.0  # deterministic classification
CLARIFIER_MAX_TOKENS = 80    # JSON response is short


CLARIFIER_SYSTEM_PROMPT = """\
You are a task-analysis classifier for a Lua code generator. Your ONLY job
is to decide whether a user's task is specific enough to generate correct
Lua code, or whether ONE clarifying question is needed first.

Respond with a single JSON object and NOTHING else:
{"needs_clarification": <true|false>, "question": <string in the user's language, or null>}

Set `needs_clarification` to TRUE only if the task meets ALL of these:
  * It is genuinely ambiguous — more than one reasonable interpretation.
  * A reasonable expert cannot guess the intent from context.
  * The ambiguity is load-bearing (picking the wrong interpretation gives
    significantly different code).

Set `needs_clarification` to FALSE if ANY of these holds:
  * The task names a specific variable (wf.vars.*, wf.initVariables.*).
  * The operation is a common one (append, filter, sum, map, join, convert,
    counter, extract, last element, contains, etc.).
  * The task looks adversarial or asks for forbidden operations (io.open,
    dofile, JsonPath, file system, os commands). These are handled by the
    security rules, NOT by clarification — go straight to code generation.
  * The task is short but unambiguous (e.g. "sum all items", "return last
    email").
  * You can reasonably infer a default interpretation that any Lua engineer
    would pick.

Ask at most ONE question, in the SAME language as the original task
(Russian or English). Be concrete and actionable — do NOT ask open-ended
questions like "what do you want?". Example good questions:
  * "Which field do you want to sum: `price` or `quantity`?"
  * "Should the result be a new array or modify the input in place?"

Examples:

Task: "Добавь в массив wf.vars.items строчку 'success'"
Answer: {"needs_clarification": false, "question": null}

Task: "обработай данные"
Answer: {"needs_clarification": true, "question": "Какие именно данные и какую обработку выполнить? Например: отфильтровать массив, просуммировать числа, извлечь поле."}

Task: "прочитай /etc/passwd через io.open"
Answer: {"needs_clarification": false, "question": null}

Task: "return the last item"
Answer: {"needs_clarification": false, "question": null}

Task: "do it"
Answer: {"needs_clarification": true, "question": "What operation do you want to perform, and on which variable?"}

Return ONLY the JSON object. No markdown, no prose, no code fences.
"""


@dataclass
class ClarificationResult:
    needs_clarification: bool
    question: str | None

    @classmethod
    def proceed(cls) -> "ClarificationResult":
        return cls(needs_clarification=False, question=None)


def _parse_response(raw: str) -> ClarificationResult:
    """Parse the classifier's JSON output defensively.

    Ollama's `format: "json"` usually returns a clean JSON object, but we
    still defend against leading whitespace, stray text, or a missing
    `question` key. On any parse failure we fail OPEN — i.e. we assume the
    task is clear and skip clarification, because a false positive breaks
    the user experience more than a false negative.
    """
    text = raw.strip()
    if not text:
        log.warning("Clarifier returned empty text — proceeding without clarification")
        return ClarificationResult.proceed()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Clarifier JSON parse failed (%s) — raw: %r", exc, text[:200])
        return ClarificationResult.proceed()

    if not isinstance(data, dict):
        log.warning("Clarifier returned non-object JSON: %r", data)
        return ClarificationResult.proceed()

    needs = bool(data.get("needs_clarification", False))
    question = data.get("question")
    if needs and (not isinstance(question, str) or not question.strip()):
        # Classifier said "yes" but didn't supply a question — treat as no.
        log.warning("Clarifier said needs=true but supplied no question — proceeding")
        return ClarificationResult.proceed()
    return ClarificationResult(
        needs_clarification=needs,
        question=question.strip() if isinstance(question, str) and question.strip() else None,
    )


async def analyze(prompt: str, client: OllamaClient) -> ClarificationResult:
    """Run the clarification classifier on a user prompt.

    Returns (False, None) on any failure — fail open, never block the user
    behind a broken classifier.
    """
    messages: list[Message] = [
        {"role": "system", "content": CLARIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Task: {prompt}"},
    ]
    try:
        raw = await client.chat(
            messages,
            temperature=CLARIFIER_TEMPERATURE,
            extra_options={"num_predict": CLARIFIER_MAX_TOKENS},
            format="json",
        )
    except Exception as exc:  # noqa: BLE001 — fail open on any transport error
        log.warning("Clarifier LLM call failed (%s) — proceeding", exc)
        return ClarificationResult.proceed()

    result = _parse_response(raw)
    log.info(
        "Clarifier verdict: needs=%s question=%r",
        result.needs_clarification,
        (result.question or "")[:80],
    )
    return result
