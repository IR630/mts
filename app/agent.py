"""Agentic loop: generate Lua code, validate, self-correct with history."""

import logging

from app.knowledge import select_examples, build_system_prompt, build_user_prompt
from app.lua_validator import extract_lua_code, validate_lua
from app.ollama_client import Message, OllamaClient

log = logging.getLogger(__name__)

MAX_RETRIES = 2
INITIAL_TEMPERATURE = 0.1
RETRY_TEMPERATURE = 0.5


async def generate(prompt: str, client: OllamaClient) -> str:
    """Generate valid Lua code for the given natural-language prompt.

    Implements an agentic self-correction loop:
      1. Select relevant few-shot examples (BM25).
      2. Build the conversation history (system + user).
      3. Call the LLM with low temperature (0.1).
      4. Extract and validate the Lua code.
      5. On syntax errors, append the assistant's broken reply and a corrective
         user message to the history, then retry with a higher temperature (0.5)
         to encourage different reasoning.
      6. Return the best result.
    """
    # Step 1 — retrieve relevant examples
    examples = select_examples(prompt, k=2)
    log.info("Selected examples: %s", [ex.title for ex in examples])

    # Step 2 — build the initial conversation
    system_prompt = build_system_prompt(examples)
    user_prompt = build_user_prompt(prompt)
    history: list[Message] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Step 3 — initial generation (low temperature for determinism)
    raw = await client.chat(history, temperature=INITIAL_TEMPERATURE)
    code = extract_lua_code(raw)
    log.debug("Initial code:\n%s", code)

    is_valid, error = validate_lua(code)
    if is_valid:
        log.info("Code valid on first attempt.")
        return code

    # Step 4 — retry loop with full history + higher temperature
    for attempt in range(1, MAX_RETRIES + 1):
        log.warning(
            "Validation failed (attempt %d/%d): %s",
            attempt,
            MAX_RETRIES,
            error,
        )

        # Extend the conversation history with the model's broken reply
        # and a corrective user message.
        history.append({"role": "assistant", "content": raw})
        history.append(
            {
                "role": "user",
                "content": (
                    f"Your code resulted in this syntax error:\n{error}\n\n"
                    f"Fix the error and return ONLY the corrected Lua code. "
                    f"Do NOT wrap it in markdown code fences."
                ),
            }
        )

        raw = await client.chat(history, temperature=RETRY_TEMPERATURE)
        code = extract_lua_code(raw)
        log.debug("Retry %d code:\n%s", attempt, code)

        is_valid, error = validate_lua(code)
        if is_valid:
            log.info("Code valid after retry %d.", attempt)
            return code

    # Exhausted retries — return best-effort code
    log.error("Exhausted %d retries. Returning best-effort code.", MAX_RETRIES)
    return code
