"""Extract Lua code from LLM output and validate syntax via luac."""

import re
import subprocess
import logging

log = logging.getLogger(__name__)

# Regex patterns for code extraction (ordered by priority)
_LUA_DELIM_RE = re.compile(r"lua\{(.*?)\}lua", re.DOTALL)
_MD_BLOCK_RE = re.compile(r"```(?:lua)?\s*\n?(.*?)```", re.DOTALL)

# Keywords that can legally start a Lua chunk
_CHUNK_STARTERS = ("if", "for", "while", "repeat", "local", "return", "function", "do")


def extract_lua_code(response: str) -> str:
    """Extract Lua code from the LLM response, trying several formats."""
    # 1. lua{...}lua delimiters
    m = _LUA_DELIM_RE.search(response)
    if m:
        return m.group(1).strip()

    # 2. Markdown code block
    m = _MD_BLOCK_RE.search(response)
    if m:
        return m.group(1).strip()

    # 3. Fallback — raw response stripped of leading/trailing whitespace
    code = response.strip()

    # Strip leftover markdown fences if present at start/end
    if code.startswith("```"):
        code = re.sub(r"^```(?:lua)?\s*\n?", "", code)
        code = re.sub(r"\n?```\s*$", "", code)

    return code.strip()


def _run_luac(code: str) -> tuple[bool, str | None]:
    """Run luac -p on the given code. Return (ok, error_message)."""
    try:
        result = subprocess.run(
            ["luac", "-p", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        # luac not installed — try luac5.4
        try:
            result = subprocess.run(
                ["luac5.4", "-p", "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError:
            log.warning("luac not found — skipping syntax validation")
            return True, None

    if result.returncode == 0:
        return True, None
    return False, result.stderr.strip()


def validate_lua(code: str) -> tuple[bool, str | None]:
    """Validate Lua syntax.

    Returns (True, None) on success or (False, error_message) on failure.
    If the code is a bare expression, tries prepending 'return '.
    """
    ok, err = _run_luac(code)
    if ok:
        return True, None

    # If code doesn't start with a statement keyword, try wrapping with return
    first_word = code.split(None, 1)[0] if code.strip() else ""
    if first_word.lower() not in _CHUNK_STARTERS:
        wrapped = f"return {code}"
        ok2, err2 = _run_luac(wrapped)
        if ok2:
            return True, None

    return False, err
