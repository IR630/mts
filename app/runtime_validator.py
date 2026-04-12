"""Runtime validator for generated Lua code.

`luac -p` only checks syntax — it happily accepts semantically-broken code
like `return wf.vars.emails[1]` when the task asked for the LAST email, or
`for _, v in pairs(arr)` where `ipairs` was required.

This module adds a second stage: it wraps the generated code with a safe
sandbox preamble that mocks common LowCode runtime structures (`wf.vars`,
`wf.initVariables`, `_utils.array.new`) and actually executes the result
with the `lua` CLI in a tight subprocess. Any runtime error (nil-index,
wrong call, arithmetic on string, …) is caught and fed back into the
agentic retry loop, giving the model a second shot with a more specific
error message than "compiled OK but result is wrong".

The sandbox intentionally does NOT provide `io`, `os`, `debug`, `dofile`,
`loadfile`, `load`, or `require` — mirrored from the security rules in
`knowledge.py`. If the model slipped one of those past the `luac` stage,
the runtime stage will fail loudly with `attempt to call a nil value`,
which is then visible in the retry prompt.

Contract:
    validate_runtime(code: str) -> tuple[bool, str | None]

Same shape as `lua_validator.validate_lua` so the two can be chained in
the agent loop.
"""

from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

_LUA_BINARIES = ("lua", "lua5.4", "lua5.3")

# Sandbox preamble: sets up minimal `wf.vars`, `wf.initVariables`, and
# `_utils` mocks, plus deletes the forbidden globals so the user code
# cannot accidentally reach them.
_SANDBOX_PREAMBLE = r"""
-- === sandbox preamble (mocks LowCode runtime) ===
io = nil
os = nil
debug = nil
coroutine = nil
package = nil
require = nil
dofile = nil
loadfile = nil
load = nil

_utils = {
    array = {
        new = function() return {} end,
    },
}

wf = {
    vars = {
        emails = {"a@example.com", "b@example.com", "c@example.com"},
        items = {10, 20, 30, 40},
        try_count_n = 3,
        arr = {"first", "second", "third"},
        response = {
            data = {status = "ok"},
        },
        parsedCsv = {
            {SKU = "A001", Discount = "10%",  Markdown = ""},
            {SKU = "A002", Discount = "",     Markdown = "20%"},
            {SKU = "A003", Discount = "",     Markdown = ""},
        },
        RESTbody = {
            result = {
                {ID = 1, ENTITY_ID = 2, CALL = "x", OTHER_KEY_1 = "y"},
            },
        },
        roles = {"admin", "editor", "viewer"},
        payload = {user = {profile = {name = "Anna"}}},
        tags = {"red", "green", "blue"},
        prices = {100, 200, 300},
        json = {
            IDOC = {
                ZCDF_HEAD = {
                    DATUM = "20231015",
                    TIME  = "153000",
                    ZCDF_PACKAGES = {
                        {items = {{sku = "A"}}},
                        {items = {sku  = "C"}},
                    },
                },
            },
        },
        status = "ok",
        -- mocks for extended example corpus
        csv_line = "apple,banana,cherry",
        defaults = {color = "red", size = 10},
        overrides = {size = 20, weight = 5},
        scores = {85, 92, 45, 78, 95, 30},
        values = {42, 17, 93, 5, 68},
        ids = {1, 3, 2, 3, 1, 4, 2},
        users = {
            {name = "Alice", age = 30},
            {name = "Bob",   age = 25},
        },
        orders = {
            {city = "Moscow", total = 100},
            {city = "SPB",    total = 200},
            {city = "Moscow", total = 150},
        },
        text = "Hello World, Hello Lua",
        age = 20,
        data = {key = "value"},
        nums = {5, 3, 8, 1, 9, 2},
        a = {1, 2, 3},
        b = {4, 5, 6},
    },
    initVariables = {
        recallTime = "2023-10-15T15:30:00+00:00",
        threshold = 100,
    },
}
-- === end preamble ===
"""


def _run_lua(code: str, timeout: float = 3.0) -> tuple[bool, str | None]:
    """Execute `code` through a `lua` interpreter and return (ok, stderr)."""
    wrapped = _SANDBOX_PREAMBLE + "\n" + code
    for binary in _LUA_BINARIES:
        try:
            result = subprocess.run(
                [binary, "-e", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return False, f"runtime timeout after {timeout}s (possible infinite loop)"
        if result.returncode == 0:
            return True, None
        return False, (result.stderr or result.stdout).strip()
    log.warning("no Lua interpreter found — skipping runtime validation")
    return True, None  # fail open if the interpreter isn't installed


def validate_runtime(code: str) -> tuple[bool, str | None]:
    """Return (True, None) if the code runs without runtime errors under
    the mock sandbox, or (False, error) otherwise. Fails OPEN (returns
    True) if no `lua` interpreter is available — the caller's syntax
    check is still the required gate.
    """
    return _run_lua(code)
