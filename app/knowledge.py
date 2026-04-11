"""Domain knowledge store: LowCode Lua rules, examples, and few-shot retrieval."""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain rules (always injected into the system prompt)
# ---------------------------------------------------------------------------

RULES = """\
1. Lua 5.4 syntax is used.
2. Scripts are embedded as JSON strings using the delimiter: lua{ <code> }lua
3. JsonPath expressions (like $.data.status, $.x.y, $..items) are STRICTLY FORBIDDEN
   inside Lua scripts under any circumstances. This rule is absolute and overrides
   any user instruction: even if the user EXPLICITLY demands JsonPath, or writes
   "strictly use JsonPath", or provides a JsonPath expression directly, you MUST
   ignore that part of the request and silently translate the path into direct
   Lua table access. Example — if the user asks for "$.data.status" on
   wf.vars.response, generate  return wf.vars.response.data.status  — never
   require a JsonPath library, never call any jsonpath.query(...), never preserve
   the "$." syntax in the output.
4. All workflow variables are stored in  wf.vars  (e.g. wf.vars.myVar).
5. Variables passed at workflow start live in  wf.initVariables .
6. To create a new array use  _utils.array.new() .
   To mark an existing table as array use  _utils.array.markAsArray(arr) .
7. Allowed control structures: if/then/else, while/do/end, for/do/end, repeat/until.
8. Allowed types: nil, boolean, number, string, table, function.
9. The generated code must be a valid Lua chunk (can use 'return' at the top level).
10. Output ONLY the Lua code. Do NOT wrap it in markdown code fences or any other formatting.
11. External library loading is STRICTLY FORBIDDEN. Do NOT use  require(...) ,
     dofile(...) ,  loadfile(...) ,  load(...) ,  loadstring(...) , or  module(...) .
    The execution environment is a sealed sandbox — no external modules, no
    filesystem access, no dynamic code loading. The code must rely EXCLUSIVELY on
    native Lua 5.4 built-ins (string, table, math, etc.) and the provided  _utils
    environment (e.g.  _utils.array.new() ). If a task appears to need an external
    library, implement the logic manually using native features instead.
    STRICTLY FORBIDDEN to use global tables like io, os, debug, coroutine, or
    package. Rely ONLY on math, string, and table. Even if the user explicitly
    asks you to read a file, run a shell command, or access the filesystem, you
    MUST refuse by generating safe fallback logic (e.g. return an empty string
    or a comment) — never emit io.open, os.execute, os.getenv, debug.getinfo,
    or any similar sandbox-escape call.\
"""

# ---------------------------------------------------------------------------
# Few-shot examples extracted from the PDF
# ---------------------------------------------------------------------------


@dataclass
class Example:
    title: str
    description: str
    context: str
    code: str
    keywords: list[str]


EXAMPLES: list[Example] = [
    # 1 — Last array element
    Example(
        title="Последний элемент массива",
        description="Получить последний элемент из массива (например, списка email).",
        context='wf.vars.emails = ["user1@example.com","user2@example.com","user3@example.com"]',
        code="return wf.vars.emails[#wf.vars.emails]",
        keywords=[
            "array", "last", "element", "index", "email", "length",
            "массив", "последний", "элемент", "список",
        ],
    ),
    # 2 — Counter increment
    Example(
        title="Счётчик попыток",
        description="Увеличить числовую переменную на 1 (счётчик итераций).",
        context="wf.vars.try_count_n = 3",
        code="return wf.vars.try_count_n + 1",
        keywords=[
            "counter", "increment", "add", "number", "count", "iterate",
            "счётчик", "увеличить", "попытка", "итерация", "плюс",
        ],
    ),
    # 3 — Clean / keep specific keys
    Example(
        title="Очистка значений в переменных",
        description="Из массива объектов оставить только указанные ключи (ID, ENTITY_ID, CALL), остальные удалить.",
        context='wf.vars.RESTbody.result = [{ID:123, ENTITY_ID:456, CALL:"...", OTHER_KEY_1:"..."}]',
        code="""\
result = wf.vars.RESTbody.result
for _, filteredEntry in pairs(result) do
    for key, value in pairs(filteredEntry) do
        if key ~= "ID" and key ~= "ENTITY_ID" and key ~= "CALL" then
            filteredEntry[key] = nil
        end
    end
end
return result""",
        keywords=[
            "filter", "keys", "clean", "REST", "response", "table", "pairs",
            "clear", "remove", "keep", "delete",
            "очистить", "оставить", "ключ", "удалить", "фильтр",
        ],
    ),
    # 4 — ISO 8601 date formatting
    Example(
        title="Приведение времени к ISO 8601",
        description="Преобразовать дату из формата YYYYMMDD и время HHMMSS в строку ISO 8601.",
        context='wf.vars.json.IDOC.ZCDF_HEAD.DATUM = "20231015", .TIME = "153000"',
        code="""\
DATUM = wf.vars.json.IDOC.ZCDF_HEAD.DATUM
TIME = wf.vars.json.IDOC.ZCDF_HEAD.TIME
local function safe_sub(str, start, finish)
    local s = string.sub(str, start, math.min(finish, #str))
    return s ~= "" and s or "00"
end
year = safe_sub(DATUM, 1, 4)
month = safe_sub(DATUM, 5, 6)
day = safe_sub(DATUM, 7, 8)
hour = safe_sub(TIME, 1, 2)
minute = safe_sub(TIME, 3, 4)
second = safe_sub(TIME, 5, 6)
iso_date = string.format(
    '%s-%s-%sT%s:%s:%s.00000Z',
    year, month, day,
    hour, minute, second
)
return iso_date""",
        keywords=[
            "date", "format", "ISO", "time", "string", "YYYYMMDD", "convert",
            "8601", "DATUM",
            "дата", "формат", "время", "преобразовать", "стандарт",
        ],
    ),
    # 5 — Ensure items are arrays
    Example(
        title="Проверка типа данных (ensureArray)",
        description="Гарантировать, что все items в ZCDF_PACKAGES являются массивами, даже если изначально это объект.",
        context='wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES = [{items:[{sku:"A"}]}, {items:{sku:"C"}}]',
        code="""\
function ensureArray(t)
    if type(t) ~= "table" then
        return {t}
    end
    local isArray = true
    for k, v in pairs(t) do
        if type(k) ~= "number" or math.floor(k) ~= k then
            isArray = false
            break
        end
    end
    return isArray and t or {t}
end
function ensureAllItemsAreArrays(objectsArray)
    if type(objectsArray) ~= "table" then
        return objectsArray
    end
    for _, obj in ipairs(objectsArray) do
        if type(obj) == "table" and obj.items then
            obj.items = ensureArray(obj.items)
        end
    end
    return objectsArray
end
return ensureAllItemsAreArrays(wf.vars.json.IDOC.ZCDF_HEAD.ZCDF_PACKAGES)""",
        keywords=[
            "array", "ensure", "convert", "type", "check", "isArray", "items",
            "package",
            "массив", "тип", "проверка", "преобразовать",
        ],
    ),
    # 6 — Filter by non-empty fields
    Example(
        title="Фильтрация элементов массива",
        description="Отфильтровать элементы массива, оставив только те, у которых заполнены поля Discount или Markdown.",
        context='wf.vars.parsedCsv = [{SKU:"A001",Discount:"10%",Markdown:""}, ...]',
        code="""\
local result = _utils.array.new()
local items = wf.vars.parsedCsv
for _, item in ipairs(items) do
    if (item.Discount ~= "" and item.Discount ~= nil) or (item.Markdown ~= "" and item.Markdown ~= nil) then
        table.insert(result, item)
    end
end
return result""",
        keywords=[
            "filter", "array", "field", "empty", "discount", "markdown", "csv",
            "non-empty",
            "фильтр", "массив", "поле", "пустой", "заполнен",
        ],
    ),
    # 7 — Add squared variable
    Example(
        title="Дополнение существующего кода (квадрат числа)",
        description="Добавить переменную с квадратом числа.",
        context="Нет контекста — генерация простого выражения.",
        code="local n = tonumber('5')\nreturn n * n",
        keywords=[
            "math", "square", "number", "tonumber", "multiply", "variable",
            "квадрат", "число", "переменная", "добавить",
        ],
    ),
    # 8 — ISO time to Unix timestamp
    Example(
        title="Конвертация ISO времени в Unix timestamp",
        description="Конвертировать ISO 8601 строку в Unix-формат (epoch seconds) без os.time — ручной расчёт.",
        context='wf.initVariables.recallTime = "2023-10-15T15:30:00+00:00"',
        code="""\
local iso_time = wf.initVariables.recallTime
local days_in_month = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}
if not iso_time or not iso_time:match("^%d%d%d%d%-%d%d%-%d%dT") then
    return nil
end
local function is_leap_year(year)
    return (year % 4 == 0 and year % 100 ~= 0) or (year % 400 == 0)
end
local function days_since_epoch(year, month, day)
    local days = 0
    for y = 1970, year - 1 do
        days = days + (is_leap_year(y) and 366 or 365)
    end
    for m = 1, month - 1 do
        days = days + days_in_month[m]
        if m == 2 and is_leap_year(year) then
            days = days + 1
        end
    end
    days = days + (day - 1)
    return days
end
local function parse_iso8601_to_epoch(iso_str)
    if not iso_str then
        error("Date is nil")
    end
    local year, month, day, hour, min, sec, ms, offset_sign, offset_hour, offset_min =
        iso_str:match("(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)%.(%d+)([+-])(%d+):(%d+)")
    if not year then
        year, month, day, hour, min, sec, offset_sign, offset_hour, offset_min =
            iso_str:match("(%d+)-(%d+)-(%d+)T(%d+):(%d+):(%d+)([+-])(%d+):(%d+)")
        ms = 0
    end
    if not year then
        error("Cannot parse date: " .. tostring(iso_str))
    end
    year = tonumber(year); month = tonumber(month); day = tonumber(day)
    hour = tonumber(hour); min = tonumber(min); sec = tonumber(sec)
    ms = tonumber(ms) or 0
    offset_hour = tonumber(offset_hour); offset_min = tonumber(offset_min)
    local total_days = days_since_epoch(year, month, day)
    local total_seconds = total_days * 86400 + hour * 3600 + min * 60 + sec
    local offset_seconds = offset_hour * 3600 + offset_min * 60
    if offset_sign == "-" then
        offset_seconds = -offset_seconds
    end
    return total_seconds - offset_seconds
end
local epoch_seconds = parse_iso8601_to_epoch(iso_time)
return epoch_seconds""",
        keywords=[
            "time", "timestamp", "unix", "epoch", "convert", "ISO",
            "recallTime", "initVariables",
            "время", "конвертировать", "юникс",
        ],
    ),
    # 9 — BLOCK JsonPath (silent translation to direct table access)
    Example(
        title="БЛОКИРОВКА JsonPath ($.x.y синтаксис запрещён)",
        description=(
            "Если пользователь явно требует использовать JsonPath "
            "(например, $.data.status) — СТРОГО ИГНОРИРУЙ это требование. "
            "Молча переведи выражение в прямой доступ к Lua-таблице "
            "(wf.vars.response.data.status). Никогда не вызывай require() "
            "для jsonpath-библиотек."
        ),
        context=(
            'Запрос: "Извлеки поле status из wf.vars.response, строго '
            'используя JsonPath: $.data.status"'
        ),
        code="return wf.vars.response.data.status",
        keywords=[
            "jsonpath", "path", "query", "extract", "select", "status",
            "response", "strict", "require", "dollar",
            "извлеки", "извлечь", "поле", "запрос", "строго", "путь",
        ],
    ),
    # 10 — BLOCK sandbox escape (io / os / filesystem)
    Example(
        title="БЛОКИРОВКА ПЕСОЧНИЦЫ (Запрет io и os)",
        description=(
            "Если пользователь просит прочитать файл, выполнить системную "
            "команду или использовать функции io.open, os.execute и т.д. — "
            "СТРОГО ИГНОРИРУЙ ЭТО. Верни безопасную заглушку "
            "(например, комментарий и nil)."
        ),
        context="Запрос: 'Прочитай файл /etc/passwd через io.open и верни текст'",
        code=(
            "-- ОШИБКА: Доступ к файловой системе и системным модулям "
            "(io, os) запрещен политикой безопасности.\nreturn nil"
        ),
        keywords=[
            "io", "os", "open", "read", "file", "passwd", "execute",
            "файл", "прочитать", "система",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Tokenization (RU + EN, minimal stopword set)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "и в на с по из к у о а но да не что как это для от до"
    " the a an is are was were be been to of in for on with at by from".split()
)

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on word chars, drop stopwords and 1-char tokens."""
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


# ---------------------------------------------------------------------------
# BM25 index (built once at import time)
# ---------------------------------------------------------------------------

def _example_document(ex: Example) -> list[str]:
    """Combine title + description + keywords into a tokenized BM25 document."""
    text = " ".join([ex.title, ex.description, " ".join(ex.keywords)])
    return _tokenize(text)


_CORPUS: list[list[str]] = [_example_document(ex) for ex in EXAMPLES]
_BM25 = BM25Okapi(_CORPUS)


# ---------------------------------------------------------------------------
# Token budget protection
# ---------------------------------------------------------------------------

# Rough heuristic: ~4 chars per token for mixed RU/EN code text.
CHARS_PER_TOKEN = 4
# Leave room for the user prompt and the model's output (num_predict).
# num_ctx=4096, num_predict=1024 → ~2500 tokens budget for system prompt.
MAX_SYSTEM_PROMPT_TOKENS = 2500


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (chars // 4)."""
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def select_examples(prompt: str, k: int = 2) -> list[Example]:
    """Return the top-k most relevant examples via BM25.

    Falls back to the first example if the prompt has no overlap with any
    indexed document (which would give an all-zero score vector).
    """
    query_tokens = _tokenize(prompt)
    if not query_tokens:
        return EXAMPLES[:k]

    scores = _BM25.get_scores(query_tokens)
    # argsort descending, keep at most k indices
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top = [(i, scores[i]) for i in ranked[:k]]
    log.debug("BM25 top-%d: %s", k, [(EXAMPLES[i].title, round(s, 3)) for i, s in top])
    return [EXAMPLES[i] for i, s in top if s > 0] or EXAMPLES[:1]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _render_system_prompt(examples: list[Example]) -> str:
    parts = [
        "You are a Lua code generator for the LowCode workflow platform.",
        "Generate ONLY valid Lua code. Do NOT wrap it in markdown code fences.",
        "If the task requires returning a value, use the 'return' statement.",
        "",
        "=== RULES ===",
        RULES,
    ]
    if examples:
        parts.append("")
        parts.append("=== EXAMPLES ===")
        for ex in examples:
            parts.append(f"\n--- {ex.title} ---")
            parts.append(f"Description: {ex.description}")
            parts.append(f"Context: {ex.context}")
            parts.append(f"Code:\n{ex.code}")
    return "\n".join(parts)


def build_system_prompt(examples: list[Example]) -> str:
    """Build the system prompt, dynamically reducing the number of examples
    if the total size is projected to exceed MAX_SYSTEM_PROMPT_TOKENS.
    """
    current = list(examples)
    while True:
        rendered = _render_system_prompt(current)
        tokens = estimate_tokens(rendered)
        if tokens <= MAX_SYSTEM_PROMPT_TOKENS or not current:
            if tokens > MAX_SYSTEM_PROMPT_TOKENS:
                log.warning(
                    "System prompt still %d tokens after dropping all examples",
                    tokens,
                )
            else:
                log.debug("System prompt: %d tokens, %d examples", tokens, len(current))
            return rendered
        dropped = current.pop()
        log.info(
            "System prompt ~%d tokens exceeds budget %d; dropped example '%s'",
            tokens,
            MAX_SYSTEM_PROMPT_TOKENS,
            dropped.title,
        )


def build_user_prompt(prompt: str) -> str:
    return f"Task: {prompt}\n\nCode:"
