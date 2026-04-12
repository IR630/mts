"""Domain knowledge store: LowCode Lua rules, examples, and few-shot retrieval.

System prompts are authored in English — qwen2.5-coder:7b follows English
instructions more reliably than Russian. User-facing prompts (tasks) can be in
either language; the BM25 keyword list for each example intentionally contains
both English AND Russian tokens so retrieval works regardless of the query
language.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain rules (always injected into the system prompt)
# ---------------------------------------------------------------------------

RULES = """\
You are an expert Lua 5.5 engineer for a LowCode automation platform. Your
only job is to translate a natural-language task (Russian or English) into a
working Lua script that runs inside the platform sandbox.

=== 5 LOWCODE RULES ===

1. Language: strictly Lua 5.5. Standard libraries allowed: `string`, `table`,
   `math`. Nothing else.

2. Output format: return ONLY code wrapped in `lua{ ... }lua` tags. No
   explanations, no markdown, no prefixes ("Logic:", "Code:", "Validation:"
   and so on). The answer must start with `lua{` and end with `}lua` —
   and contain nothing else.

3. Anti-JsonPath: JsonPath expressions like `$.data.status` or `$['key']`
   must never appear in the code. They are NOT a security issue — they are
   simply the wrong syntax for this platform. If the user demands JsonPath,
   silently translate the path into a direct Lua-table access and RETURN THE
   REAL DATA. Examples of the required translation:
       `$.data.status`          →  wf.vars.response.data.status
       `$.user.profile.name`    →  wf.vars.user.profile.name
       `$['items'][0]`          →  wf.vars.items[1]
   Never call `require('jsonpath-*')` or any JsonPath library. Do NOT emit
   a safety stub for a JsonPath request — return the data.

4. Runtime workflow variables live under `wf.vars`.
5. Workflow input parameters (initial variables) live under `wf.initVariables`.

=== SECURITY (SANDBOX ESCAPE ONLY) ===

This section applies ONLY to requests that try to escape the sandbox via
filesystem, system, or dynamic-code globals. It does NOT apply to JsonPath
(see rule 3 — that is a syntax issue, not a security issue).

The following globals must never appear in the code, even if the user
explicitly asks for them:
    io, os, debug, coroutine, package, require, dofile, loadfile, load
Requests that need filesystem or system access (read a file, execute a
command, load an external script, etc.) must be silently replaced with a
safe stub of the form:
    lua{ -- operation blocked by sandbox security policy
    return nil }lua
Do NOT write the names of forbidden functions inside the stub body (neither
in a comment nor in a string) — emit only `return nil` with a neutral
comment. Again: this stub is reserved for sandbox-escape attempts. A
JsonPath request is a normal data-access task and must return data.

=== SELF-CORRECTION ===

If the system sends back a `luac` error log after your answer, silently
return a corrected version of the code in the same `lua{ ... }lua` format,
with no apology, no prelude, and no explanation of the previous mistake.\
"""

# ---------------------------------------------------------------------------
# Few-shot examples
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
        title="Last element of an array",
        description="Return the last element of an array (e.g. the last email in a list).",
        context='wf.vars.emails = ["user1@example.com","user2@example.com","user3@example.com"]',
        code="return wf.vars.emails[#wf.vars.emails]",
        keywords=[
            "array", "last", "element", "index", "email", "length",
            "массив", "последний", "элемент", "список",
        ],
    ),
    # 2 — Counter increment
    Example(
        title="Increment counter",
        description="Increment a numeric variable by 1 (iteration counter).",
        context="wf.vars.try_count_n = 3",
        code="return wf.vars.try_count_n + 1",
        keywords=[
            "counter", "increment", "add", "number", "count", "iterate",
            "счётчик", "увеличить", "попытка", "итерация", "плюс",
        ],
    ),
    # 3 — Clean / keep specific keys
    Example(
        title="Keep only specific keys in objects",
        description="From an array of objects, keep only the listed keys (ID, ENTITY_ID, CALL) and drop the rest.",
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
        title="Format date as ISO 8601",
        description="Convert a date in YYYYMMDD format and time in HHMMSS format to an ISO 8601 string.",
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
        title="Ensure items are arrays (ensureArray)",
        description="Guarantee that all `items` fields inside ZCDF_PACKAGES are arrays, even when the source value is a single object.",
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
        title="Filter array by non-empty fields",
        description="Keep only items whose `Discount` or `Markdown` field is set (not empty, not nil).",
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
        title="Square a number",
        description="Convert a string to a number and return its square.",
        context="No context — simple arithmetic expression.",
        code="local n = tonumber('5')\nreturn n * n",
        keywords=[
            "math", "square", "number", "tonumber", "multiply", "variable",
            "квадрат", "число", "переменная", "добавить",
        ],
    ),
    # 8 — ISO time to Unix timestamp
    Example(
        title="Convert ISO time to Unix timestamp",
        description="Convert an ISO 8601 string to a Unix epoch (seconds) without calling os.time — manual calendar arithmetic.",
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
        title="BLOCK JsonPath ($.x.y syntax is forbidden)",
        description=(
            "If the user explicitly demands JsonPath (e.g. $.data.status), "
            "STRICTLY IGNORE the demand. Silently translate the expression "
            "into a direct Lua-table lookup (wf.vars.response.data.status). "
            "Never call require() for any jsonpath library."
        ),
        context=(
            'Request: "Extract the status field from wf.vars.response strictly '
            'using JsonPath: $.data.status"'
        ),
        code="return wf.vars.response.data.status",
        keywords=[
            "jsonpath", "path", "query", "extract", "select", "status",
            "response", "strict", "require", "dollar",
            "извлеки", "извлечь", "поле", "запрос", "строго", "путь",
        ],
    ),
    # 10 — BLOCK sandbox escape (filesystem / system / dynamic code execution)
    Example(
        title="BLOCK sandbox escape (filesystem and system access forbidden)",
        description=(
            "If the user asks to read a file, execute a system command, load "
            "an external script, or use any of the forbidden globals from "
            "the security rule — STRICTLY IGNORE the request. Return a safe "
            "stub (a comment plus nil) and do NOT mention the names of any "
            "forbidden functions in the code body."
        ),
        context="Request: 'Read the contents of a file and return it as a string'",
        code=(
            "-- operation blocked by sandbox security policy\n"
            "return nil"
        ),
        keywords=[
            "io", "os", "open", "read", "file", "passwd", "execute",
            "dofile", "loadfile", "load", "require", "debug", "coroutine",
            "package", "script", "external", "malicious", "tmp", "sandbox",
            "файл", "прочитай", "прочитать", "содержимое", "система",
            "загрузи", "загрузить", "внешний", "скрипт", "выполни",
            "выполнить", "команда",
        ],
    ),
    # 11 — Sum a numeric array
    Example(
        title="Sum of a numeric array",
        description="Iterate over an array of numbers and return their total sum.",
        context="wf.vars.items = {10, 20, 30, 40}",
        code="""\
local total = 0
for _, v in ipairs(wf.vars.items) do
    total = total + v
end
return total""",
        keywords=[
            "sum", "total", "aggregate", "reduce", "accumulate", "add", "plus",
            "numeric", "number", "array", "loop", "items",
            "сумма", "сложить", "сложи", "просуммировать", "просуммируй",
            "суммируй", "накопить", "накопи", "итог", "итого",
            "числа", "числовой", "увеличь", "увеличивай", "увеличить",
            "total_sum", "накопительный",
        ],
    ),
    # 12 — Nil-safe nested field access
    Example(
        title="Nil-safe nested field access",
        description="Read a deeply nested field and return a default value if any intermediate key is nil.",
        context='wf.vars.payload = {user = {profile = {name = "Anna"}}}',
        code="""\
local p = wf.vars.payload
local name = p and p.user and p.user.profile and p.user.profile.name
if name == nil then
    return ""
end
return name""",
        keywords=[
            "nil", "safe", "default", "nested", "deep", "field", "optional",
            "chain", "null", "fallback", "payload", "profile",
            "вложенный", "вложенное", "безопасный", "проверка",
            "по умолчанию", "необязательный", "отсутствует", "пустой",
        ],
    ),
    # 13 — Check if array contains a value
    Example(
        title="Contains check on an array",
        description="Return true if a given value is present in an array, false otherwise.",
        context='wf.vars.roles = {"admin", "editor", "viewer"}',
        code="""\
local needle = "editor"
for _, v in ipairs(wf.vars.roles) do
    if v == needle then
        return true
    end
end
return false""",
        keywords=[
            "contains", "includes", "has", "in", "find", "search", "member",
            "present", "exists", "check", "needle", "roles",
            "содержит", "содержится", "есть", "поиск", "найти", "ищи",
            "проверить", "проверь", "наличие", "присутствует",
        ],
    ),
    # 14 — Map/transform array
    Example(
        title="Map array (transform each element)",
        description="Build a new array by transforming each element of the source array.",
        context="wf.vars.prices = {100, 200, 300}",
        code="""\
local result = {}
for i, v in ipairs(wf.vars.prices) do
    result[i] = v * 1.2
end
return result""",
        keywords=[
            "map", "transform", "apply", "convert", "project", "build",
            "foreach", "prices", "multiply", "new",
            "преобразовать", "преобразуй", "применить", "примени",
            "новый", "массив", "обход", "умножь", "умножить",
        ],
    ),
    # 15 — Join string array with a separator
    Example(
        title="Join string array with separator",
        description="Concatenate an array of strings using a separator (comma, space, etc.).",
        context='wf.vars.tags = {"red", "green", "blue"}',
        code="return table.concat(wf.vars.tags, \", \")",
        keywords=[
            "join", "concat", "concatenate", "separator", "delimiter",
            "string", "implode", "merge", "tags", "comma",
            "объединить", "объедини", "соединить", "соедини", "склеить",
            "склей", "разделитель", "запятая", "запятую", "через",
        ],
    ),
    # 16 — Split string by delimiter
    Example(
        title="Split string by delimiter",
        description="Split a string into an array using a delimiter character.",
        context='wf.vars.csv_line = "apple,banana,cherry"',
        code="""\
local result = {}
for part in string.gmatch(wf.vars.csv_line, "([^,]+)") do
    table.insert(result, part)
end
return result""",
        keywords=[
            "split", "string", "delimiter", "separator", "csv", "parse",
            "gmatch", "pattern", "tokenize",
            "разделить", "разделитель", "строка", "разбить", "парсить",
        ],
    ),
    # 17 — Merge two tables (shallow)
    Example(
        title="Merge two tables (shallow)",
        description="Merge all key-value pairs from one table into another, overwriting on conflict.",
        context='wf.vars.defaults = {color="red", size=10}, wf.vars.overrides = {size=20, weight=5}',
        code="""\
local merged = {}
for k, v in pairs(wf.vars.defaults) do
    merged[k] = v
end
for k, v in pairs(wf.vars.overrides) do
    merged[k] = v
end
return merged""",
        keywords=[
            "merge", "combine", "table", "dict", "dictionary", "extend",
            "overwrite", "union", "shallow", "copy",
            "объединить", "объедини", "слить", "словарь", "таблица",
            "совместить", "перезаписать",
        ],
    ),
    # 18 — Count elements matching a condition
    Example(
        title="Count elements matching a condition",
        description="Count how many elements in an array satisfy a given condition.",
        context="wf.vars.scores = {85, 92, 45, 78, 95, 30}",
        code="""\
local count = 0
for _, v in ipairs(wf.vars.scores) do
    if v >= 80 then
        count = count + 1
    end
end
return count""",
        keywords=[
            "count", "condition", "match", "filter", "satisfy", "threshold",
            "greater", "less", "equal", "how many",
            "посчитать", "посчитай", "количество", "условие", "сколько",
            "больше", "меньше", "равно", "удовлетворяет",
        ],
    ),
    # 19 — Find min and max
    Example(
        title="Find minimum and maximum in array",
        description="Return the minimum and maximum values from a numeric array.",
        context="wf.vars.values = {42, 17, 93, 5, 68}",
        code="""\
local items = wf.vars.values
local mn = items[1]
local mx = items[1]
for i = 2, #items do
    if items[i] < mn then mn = items[i] end
    if items[i] > mx then mx = items[i] end
end
return {min = mn, max = mx}""",
        keywords=[
            "min", "max", "minimum", "maximum", "smallest", "largest",
            "extreme", "range", "find",
            "минимум", "максимум", "минимальный", "максимальный",
            "наименьший", "наибольший", "найти",
        ],
    ),
    # 20 — Remove duplicates
    Example(
        title="Remove duplicates from array",
        description="Return a new array with only unique elements, preserving order.",
        context="wf.vars.ids = {1, 3, 2, 3, 1, 4, 2}",
        code="""\
local seen = {}
local result = {}
for _, v in ipairs(wf.vars.ids) do
    if not seen[v] then
        seen[v] = true
        table.insert(result, v)
    end
end
return result""",
        keywords=[
            "unique", "distinct", "deduplicate", "dedup", "duplicate",
            "remove", "set", "seen",
            "уникальный", "уникальные", "дубликат", "дубликаты",
            "удалить", "убрать", "повторы", "повторяющиеся",
        ],
    ),
    # 21 — Reverse an array
    Example(
        title="Reverse an array",
        description="Return a new array with elements in reverse order.",
        context="wf.vars.arr = {1, 2, 3, 4, 5}",
        code="""\
local result = {}
local src = wf.vars.arr
for i = #src, 1, -1 do
    table.insert(result, src[i])
end
return result""",
        keywords=[
            "reverse", "backward", "invert", "flip", "order",
            "перевернуть", "переверни", "обратный", "обратном",
            "порядок", "реверс",
        ],
    ),
    # 22 — Extract field from array of objects (pluck)
    Example(
        title="Extract field from array of objects (pluck)",
        description="Extract a single field from each object in an array.",
        context='wf.vars.users = {{name="Alice", age=30}, {name="Bob", age=25}}',
        code="""\
local result = {}
for _, u in ipairs(wf.vars.users) do
    table.insert(result, u.name)
end
return result""",
        keywords=[
            "extract", "pluck", "field", "column", "select", "property",
            "attribute", "name", "users", "objects",
            "извлечь", "извлеки", "поле", "свойство", "атрибут",
            "выбрать", "выбери", "колонка", "столбец",
        ],
    ),
    # 23 — Group by key
    Example(
        title="Group array elements by a field",
        description="Group an array of objects by a given key, returning a table of arrays.",
        context='wf.vars.orders = {{city="Moscow", total=100}, {city="SPB", total=200}, {city="Moscow", total=150}}',
        code="""\
local groups = {}
for _, order in ipairs(wf.vars.orders) do
    local key = order.city
    if not groups[key] then
        groups[key] = {}
    end
    table.insert(groups[key], order)
end
return groups""",
        keywords=[
            "group", "groupby", "categorize", "bucket", "partition",
            "key", "field", "classify", "city",
            "группировать", "группируй", "сгруппировать", "сгруппируй",
            "категория", "разбить", "разбей",
        ],
    ),
    # 24 — String replace
    Example(
        title="Replace substring in string",
        description="Replace all occurrences of a substring within a string.",
        context='wf.vars.text = "Hello World, Hello Lua"',
        code='return string.gsub(wf.vars.text, "Hello", "Hi")',
        keywords=[
            "replace", "substitute", "gsub", "string", "change", "swap",
            "occurrence", "text",
            "заменить", "замени", "замена", "подстрока", "строка",
            "текст", "подставить",
        ],
    ),
    # 25 — Conditional / ternary value
    Example(
        title="Conditional value (ternary pattern)",
        description="Return one of two values based on a condition (Lua and/or idiom).",
        context="wf.vars.age = 20",
        code="""\
local age = wf.vars.age
local label = age >= 18 and "adult" or "minor"
return label""",
        keywords=[
            "ternary", "conditional", "inline", "if", "else", "decide",
            "choice", "pick", "select", "branch",
            "тернарный", "условие", "условный", "выбор", "выбрать",
            "если", "иначе", "определить",
        ],
    ),
    # 26 — Check if table is empty
    Example(
        title="Check if a table is empty",
        description="Return true if the table has no elements, false otherwise.",
        context="wf.vars.data = {}",
        code="""\
if next(wf.vars.data) == nil then
    return true
end
return false""",
        keywords=[
            "empty", "check", "nil", "next", "length", "size", "blank",
            "zero", "none", "has",
            "пустой", "пустая", "проверить", "проверь", "проверка",
            "пуст", "размер",
        ],
    ),
    # 27 — Read workflow input parameter (initVariables)
    Example(
        title="Read workflow input parameter",
        description="Access a workflow input parameter from wf.initVariables and use it to filter.",
        context="wf.initVariables.threshold = 100",
        code="""\
local threshold = wf.initVariables.threshold
local result = {}
for _, v in ipairs(wf.vars.items) do
    if v > threshold then
        table.insert(result, v)
    end
end
return result""",
        keywords=[
            "initVariables", "input", "parameter", "config", "threshold",
            "initial", "setting", "workflow",
            "входные", "входной", "параметр", "настройка", "конфиг",
            "начальный", "пороговый", "порог",
        ],
    ),
    # 28 — Sort array
    Example(
        title="Sort an array",
        description="Sort an array of numbers in ascending order (copy first to avoid mutating).",
        context="wf.vars.nums = {5, 3, 8, 1, 9, 2}",
        code="""\
local nums = {}
for i, v in ipairs(wf.vars.nums) do
    nums[i] = v
end
table.sort(nums)
return nums""",
        keywords=[
            "sort", "order", "ascending", "descending", "arrange", "rank",
            "sorted",
            "сортировать", "сортируй", "отсортировать", "отсортируй",
            "порядок", "упорядочить", "упорядочь", "возрастание",
            "убывание",
        ],
    ),
    # 29 — Concatenate two arrays
    Example(
        title="Concatenate two arrays",
        description="Merge two arrays into one, appending the second after the first.",
        context="wf.vars.a = {1, 2, 3}, wf.vars.b = {4, 5, 6}",
        code="""\
local result = {}
for _, v in ipairs(wf.vars.a) do
    table.insert(result, v)
end
for _, v in ipairs(wf.vars.b) do
    table.insert(result, v)
end
return result""",
        keywords=[
            "concatenate", "append", "merge", "combine", "array", "join",
            "extend", "add", "union", "two",
            "конкатенация", "присоединить", "добавить", "объединить",
            "массивы", "два", "соединить", "склеить",
        ],
    ),
    # 30 — String length and substring
    Example(
        title="Get string length and extract substring",
        description="Get the length of a string and extract a portion of it.",
        context='wf.vars.text = "Hello, World!"',
        code="""\
local text = wf.vars.text
local len = #text
local sub = string.sub(text, 1, 5)
return {length = len, first_five = sub}""",
        keywords=[
            "length", "substring", "sub", "slice", "portion", "part",
            "characters", "extract", "cut", "trim",
            "длина", "подстрока", "часть", "вырезать", "извлечь",
            "символы", "обрезать",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Tokenization (RU + EN, minimal stopword set)
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "и в на с по из к у о а но да не что как это для от до"
    " the a an is are was were be been to of in for on with at by from"
    # Universal LowCode markers — they appear in virtually every prompt AND
    # every example's context, so they carry no task-discriminating signal.
    # Without this, BM25 over-rewards whichever example mentions `wf.vars`
    # in its description text (e.g. the JsonPath anti-example) even for
    # unrelated queries.
    " wf vars initvariables _utils".split()
)

_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on word chars, drop stopwords and 1-char tokens."""
    return [
        t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1
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
# Hackathon-fixed params: num_ctx=4096, num_predict=256. Leave room for the
# user prompt (~100 tokens) and model output (256 tokens) → ~3700 tokens
# available for the system prompt, but we cap lower to stay safe.
MAX_SYSTEM_PROMPT_TOKENS = 3500


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
    parts = [RULES]
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


def build_user_prompt(
    prompt: str,
    *,
    plan: str | None = None,
    previous_code: str | None = None,
    schema_info: str | None = None,
) -> str:
    """Assemble the user-role message.

    Optional enrichments (all injected before the task line so the model
    sees context first):
      * schema_info — data-path description from sample_wf_vars.
      * previous_code — last generated code for multi-turn refinement.
      * plan — brief approach outline from the Planner stage.
    """
    parts: list[str] = []
    if schema_info:
        parts.append(f"Available data paths:\n{schema_info}")
    if previous_code:
        parts.append(
            f"Previously generated code (refine it according to the request below):\n"
            f"{previous_code}"
        )
    parts.append(f"Task: {prompt}")
    if plan:
        parts.append(f"Plan: {plan}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Schema inference (for sample_wf_vars grounding)
# ---------------------------------------------------------------------------


def infer_schema(data: object, prefix: str = "wf.vars") -> str:
    """Walk a sample payload recursively and return a human-readable list of
    available data paths with their types. This is injected into the user
    prompt so the model knows exactly which fields exist.
    """
    lines: list[str] = []

    def _walk(obj: object, path: str, depth: int = 0) -> None:
        if depth > 8:  # prevent runaway recursion on cyclic / huge payloads
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path}.{k}", depth + 1)
        elif isinstance(obj, list):
            lines.append(f"  {path} : array ({len(obj)} items)")
            if obj:
                _walk(obj[0], f"{path}[1]", depth + 1)
        elif isinstance(obj, bool):
            lines.append(f"  {path} : boolean = {obj}")
        elif isinstance(obj, (int, float)):
            lines.append(f"  {path} : number = {obj}")
        elif isinstance(obj, str):
            # Truncate long string values in the schema description.
            shown = obj if len(obj) <= 40 else obj[:37] + "..."
            lines.append(f"  {path} : string = {shown!r}")
        elif obj is None:
            lines.append(f"  {path} : nil")

    _walk(data, prefix)
    return "\n".join(lines) if lines else ""
