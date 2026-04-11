# Архитектура LocalScript (C4-модель)

Документ описывает архитектуру системы **LocalScript** — локального
агентного генератора Lua-кода для LowCode-платформы — на двух уровнях
модели C4: **Container Diagram (Уровень 2)** и **Component Diagram
(Уровень 3)**. Диаграммы написаны на Mermaid.js и отображаются прямо в
браузере (GitHub, VS Code, MkDocs и т. п.).

---

## Уровень 2 — Container Diagram (макро-инфраструктура)

Эта диаграмма отражает три главных ограничения хакатона:

1. **100 % локальное исполнение.** На ней нет ни одного облачного сервиса,
   ни одного внешнего API — всё, что нужно пользователю (веб-приложение,
   LLM-движок, валидатор Lua), упаковано в один `docker compose up` и
   работает на его собственной машине. Данные пользователя (промпты,
   сгенерированный код) не покидают хост.
2. **Экономия VRAM < 8 ГБ.** Вместо крупной универсальной модели мы
   держим на GPU один-единственный квантованный `qwen2.5-coder:7b-
   instruct-q4_K_M` (~4.7 ГБ Q4). Контейнер `ollama` запускается с
   директивой `OLLAMA_NUM_PARALLEL=1`, что не даёт движку дублировать
   веса под конкурентные запросы и вписывает систему в лимит 8 ГБ даже
   на RTX 3060 / 4060.
3. **Автономная самокоррекция.** Валидатор `luac` — это не отдельный
   сервис и не облачный линтер, а локальный дочерний процесс, который
   FastAPI-контейнер запускает через `subprocess`. Благодаря этому
   агент может мгновенно (без сетевых задержек) получить обратную связь
   о синтаксической ошибке и передать её обратно в LLM внутри одного
   HTTP-запроса пользователя.

```mermaid
C4Container
    title Диаграмма контейнеров — LocalScript (Уровень 2)

    Person(user, "Пользователь", "Разработчик LowCode-платформы. Формулирует задачу на русском языке и ожидает готовый Lua-скрипт.")

    System_Boundary(localscript, "LocalScript (единый docker compose)") {
        Container(app, "FastAPI-приложение", "Python 3.11, FastAPI, uvicorn, httpx", "Принимает POST /generate, оркестрирует агентный цикл: поиск примеров → генерация → валидация → самокоррекция.")

        Container(ollama, "Сервер Ollama", "Docker-контейнер, NVIDIA GPU", "Хостит квантованную модель qwen2.5-coder:7b-instruct-q4_K_M (~4.7 ГБ VRAM). Отдаёт ответы через /api/chat.")

        Container(luac, "Валидатор luac", "Локальный процесс Lua 5.4", "Дочерний процесс, запускаемый FastAPI через subprocess. Проверяет синтаксис сгенерированного кода командой luac -p.")
    }

    Rel(user, app, "Отправляет промпт на естественном языке", "HTTP POST /generate (JSON)")
    Rel(app, ollama, "Запрашивает генерацию кода с историей диалога", "HTTP POST /api/chat")
    Rel(app, luac, "Передаёт Lua-код на проверку через stdin", "subprocess.run")
    Rel(ollama, app, "Возвращает сгенерированный Lua-код", "HTTP JSON")
    Rel(luac, app, "Возвращает стёрр с ошибкой или код возврата 0", "stdout / stderr")
    Rel(app, user, "Возвращает валидный Lua-код", "HTTP JSON")

    UpdateRelStyle(user, app, $offsetX="-40", $offsetY="-20")
    UpdateRelStyle(app, ollama, $offsetX="10", $offsetY="-10")
    UpdateRelStyle(app, luac, $offsetX="10", $offsetY="10")
```

---

## Уровень 3 — Component Diagram (внутренности FastAPI-приложения)

Эта диаграмма показывает, как пять модулей внутри `app/` распределяют
между собой три ключевые обязанности агента и как именно они удерживают
систему в рамках хакатонных ограничений:

1. **Целевой BM25-поиск против разрастания контекста.** Модуль
   `knowledge.py` держит корпус из 10 few-shot примеров (включая два
   anti-example для JsonPath и sandbox-escape). На каждый запрос BM25
   выбирает только top-2 наиболее релевантных — остальные в системный
   промпт не попадают. Дополнительно `build_system_prompt` считает
   приблизительное число токенов (`chars // 4`) и, если промпт
   приближается к бюджету `MAX_SYSTEM_PROMPT_TOKENS = 2500`, **сам
   выкидывает** лишние примеры, не давая превысить `num_ctx = 4096`
   модели. Это прямой механизм защиты от переполнения VRAM на
   карточках до 8 ГБ.
2. **Автономный цикл самокоррекции.** Модуль `agent.py` держит полную
   `history` из сообщений чата. После первой генерации при
   `temperature = 0.1` результат извлекается через `lua_validator.py`
   (`extract_lua_code` умеет разбирать `lua{...}lua`, markdown-фенсы и
   голый текст) и прогоняется через `validate_lua` → `luac -p`. При
   ошибке цикл добавляет в `history` сломанный ответ ассистента и
   корректирующее сообщение пользователя, поднимает температуру до
   `0.5` и повторяет — до двух раз. Это *в точности* то, что требовал
   хакатон: агент, который сам чинит свои ошибки, без участия человека.
3. **Полная изоляция инфраструктуры от бизнес-логики.** Модуль
   `ollama_client.py` — единственная точка, знающая про HTTP и httpx.
   Модуль `lua_validator.py` — единственная точка, знающая про
   `subprocess` и `luac`. `main.py` не импортирует ни то, ни другое
   напрямую — только `agent.generate(...)`, что делает код легко
   тестируемым и позволяет подменять инфраструктурные зависимости.

```mermaid
C4Component
    title Диаграмма компонентов — FastAPI-приложение (Уровень 3)

    Person(user, "Пользователь", "Разработчик LowCode-платформы")

    Container_Boundary(app, "FastAPI-приложение (контейнер app)") {
        Component(main, "main.py", "FastAPI, Pydantic", "Входная точка HTTP. Определяет POST /generate и GET /health, управляет жизненным циклом OllamaClient через lifespan, оборачивает исключения в HTTPException.")

        Component(agent, "agent.py", "Python async", "Агентный цикл самокоррекции. Строит историю диалога, вызывает LLM при temperature=0.1, при синтаксической ошибке повторяет с temperature=0.5 и добавленным фидбеком (до 2 ретраев).")

        Component(knowledge, "knowledge.py", "rank-bm25, dataclasses", "База знаний: 10 правил LowCode + 10 few-shot примеров (включая anti-examples). BM25-поиск top-2 релевантных примеров и сборка системного промпта с защитой от превышения бюджета токенов (~2500).")

        Component(validator, "lua_validator.py", "re, subprocess", "Извлечение Lua-кода из ответа LLM (lua{...}lua → markdown-фенсы → fallback). Синтаксическая проверка через дочерний процесс luac -p.")

        Component(ollama_client, "ollama_client.py", "httpx.AsyncClient", "Асинхронный HTTP-клиент Ollama. Вызывает /api/chat со всей историей сообщений, /api/tags для проверки здоровья, управляет таймаутом 300 секунд.")
    }

    System_Ext(ollama_ext, "Ollama-сервер", "Docker-контейнер с qwen2.5-coder на GPU")
    System_Ext(luac_ext, "Компилятор luac", "Локальный процесс Lua 5.4")

    Rel(user, main, "POST /generate с JSON {prompt}", "HTTP")
    Rel(main, agent, "Вызывает generate(prompt, client)", "Python async")

    Rel(agent, knowledge, "Просит top-2 примера и собранный системный промпт", "select_examples, build_system_prompt")
    Rel(agent, ollama_client, "Отправляет полную историю чата с температурой", "chat(history, temperature)")
    Rel(agent, validator, "Извлекает код и проверяет синтаксис", "extract_lua_code, validate_lua")

    Rel(ollama_client, ollama_ext, "Запрашивает генерацию", "HTTP POST /api/chat")
    Rel(validator, luac_ext, "Передаёт код на stdin и читает ошибки", "subprocess luac -p -")

    Rel(main, user, "Возвращает JSON {code}", "HTTP 200")

    UpdateRelStyle(user, main, $offsetX="-20", $offsetY="-10")
    UpdateRelStyle(agent, knowledge, $offsetX="-30", $offsetY="0")
    UpdateRelStyle(agent, ollama_client, $offsetX="10", $offsetY="-10")
    UpdateRelStyle(agent, validator, $offsetX="10", $offsetY="10")
```

---

## Связь диаграмм с хакатонными ограничениями (резюме)

| Ограничение жюри | Как решает архитектура | Где видно на диаграмме |
|---|---|---|
| 100 % локальное исполнение, без облаков | Единый `docker compose`, все контейнеры локальные, `luac` — дочерний процесс | **Уровень 2**: отсутствие внешних систем за границей `System_Boundary` |
| VRAM ≤ 8 ГБ | Квантованная Q4-модель (~4.7 ГБ) + `OLLAMA_NUM_PARALLEL=1` + BM25 top-2 + token-budget guard | **Уровень 2**: `ollama` с GPU; **Уровень 3**: `knowledge.py` с защитой бюджета |
| Автономная самокоррекция | Агентный цикл `generate → validate → retry` с ростом температуры и накоплением истории | **Уровень 3**: стрелки `agent → validator → agent → ollama_client` |
| Изоляция / тестируемость | HTTP живёт только в `ollama_client.py`, subprocess только в `lua_validator.py` | **Уровень 3**: только эти два компонента имеют `Rel` к `System_Ext` |
