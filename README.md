# LocalScript — Агентная генерация Lua-кода

Локальная AI-система, которая переводит запросы на естественном языке (RU/EN) в рабочие Lua-скрипты (Lua 5.5) для платформы LowCode. Полностью автономная — никаких обращений к внешним AI API.

## Ключевые возможности

| Возможность | Описание |
|---|---|
| **Planner → Coder pipeline** | Два этапа: сначала модель планирует подход, затем генерирует код по плану |
| **Clarification Agent** | Автоматически задаёт уточняющий вопрос, если задача двусмысленна |
| **Многотуровый диалог** | Сессионный API: пользователь может итеративно улучшать код («добавь проверку на nil», «переименуй переменную») |
| **Двухступенчатая валидация** | Синтаксис через `luac -p` + runtime-проверка в песочнице Lua с mock `wf.vars` |
| **Self-correction** | Автоматическое исправление ошибок — до 2 повторных генераций с фидбеком от валидатора |
| **BM25 Few-shot retrieval** | 30 релевантных примеров (включая anti-examples) с RU/EN ключевыми словами |
| **Schema inference** | Опциональный `sample_wf_vars` — система выводит доступные пути данных и заземляет генерацию |
| **Web UI** | Интерактивный чат-интерфейс с отображением плана, кода и телеметрии |
| **Sandbox-защита** | Блокировка `io`, `os`, `debug`, `dofile`, `loadfile`, `load`, `require` |

## Параметры запуска (фиксированы жюри)

| Параметр              | Значение | Где задаётся |
|---|---|---|
| `num_ctx`             | `4096`   | [app/ollama_client.py](app/ollama_client.py) → `BASE_OPTIONS` |
| `num_predict`         | `256`    | [app/ollama_client.py](app/ollama_client.py) → `BASE_OPTIONS` |
| `num_batch`           | `1`      | [app/ollama_client.py](app/ollama_client.py) → `BASE_OPTIONS` |
| `OLLAMA_NUM_PARALLEL` | `1`      | [docker-compose.yml](docker-compose.yml) (env сервиса `ollama`) |

**Пиковое потребление VRAM ≤ 8.0 ГБ.** Модель квантована в Q4_K_M (~4.7 ГБ на GPU), `OLLAMA_NUM_PARALLEL=1` не даёт Ollama дублировать веса. Замеры — через `nvidia-smi --query-gpu=memory.used` (peak memory) или эндпоинт `/metrics`.

## Используемая модель

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
```

Тег — именно такой (Q4_K_M, ~4.7 ГБ). Автоматически скачивается [entrypoint.sh](entrypoint.sh) при первом `docker compose up`.

---

## Архитектура (Pipeline)

```
Пользователь ──POST /session/{id}/message──>  FastAPI-приложение
                                                │
                          1. Clarification Agent (нужно ли уточнение?)
                             ├─ Да → вернуть вопрос, ждать ответа
                             └─ Нет ↓
                          2. Planner — LLM-вызов: описать подход (2-4 пункта)
                          3. BM25 Retrieval — top-2 из 30 few-shot примеров
                          4. Coder — LLM-вызов: сгенерировать Lua-код по плану
                          5. Извлечь код из ответа (lua{...}lua / markdown / raw)
                          6. Валидация Stage 1: luac -p (синтаксис)
                          7. Валидация Stage 2: lua sandbox (runtime)
                          8. Если ошибка → Self-correction (до 2 ретраёв)
                          9. Вернуть {code, plan, retries, examples_used, ...}
                          
          ──Следующее сообщение──>  Refinement (предыдущий код + правка)
```

**Стек:** Python 3.11, FastAPI, Ollama, qwen2.5-coder:7b-instruct-q4_K_M, Lua 5.4 (`luac -p` + runtime sandbox)

---

## Требования к системе

| Компонент | Минимум |
|---|---|
| ОС | Linux (Ubuntu 20.04+, Manjaro, Arch и др.) |
| Docker | версия 24+ с поддержкой `docker compose` (v2) |
| NVIDIA GPU | 8 ГБ+ VRAM (RTX 3060, 3070, 4060 и выше) |
| nvidia-container-toolkit | для проброса GPU в Docker |
| Свободное место на диске | ~7 ГБ (модель ~4 ГБ + Docker-образы ~3 ГБ) |
| Оперативная память | 8 ГБ+ |

---

## Пошаговая инструкция по запуску

### Шаг 1. Установить Docker

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2

# Manjaro / Arch
sudo pacman -S docker docker-compose

sudo usermod -aG docker $USER && newgrp docker
```

### Шаг 2. Установить NVIDIA Container Toolkit

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Manjaro / Arch
sudo pacman -S nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверить: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

### Шаг 3. Склонировать и запустить

```bash
git clone <URL-репозитория>
cd mts
docker compose up --build
```

Дождитесь в логах:
```
=== Starting FastAPI ===
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Шаг 4. Проверить работу

```bash
# Здоровье
curl http://localhost:8080/health
# → {"status":"ok"}

# Web UI
# Откройте http://localhost:8080/ui в браузере

# API генерация
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Из полученного списка email получи последний"}'
```

---

## Примеры запросов

### Простая генерация (POST /generate)

```bash
# Счётчик попыток
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Увеличь значение переменной try_count_n на 1"}'

# С заземлением через schema inference
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Извлеки статус из ответа",
    "sample_wf_vars": {"response": {"data": {"status": "active"}}}
  }'
```

### Многотуровая сессия

```bash
# 1. Создать сессию
curl -X POST http://localhost:8080/session/start
# → {"session_id": "abc123"}

# 2. Первый запрос → clarification или code
curl -X POST http://localhost:8080/session/abc123/message \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Просуммируй массив wf.vars.items"}'
# → {"kind": "code", "code": "...", "plan": "...", ...}

# 3. Уточнение (многотуровое улучшение)
curl -X POST http://localhost:8080/session/abc123/message \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Добавь проверку на nil"}'
# → {"kind": "code", "code": "...", "plan": "...", ...}

# 4. История сессии
curl http://localhost:8080/session/abc123
```

---

## Структура проекта

```
mts/
├── docker-compose.yml          # Сервисы: Ollama + FastAPI
├── Dockerfile                  # Python 3.11 + lua5.4
├── entrypoint.sh               # Ожидание Ollama, pull модели, запуск uvicorn
├── requirements.txt            # Python-зависимости
├── ARCHITECTURE.md             # C4 Level 2/3 диаграммы (Mermaid)
├── README.md                   # Этот файл
├── app/
│   ├── main.py                 # FastAPI: /generate, /session/*, /metrics, /health
│   ├── agent.py                # Planner→Coder pipeline + retry loop
│   ├── clarifier.py            # Clarification Agent (JSON classifier)
│   ├── knowledge.py            # 30 few-shot примеров + BM25 + schema inference
│   ├── lua_validator.py        # Извлечение кода + luac -p валидация
│   ├── runtime_validator.py    # Runtime sandbox с mock wf.vars
│   ├── ollama_client.py        # Async HTTP-клиент для Ollama API
│   └── static/
│       └── index.html          # Web UI (чат-интерфейс)
└── tests/
    └── test_e2e.py             # 10 E2E-тестов (baseline, self-correction,
                                #   sandbox, session, schema, planner)
```

---

## API-спецификация

### POST /generate — Одноразовая генерация

```json
// Запрос
{"prompt": "...", "sample_wf_vars": {"optional": "payload"}}

// Ответ
{
  "code": "return wf.vars.emails[#wf.vars.emails]",
  "plan": "- Read wf.vars.emails\n- Return last element using # operator",
  "retries": 0,
  "examples_used": ["Last element of an array"],
  "status": "ok",
  "llm_ms": 2340,
  "validation_ms": 12,
  "last_error": null
}
```

### POST /session/start — Создать сессию

```json
// Ответ
{"session_id": "a1b2c3d4e5f6"}
```

### POST /session/{id}/message — Сообщение в сессию

```json
// Запрос
{"prompt": "...", "is_clarification_answer": false, "sample_wf_vars": null}

// Ответ (код)
{"kind": "code", "code": "...", "plan": "...", "retries": 0, ...}

// Ответ (уточнение)
{"kind": "clarification", "question": "Какие именно данные обработать?"}
```

### GET /metrics — VRAM и статистика

```json
{
  "vram_used_mb": 4760,
  "vram_total_mb": 8192,
  "vram_peak_mb": 4780,
  "vram_compliant": true,
  "model": "qwen2.5-coder:7b-instruct-q4_K_M",
  "sessions_open": 1
}
```

### GET /health — Проверка здоровья

```json
{"status": "ok"}
```

---

## Тестирование

```bash
# Установить зависимости тестов
pip install -r requirements-dev.txt

# Запустить E2E-тесты (сервер должен быть запущен)
pytest tests/test_e2e.py -v
```

10 тестов покрывают:
- Baseline-генерацию (массив + append)
- Self-correction Python→Lua (`+=`, `elif`, `!=`)
- Anti-sycophancy JsonPath
- Sandbox-escape (`io.open`, `dofile`)
- Markdown-resilience
- Planner (наличие поля `plan`)
- Schema inference (grounded field access)
- Multi-turn session refinement

---

## Остановка и управление

```bash
docker compose down          # Остановить
docker compose down -v       # Остановить + удалить модель
docker compose logs -f app   # Логи приложения
docker compose restart app   # Перезапуск без Ollama
```

## Запуск без Docker (для разработки)

```bash
# 1. Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b-instruct-q4_K_M
ollama serve  # в отдельном терминале

# 2. Lua
sudo apt install lua5.4        # Ubuntu/Debian
sudo pacman -S lua             # Manjaro/Arch

# 3. Python
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## Решение проблем

| Проблема | Решение |
|---|---|
| `could not select device driver` | Установить nvidia-container-toolkit (Шаг 2) |
| Модель скачивается долго | Нормально (~4 ГБ), далее из кеша |
| `Connection refused` на 8080 | Дождаться `Uvicorn running` в логах |
| `out of memory` (GPU) | Убедиться, что нет других GPU-процессов: `nvidia-smi` |
| Некорректные ответы | Переформулировать запрос или добавить `sample_wf_vars` |
