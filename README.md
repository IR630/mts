# LocalScript — Агентная генерация Lua-кода

Локальная AI-система, которая переводит запросы на естественном языке в рабочие Lua-скрипты для платформы LowCode.

## Архитектура

```
Пользователь ──POST /generate──>  FastAPI-приложение
                                      │
                         1. Выбрать 1-2 релевантных примера (keyword matching)
                         2. Собрать системный промпт (правила + примеры)
                         3. Отправить запрос к локальной LLM через Ollama
                         4. Извлечь Lua-код из ответа модели
                         5. Валидировать синтаксис через luac -p
                         6. Если ошибка → повторить с фидбеком (макс. 2 раза)
                         7. Вернуть {"code": "..."}
```

**Стек:** Python 3.11, FastAPI, Ollama, qwen2.5-coder:7b-instruct-q4_K_M

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

Если Docker ещё не установлен:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2

# Manjaro / Arch
sudo pacman -S docker docker-compose

# Добавить себя в группу docker (чтобы не писать sudo)
sudo usermod -aG docker $USER
# После этого перелогиниться или выполнить:
newgrp docker
```

Проверить, что Docker работает:
```bash
docker --version
docker compose version
```

### Шаг 2. Установить NVIDIA Container Toolkit

Это нужно для проброса GPU внутрь Docker-контейнеров.

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Manjaro / Arch
sudo pacman -S nvidia-container-toolkit

# Настроить Docker runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Проверить, что GPU видна в Docker:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```
Должна отобразиться таблица с информацией о вашей видеокарте.

### Шаг 3. Склонировать проект

```bash
git clone <URL-репозитория>
cd mts
```

Или если проект уже на диске:
```bash
cd /путь/к/mts
```

### Шаг 4. Запустить одной командой

```bash
docker compose up --build
```

**Что произойдёт:**
1. Соберётся Docker-образ Python-приложения (с установкой lua5.4 для валидации)
2. Запустится сервер Ollama с проброшенной GPU
3. Автоматически скачается модель `qwen2.5-coder:7b-instruct-q4_K_M` (~4 ГБ)
4. Запустится FastAPI-приложение на порту **8080**

Первый запуск займёт время из-за скачивания модели. Последующие запуски будут быстрыми — модель сохраняется в Docker volume.

Дождитесь в логах строки:
```
=== Starting FastAPI ===
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Шаг 5. Проверить работу

Открыть новый терминал и выполнить:

```bash
# Проверка здоровья сервиса
curl http://localhost:8080/health
```

Ожидаемый ответ: `{"status":"ok"}`

```bash
# Сгенерировать Lua-код
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Из полученного списка email получи последний"}'
```

Ожидаемый ответ:
```json
{
  "code": "return wf.vars.emails[#wf.vars.emails]"
}
```

---

## Примеры запросов

```bash
# Счётчик попыток
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Увеличивай значение переменной try_count_n на каждой итерации"}'

# Фильтрация массива
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Отфильтруй элементы из массива, оставив только те, у которых заполнено поле Discount"}'

# Конвертация времени в Unix
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Конвертируй время в переменной recallTime в unix-формат"}'
```

---

## Структура проекта

```
mts/
├── docker-compose.yml      # Сервисы: Ollama + FastAPI-приложение
├── Dockerfile              # Образ Python-приложения с lua5.4 для валидации
├── entrypoint.sh           # Скрипт запуска: ожидание Ollama, скачивание модели, запуск uvicorn
├── requirements.txt        # Python-зависимости
├── README.md               # Этот файл
└── app/
    ├── __init__.py
    ├── main.py             # FastAPI-эндпоинты (POST /generate, GET /health)
    ├── agent.py            # Агентный цикл: генерация → валидация → самокоррекция
    ├── knowledge.py        # Доменные правила, 8 примеров, поиск релевантных примеров
    ├── lua_validator.py    # Извлечение кода из ответа LLM + валидация синтаксиса через luac
    └── ollama_client.py    # Асинхронный HTTP-клиент для Ollama API
```

---

## Как это работает

1. **Поиск примеров (Few-Shot Retrieval):** При каждом запросе система ищет 1-2 наиболее релевантных примера из 8 заложенных, используя пересечение ключевых слов. Найденные примеры вставляются в системный промпт модели.

2. **Генерация кода (LLM):** Промпт с правилами и примерами отправляется в локальную модель qwen2.5-coder через Ollama. Модель возвращает Lua-код.

3. **Валидация синтаксиса:** Сгенерированный код проверяется компилятором `luac -p` (режим только парсинга, без генерации байткода).

4. **Самокоррекция:** Если валидация не прошла — сообщение об ошибке автоматически отправляется обратно в модель для исправления (до 2 попыток).

---

## Остановка и управление

```bash
# Остановить все сервисы
docker compose down

# Остановить и удалить скачанную модель (освободить место)
docker compose down -v

# Посмотреть логи
docker compose logs -f app
docker compose logs -f ollama

# Перезапустить только приложение (без Ollama)
docker compose restart app
```

---

## Запуск без Docker (для разработки)

Если нужно запустить приложение локально без Docker:

```bash
# 1. Установить Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Скачать модель
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# 3. Запустить Ollama-сервер (в отдельном терминале)
ollama serve

# 4. Установить lua для валидации
sudo apt install lua5.4        # Ubuntu/Debian
sudo pacman -S lua             # Manjaro/Arch

# 5. Установить Python-зависимости
pip install -r requirements.txt

# 6. Запустить приложение
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

---

## API-спецификация

### POST /generate

Сгенерировать Lua-код по текстовому описанию задачи.

**Запрос:**
```json
{
  "prompt": "Текст задачи на естественном языке"
}
```

**Ответ (200):**
```json
{
  "code": "return wf.vars.emails[#wf.vars.emails]"
}
```

### GET /health

Проверка доступности сервиса и модели.

**Ответ (200):**
```json
{
  "status": "ok"
}
```

---

## Решение проблем

| Проблема | Решение |
|---|---|
| `docker: Error response from daemon: could not select device driver` | Установить nvidia-container-toolkit (см. Шаг 2) |
| Модель скачивается слишком долго | Нормально для первого запуска (~4 ГБ). При последующих запусках модель берётся из кеша |
| `Connection refused` на порту 8080 | Дождаться полного запуска — в логах должно появиться `Uvicorn running` |
| Ответы модели некорректные | Попробовать переформулировать запрос, добавить контекст (какие переменные доступны) |
| `out of memory` (GPU) | Убедиться, что нет других процессов, использующих GPU. Проверить: `nvidia-smi` |
