# PDF Translation Framework

Чистый локальный каркас для воспроизводимого перевода больших PDF. Он извлекает
текст, таблицы и координаты, переводит документ независимыми шардами через
локальный каскад моделей, объединяет результаты и блокирует выпуск при неполном
покрытии или изменении защищаемых данных.

Каркас намеренно не содержит предметной логики, оформления или данных исходного
проекта. Он выпускает проверенный структурированный набор переводов.
Рендерер PDF подключается отдельным адаптером под конкретное издание.

## Что входит

- детерминированное извлечение PDF через PyMuPDF;
- атомарные постраничные checkpoints и безопасное возобновление;
- конфигурируемый локальный каскад primary → fallback;
- SQLite-кэш только для принятых переводов;
- балансировка и параллельный запуск шардов;
- fail-closed проверка чисел, единиц, интервалов, отрицаний и глоссария;
- детерминированное объединение результатов;
- глобальный QA и выпуск только при статусе PASS.

## Быстрый старт

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
Copy-Item profiles\example.json profiles\local.json
```

Измените пути, языки и локальные endpoints в `profiles/local.json`, затем:

```powershell
pdf-translate extract  --profile profiles/local.json
pdf-translate plan     --profile profiles/local.json --shards 4
pdf-translate translate --profile profiles/local.json --all-shards --workers 4
pdf-translate merge    --profile profiles/local.json
pdf-translate validate --profile profiles/local.json
pdf-translate release  --profile profiles/local.json
```

Первый запуск лучше ограничить несколькими страницами:

```powershell
pdf-translate extract --profile profiles/local.json --pages 1-5
pdf-translate plan --profile profiles/local.json --shards 1
pdf-translate translate --profile profiles/local.json --all-shards
```

## Поддерживаемые локальные API

- OpenAI-compatible `/v1/chat/completions`;
- llama.cpp `/completion`;
- Ollama `/api/chat`.

Для TranslateGemma укажите 4B endpoint как `primary`, а 12B как `fallback`.
Секреты задаются только через переменную окружения из поля `api_key_env`.

## Ограничения

- нужен качественный текстовый слой PDF; OCR пока является внешним адаптером;
- автоматического универсального восстановления исходного дизайна нет;
- правила отрицаний и единиц необходимо адаптировать к языковой паре и домену;
- выпуск — JSON/JSONL-каталог, а не заново свёрстанный PDF.

Контракты артефактов описаны в [docs/CONTRACTS.md](docs/CONTRACTS.md).
