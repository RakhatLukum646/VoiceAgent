# Голосовой RAG: оформление и получение Kaspi Gold

Настоящий браузерный звонок LiveKit Web SDK, Gemini Live ASR, Gemini generator и **обязательный независимый checker**, Deepdub TTS. Демо отвечает только об оформлении/получении Gold. Другие продукты сопровождаются фразой перевода и завершением звонка; посторонняя тема возвращает разговор к услуге. Реального специалиста/телефонного номера в демо нет.

## Запуск

Windows, Python3.12, Docker Desktop, браузер с микрофоном. Заполните `.env` по `.env.example`: Google key, Deepdub key/voice. Секреты остаются в backend.

Два равноценных способа запуска — выберите один, не оба сразу (оба слушают8010 и конфликтуют):

**A. Полностью в Docker** (backend, LiveKit и Qdrant в контейнерах):

```powershell
docker compose up -d --build
```

Пересборка нужна только после изменения кода (`--build`). Данные `data/` смонтированы volume'ом, так что переиндексация не требуется при каждом старте.

**B. Host Python + Docker-инфраструктура** (быстрее при частой правке кода, без пересборки образа на каждое изменение):

Дважды нажмите **`start_demo.cmd`**, либо:

```powershell
powershell -ExecutionPolicy Bypass -File start_demo.ps1 -Background
```

Скрипт поднимает LiveKit и Qdrant в Docker, а текущий Python backend — скрыто на Windows. После проверки актуального API/официального индекса `.cmd` открывает [демо](http://127.0.0.1:8010). Логи и PID запуска — `artifacts/runtime`. Без `-Background` backend работает в текущем окне, Ctrl+C завершает его. uvicorn запущен без `--reload`: после правки кода backend нужно перезапустить той же командой.

Первый запуск без manifest загружает официальные страницы и публикует индекс; последующие переиспользуют его. Обновление источников: добавьте `-Reindex` при остановленном backend либо выполните offline команды ниже. Если порт занят посторонним процессом, launcher сообщает PID и не завершает его.

`docker-compose.yml` поднимает все три сервиса (`livekit`, `qdrant`, `voice-rag`) с локальными портами7880/7881/7882,6333 и8010. Внутри контейнера `voice-rag` обращается к Qdrant/LiveKit по именам сервисов докер-сети (`qdrant`, `livekit`), а браузеру отдаётся `ws://localhost:7880` — так WebRTC media path работает и для контейнеризированного backend. Local LiveKit `--dev` credentials предназначены для localhost.

## Что изменилось в архитектуре

```text
ASR final + история + server-owned service scope
 ├─ router
 └─ один query embedding → main hybrid retrieval + independent evidence retrieval
                         ↓
             короткий grounded JSON candidate
 ├─ exact quote checks + batched evidence expansion + independent full checker
 └─ Deepdub → PRIVATE bounded PCM, без подключения к аудиотреку
                         ↓
      только PASS той же версии candidate/call/turn/generation
                         ↓
             LiveKit PCM track → браузер
```

**RAG сохранён.** Онлайн поиск больше не читает/десериализует все SQLite vectors и не пересчитывает BM25 в Python. Qdrant **v1.19.2** использует named dense vectors Gemini и встроенный локальный `qdrant/bm25` с русским tokenizer/stemming, fusion RRF. Dense/BM25 branches и финальный поиск имеют обязательные фильтры tenant, service, language, corpus_version, embedding model. Прямой UUID lookup также проверяет payload scope; malformed IDs дают отсутствие evidence, а не ошибку400. RRF score — ранг, не confidence/cosine; разрешение ответа зависит от evidence и checker.

Router и retrieval стартуют параллельно. Main/verifier делят один query-vector future, но получают разные по ширине evidence pools; независимость checker не подменяется доверием генератору. До3разных claim queries отправляются одним embedding batch. Уточнения используют только недавний Gold-контекст; погода не подмешивается в новый явный вопрос Gold. Query cache ограничен256точными нормализованными строками, относится к одному configured model; это не semantic answer cache.

**Непроверенное аудио не публикуется.** TTS готовит точный candidate во внутреннюю очередь до384000байт (около12с PCM). Глобально до3speculative producers;35с timeout включает ожидание слота/буфера, общий синтез ограничен35с PCM на candidate. Очередь применяет backpressure. Binding содержит call, turn, generation, attempt и SHA-256 текста. RETRY, ошибка и перебивание отменяют producer и очищают очередь. Полностью синтезированная короткая фраза до PASS также остаётся приватной. Ошибка TTS после PASS видна пользователю и не заменяется непроверенной речью. Спекуляция может потратить оплачиваемый TTS впустую; память/задачи ограничены, отмена не гарантирует отмену биллинга провайдера.

Checker по-прежнему отдельный полный model call до речи. На той же `gemini-3.5-flash` после ограниченного сравнения включён `GOOGLE_CHECKER_THINKING_LEVEL=minimal`; параметр отдельный от generator. Пустое значение возвращает provider default. Это настройка вычисления, не удаление checker; небольшой проверочный набор не доказывает равенство качества на всех будущих данных. Формат verdict, deterministic quotes и PASS gate сохранены.

Короткий ACK «Сейчас проверю» идёт параллельно вместо длинного повторения вопроса. ACK — не содержательный ответ. RTC packetizer собирает ровные20мс PCM frames, native capture при отмене присоединяется до освобождения source lock; старый producer не может заполнить очередь после clear. Transfer остаётся terminal: TTS→source drain/тишина→browser availability ACK→RoomService.RemoveParticipant. При ошибке финальной речи — видимый отказ и закрытие с `transfer_audio_failed`.

## Официальная база и обновления

Активны8страниц [Kaspi Гид по получению Gold](https://guide.kaspi.kz/client/ru/gold/receive/general/q27): q27, q43, q12371, q13478, q12508, q14620, q13483, q17279. Они охватывают места оформления/получения, документы для отделения, возраст самостоятельного оформления, приложение, причины отказа, условия иностранцев, выпуск пластика и срок карты нерезидента. Другие банковские продукты остаются вне scope.

`data/official_snapshots` хранит raw HTML, точное тело, дату страницы/получения, hashes и правила извлечения. В q13478 включены буквальные application/collection paragraphs с offsets; список возможностей других продуктов и изображение с неполным текстом исключены. Условные списки иностранцев сохранены целиком. Телефон для получения в Kartomat нельзя обобщать на первое оформление в отделении, где нужен также документ. Возраст/ВНЖ/дата выдачи проверяются по конкретному источнику; содержание связанных внешних памяток не выдумывается.

```powershell
.\.venv\Scripts\python.exe -m scripts.fetch_corpus
.\.venv\Scripts\python.exe -m scripts.publish_corpus
```

Публикация offline: UUIDv5 chunks по URL/тексту, content/model embedding cache, immutable collection, keyword indexes, scoped count+hybrid smoke, затем atomic replace `data/active_corpus.json`. Turn фиксирует collection/version при старте; прежние версии остаются доступны активным turns. Отсутствующие документы не попадают в новый snapshot. Неполная staging collection восстанавливается идемпотентными upserts. Старые коллекции не удаляются автоматически: для долгой эксплуатации нужна retention после завершения старых turns.

По умолчанию разрешено не более32**новых оплачиваемых** document embeddings за offline publication; неизменённые тексты не эмбеддятся заново. Больший job требует явного `--max-new-embeddings N`. Текущий импорт выполнял всего8реальных document embeddings, а не10тысяч. Review-sized parent chunks сохраняют условия; generic crawler/PDF parser и distributed ingestion queue не входят в этот демо.

SQLite корпуса и прежняя документация сохранены архивно; основной call path использует только published official Qdrant collection. Синтетический индекс не может стать активным через official manifest validation.

## Проверки и результаты

Подробные фактические значения и ограничения — [verification.md](verification.md). Основные команды:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate_official
.\.venv\Scripts\python.exe -m scripts.benchmark_index
```

`evaluate_official` — небольшой реальный source/evidence/LLM набор с конфликтами документов, возраста, ВНЖ и дат. `benchmark_index` — **10 000 разных синтетических небанковских текстов**, local clustered768D vectors, никаких платных embeddings, отдельная `bench_synthetic_*` collection. Он проверяет индекс, фильтры и ANN Recall@8 против exact search; не доказывает банковскую семантику, voice quality или production throughput. Не запускайте как production banking corpus.

Браузерные software tests требуют Node+Playwright+Edge (`PLAYWRIGHT_MODULE` может указывать пакет): `scripts/check_latency_cohort.cjs` использует ту же синтетическую WAV речь и настоящий Web SDK/providers; `scripts/check_browser_lifecycle.cjs` проверяет отказ микрофона, barge-in, mute и повторный звонок. Это не физический микрофон/динамик.

## Что ещё ограничивает задержку

Current measured response пока не обещает естественные2–4с. ASR endpoint, router, полная генерация и semantic checker всё ещё стоят в critical path. Небольшие до/после выборки не являются p95 и менялись одновременно с корпусом1→8; кэш/сеть/вычисление влияют на результат. Метрика server first PCM фиксируется до RTC очереди. Browser probe измеряет декодированный onset после answer и паузы в той же browser clock domain; это приближение начала содержательного аудио, не акустическая слышимость.

Следующие отдельные измеримые этапы — provider-aware admission/backpressure под реальными QPS, isolated A/B на одном корпусе, более широкий adversarial benchmark ролей и при необходимости тёплый multi-turn Deepdub. Persistent WS, partial-ASR prefetch и distributed multi-worker admission сейчас не внедрены. SIP номер/реальный перевод/лимиты транка остаются дальнейшим развитием. Исследование контрактов — `architecture_latency_research.md` и `qdrant_corpus_research_addendum.md`.
