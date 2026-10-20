from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.config import Settings
from app.providers import create_asr, create_embeddings, create_chat
from app.rag.vector_store import SQLiteVectorStore
from app.rag.retriever import Retriever
from app.rag.indexed import QdrantIndex, IndexedRetriever
from app.agents.generator import AnswerGenerator
from app.agents.checker import AnswerChecker
from app.agents.orchestrator import Orchestrator
from app.tts.deepdub import DeepdubTTSProvider
from app.call.livekit_transport import LiveKitCall, room_token
from app.agents.scope import ScopedOrchestrator
import asyncio
import uuid
from app.observability.logging import event
from app.call.session import CallSession
from app.observability.errors import safe_error_fields


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    store = SQLiteVectorStore(config.db_path)
    embeddings = create_embeddings(config)
    generator_chat = create_chat(config)
    checker_chat = create_chat(config, role="checker")
    index_store = QdrantIndex(config.qdrant_url)
    retriever = IndexedRetriever(embeddings, index_store, config.corpus_manifest, config.tenant_scope, config.service_scope, config.embedding_model)
    rag = Orchestrator(retriever, AnswerGenerator(generator_chat), AnswerChecker(checker_chat, retriever), config.min_similarity)
    router_chat = create_chat(config, role="router")
    orchestrator = ScopedOrchestrator(rag, router_chat)
    calls: dict[str, LiveKitCall] = {}
    watchers: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await asyncio.gather(*(call.close() for call in list(calls.values())), return_exceptions=True)
        await asyncio.gather(*watchers, return_exceptions=True)
        await router_chat.client.aclose()
        await embeddings.client.aclose()
        await index_store.client.aclose()
        await generator_chat.client.aclose()
        await checker_chat.client.aclose()

    app = FastAPI(title="Kaspi Gold Voice Demo", lifespan=lifespan)
    recent_metrics: list[dict] = []

    def record_metrics(value: dict):
        recent_metrics.append(value)
        del recent_metrics[:-100]
    app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")

    @app.get("/")
    async def index():
        return HTMLResponse(Path(__file__).with_name("static").joinpath("index.html").read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        missing = config.missing_credentials() + config.missing_livekit_credentials()
        count, version = 0, None
        try:
            manifest, scope = retriever.manifest()
            count = await index_store.count(scope)
            version = scope.version
            if count != manifest["passages"]:
                count = 0
        except Exception:
            pass
        return {"status": "ready" if not missing and count else "not_ready", "missing_credentials": missing, "indexed_passages": count, "corpus_version": version, "retrieval": "qdrant-hybrid", "embedding_model": config.embedding_model, "ai_provider": config.ai_provider, "asr_provider": config.asr_provider}

    @app.get("/ready")
    async def ready():
        value = await health()
        payload = {"ready": value["status"] == "ready", "missing_credentials": value["missing_credentials"], "indexed_passages": value["indexed_passages"]}
        return JSONResponse(payload, status_code=200 if payload["ready"] else 503)

    @app.get("/metrics")
    async def metrics():
        return {"recent_turns": recent_metrics}

    @app.post("/api/calls")
    async def start_call(request: Request):
        # This local demo accepts only same-origin browser requests. No tokens,
        # arbitrary room names or provider secrets are supplied by the client.
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Недопустимый источник запроса")
        if (await health())["status"] != "ready":
            raise HTTPException(503, "Сервис пока не готов к звонку")
        name = "gold-" + uuid.uuid4().hex
        caller = "caller-" + uuid.uuid4().hex
        def session_factory(transport):
            return CallSession(transport, create_asr(config), orchestrator,
                DeepdubTTSProvider(config.deepdub_api_key, config.deepdub_voice_id,
                                   config.deepdub_base_url, config.deepdub_locale), record_metrics)
        call = LiveKitCall(config, name, caller, session_factory)
        calls[name] = call
        try:
            await call.start()
        except Exception as exc:
            event("rtc_start_error", call.session.call_id, 0, **safe_error_fields(exc))
            await call.close()
            calls.pop(name, None)
            raise HTTPException(503, "Не удалось соединиться. Попробуйте ещё раз.") from None
        async def watch():
            try:
                await call.watch()
            finally:
                calls.pop(name, None)
        task = asyncio.create_task(watch())
        watchers.add(task)
        task.add_done_callback(watchers.discard)
        return JSONResponse({"url": config.livekit_public_url,
                             "token": room_token(config, name, caller, browser=True),
                             "room": name, "identity": caller}, headers={"Cache-Control": "no-store"})

    return app


app = create_app()
