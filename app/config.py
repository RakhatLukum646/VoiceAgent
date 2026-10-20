from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ai_provider: str = field(default_factory=lambda: os.getenv("AI_PROVIDER", "google"))
    asr_provider: str = field(default_factory=lambda: os.getenv("ASR_PROVIDER", "google"))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_base_url: str = field(default_factory=lambda: os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    google_embedding_model: str = field(default_factory=lambda: os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"))
    google_llm_model: str = field(default_factory=lambda: os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash"))
    google_generator_thinking_level: str = field(default_factory=lambda: os.getenv("GOOGLE_GENERATOR_THINKING_LEVEL", "minimal"))
    google_checker_thinking_level: str = field(default_factory=lambda: os.getenv("GOOGLE_CHECKER_THINKING_LEVEL", "minimal"))
    google_asr_model: str = field(default_factory=lambda: os.getenv("GOOGLE_ASR_MODEL", "gemini-3.5-transcribe-live"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    openai_llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"))
    deepgram_api_key: str = field(default_factory=lambda: os.getenv("DEEPGRAM_API_KEY", ""))
    deepgram_model: str = field(default_factory=lambda: os.getenv("DEEPGRAM_MODEL", "nova-3"))
    asr_language: str = field(default_factory=lambda: os.getenv("ASR_LANGUAGE", "ru-RU"))
    deepdub_api_key: str = field(default_factory=lambda: os.getenv("DEEPDUB_API_KEY", ""))
    deepdub_base_url: str = field(default_factory=lambda: os.getenv("DEEPDUB_BASE_URL", "wss://wss.deepdub.ai/ws"))
    deepdub_voice_id: str = field(default_factory=lambda: os.getenv("DEEPDUB_VOICE_ID", ""))
    deepdub_locale: str = field(default_factory=lambda: os.getenv("DEEPDUB_LOCALE", "ru-RU"))
    db_path: str = field(default_factory=lambda: os.getenv("VECTOR_DB_PATH", "data/service.sqlite3"))
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
    corpus_manifest: str = field(default_factory=lambda: os.getenv("CORPUS_MANIFEST", "data/active_corpus.json"))
    service_scope: str = "gold-receive"
    tenant_scope: str = "kaspi-demo"
    retrieval_timeout: float = 15.0
    max_speculative_audio_bytes: int = 16000 * 2 * 12
    livekit_url: str = field(default_factory=lambda: os.getenv("LIVEKIT_URL", "ws://127.0.0.1:7880"))
    livekit_public_url: str = field(default_factory=lambda: os.getenv("LIVEKIT_PUBLIC_URL", "ws://localhost:7880"))
    livekit_api_key: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_KEY", ""))
    livekit_api_secret: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_SECRET", ""))
    max_concurrent_calls: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_CALLS", "4")))
    max_call_seconds: int = field(default_factory=lambda: int(os.getenv("MAX_CALL_SECONDS", "600")))
    call_connect_seconds: int = field(default_factory=lambda: int(os.getenv("CALL_CONNECT_SECONDS", "45")))
    max_audio_seconds: int = 30
    top_k: int = 5
    min_similarity: float = 0.28

    def missing_livekit_credentials(self) -> list[str]:
        return [name for name, value in (("LIVEKIT_URL", self.livekit_url), ("LIVEKIT_API_KEY", self.livekit_api_key), ("LIVEKIT_API_SECRET", self.livekit_api_secret)) if not value]

    @property
    def embedding_model(self) -> str:
        model = self.google_embedding_model if self.ai_provider == "google" else self.openai_embedding_model
        return f"{self.ai_provider}:{model}"

    @property
    def llm_model(self) -> str:
        return self.google_llm_model if self.ai_provider == "google" else self.openai_llm_model

    def missing_credentials(self) -> list[str]:
        missing = []
        if self.ai_provider == "google":
            if not self.google_api_key: missing.append("GOOGLE_API_KEY")
        elif self.ai_provider == "openai":
            if not self.openai_api_key: missing.append("OPENAI_API_KEY")
        else:
            missing.append("AI_PROVIDER=google|openai")
        if self.asr_provider == "google":
            if not self.google_api_key: missing.append("GOOGLE_API_KEY")
        elif self.asr_provider == "deepgram":
            if not self.deepgram_api_key: missing.append("DEEPGRAM_API_KEY")
        else:
            missing.append("ASR_PROVIDER=google|deepgram")
        for name, value in (("DEEPDUB_API_KEY", self.deepdub_api_key), ("DEEPDUB_BASE_URL", self.deepdub_base_url), ("DEEPDUB_VOICE_ID", self.deepdub_voice_id)):
            if not value: missing.append(name)
        return list(dict.fromkeys(missing))
