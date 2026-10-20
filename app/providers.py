from app.config import Settings
from app.rag.embeddings import OpenAIEmbeddings, GoogleEmbeddings
from app.agents.generator import OpenAIChat, GoogleChat
from app.asr.deepgram import DeepgramASR
from app.asr.google_live import GoogleLiveASR


def create_embeddings(config: Settings):
    if config.ai_provider == "google":
        return GoogleEmbeddings(config.google_api_key, config.google_base_url, config.google_embedding_model)
    if config.ai_provider == "openai":
        return OpenAIEmbeddings(config.openai_api_key, config.openai_base_url, config.openai_embedding_model)
    raise ValueError("AI_PROVIDER must be google or openai")


def create_chat(config: Settings, role: str = "generator"):
    if config.ai_provider == "google":
        thinking = config.google_checker_thinking_level if role == "checker" else config.google_generator_thinking_level
        return GoogleChat(config.google_api_key, config.google_base_url, config.google_llm_model, thinking_level=thinking, explicit_thinking=role == "checker" and bool(thinking))
    if config.ai_provider == "openai":
        return OpenAIChat(config.openai_api_key, config.openai_base_url, config.openai_llm_model)
    raise ValueError("AI_PROVIDER must be google or openai")


def create_asr(config: Settings):
    if config.asr_provider == "google":
        return GoogleLiveASR(config.google_api_key, config.google_asr_model, config.asr_language)
    if config.asr_provider == "deepgram":
        return DeepgramASR(config.deepgram_api_key, config.deepgram_model, config.asr_language.split("-")[0])
    raise ValueError("ASR_PROVIDER must be google or deepgram")
