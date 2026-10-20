import json
import logging


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("voice_rag")


def event(component: str, call_id: str, turn_id: int, **fields) -> None:
    logger.info(json.dumps({"component": component, "call_id": call_id, "turn_id": turn_id, **fields}, ensure_ascii=False, default=str))
