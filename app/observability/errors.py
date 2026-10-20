"""Safe diagnostics for call failures; never include exception text or locals."""

import traceback
from pathlib import Path


_CODES = {
    "Prior Google Live transcription did not finalize": "asr_prior_final_timeout",
    "Google Live transcription stream unavailable": "asr_stream_unavailable",
    "Google Live transcription connection or setup failed": "asr_setup_failed",
    "Google Live transcription stream failed": "asr_stream_failed",
    "Google Live transcription provider error": "asr_provider_error",
    "Google Live transcription stream closed": "asr_stream_closed",
}


def safe_error_fields(exc: Exception) -> dict:
    message = str(exc)
    code = _CODES.get(message)
    if code is None:
        name = type(exc).__name__
        if name in {"ConnectionClosed", "ConnectionClosedError", "ConnectionClosedOK"}:
            code = "websocket_connection_closed"
        elif isinstance(exc, RuntimeError):
            code = "runtime_error"
        else:
            code = name.lower()
    frames = traceback.extract_tb(exc.__traceback__)
    safe_frames = [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name} for frame in frames[-8:]]
    return {"error_code": code, "error_type": type(exc).__name__, "trace": safe_frames}
