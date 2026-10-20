from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class TurnMetrics:
    call_id: str
    turn_id: int
    started: float = field(default_factory=perf_counter)
    stamps: dict[str, float] = field(default_factory=dict)
    durations_ms: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.stamps[name] = perf_counter()

    def duration(self, name: str, start: str, end: str) -> float:
        value = max(0.0, (self.stamps[end] - self.stamps[start]) * 1000)
        self.durations_ms[name] = round(value, 2)
        return value

    def elapsed(self, name: str, start: float) -> float:
        value = max(0.0, (perf_counter() - start) * 1000)
        self.durations_ms[name] = round(value, 2)
        return value

    def as_event(self) -> dict:
        return {"call_id": self.call_id, "turn_id": self.turn_id, "metrics_ms": dict(self.durations_ms)}
