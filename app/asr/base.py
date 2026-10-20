from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable


class ASRProvider(ABC):
    @abstractmethod
    async def start(self, on_partial: Callable[[str], Awaitable[None]], on_final: Callable[[str], Awaitable[None]]) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: bytes) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
