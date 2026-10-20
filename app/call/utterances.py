import re


FILLER_WORDS = {"вот", "ну", "угу", "ага", "эм", "ээ", "мм", "этот", "эта", "это", "того", "что", "а"}
STOP_WORDS = {"стоп", "остановись", "прекрати", "stop"}


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold())


def is_filler(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and len(tokens) <= 8 and all(token in FILLER_WORDS for token in tokens)


def is_filler_partial(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and len(tokens) <= 8 and all(token in FILLER_WORDS for token in tokens[:-1]) and (tokens[-1] in FILLER_WORDS or any(word.startswith(tokens[-1]) for word in FILLER_WORDS))


def is_stop(text: str) -> bool:
    tokens = words(text)
    return bool(tokens) and len(tokens) <= 3 and any(token in STOP_WORDS for token in tokens) and all(token in STOP_WORDS | FILLER_WORDS for token in tokens)


def is_contentful(text: str) -> bool:
    return is_stop(text) or (sum(char.isalpha() for char in text) >= 3 and not is_filler_partial(text))
