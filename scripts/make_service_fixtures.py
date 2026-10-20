"""Generate explicit test speech with the configured real Deepdub provider."""
import asyncio
import wave
from pathlib import Path
from app.config import Settings
from app.tts.deepdub import DeepdubTTSProvider

QUESTIONS = {
    'gold': 'Где я могу оформить Каспи Голд?',
    'weather': 'Какая сегодня погода?',
    'unknown': 'Подскажите, какая комиссия за обслуживание карты Каспи Голд?',
    'credit': 'Я хочу оформить кредит наличными.',
}


async def run():
    config = Settings()
    tts = DeepdubTTSProvider(config.deepdub_api_key, config.deepdub_voice_id, config.deepdub_base_url, config.deepdub_locale)
    for name, text in QUESTIONS.items():
        pcm = b''.join([chunk async for chunk in tts.synthesize_stream(text)])
        path = Path('artifacts/service-' + name + '.wav')
        with wave.open(str(path), 'wb') as out:
            out.setnchannels(1); out.setsampwidth(2); out.setframerate(16000)
            out.writeframes(pcm)
        print(name + ': ' + str(len(pcm) // 2) + ' samples')


if __name__ == '__main__':
    asyncio.run(run())
