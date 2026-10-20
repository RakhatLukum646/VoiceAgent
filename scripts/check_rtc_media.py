"""Two real RTC peers: publish PCM, verify decoded audio, reliable data and hangup."""
import asyncio
import json
import math
import struct
import uuid
from datetime import timedelta
from pathlib import Path
from livekit import api, rtc
from app.config import Settings


async def run():
    config = Settings()
    name = 'media-smoke-' + uuid.uuid4().hex
    rooms = [rtc.Room(), rtc.Room()]
    sources = [rtc.AudioSource(16000, 1), rtc.AudioSource(16000, 1)]
    received = [asyncio.Event(), asyncio.Event()]
    data_received, removed = asyncio.Event(), asyncio.Event()
    tasks = []
    report = {'room': name, 'pcm_frames': [0, 0], 'peak_pcm': [0, 0]}
    async def read(track, index):
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
        try:
            async for item in stream:
                pcm = bytes(item.frame.data)
                values = struct.unpack('<' + 'h' * (len(pcm) // 2), pcm)
                report['pcm_frames'][index] += 1
                report['peak_pcm'][index] = max(report['peak_pcm'][index], max(map(abs, values), default=0))
                if report['peak_pcm'][index] > 500:
                    received[index].set()
        finally:
            await stream.aclose()
    for index, room in enumerate(rooms):
        room.on('track_subscribed', lambda track, publication, participant, i=index: tasks.append(asyncio.create_task(read(track, i))))
    rooms[1].on('data_received', lambda packet: data_received.set() if packet.data == b'rtc-smoke' else None)
    rooms[1].on('disconnected', lambda reason: removed.set())
    client = api.LiveKitAPI(config.livekit_url, config.livekit_api_key, config.livekit_api_secret)
    try:
        for index, room in enumerate(rooms):
            token = api.AccessToken(config.livekit_api_key, config.livekit_api_secret).with_identity('peer-' + str(index)).with_ttl(timedelta(minutes=2)).with_grants(api.VideoGrants(room_join=True, room=name)).to_jwt()
            await room.connect(config.livekit_url, token)
            await room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track('pcm', sources[index]))
        await asyncio.sleep(1)
        async def send(source, frequency):
            for tick in range(150):
                pcm = struct.pack('<' + 'h' * 320, *[int(5000 * math.sin(2 * math.pi * frequency * (tick * 320 + n) / 16000)) for n in range(320)])
                await source.capture_frame(rtc.AudioFrame(pcm, 16000, 1, 320))
            await source.wait_for_playout()
        await asyncio.gather(send(sources[0], 440), send(sources[1], 660))
        await asyncio.wait_for(asyncio.gather(*(value.wait() for value in received)), 10)
        await rooms[0].local_participant.publish_data(b'rtc-smoke', reliable=True, destination_identities=['peer-1'])
        await asyncio.wait_for(data_received.wait(), 5)
        await client.room.remove_participant(api.RoomParticipantIdentity(room=name, identity='peer-1'))
        await asyncio.wait_for(removed.wait(), 5)
        report.update(status='pass', reliable_data=True, infrastructure_hangup=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(room.disconnect() for room in rooms), return_exceptions=True)
        await asyncio.gather(*(source.aclose() for source in sources))
        await client.room.delete_room(api.DeleteRoomRequest(room=name))
        await client.aclose()
        Path('artifacts/livekit-media-smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))


if __name__ == '__main__':
    asyncio.run(run())
