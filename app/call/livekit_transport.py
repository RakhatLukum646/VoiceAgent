"""Real LiveKit audio tracks and room lifecycle around the existing voice pipeline."""
import asyncio
import json
from datetime import timedelta
from time import perf_counter
from livekit import api, rtc
from app.observability.logging import event
from app.observability.errors import safe_error_fields


def room_token(config, room: str, identity: str, browser: bool = False) -> str:
    grants = api.VideoGrants(room_join=True, room=room, can_publish=True,
                             can_subscribe=True, can_publish_data=True,
                             can_publish_sources=['microphone'] if browser else None)
    return (api.AccessToken(config.livekit_api_key, config.livekit_api_secret)
            .with_identity(identity).with_ttl(timedelta(minutes=5))
            .with_grants(grants).to_jwt())


class LiveKitTransport:
    def __init__(self, room, source, client, room_name, caller, finished):
        self.room, self.source, self.client = room, source, client
        self.room_name, self.caller, self.finished = room_name, caller, finished
        self.generation = 0
        self._audio_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._remainder = b''
        self._first = True
        self._resume_after = 0.0
        self._hung_up = False

    async def event(self, payload):
        if payload.get('type') == 'interrupt':
            self.generation = payload['generation']
            self.source.clear_queue()
            self._remainder = b''
            self._first = True
            # Let already transmitted RTC packets leave the receiver jitter buffer.
            self._resume_after = perf_counter() + .35
        async with self._send_lock:
            await self.room.local_participant.publish_data(json.dumps(payload, ensure_ascii=False), reliable=True,
                destination_identities=[self.caller], topic='call.events')

    async def clear_audio(self, generation):
        # Called after joining the cancelled producer. It cannot refill the source.
        async with self._audio_lock:
            self.generation = generation
            self.source.clear_queue()
            self._remainder = b''

    async def audio(self, pcm, generation):
        async with self._audio_lock:
            if generation != self.generation or self._hung_up:
                return
            if self._first:
                await asyncio.sleep(max(0, self._resume_after - perf_counter()))
                if generation != self.generation:
                    return
                self._first = False
                await self.event({'type': 'audio_start', 'generation': generation})
            data = self._remainder + pcm
            # Provider boundaries need not align with PCM samples or RTC frames.
            # Publish fixed 20 ms frames; retain the tail until the next chunk.
            complete = len(data) // 640 * 640
            self._remainder = data[complete:]
            data = data[:complete]
            for offset in range(0, len(data), 640):
                if generation != self.generation:
                    return
                chunk = data[offset:offset + 640]
                capture = asyncio.create_task(self.source.capture_frame(rtc.AudioFrame(chunk, 16000, 1, len(chunk) // 2)))
                try:
                    await asyncio.shield(capture)
                except asyncio.CancelledError:
                    # Cancelling the Python waiter does not cancel the native
                    # FFI push. Join it BEFORE releasing the source lock/clearing
                    # the queue, otherwise the next generation can race it.
                    try:
                        await asyncio.wait_for(capture, .5)
                    except Exception:
                        pass
                    raise

    async def drain(self, generation):
        if generation != self.generation:
            return
        if len(self._remainder) % 2:
            raise ValueError('TTS ended with incomplete PCM sample')
        if self._remainder:
            await self.audio(b'\0' * (640 - len(self._remainder)), generation)
        # Server drain followed by a short silent tail; browser confirms playback
        # availability after receipt. Neither is proof of physical loudspeaker sound.
        await self.audio(b'\0' * 16000, generation)
        await self.source.wait_for_playout()

    async def hangup(self, reason):
        if self._hung_up:
            return
        self._hung_up = True
        try:
            await self.event({'type': 'ended', 'reason': reason})
            await self.client.room.remove_participant(api.RoomParticipantIdentity(room=self.room_name, identity=self.caller))
        finally:
            self.finished.set()


class LiveKitCall:
    def __init__(self, config, name, caller, session_factory):
        self.config, self.name, self.caller = config, name, caller
        self.room = rtc.Room()
        self.source = rtc.AudioSource(16000, 1, queue_size_ms=200)
        self.client = api.LiveKitAPI(config.livekit_url, config.livekit_api_key, config.livekit_api_secret)
        self.finished, self.joined = asyncio.Event(), asyncio.Event()
        self.transport = LiveKitTransport(self.room, self.source, self.client, name, caller, self.finished)
        self.session = session_factory(self.transport)
        self.tasks = set()
        self.audio_task = None
        self.closed = False
        self.room.on('track_subscribed', self._track)
        self.room.on('participant_connected', self._participant)
        self.room.on('participant_disconnected', self._departed)
        self.room.on('disconnected', lambda reason: self.finished.set())
        self.room.on('data_received', self._data)

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        def done(value):
            self.tasks.discard(value)
            if not value.cancelled() and value.exception():
                event('rtc_task_error', self.session.call_id, self.session.turn_id, **safe_error_fields(value.exception()))
                self.finished.set()
        task.add_done_callback(done)
        return task

    def _participant(self, participant):
        if participant.identity == self.caller:
            self.joined.set()

    def _departed(self, participant):
        if participant.identity == self.caller:
            self.finished.set()

    def _track(self, track, publication, participant):
        if participant.identity != self.caller or track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        self.joined.set()
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
        self.audio_task = self._spawn(self._read_audio(track))

    async def _read_audio(self, track):
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1, frame_size_ms=20, capacity=100)
        try:
            async for item in stream:
                if self.finished.is_set():
                    break
                await self.session.feed_audio(bytes(item.frame.data))
        finally:
            await stream.aclose()

    def _data(self, packet):
        if not packet.participant or packet.participant.identity != self.caller or packet.topic != 'call.control' or len(packet.data) > 2048:
            return
        try:
            message = json.loads(packet.data)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(message, dict):
            return
        if message.get('type') == 'playback_done':
            self.session.acknowledge_playback(message.get('generation'))
        elif message.get('type') == 'interrupt':
            self._spawn(self.session.interrupt())
        elif message.get('type') == 'end':
            self.finished.set()
        elif message.get('type') == 'ready':
            self.joined.set()
            self._spawn(self.transport.event({'type': 'connected', 'generation': self.session.generation, 'call_id': self.session.call_id}))

    async def start(self):
        await asyncio.wait_for(self.room.connect(self.config.livekit_url, room_token(self.config, self.name, 'assistant')), 15)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(rtc.LocalAudioTrack.create_audio_track('assistant-voice', self.source), options)
        await self.session.start()

    async def watch(self):
        joined_task = asyncio.create_task(self.joined.wait())
        finished_task = asyncio.create_task(self.finished.wait())
        try:
            done, _ = await asyncio.wait({joined_task, finished_task}, timeout=self.config.call_connect_seconds, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise asyncio.TimeoutError
            if not self.finished.is_set():
                await asyncio.wait_for(self.finished.wait(), self.config.max_call_seconds)
        except asyncio.TimeoutError:
            event('rtc_call_timeout', self.session.call_id, self.session.turn_id)
        finally:
            for task in (joined_task, finished_task):
                task.cancel()
            await asyncio.gather(joined_task, finished_task, return_exceptions=True)
            await self.close()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.finished.set()
        await self.session.close()
        for task in self.tasks.copy():
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.source.clear_queue()
        try:
            # Remove remaining participants and room resources. Short-lived JWT
            # could recreate a room until expiry, but never creates a new bot.
            await self.client.room.delete_room(api.DeleteRoomRequest(room=self.name))
        except Exception as exc:
            event('rtc_cleanup_error', self.session.call_id, self.session.turn_id, **safe_error_fields(exc))
        await self.room.disconnect()
        await self.source.aclose()
        await self.client.aclose()
