import asyncio
import pytest
from app.call.session import CallSession
from app.agents.orchestrator import Orchestrator
from app.agents.generator import AnswerGenerator
from app.agents.prompts import FALLBACK


class ASR:
    async def start(self,partial,final):self.partial,self.final=partial,final
    async def close(self):pass


class Transport:
    def __init__(self):self.events=[];self.pcm=[]
    async def event(self,value):self.events.append(value)
    async def audio(self,value,generation):self.pcm.append(value)


@pytest.mark.asyncio
async def test_full_response_queue_pending_checker_no_secret_pcm_then_barge_in_discards():
    blocked=asyncio.Event();cancelled=asyncio.Event()
    hit={'article':'a','text':'official','source':'s','section':'title','score':1.}
    class Prepared:
        timings={}
        async def search(self,q):return [hit]
        async def verifier_search(self,q):return [hit]
        async def close(self):pass
    class Generator:
        async def generate(self,*args):return {'answer':'private candidate','cited_articles':['a'],'claims':[{'text':'fact','article':'a','quote':'official'}],'insufficient_context':False},None
    class Checker:
        async def check(self,*args,**kwargs):blocked.set();await asyncio.sleep(60)
    class TTS:
        async def synthesize_stream(self,text):
            if text!='private candidate':yield b'ACK';return
            try:
                for _ in range(300):yield b'SECRET'*530
            finally:cancelled.set()
    rag=Orchestrator(Prepared(),Generator(),Checker())
    class Scoped:
        def prepare(self,*args):return Prepared()
        async def answer(self,*args,**kwargs):return await rag.answer(*args,**kwargs)
    transport=Transport();session=CallSession(transport,ASR(),Scoped(),TTS())
    await session.start();await session.asr.final('test question')
    await asyncio.wait_for(blocked.wait(),1);await asyncio.sleep(.02)
    assert all(b'SECRET' not in item for item in transport.pcm)
    await asyncio.wait_for(session.interrupt(),1)
    assert cancelled.is_set() and session.history==[]
    assert all(b'SECRET' not in item for item in transport.pcm)
    await session.close()


@pytest.mark.asyncio
async def test_empty_insufficient_context_is_safe_fallback_not_invalid_json():
    class Chat:
        async def json_completion(self,*args,**kwargs):return {'answer':'','cited_articles':[],'claims':[],'insufficient_context':True},None
    answer,_=await AnswerGenerator(Chat()).generate('unknown',[],[])
    assert answer['answer']==FALLBACK and answer['insufficient_context']
