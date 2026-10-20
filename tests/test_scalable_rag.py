import asyncio
import json
from types import SimpleNamespace
import httpx
import pytest
from app.rag.indexed import Scope, QdrantIndex, TurnRetrieval, read_manifest
from app.tts.speculative import CandidateAudio
from app.agents.orchestrator import Orchestrator
from app.observability.latency import TurnMetrics
from app.agents.prompts import FALLBACK
from scripts.publish_corpus import chunk_id

SCOPE = Scope('official_test', 'tenant-a', 'gold-receive', 'v2', 'model')


@pytest.mark.asyncio
async def test_hybrid_filters_each_branch_and_direct_id_rejects_wrong_scope():
    bodies=[]
    def handle(request):
        body=json.loads(request.content);bodies.append(body)
        if request.url.path.endswith('query'):
            assert body['filter']==SCOPE.filter()
            assert all(p['filter']==SCOPE.filter() for p in body['prefetch'])
            return httpx.Response(200,json={'result':{'points':[{'id':'a','payload':SCOPE.fields()|{'article':'a','text':'evidence'},'score':.01}]}})
        return httpx.Response(200,json={'result':[{'id':'wrong','payload':SCOPE.fields()|{'service':'deposit'}},{'id':'right','payload':SCOPE.fields()}]})
    async with httpx.AsyncClient(base_url='http://index',transport=httpx.MockTransport(handle)) as client:
        store=QdrantIndex('',client)
        assert await store.lookup(['invented-passage-id'],SCOPE)==[]
        assert (await store.search([1.,0.],'query',SCOPE,5))[0]['hybrid_match']
        assert [h['id'] for h in await store.lookup(['aaaaaaaa-0000-0000-0000-000000000000','bbbbbbbb-0000-0000-0000-000000000000'],SCOPE)]==['right']


@pytest.mark.asyncio
async def test_one_embedding_shared_main_independent_verifier_and_cancelled_work_joined():
    started=asyncio.Event(); release=asyncio.Event(); calls=[]
    class Owner:
        async def embed(self,texts):calls.append(texts);started.set();await release.wait();return [[1.,0.]]
        index=SimpleNamespace()
    queries=[]
    async def search(vector,text,scope,limit):queries.append(limit);return [{'article':str(limit)}]
    owner=Owner();owner.index.search=search
    turn=TurnRetrieval(owner,'query',[],SCOPE)
    await started.wait();release.set()
    main, verifier=await asyncio.gather(turn.search(),turn.verifier_search())
    assert len(calls)==1 and queries==[5,10] and main!=verifier
    await turn.close()
    release.clear();other=TurnRetrieval(owner,'query',[],SCOPE)
    await asyncio.sleep(0);await other.close()
    assert other._main.done() and other._verifier.done() and other._vector_task.done()


@pytest.mark.asyncio
async def test_private_audio_stays_bounded_and_cannot_stream_before_approval():
    class TTS:
        async def synthesize_stream(self,text):
            for _ in range(30):yield b'x'*3200
    gate=CandidateAudio(TTS(),'call',1,2,max_bytes=6400)
    key=await gate.start('exact candidate',0)
    await asyncio.sleep(.02)
    assert gate.queue.qsize()==2 and not gate.task.done()
    with pytest.raises(ValueError):
        await anext(gate.stream('exact candidate',('call',1,2)))
    gate.approve(key,'exact candidate')
    with pytest.raises(ValueError):await anext(gate.stream('changed candidate',('call',1,2)))
    with pytest.raises(ValueError):await anext(gate.stream('exact candidate',('different-call',1,2)))
    assert len(b''.join([x async for x in gate.stream('exact candidate',('call',1,2))]))==96000
    await gate.discard()
    assert gate.queue.empty() and not gate.approved


@pytest.mark.asyncio
async def test_finite_audio_completes_before_checker_and_retry_destroys_stale_pcm():
    class TTS:
        async def synthesize_stream(self,text):yield text.encode()
    gate=CandidateAudio(TTS(),'call',1,1)
    stale=await gate.start('rejected',0);await gate.task
    fresh=await gate.start('accepted',1);await gate.task
    with pytest.raises(ValueError):gate.approve(stale,'rejected')
    gate.approve(fresh,'accepted')
    assert b''.join([x async for x in gate.stream('accepted',('call',1,1))])==b'accepted'
    await gate.discard()


@pytest.mark.asyncio
async def test_synthesis_failure_never_unblocks_unapproved_audio_and_does_not_hang():
    class Broken:
        async def synthesize_stream(self,text):
            raise RuntimeError('failure');yield b''
    gate=CandidateAudio(Broken(),'call',1,1)
    key=await gate.start('candidate',0);await gate.task
    with pytest.raises(ValueError):await anext(gate.stream('candidate',('call',1,1)))
    gate.approve(key,'candidate')
    with pytest.raises(RuntimeError):await asyncio.wait_for(anext(gate.stream('candidate',('call',1,1))),1)
    await gate.discard()


@pytest.mark.asyncio
async def test_orchestrator_rejected_candidate_never_approved_retry_exact_binding():
    hit={'article':'a','text':'source','source':'official','section':'title','score':.01,'hybrid_match':True}
    class Search:
        async def search(self,q):return [hit]
    class Generator:
        i=0
        async def generate(self,*args):
            self.i+=1;return {'answer':str(self.i),'cited_articles':['a'],'claims':[{'text':'fact','article':'a','quote':'source'}],'insufficient_context':False},None
    class Checker:
        i=0
        async def check(self,q,hits,candidate):
            self.i+=1;await asyncio.sleep(.01);return {'action':'RETRY' if self.i==1 else 'PASS'}
    class TTS:
        async def synthesize_stream(self,text):yield text.encode()
    gate=CandidateAudio(TTS(),'c',1,1)
    async def state(value):pass
    answer,hits=await Orchestrator(Search(),Generator(),Checker()).answer('q',[],TurnMetrics('c',1),state,candidate_audio=gate)
    assert answer=='2' and gate.approved
    assert b''.join([x async for x in gate.stream(answer,('c',1,1))])==b'2'
    await gate.discard()


def test_manifest_rejects_synthetic_wrong_service_and_changed_scope(tmp_path):
    path=tmp_path/'active.json'
    item={'dataset_kind':'synthetic','collection':'bench_test','tenant':'a','service':'gold-receive','version':'v','model':'m','passages':10000}
    path.write_text(json.dumps(item))
    with pytest.raises(ValueError):read_manifest(path,'a','gold-receive','m')
    item.update(dataset_kind='official',collection='official_test');path.write_text(json.dumps(item))
    _,scope=read_manifest(path,'a','gold-receive','m')
    assert not scope.accepts(scope.fields()|{'corpus_version':'stale'})
    with pytest.raises(ValueError):read_manifest(path,'a','deposit','m')


def test_stable_content_ids_and_changed_source_version():
    assert chunk_id('source','text')==chunk_id('source','text')
    assert chunk_id('source','text')!=chunk_id('source','new text')


@pytest.mark.asyncio
async def test_weather_history_does_not_contaminate_explicit_gold_retrieval():
    queries=[]
    class Owner:
        async def embed(self,texts):queries.extend(texts);return [[1.,0.]]
        index=SimpleNamespace()
    async def search(*args):return []
    owner=Owner();owner.index.search=search
    history=[{'role':'user','content':'Хочу оформить Gold'},{'role':'user','content':'Какая погода?'}]
    turn=TurnRetrieval(owner,'Где оформить Gold?',history,SCOPE)
    await turn.search();await turn.close()
    assert queries==['Где оформить Gold?']
