"""10k DISTINCT synthetic non-banking records, local vectors, real indexed Qdrant.
Never publishes an official manifest. No embedding/model provider is called.
"""
import asyncio
import json
import math
import random
import statistics
import time
import uuid
from pathlib import Path
from app.config import Settings
from app.rag.indexed import QdrantIndex, Scope, bm25

N=10000
DIM=768
TOPICS=['астрономия','ботаника','геология','акустика','механика','оптика','экология','археология','керамика','геометрия','робототехника','метеорология']


def vector(rng, centre):
    return [value+rng.gauss(0,.04) for value in centre]


def percentile(values,q):
    return sorted(values)[min(len(values)-1,math.ceil(q*len(values))-1)]


async def run():
    index=QdrantIndex(Settings().qdrant_url)
    collection='bench_synthetic_10000_v1'
    scope=Scope(collection,'benchmark','catalog-a','v2','synthetic-local-768')
    rng=random.Random(2197)
    centres=[[rng.gauss(0,.3) for _ in range(DIM)] for _ in range(96)]
    report={'dataset_kind':'synthetic','banking_relevance_claim':False,'paid_embeddings':0,'points':N,'dimension':DIM,'collection':collection,'concurrency':4}
    start=time.perf_counter()
    try:
        exists=await index.client.get(f'/collections/{collection}')
        if exists.status_code==200:
            await index.request('DELETE',f'/collections/{collection}')
        await index.request('PUT',f'/collections/{collection}',{'vectors':{'dense':{'size':DIM,'distance':'Cosine'}},'sparse_vectors':{'bm25':{'modifier':'idf'}},'optimizers_config':{'indexing_threshold':1000},'hnsw_config':{'m':32,'ef_construct':200,'full_scan_threshold':10}})
        for field in scope.fields():await index.request('PUT',f'/collections/{collection}/index?wait=true',{'field_name':field,'field_schema':'keyword'})
        for offset in range(0,N,100):
            points=[]
            for i in range(offset,min(offset+100,N)):
                topic=TOPICS[i%len(TOPICS)]
                text=f'СИНТЕТИЧЕСКИЙ ТЕСТ. Каталог {topic}. Образец X{i:05d}. Серия {i//96}. Наблюдение {i*17%1009}; глубина {i%407} условных единиц; режим {i%7}. '+('Исследование измерений образца и контроль точности. '*(1+i%5))
                fields=scope.fields()|{'service':'catalog-b' if i%4==0 else 'catalog-a','corpus_version':'v1' if i%9==0 else 'v2','language':'kk' if i%11==0 else 'ru'}
                payload=fields|{'text':text,'article':str(uuid.uuid5(uuid.NAMESPACE_URL,'synthetic:'+str(i))),'section':topic,'source':'synthetic://catalog/'+str(i),'document':f'Образец X{i:05d}','dataset_kind':'synthetic','version':'v2','retrieved_at':'synthetic'}
                points.append({'id':payload['article'],'vector':{'dense':vector(rng,centres[i%96]),'bm25':bm25(text)},'payload':payload})
            await index.request('PUT',f'/collections/{collection}/points?wait=true',{'points':points})
        report['ingestion_seconds']=round(time.perf_counter()-start,2)
        deadline=time.monotonic()+180
        while True:
            info=await index.request('GET',f'/collections/{collection}')
            if info['indexed_vectors_count']>=2*N and info['status']=='green':break
            if time.monotonic()>deadline:raise RuntimeError('HNSW indexing did not finish; no indexed benchmark claim')
            await asyncio.sleep(1)
        report['index']={k:info[k] for k in ('points_count','indexed_vectors_count','status','optimizer_status')}
        report['hnsw']=info['config']['hnsw_config']
        report['in_scope_count']=await index.count(scope)
        timings=[];recalls=[];leaks=0
        semaphore=asyncio.Semaphore(4)
        async def probe(number):
            nonlocal leaks
            local=random.Random(number+773)
            query=vector(local,centres[number%96]);text=TOPICS[number%12]+' образец измерения X'+str(number*173%N).zfill(5)
            async with semaphore:
                start=time.perf_counter()
                hits=await index.search(query,text,scope,8)
                timings.append((time.perf_counter()-start)*1000)
                leaks+=sum(not scope.accepts(hit) for hit in hits)
                body={'query':query,'using':'dense','filter':scope.filter(),'limit':8,'with_payload':False,'params':{'hnsw_ef':256}}
                approximate=await index.request('POST',f'/collections/{collection}/points/query',body)
                body['params']={'exact':True}
                exact=await index.request('POST',f'/collections/{collection}/points/query',body)
                expected={p['id'] for p in exact['points']}
                recalls.append(len(expected&{p['id'] for p in approximate['points']})/len(expected))
        await asyncio.gather(*(probe(i) for i in range(40)))
        report.update(status='pass' if leaks==0 and statistics.mean(recalls)>=.9 else 'fail',scope_leaks=leaks,dense_ann_recall_at8=statistics.mean(recalls),query_hnsw_ef=256,queries=40,hybrid_p50_ms=statistics.median(timings),hybrid_p95_ms=percentile(timings,.95),notes='Synthetic clustered vectors + distinct nonbanking text. indexed_vectors_count counts dense+sparse. Tests ANN/storage/scope only; no claim about real Gemini semantics, voice or p95 under production load.')
    finally:
        await index.client.aclose()
        Path('artifacts/index-benchmark-10000.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':asyncio.run(run())
