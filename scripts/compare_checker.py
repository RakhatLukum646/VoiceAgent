"""Bounded default-vs-minimal thinking comparison; independent semantic gate kept."""
import asyncio
import json
import statistics
from pathlib import Path
from time import perf_counter
from app.config import Settings
from app.agents.generator import GoogleChat
from app.agents.checker import AnswerChecker
from app.rag.indexed import IndexedRetriever,QdrantIndex,read_manifest


async def run():
    c=Settings();manifest,scope=read_manifest(c.corpus_manifest,c.tenant_scope,c.service_scope,c.embedding_model)
    index=QdrantIndex(c.qdrant_url)
    hits=await index.lookup([d['article'] for d in manifest['documents']],scope)
    by_url={hit['source'].rsplit('/',1)[-1]:hit for hit in hits}
    phone=by_url['q27'];branch=by_url['q43'];foreign=by_url['q14620']
    def candidate(text,hit,quote):
        return {'answer':text,'cited_articles':[hit['article']],'claims':[{'text':text,'article':hit['article'],'quote':quote}],'insufficient_context':False}
    phone_quote='Понадобится только мобильный телефон.'
    branch_quote=branch['text'].split('. ')[0]+'.'
    cases=[
        ('collection_supported','Я уже оформил Gold онлайн, что взять для получения в Kartomat?',candidate('Для получения в Kartomat понадобится только мобильный телефон.',phone,phone_quote),True),
        ('branch_supported','Что нужно для оформления Gold в отделении?',candidate('Понадобится мобильный телефон и оригинал документа, удостоверяющего личность, или его цифровая версия.',branch,branch_quote),True),
        ('wrong_applicability','Впервые открываю Gold в отделении. Достаточно только телефона?',candidate('Да, для первого открытия Gold в отделении достаточно только мобильного телефона.',phone,phone_quote),False),
        ('invented_fee','Сколько стоит обслуживание Gold?',candidate('Годовое обслуживание Gold всегда бесплатно.',phone,phone['text'].split('. ')[0]+'.'),False),
        ('foreign_condition','У меня нет ВНЖ Казахстана. Могу открыть Gold в приложении?',candidate('Да, вы можете оформить Gold в приложении без ВНЖ.',foreign,foreign['text'][:foreign['text'].index('. ')+1]),False),
    ]
    report={'model':c.google_llm_model,'cases':[],'production_gate':'independent full checker; no audio speculation published in this test'}
    try:
        for mode in ('default','minimal'):
            chat=GoogleChat(c.google_api_key,c.google_base_url,c.google_llm_model,thinking_level='minimal',explicit_thinking=mode=='minimal')
            try:
                checker=AnswerChecker(chat)
                for name,question,answer,should_pass in cases:
                    start=perf_counter();verdict=await checker.check(question,hits,answer)
                    item={'mode':mode,'case':name,'milliseconds':round((perf_counter()-start)*1000,2),'action':verdict['action'],'expected_pass':should_pass,'pass':(verdict['action']=='PASS')==should_pass}
                    report['cases'].append(item);print(json.dumps(item),flush=True)
            finally:await chat.client.aclose()
        report['status']='pass' if all(item['pass'] for item in report['cases']) else 'fail'
        report['median_ms']={mode:statistics.median(item['milliseconds'] for item in report['cases'] if item['mode']==mode) for mode in ('default','minimal')}
    finally:
        await index.client.aclose()
        Path('artifacts/checker-thinking-comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':asyncio.run(run())
