/* Same synthetic microphone recording, real providers and decoded remote audio.
 * Run COHORT_LABEL=before|after. Small samples are observations, not p95 claims. */
const fs=require('fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
async function run(){
 const label=process.env.COHORT_LABEL||'after', count=Number(process.env.COHORT_TURNS||3);
 const report={label,input:'artifacts/service-gold.wav',syntheticInput:true,turns:[],browserErrors:[]};
 const browser=await chromium.launch({channel:'msedge',headless:true,args:['--autoplay-policy=no-user-gesture-required']});
 const page=await browser.newPage();page.on('pageerror',e=>report.browserErrors.push(e.message));
 try{
 await page.addInitScript(()=>{window.micContext=new AudioContext({sampleRate:48000});window.micOut=micContext.createMediaStreamDestination();const osc=micContext.createOscillator(),gain=micContext.createGain();gain.gain.value=0;osc.connect(gain).connect(micOut);osc.start();navigator.mediaDevices.getUserMedia=async()=>{await micContext.resume();return micOut.stream;};});
 await page.goto('http://127.0.0.1:8010');await page.click('#start');await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ПОДКЛЮЧЕН',{},{timeout:25000});
 await page.evaluate(()=>{
 window.events=[];window.probe={};
 room.on(LK.RoomEvent.DataReceived,(bytes,p,k,topic)=>{if(p?.identity!=='assistant'||topic!=='call.events')return;const item=JSON.parse(new TextDecoder().decode(bytes));events.push({...item,browserAt:Date.now()});if(item.type==='answer'){probe.answerAt=Date.now();probe.silence=false;}});
 const attach=track=>{if(track.kind!==LK.Track.Kind.Audio)return;const ctx=new AudioContext();ctx.resume();const source=ctx.createMediaStreamSource(new MediaStream([track.mediaStreamTrack])),meter=ctx.createScriptProcessor(512,1,1);source.connect(meter).connect(ctx.destination);meter.onaudioprocess=e=>{let peak=0;for(const value of e.inputBuffer.getChannelData(0))peak=Math.max(peak,Math.abs(value));if(probe.answerAt&&Date.now()>probe.answerAt+100){if(peak<.005)probe.silence=true;if(peak>.015&&probe.silence&&!probe.decodedAt)probe.decodedAt=Date.now();}};};
 room.on(LK.RoomEvent.TrackSubscribed,attach);for(const p of room.remoteParticipants.values())for(const publication of p.trackPublications.values())if(publication.track)attach(publication.track);
 });
 const wav=fs.readFileSync(report.input).toString('base64');
 for(let i=0;i<count;i++){
 const before=await page.evaluate(()=>events.filter(e=>e.type==='metrics').length);
 await page.evaluate(async wav=>{window.probe={};const bytes=Uint8Array.from(atob(wav),c=>c.charCodeAt(0)),buffer=await micContext.decodeAudioData(bytes.buffer),source=micContext.createBufferSource();source.buffer=buffer;source.connect(micOut);source.start();await new Promise(resolve=>source.onended=()=>{probe.inputEndedAt=Date.now();resolve();});},wav);
 await page.waitForFunction(n=>events.filter(e=>e.type==='metrics').length>n||events.some(e=>e.type==='error'),before,{timeout:90000});
 if(await page.evaluate(()=>events.some(e=>e.type==='error')))throw Error('Call error; inspect runtime trace');
 report.turns.push(await page.evaluate(()=>({answer:events.filter(e=>e.type==='answer').at(-1),metrics:events.filter(e=>e.type==='metrics').at(-1),probe,decodedAfterInputMs:probe.decodedAt-probe.inputEndedAt})));
 }
 await page.click('#end');report.status=report.browserErrors.length?'fail':'pass';
 }catch(e){report.status='fail';report.failure=e.message;}
 finally{await browser.close();fs.writeFileSync('artifacts/latency-'+label+'.json',JSON.stringify(report,null,2));}
 console.log(JSON.stringify({label,status:report.status,failure:report.failure,turns:report.turns.map(t=>({serverPCM:t.metrics.metrics_ms.end_of_turn_to_first_substantive_audio,browserDecoded:t.decodedAfterInputMs}))}));if(report.status!=='pass')process.exitCode=1;
}
run().catch(e=>{console.error(e.message);process.exitCode=1;});
