/* Integration test: actual Web SDK, synthetic microphone stream and remote RTC audio.
 * This is browser automation for software testing, not a physical microphone test. */
const fs=require('fs');
const path=require('path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
async function run(){
 const report={test:'LiveKit Web SDK synthetic speech',events:[],browserErrors:[],scenarios:[]};
 const browser=await chromium.launch({channel:process.env.TEST_BROWSER_CHANNEL||'msedge',headless:true,args:['--autoplay-policy=no-user-gesture-required','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']});
 const page=await browser.newPage({viewport:{width:900,height:1100}});
 page.on('pageerror',e=>report.browserErrors.push(e.message));
 try{
  await page.addInitScript(()=>{
   window.testMic=new AudioContext({sampleRate:48000});
   window.testDestination=window.testMic.createMediaStreamDestination();
   const oscillator=window.testMic.createOscillator(),gain=window.testMic.createGain();gain.gain.value=0;
   oscillator.connect(gain).connect(window.testDestination);oscillator.start();
   navigator.mediaDevices.getUserMedia=async()=>{await window.testMic.resume();return window.testDestination.stream;};
  });
  await page.goto('http://127.0.0.1:8010');
  await page.click('#start');
  await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ПОДКЛЮЧЕН',{},{timeout:25000});
  await page.evaluate(()=>{
   window.testEvents=[];window.testPeak=0;window.testReceivedSamples=0;
   room.on(LK.RoomEvent.DataReceived,(bytes,p,kind,topic)=>{if(p?.identity==='assistant'&&topic==='call.events')testEvents.push({...JSON.parse(new TextDecoder().decode(bytes)),testAt:Date.now()});});
   const attach=(track)=>{if(track.kind!==LK.Track.Kind.Audio)return;const context=new AudioContext();context.resume();const source=context.createMediaStreamSource(new MediaStream([track.mediaStreamTrack]));const meter=context.createScriptProcessor(1024,1,1);source.connect(meter);meter.connect(context.destination);meter.onaudioprocess=e=>{const input=e.inputBuffer.getChannelData(0);for(const sample of input)window.testPeak=Math.max(window.testPeak,Math.abs(sample));window.testReceivedSamples+=input.length;};};
   room.on(LK.RoomEvent.TrackSubscribed,attach);
   for(const participant of room.remoteParticipants.values())for(const publication of participant.trackPublications.values())if(publication.track)attach(publication.track);
  });
  const sequence=process.argv.includes('--quick')?['credit']:['gold','weather','unknown','credit'];
  for(const name of sequence){
   const before=await page.evaluate(()=>window.testEvents.filter(e=>e.type==='metrics').length);
   const base64=fs.readFileSync(path.join('artifacts','service-'+name+'.wav')).toString('base64');
   const started=Date.now();
   await page.evaluate(async base64=>{const bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0));const buffer=await window.testMic.decodeAudioData(bytes.buffer);const source=window.testMic.createBufferSource();source.buffer=buffer;source.connect(window.testDestination);source.start();await new Promise(resolve=>source.onended=resolve);},base64);
   await page.waitForFunction(n=>window.testEvents.filter(e=>e.type==='metrics').length>n,before,{timeout:90000});
   const last=await page.evaluate(()=>({answer:window.testEvents.filter(e=>e.type==='answer').at(-1),metrics:window.testEvents.filter(e=>e.type==='metrics').at(-1),peak:window.testPeak,receivedSamples:window.testReceivedSamples}));
   report.scenarios.push({name,...last,wallMs:Date.now()-started});
   if(name==='credit')await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ЗАВЕРШЁН',{},{timeout:10000});
  }
  report.events=await page.evaluate(()=>window.testEvents);
  report.disconnected=await page.locator('#connection').textContent()==='ЗВОНОК ЗАВЕРШЁН';
  const expected={gold:'RAG',weather:'REDIRECT',unknown:'RAG',credit:'TRANSFER'};
  for(const result of report.scenarios){if(result.answer.action!==expected[result.name]||result.peak<.01)throw Error('Scenario or media failed: '+result.name);}
  if(!report.disconnected||report.browserErrors.length)throw Error('Browser lifecycle failed');
  report.status='pass';
  await page.screenshot({path:'artifacts/browser-service-demo.png',fullPage:true});
 }catch(error){report.status='fail';report.failure=error.message;await page.screenshot({path:'artifacts/browser-service-failure.png',fullPage:true});}
 finally{fs.writeFileSync(process.argv.includes('--quick')?'artifacts/browser-launch-smoke.json':'artifacts/browser-service-smoke.json',JSON.stringify(report,null,2));await browser.close();}
 console.log(JSON.stringify({status:report.status,failure:report.failure,scenarios:report.scenarios.map(x=>({name:x.name,action:x.answer?.action,peak:x.peak,wallMs:x.wallMs})),browserErrors:report.browserErrors}));
 if(report.status!=='pass')process.exitCode=1;
}
run().catch(error=>{console.error(error.message);process.exitCode=1;});
