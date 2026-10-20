/* Addressed real Web SDK lifecycle tests; input is synthetic, never physical mic. */
const fs=require('fs');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
async function run(){
 const report={cases:[],browserErrors:[],rooms:[]};
 const browser=await chromium.launch({channel:process.env.TEST_BROWSER_CHANNEL||'msedge',headless:true,args:['--autoplay-policy=no-user-gesture-required']});
 const page=await browser.newPage({viewport:{width:900,height:1100}});
 page.on('pageerror',error=>report.browserErrors.push(error.message));
 let requests=0;
 page.on('response',async response=>{if(response.url().endsWith('/api/calls')){requests++;if(response.ok()){const item=await response.json();report.rooms.push(item.room);}}});
 try{
  await page.addInitScript(()=>{
   window.testAllow=false;window.testMic=new AudioContext({sampleRate:48000});window.testDestination=window.testMic.createMediaStreamDestination();
   const oscillator=testMic.createOscillator(),gain=testMic.createGain();gain.gain.value=0;oscillator.connect(gain).connect(testDestination);oscillator.start();
   navigator.mediaDevices.getUserMedia=async()=>{if(!testAllow)throw new DOMException('Denied for test','NotAllowedError');await testMic.resume();return testDestination.stream;};
  });
  await page.goto('http://127.0.0.1:8010');await page.click('#start');
  await page.waitForFunction(()=>document.getElementById('state').textContent.includes('Разрешите доступ'));
  if(requests!==0)throw Error('Denied microphone should not allocate a server call');
  report.cases.push({name:'microphone_refused',pass:true,serverCallsCreated:requests});
  await page.evaluate(()=>window.testAllow=true);await page.click('#start');
  await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ПОДКЛЮЧЕН',{},{timeout:25000});
  await page.evaluate(()=>{window.testEvents=[];room.on(LK.RoomEvent.DataReceived,(bytes,p,kind,topic)=>{if(p?.identity==='assistant'&&topic==='call.events')testEvents.push(JSON.parse(new TextDecoder().decode(bytes)));});});
  async function speak(name){const data=fs.readFileSync('artifacts/service-'+name+'.wav').toString('base64');await page.evaluate(async base64=>{const bytes=Uint8Array.from(atob(base64),c=>c.charCodeAt(0));const buffer=await testMic.decodeAudioData(bytes.buffer);const source=testMic.createBufferSource();source.buffer=buffer;source.connect(testDestination);source.start();await new Promise(resolve=>source.onended=resolve);},data);}
  await speak('gold');
  await page.waitForFunction(()=>testEvents.some(e=>e.type==='answer'&&e.action==='RAG'),{},{timeout:60000});
  const firstGeneration=await page.evaluate(()=>testEvents.find(e=>e.type==='answer').generation);
  await page.waitForTimeout(1500);
  await speak('weather');
  await page.waitForFunction(()=>testEvents.some(e=>e.type==='metrics'&&e.turn_id===2),{},{timeout:60000});
  const events=await page.evaluate(()=>testEvents);
  const latest=events.filter(e=>e.type==='answer').at(-1),completed=events.filter(e=>e.type==='metrics');
  if(latest.action!=='REDIRECT'||latest.generation<=firstGeneration||completed.some(e=>e.turn_id===1)||completed.at(-1).history_turns!==1)throw Error('Interruption did not cancel old response/history');
  if(events.some(e=>e.type==='audio_end'&&e.generation===firstGeneration))throw Error('Old audio completed after interruption');
  report.cases.push({name:'barge_in_cancels_old_turn',pass:true,oldGeneration:firstGeneration,newGeneration:latest.generation,completedTurns:completed.map(e=>e.turn_id),historyTurns:completed.at(-1).history_turns});
  await page.click('#mute');
  await page.waitForFunction(()=>[...room.localParticipant.audioTrackPublications.values()].every(p=>p.isMuted));
  await page.click('#mute');
  await page.waitForFunction(()=>[...room.localParticipant.audioTrackPublications.values()].every(p=>!p.isMuted));
  report.cases.push({name:'microphone_mute_unmute',pass:true});
  await page.click('#end');await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ЗАВЕРШЁН');
  report.cases.push({name:'manual_hangup',pass:true});
  await page.evaluate(()=>{testAllow=true;testDestination=testMic.createMediaStreamDestination();const gain=testMic.createGain(),oscillator=testMic.createOscillator();gain.gain.value=0;oscillator.connect(gain).connect(testDestination);oscillator.start();});
  await page.click('#start');await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ПОДКЛЮЧЕН',{},{timeout:25000});
  if(await page.locator('#transcript').textContent()!=='')throw Error('Repeated call retained prior transcript');
  await page.click('#end');await page.waitForFunction(()=>document.getElementById('connection').textContent==='ЗВОНОК ЗАВЕРШЁН');
  if(report.rooms.length!==2||report.rooms[0]===report.rooms[1])throw Error('Repeated call did not get separate room');
  report.cases.push({name:'repeat_call_isolated',pass:true});
  report.events=events;report.status=report.browserErrors.length?'fail':'pass';
 }catch(error){report.status='fail';report.failure=error.message;await page.screenshot({path:'artifacts/browser-lifecycle-failure.png',fullPage:true});}
 finally{await browser.close();fs.writeFileSync('artifacts/browser-lifecycle-smoke.json',JSON.stringify(report,null,2));}
 console.log(JSON.stringify({status:report.status,failure:report.failure,cases:report.cases,browserErrors:report.browserErrors}));
 if(report.status!=='pass')process.exitCode=1;
}
run().catch(error=>{console.error(error.message);process.exitCode=1;});
