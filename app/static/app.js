/* Official LiveKit Web SDK owns microphone and speaker audio tracks. */
const el = id => document.getElementById(id);
const LK = LivekitClient;
const labels = {connected:'Слушаю вас',listening:'Слушаю вас…',finalizing:'Распознаю вопрос…',routing:'Уточняю тему…',retrieving:'Проверяю информацию…',generating:'Готовлю ответ…',checking:'Проверяю ответ…',speaking:'Отвечаю…'};
let room=null, timer=null, started=0, muted=false, generation=0, partial=null, closing=false, currentState='', epoch=0;
let turnClock=0, liveTimer=null, liveSpan=null;
function stopLiveTimer(){clearInterval(liveTimer);liveTimer=null;if(liveSpan){liveSpan.classList.remove('live');liveSpan=null;}}
function startLiveTimer(span){stopLiveTimer();liveSpan=span;span.classList.add('live');liveTimer=setInterval(()=>{span.textContent='+'+((performance.now()-turnClock)/1000).toFixed(1)+' с';},100);}
let interrupted = false;
const decoder = new TextDecoder();
const audioElements = new Set();
function state(value) { currentState=value; el('state').textContent=labels[value]||value; el('orb').classList.toggle('speaking',value==='speaking'); }
function append(who,text) {
  const p=document.createElement('p');p.className='line '+who;const b=document.createElement('b');b.textContent=who==='user'?'Вы: ':'Помощник: ';p.append(b,document.createTextNode(text));
  if(who==='assistant'&&turnClock){const ts=document.createElement('span');ts.className='ts';ts.textContent='+'+((performance.now()-turnClock)/1000).toFixed(1)+' с';p.append(ts);}
  el('transcript').append(p);el('transcript').scrollTop=el('transcript').scrollHeight;return p;
}
async function control(payload, target=room) {if(target?.state===LK.ConnectionState.Connected) await target.localParticipant.publishData(new TextEncoder().encode(JSON.stringify(payload)),{reliable:true,topic:'call.control',destinationIdentities:['assistant']});}
function speakerMuted(value) {for(const audio of audioElements) audio.muted=value;}
function onEvent(bytes,participant,kind,topic) {
  if(participant?.identity!=='assistant'||topic!=='call.events') return;
  let m;try{m=JSON.parse(decoder.decode(bytes));}catch{return;}
  if(m.type==='interrupt'){generation=m.generation;interrupted=true;speakerMuted(true);state('Слушаю вас…');stopLiveTimer();}
  if(m.generation!==undefined&&m.generation<generation)return;
  if(m.type==='audio_start'){generation=m.generation;interrupted=false;speakerMuted(false);}
  if(m.type==='state')state(m.state);
  if(m.type==='connected')state('listening');
  if(m.type==='partial'){if(!partial)partial=append('user',m.text);else partial.lastChild.textContent=m.text;}
  if(m.type==='final'){turnClock=performance.now();stopLiveTimer();if(partial){partial.lastChild.textContent=m.text;partial=null;}else append('user',m.text);}
  if(m.type==='discard_partial'&&partial){partial.remove();partial=null;}
  if(['answer','acknowledgement','clarification'].includes(m.type)){
    const line=append('assistant',m.text);
    if(m.type==='acknowledgement')startLiveTimer(line.querySelector('.ts'));
    else stopLiveTimer();
  }
  if(m.type==='answer') {el('sources').textContent=m.sources?.length?'Ответ проверен по Kaspi Гид':'';if(m.action==='TRANSFER')state('Перевод на специалиста…');}
  if(m.type==='metrics'){const ms=m.metrics_ms.end_of_turn_to_first_substantive_audio;el('latency').textContent=ms===undefined?'':'Ответ начался через '+(ms/1000).toFixed(1)+' с после речи';}
  if(m.type==='audio_end') {
    // Server queue drained, including a silent tail. Playback availability and
    // bounded grace do not prove physical loudspeaker sound.
    const target=room, callEpoch=epoch;
    setTimeout(async()=>{if(epoch===callEpoch&&room===target&&m.generation===generation&&target?.canPlaybackAudio&&[...audioElements].some(a=>!a.paused&&!a.muted))await control({type:'playback_done',generation},target);},500);
  }
  if(m.type==='ended'){state(m.reason==='transfer'?'Демонстрационный перевод завершён':'Звонок завершён');el('endReason').textContent=m.reason==='transfer'?'Фраза перевода озвучена, звонок завершён. В демо специалист не подключается.':m.reason==='transfer_audio_failed'?'Фраза перевода не завершилась из-за ошибки воспроизведения.':'';}
  if(m.type==='error'){stopLiveTimer();state(m.message);}
  if(m.type==='ended')stopLiveTimer();
}
async function start() {
  const myEpoch=++epoch;el('start').disabled=true;closing=false;muted=false;generation=0;partial=null;interrupted=false;turnClock=0;
  el('transcript').replaceChildren();el('sources').textContent='';el('latency').textContent='';el('endReason').textContent='';stopLiveTimer();state('Подключение…');
  const candidate=new LK.Room({adaptiveStream:false,dynacast:false});room=candidate;
  candidate.on(LK.RoomEvent.DataReceived,onEvent);
  candidate.on(LK.RoomEvent.TrackSubscribed,(track,publication,participant)=>{if(track.kind===LK.Track.Kind.Audio&&participant.identity==='assistant'&&room===candidate){const audio=track.attach();audio.autoplay=true;audio.muted=interrupted;audioElements.add(audio);el('audio').append(audio);audio.play().catch(()=>{if(room===candidate)el('enableAudio').hidden=false;});}});
  candidate.on(LK.RoomEvent.TrackUnsubscribed,track=>{for(const audio of track.detach()){audioElements.delete(audio);audio.remove();}});
  candidate.on(LK.RoomEvent.AudioPlaybackStatusChanged,()=>{el('enableAudio').hidden=candidate.canPlaybackAudio;});
  candidate.on(LK.RoomEvent.Reconnecting,()=>state('Восстанавливаю соединение…'));
  candidate.on(LK.RoomEvent.Reconnected,()=>state('listening'));
  candidate.on(LK.RoomEvent.Disconnected,()=>{if(room===candidate)end(true);});
  try {
    await candidate.startAudio();
    const mic=await LK.createLocalAudioTrack({echoCancellation:true,noiseSuppression:true,channelCount:1});
    if(myEpoch!==epoch){mic.stop();return;}
    let response;
    try {response=await fetch('/api/calls',{method:'POST'});}catch(error){mic.stop();throw error;}
    if(!response.ok){mic.stop();const detail=await response.json();throw Error(detail.detail||'Не удалось соединиться');}
    const credentials=await response.json();
    try{await candidate.connect(credentials.url,credentials.token);await candidate.localParticipant.publishTrack(mic,{source:LK.Track.Source.Microphone});}catch(error){mic.stop();throw error;}
    if(myEpoch!==epoch){await candidate.disconnect();return;}
    await candidate.startAudio();await control({type:'ready'},candidate);
    el('connection').textContent='ЗВОНОК ПОДКЛЮЧЕН';el('start').hidden=true;el('mute').disabled=false;el('end').disabled=false;el('mute').textContent='Выключить микрофон';started=Date.now();
    timer=setInterval(()=>{const s=Math.floor((Date.now()-started)/1000);el('timer').textContent=String(Math.floor(s/60)).padStart(2,'0')+':'+String(s%60).padStart(2,'0');},1000);state('listening');
  } catch(error){if(myEpoch===epoch&&room===candidate){await end(true);state(error.name==='NotAllowedError'?'Разрешите доступ к микрофону и попробуйте снова.':error.message);}else{await candidate.disconnect(true);}}
}
async function end(remote=false) {
  if(closing)return;closing=true;++epoch;clearInterval(timer);stopLiveTimer();const old=room;room=null;
  if(!remote){try{await control({type:'end'},old);}catch{}state('Звонок завершён');}
  try{await old?.disconnect(true);}catch{}
  for(const audio of audioElements){audio.pause();audio.srcObject=null;audio.remove();}audioElements.clear();
  el('start').hidden=false;el('start').disabled=false;el('mute').disabled=true;el('end').disabled=true;el('enableAudio').hidden=true;el('connection').textContent='ЗВОНОК ЗАВЕРШЁН';
  if(remote&&currentState!=='Демонстрационный перевод завершён'&&!el('endReason').textContent)state('Звонок завершён');
}
el('start').onclick=start;el('end').onclick=()=>end();
el('mute').onclick=async()=>{if(!room)return;muted=!muted;await room.localParticipant.setMicrophoneEnabled(!muted);el('mute').textContent=muted?'Включить микрофон':'Выключить микрофон';};
el('enableAudio').onclick=async()=>{await room?.startAudio();for(const audio of audioElements)await audio.play();};
window.addEventListener('pagehide',()=>{room?.disconnect(true);});
