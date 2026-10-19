/* Chat panel. The model is a text completer, not a chatbot: you type the start of a line and it continues it.
   Each reply streams in one character at a time from /api/generate/stream. */
(function(){
'use strict';
const API=window.API_BASE||'';
const $=s=>document.querySelector(s);
const SUGGESTIONS=['ROMEO:','First Citizen:','HAMLET:','KING RICHARD III:','To be, or not to be'];

$('#tabs').innerHTML=`<button class="btn" data-tab="info" aria-pressed="true">Explain</button><button class="btn" data-tab="chat" aria-pressed="false">Chat</button>`;
$('#chat').innerHTML=`
  <div class="msgs" id="msgs"></div>
  <div class="sugg" id="sugg">${SUGGESTIONS.map(s=>`<button class="btn" data-sugg="${s}">${s}</button>`).join('')}</div>
  <label class="chatopt">Temperature <input type="range" id="ctemp" min="0.2" max="1.5" step="0.1" value="0.8"><b id="ctv">0.8</b></label>
  <form class="composer" id="cform"><input id="cin" placeholder="Start a line…" maxlength="200" autocomplete="off" spellcheck="false"><button class="btn" id="csend">Send</button></form>`;

const msgs=$('#msgs');
function bubble(kind,html){
  const b=document.createElement('div');b.className='bub '+kind;b.innerHTML=html;msgs.appendChild(b);msgs.scrollTop=msgs.scrollHeight;return b;
}
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
bubble('bot','Hi! I’m a tiny <b>text completer</b> trained on Shakespeare. I’m not a chatbot, so I won’t answer questions. Type the start of a line and I’ll continue it, one character at a time.');

function showTab(name){
  $('#info').hidden=name!=='info';$('#chat').hidden=name!=='chat';
  document.querySelectorAll('[data-tab]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.tab===name));
}
showTab('info');

let busy=false;
async function send(text){
  if(busy)return;busy=true;$('#csend').disabled=true;
  bubble('me',esc(text));
  const out=bubble('bot','<span class="gen"></span>');
  const gen=out.querySelector('.gen');let full='';
  try{
    const res=await fetch(API+'/api/generate/stream',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,max_new_tokens:150,temperature:+$('#ctemp').value})});
    if(!res.ok)throw new Error('the model returned '+res.status);
    const reader=res.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){
      const {done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      let i;
      while((i=buf.indexOf('\n\n'))>=0){
        const line=buf.slice(0,i).replace(/^data: /,'');buf=buf.slice(i+2);
        if(line==='[DONE]')continue;
        full+=JSON.parse(line);gen.textContent=full;msgs.scrollTop=msgs.scrollHeight;
      }
    }
    const tools=document.createElement('div');tools.className='tools';
    tools.innerHTML='<button class="btn">Visualize this</button>';
    tools.firstChild.onclick=()=>visualize(text+full);
    out.appendChild(tools);
  }catch(e){
    out.innerHTML='Can’t reach the model right now ('+esc(e.message)+'). Free hosting sleeps when idle, so give it a minute and try again.';
  }
  busy=false;$('#csend').disabled=false;msgs.scrollTop=msgs.scrollHeight;
}
async function visualize(full){
  $('#prompt').value=full;
  showTab('info');
  await window.gptApp.applyText(full);
  window.gptApp.go('attn');
}

$('#cform').addEventListener('submit',e=>{
  e.preventDefault();const v=$('#cin').value;
  if(!v.trim())return;
  $('#cin').value='';send(v);
});
document.addEventListener('click',e=>{
  const t=e.target.closest('[data-tab],[data-sugg]');if(!t)return;
  if(t.dataset.tab)showTab(t.dataset.tab);
  else send(t.dataset.sugg);
});
document.addEventListener('input',e=>{if(e.target.id==='ctemp')$('#ctv').textContent=(+e.target.value).toFixed(1)});
})();
