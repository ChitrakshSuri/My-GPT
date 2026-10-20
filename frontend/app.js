/* GPT Anatomy: draws the tensors that /api/trace returns for the trained model.
   The page shows small slices (first 16 of 384 numbers, 64 of 1536 neurons) so every cell stays readable. */
(function(){
'use strict';
const API=window.API_BASE||'';
const $=s=>document.querySelector(s);
const D=16, FF=64, MAXT=12;           // numbers shown per token, hidden neurons shown, characters shown
let CHARS=[], NH=6, NLAYER=6, TOKTABLE=[], PARAMS=null;
const show=c=>c==='\n'?'↵':c===' '?'␣':c;
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const maxAbs=M=>Math.max(1e-9,...M.flat(Infinity).map(Math.abs));
const softmax=(a,t=1)=>{const m=Math.max(...a);const e=a.map(x=>Math.exp((x-m)/t));const s=e.reduce((p,q)=>p+q,0);return e.map(x=>x/s)};

/* ---------- talking to the backend ---------- */
async function api(path,body){
  const res=await fetch(API+path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:undefined);
  if(!res.ok)throw new Error(path+' returned '+res.status);
  return res.json();
}
let reqId=0,busyTimer=null;
function setBusy(on){
  clearTimeout(busyTimer);
  if(on)busyTimer=setTimeout(()=>$('#busy').classList.add('on'),350);   // only show it if the model is slow
  else $('#busy').classList.remove('on');
}
function showError(e){
  $('#err').innerHTML=`Can’t reach the model right now (${e.message}). Free hosting goes to sleep when nobody visits, so it may need a minute. <button class="btn" id="retry">Try again</button>`;
  $('#err').classList.add('on');
}

/* ---------- the data the scenes draw ---------- */
const S={scene:'overview',head:0,layer:0,sel:null,hov:null,mask:true,explode:.35,resid:1,temp:1,hoverPart:null,valid:''};
let tokens=[],R=null,T=0;
function newLive(){return{n:0,done:false,fitted:false,steps:[],shown:-1,playing:true,timer:null}}   // steps = every letter so far, shown = the one on screen
S.live=newLive();S.liveLayer=5;S.liveHead='avg';   // the live view: latest step, block and head being watched
async function load(text){
  const id=++reqId;setBusy(true);
  try{
    const d=await api('/api/trace',{text,max_tokens:MAXT,layers:[S.layer],include_hidden:true});
    if(id!==reqId)return false;
    adopt(d);setBusy(false);$('#err').classList.remove('on');return true;
  }catch(e){if(id===reqId){setBusy(false);showError(e)}return false}
}
function adopt(d){
  NH=d.config.n_head;NLAYER=d.config.n_layer;
  tokens=d.ids.map((id,i)=>({ch:d.tokens[i],id}));T=d.T;
  const L=d.layers[0],sl=M=>M.map(r=>r.slice(0,D));
  const stat=L['in'].map(r=>{const m=r.reduce((p,q)=>p+q,0)/r.length;return{m,sd:Math.sqrt(r.reduce((p,q)=>p+(q-m)*(q-m),0)/r.length)}});
  R={T,tok:sl(d.embeddings.token),pos:sl(d.embeddings.position),x0:sl(d.embeddings.sum),
     xin:sl(L['in']),n1:sl(L.ln1),n2:sl(L.ln2),attn:sl(L.attn),x1:sl(L.after_attn),ffn:sl(L.ffn),x2:sl(L.out),stat,
     heads:L.heads.map(h=>({A:h.weights,S:h.scores})),fv:L.ffn_view,logits:d.final.logits};
  if(S.sel===null||S.sel>=T)S.sel=T-1;
}
/* raw text from the message box -> the characters the model knows, shown in a window of the last MAXT */
/* remembers the usable characters, writes the note under the flow chart, and returns the window to show (or null) */
function describeText(raw){
  const chars=[...raw],ok=chars.filter(c=>CHARS.includes(c)),bad=[...new Set(chars.filter(c=>!CHARS.includes(c)))];
  S.valid=ok.join('');
  const st=$('#pstat');
  if(ok.length<2){st.textContent='Type at least 2 characters the model knows (letters, digits and basic punctuation).';return null}
  st.textContent=(ok.length>MAXT?`Showing the last ${MAXT} of ${ok.length} characters, because a GPT only sees a limited window (block_size in gpt.py). `:'')+
    (bad.length?`Skipped, not in the 65-character vocabulary: ${bad.map(show).join(' ')}`:'');
  return ok.slice(-MAXT).join('');
}
async function applyText(raw,fresh=true){
  const win=describeText(raw);
  if(win===null)return false;
  const good=await load(win);
  if(good){renderFlow();build(!fresh)}
  return good;
}
/* no network: only updates the note and the first flow chart box (the live view does not need a trace) */
function previewText(raw){
  const win=describeText(raw);
  if(win===null)return false;
  const n=document.querySelector('.node[data-part="text"] .tx');
  if(n)n.textContent='“'+[...win].map(show).join('')+'”';
  return true;
}

/* ---------- what each part is (paraphrasing Karpathy's lecture) ---------- */
const PARTS={
 text:{name:'Text',tag:'input',bg:'#f3e3a0',scene:'embed',short:'Raw characters. The model will extend them one letter at a time.',
  desc:'This is the message you typed. A GPT does one job: look at the text so far and guess the next character, then do it again, and again.',code:'gpt.py:230  context',shape:'string'},
 tokens:{name:'Tokens',tag:'turn text into numbers',bg:'#f3e3a0',scene:'embed',short:'Each character becomes an integer id.',
  desc:'Neural nets only understand numbers, so every character gets an id. This version is character-level: tiny Shakespeare has 65 distinct characters, so ids run from 0 to 64. Real GPTs use bigger chunks of text as tokens.',code:'gpt.py:28-34  stoi / encode',shape:'(T,)  ints in [0, 65)'},
 embed:{name:'Embeddings',tag:'lookup table',bg:'#f3c6ec',scene:'embed',short:'Token vector + position vector.',
  desc:'Each id picks one row of a learned table: a list of numbers that describes that character. A second table gives each position its own vector, because attention alone has no sense of where a token sits. The two vectors are added.',code:'gpt.py:145-146,166-168',shape:'(T,) → (T, C)   C = 384'},
 norm1:{name:'Layer norm',tag:'keeps numbers tame',bg:'#e4eeb2',scene:'norm',short:'Rescales each token’s vector to mean 0, spread 1.',
  desc:'Like batch norm, except it works on each token’s own vector instead of across the batch. Every row ends up with mean 0 and standard deviation 1, which keeps a deep stack of layers trainable. Your gpt.py normalises before attention (pre-norm).',code:'gpt.py:132,136  ln1',shape:'(T, C) → (T, C)'},
 norm2:{name:'Layer norm',tag:'keeps numbers tame',bg:'#e4eeb2',scene:'norm',short:'Rescales each token’s vector to mean 0, spread 1.',
  desc:'The same normalisation, applied again before the feed-forward layer.',code:'gpt.py:133,137  ln2',shape:'(T, C) → (T, C)'},
 normf:{name:'Final norm',tag:'keeps numbers tame',bg:'#e4eeb2',scene:'norm',short:'One last normalisation before predicting.',
  desc:'After the last block, one more layer norm tidies the vectors before they are turned into scores.',code:'gpt.py:148,170  ln_f',shape:'(T, C) → (T, C)'},
 attn:{name:'Attention',tag:'communication',bg:'#f2d9b1',scene:'attn',short:'Tokens look back at earlier tokens and pull in what matters.',
  desc:'Every token emits a query (what am I looking for?), a key (what do I contain?) and a value (what I will give you). Query times key says how much two tokens care about each other. A triangular mask hides the future, softmax turns scores into weights, and each token takes a weighted sum of the values. Six heads do this in parallel, so there are six kinds of conversation.',code:'gpt.py:78-92  Head · 94-106  MultiHeadAttention',shape:'(T, C) → (T, C)'},
 add1:{name:'Add',tag:'residual connection',bg:'#ffffff',scene:'resid',short:'Keep the input, add the sub-layer’s result.',
  desc:'The input skips past the sub-layer and the sub-layer’s output is added on top. That gives gradients a straight road from the loss back to the input, so each block only has to learn a small correction.',code:'gpt.py:136  x = x + self.sa(…)',shape:'(T, C) + (T, C)'},
 add2:{name:'Add',tag:'residual connection',bg:'#ffffff',scene:'resid',short:'Keep the input, add the sub-layer’s result.',
  desc:'The same shortcut again, around the feed-forward layer. After this the block is finished and the result goes into the next block.',code:'gpt.py:137  x = x + self.ffwd(…)',shape:'(T, C) + (T, C)'},
 ffn:{name:'Feed forward',tag:'computation',bg:'#bfeaea',scene:'ffn',short:'Each token thinks on its own: widen 4x, ReLU, narrow.',
  desc:'Attention was the talking between tokens. This is the thinking: every token goes through the same small network by itself. A Linear layer makes the vector 4 times wider (384 to 1536), ReLU switches off the negative values, and a second Linear brings it back to 384.',code:'gpt.py:108-121  FeedFoward',shape:'(C) → (4C) → (C)'},
 linear:{name:'Linear',tag:'scores',bg:'#c3cff0',scene:'output',short:'One score for each of the 65 possible next characters.',
  desc:'A last Linear layer (lm_head) turns the final vector of the last token into 65 raw scores, called logits. A higher score means that character looks more likely to come next.',code:'gpt.py:149,171  lm_head',shape:'(T, C) → (T, 65)'},
 softmax:{name:'Softmax',tag:'probabilities',bg:'#bfe9b8',scene:'output',short:'Scores become probabilities that add up to 1.',
  desc:'Softmax exponentiates the scores and divides by their sum, so you get probabilities. Dividing the scores by a temperature first makes the choice more predictable (low) or more adventurous (high).',code:'gpt.py:193  F.softmax',shape:'(65,) → (65,)'},
 next:{name:'Next letter',tag:'output',bg:'#f5d98a',scene:'output',short:'Sample a character, append it, run it all again.',
  desc:'The next character is drawn at random in proportion to those probabilities, added to the text, and the whole pipeline runs again on the longer text. That loop is how a GPT writes.',code:'gpt.py:195-197  multinomial, cat',shape:'1 id'},
 q:{name:'Query',tag:'what am I looking for?',bg:'#f29a9a',short:'The question this token asks.',desc:'Each token multiplies its vector by a learned matrix to make a query: what it is looking for among earlier tokens.',code:'gpt.py:71,83',shape:'(T, head_size)'},
 k:{name:'Key',tag:'what do I contain?',bg:'#9ab9f2',short:'What this token offers to others.',desc:'The key is the token’s label: what it contains. A query that lines up with a key (large dot product) means those two tokens should talk.',code:'gpt.py:70,82',shape:'(T, head_size)'},
 v:{name:'Value',tag:'what I hand over',bg:'#a9dc9c',short:'What a token passes on if it is attended to.',desc:'The value is the actual information a token shares. The weights decide how much of each value flows into the receiving token.',code:'gpt.py:72,90',shape:'(T, head_size)'},
 o:{name:'Head output',tag:'weighted mix',bg:'#f5d98a',short:'Each token’s weighted sum of values.',desc:'The output for a token is the sum of all the value vectors it can see, each scaled by its attention weight. Outputs from all six heads are joined and projected back to 384 numbers.',code:'gpt.py:91,104-105',shape:'(T, head_size)'},
 hid:{name:'Hidden neurons',tag:'the wide layer',bg:'#bfeaea',short:'4x wider. Empty ones were switched off by ReLU.',desc:'The wide middle layer. ReLU turns every negative value into zero, so only some neurons fire for a given token. Here the first 64 of the 1536 are drawn.',code:'gpt.py:114-115',shape:'(4C)'}
};

/* ---------- three.js ---------- */
const INK=0x111214;
const COL={tok:0xf3e3a0,embed:0xf3c6ec,norm:0xe4eeb2,attn:0xf2d9b1,ffn:0xbfeaea,add:0xffffff,lin:0xc3cff0,soft:0xbfe9b8,q:0xf29a9a,k:0x9ab9f2,v:0xa9dc9c,o:0xf5d98a,next:0xf5d98a};
const POS=new THREE.Color(0x2a4fd6),NEG=new THREE.Color(0xd8602a),WHITE=new THREE.Color(0xffffff);
const cvs=$('#gl'),stage=$('#stage');
const renderer=new THREE.WebGLRenderer({canvas:cvs,antialias:true});
renderer.setClearColor(0xffffff,1);renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
const scene=new THREE.Scene();
const camera=new THREE.PerspectiveCamera(42,1,.1,600);
const controls=new THREE.OrbitControls(camera,cvs);
controls.enableDamping=true;controls.dampingFactor=.08;controls.autoRotateSpeed=1.2;
scene.add(new THREE.AmbientLight(0xffffff,.78));
const sun=new THREE.DirectionalLight(0xffffff,.6);sun.position.set(12,24,18);scene.add(sun);
let root=new THREE.Group();scene.add(root);
let interactives=[],camGoal=null,tween=0,viewDist=40;

function resize(){const w=cvs.clientWidth,h=cvs.clientHeight;if(!w||!h)return;renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix()}
new ResizeObserver(resize).observe(cvs);

function clearRoot(){
  scene.remove(root);
  root.traverse(o=>{if(o.geometry)o.geometry.dispose();if(o.material&&!o.isSprite)o.material.dispose()});
  root=new THREE.Group();scene.add(root);interactives=[];
}
const lambert=(c,o)=>new THREE.MeshLambertMaterial(Object.assign({color:c},o||{}));
function box(parent,w,h,d,color,pos,part,o={}){
  const g=new THREE.BoxGeometry(w,h,d);
  const m=new THREE.Mesh(g,lambert(color,o.opacity?{transparent:true,opacity:o.opacity,depthWrite:false}:{}));
  m.position.set(pos[0],pos[1],pos[2]);
  m.add(new THREE.LineSegments(new THREE.EdgesGeometry(g),new THREE.LineBasicMaterial({color:INK,transparent:true,opacity:o.edge??1})));
  if(part){m.userData.part=part;interactives.push(m)}
  Object.assign(m.userData,o.data||{});parent.add(m);return m;
}
function ball(parent,r,color,pos,part,data){
  const m=new THREE.Mesh(new THREE.SphereGeometry(r,18,14),lambert(color));
  m.position.set(pos[0],pos[1],pos[2]);
  m.add(new THREE.Mesh(new THREE.SphereGeometry(r*1.02,18,14),new THREE.MeshBasicMaterial({color:INK,wireframe:true,transparent:true,opacity:.18})));
  if(part){m.userData.part=part;interactives.push(m)}
  Object.assign(m.userData,data||{});parent.add(m);return m;
}
const texCache={};
function label(parent,text,pos,size=1,o={}){
  const font=o.font||'Humor Sans',color=o.color||'#111214',key=text+'|'+font+'|'+color;
  let t=texCache[key];
  if(!t){
    const c=document.createElement('canvas'),x=c.getContext('2d');
    x.font=`700 72px "${font}"`;const w=Math.ceil(x.measureText(text).width)+24;
    c.width=w;c.height=110;x.font=`700 72px "${font}"`;x.fillStyle=color;x.textAlign='center';x.textBaseline='middle';x.fillText(text,w/2,58);
    const tex=new THREE.CanvasTexture(c);tex.minFilter=THREE.LinearFilter;
    t=texCache[key]={tex,aspect:w/110};
  }
  const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t.tex,transparent:true,depthTest:false}));
  s.scale.set(size*t.aspect,size,1);s.position.set(pos[0],pos[1],pos[2]);s.renderOrder=20;
  parent.add(s);return s;
}
function edge(parent,a,b,r,color,opacity=1){
  const A=new THREE.Vector3(...a),B=new THREE.Vector3(...b),len=A.distanceTo(B);
  const m=new THREE.Mesh(new THREE.CylinderGeometry(r,r,len,6,1),new THREE.MeshBasicMaterial({color,transparent:opacity<1,opacity,depthWrite:opacity>=.9}));
  m.position.copy(A).add(B).multiplyScalar(.5);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),B.clone().sub(A).normalize());
  parent.add(m);return m;
}
function arrow(parent,a,b,r=.07){
  edge(parent,a,b,r,INK);
  const dir=new THREE.Vector3(b[0]-a[0],b[1]-a[1],b[2]-a[2]).normalize();
  const cone=new THREE.Mesh(new THREE.ConeGeometry(r*4,r*10,10),new THREE.MeshBasicMaterial({color:INK}));
  cone.position.set(b[0],b[1],b[2]);cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);parent.add(cone);
}
function tube(parent,pts,r,color){
  const curve=new THREE.CatmullRomCurve3(pts.map(p=>new THREE.Vector3(...p)));
  parent.add(new THREE.Mesh(new THREE.TubeGeometry(curve,40,r,6,false),new THREE.MeshBasicMaterial({color})));
}
const dcolor=(v,m)=>WHITE.clone().lerp(v>=0?POS:NEG,clamp(Math.abs(v)/m,0,1)*.92);
/* a flat wall of coloured cells; m is the value that gets the strongest colour */
function heatBlock(parent,M,ox,oy,oz,cw,ch,depth,part,hover,m){
  const rows=M.length,cols=M[0].length,inst=new THREE.InstancedMesh(new THREE.BoxGeometry(cw*.88,ch*.88,depth),new THREE.MeshLambertMaterial({color:0xffffff}),rows*cols);
  const m4=new THREE.Matrix4();m=m||maxAbs(M);
  for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){const k=i*cols+j;m4.setPosition(ox+j*cw,oy-i*ch,oz);inst.setMatrixAt(k,m4);inst.setColorAt(k,dcolor(M[i][j],m))}
  inst.instanceColor.needsUpdate=true;inst.instanceMatrix.needsUpdate=true;
  inst.userData.cols=cols;if(hover)inst.userData.hover=hover;
  if(part){inst.userData.part=part;interactives.push(inst)}
  parent.add(inst);
  const fr=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(cols*cw,rows*ch,depth*1.2)),new THREE.LineBasicMaterial({color:INK}));
  fr.position.set(ox+(cols-1)*cw/2,oy-(rows-1)*ch/2,oz);parent.add(fr);
  return inst;
}
/* bars standing on a floor: rows go along z, columns along x. scale = height per unit */
function barGrid(parent,M,ox,oy,oz,sx,sz,scale,o={}){
  const rows=M.length,cols=M[0].length,inst=new THREE.InstancedMesh(new THREE.BoxGeometry(sx*.7,1,sz*.7),new THREE.MeshLambertMaterial({color:0xffffff}),rows*cols);
  const m4=new THREE.Matrix4(),p=new THREE.Vector3(),q=new THREE.Quaternion(),s=new THREE.Vector3();
  for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
    const v=M[i][j],h=Math.max(Math.abs(v)*scale,.02);
    p.set(ox+j*sx,oy+Math.sign(v||1)*h/2,oz+i*sz);s.set(1,h,1);m4.compose(p,q,s);inst.setMatrixAt(i*cols+j,m4);
    const base=o.color?new THREE.Color(o.color(i,j,v)):(v>=0?POS.clone():NEG.clone());
    if(o.dim&&o.dim(i))base.lerp(WHITE,.6);
    inst.setColorAt(i*cols+j,base);
  }
  inst.instanceColor.needsUpdate=true;inst.instanceMatrix.needsUpdate=true;
  inst.userData.cols=cols;if(o.hover)inst.userData.hover=o.hover;if(o.part){inst.userData.part=o.part;interactives.push(inst)}
  parent.add(inst);
  const fl=new THREE.Mesh(new THREE.BoxGeometry(cols*sx+.2,.04,rows*sz+.2),new THREE.MeshBasicMaterial({color:0xcfd6e4}));
  fl.position.set(ox+(cols-1)*sx/2,oy,oz+(rows-1)*sz/2);parent.add(fl);
  return inst;
}
function setCam(pos,look){camGoal={pos:new THREE.Vector3(...pos),look:new THREE.Vector3(...look)};tween=70;viewDist=new THREE.Vector3(...pos).distanceTo(new THREE.Vector3(...look))}
function fitCam(c,halfW,halfH,dir){
  const t=Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
  const dist=Math.max(halfH/t,halfW/(t*camera.aspect))*1.17;
  const d=new THREE.Vector3(dir[0],dir[1],dir[2]).normalize().multiplyScalar(dist);
  setCam([c[0]+d.x,c[1]+d.y,c[2]+d.z],c);
}

/* ---------- scenes ---------- */
const BUILD={};
const layerHtml=()=>`<div class="grp"><span style="font-size:15px;color:#4d5666">Block</span>${Array.from({length:NLAYER},(_,i)=>`<button class="chipbtn" data-layer="${i}" aria-pressed="${S.layer===i}">${i+1}</button>`).join('')}</div>`;
const chipsHtml=()=>`<div class="grp">${tokens.map((k,i)=>`<button class="chipbtn" data-sel="${i}" aria-pressed="${S.sel===i}">${show(k.ch)}</button>`).join('')}</div>`;

BUILD.overview=()=>{
  const e=.7+S.explode*.9,X=v=>v*e,g=root,stepY=1.05;
  tokens.forEach((k,i)=>{const y=((T-1)/2-i)*stepY;box(g,.85,.85,.85,COL.tok,[X(-26),y,0],'tokens');label(g,show(k.ch),[X(-26),y,.8],.75)});
  arrow(g,[X(-24.6),0,0],[X(-22),0,0]);
  box(g,3,9,1.6,COL.embed,[X(-19.5),0,0],'embed');box(g,3,4.5,1.6,COL.embed,[X(-19.5),-2.2,-2.4],'embed');
  label(g,'token table',[X(-19.5),5.2,0],.8,{font:'xkcd Script'});label(g,'position table',[X(-19.5),-5.4,-2.4],.8,{font:'xkcd Script'});
  arrow(g,[X(-17.6),0,0],[X(-14.5),0,0]);
  for(let k=5;k>=1;k--)box(g,X(24),15,8,0xe9edf5,[X(-0.5),0,-k*3.4-1],null,{opacity:.12,edge:.35});
  box(g,X(24),15,8,0xf3f5fa,[X(-0.5),0,0],null,{opacity:.28,edge:.9});
  label(g,'Transformer block  × '+NLAYER,[X(-0.5),8.8,0],1.15);
  box(g,.7,6,6,COL.norm,[X(-10.5),0,0],'norm1');
  for(let h=0;h<NH;h++)box(g,2.4,5.2,.8,COL.attn,[X(-6.2),0,(h-(NH-1)/2)*1.05],'attn',{data:{head:h}});
  ball(g,.9,COL.add,[X(-1.5),0,0],'add1');
  box(g,.7,6,6,COL.norm,[X(1.6),0,0],'norm2');
  box(g,1.2,5,5,COL.ffn,[X(4.2),0,0],'ffn');box(g,1.8,9,5,COL.ffn,[X(6.6),0,0],'ffn');box(g,1.2,5,5,COL.ffn,[X(9),0,0],'ffn');
  ball(g,.9,COL.add,[X(11),0,0],'add2');
  arrow(g,[X(-14.5),0,0],[X(-11.4),0,0]);arrow(g,[X(-9.6),0,0],[X(-7.8),0,0]);arrow(g,[X(-4.6),0,0],[X(-2.5),0,0]);
  arrow(g,[X(-0.5),0,0],[X(.8),0,0]);arrow(g,[X(2.4),0,0],[X(3.2),0,0]);arrow(g,[X(9.6),0,0],[X(10.1),0,0]);
  arrow(g,[X(12),0,0],[X(15.2),0,0]);
  tube(g,[[X(-12.5),0,0],[X(-12.5),7.2,0],[X(-1.5),7.2,0],[X(-1.5),1,0]],.07,0x6e7b91);
  tube(g,[[X(-1.5),0,0],[X(-1.5),-7,0],[X(11),-7,0],[X(11),-1,0]],.07,0x6e7b91);
  label(g,'residual shortcut',[X(-7),7.9,0],.75,{font:'xkcd Script',color:'#4d5666'});label(g,'residual shortcut',[X(5),-7.9,0],.75,{font:'xkcd Script',color:'#4d5666'});
  box(g,.7,5,5,COL.norm,[X(16),0,0],'normf');
  arrow(g,[X(16.6),0,0],[X(18.4),0,0]);
  box(g,1.6,5,5,COL.lin,[X(19.6),0,0],'linear');
  arrow(g,[X(20.6),0,0],[X(22.6),0,0]);
  const probs=softmax(R.logits[T-1]),top=probs.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]).slice(0,7);
  top.forEach(([p,i],k)=>box(g,.8,Math.max(.4,p*14),.8,COL.soft,[X(24.2),-3+Math.max(.4,p*14)/2,(k-3)*1.05],'softmax'));
  arrow(g,[X(25.4),0,0],[X(27),0,0]);
  box(g,1.3,1.3,1.3,COL.next,[X(28.3),0,0],'next');label(g,show(CHARS[top[0][1]]),[X(28.3),0,1],1);
  const cap=(t,x,y=-8.8)=>label(g,t,[X(x),y,2],.95);
  cap('Tokens',-26,-7.2);cap('Embeddings',-19.5,-7.2);cap('Norm',-10.5);cap('Attention',-6.2);cap('Add',-1.5);cap('Norm',1.6);cap('Feed forward',6.6);cap('Add',11);
  cap('Norm',16,-7.2);cap('Linear',19.6,-7.2);cap('Softmax',24.2,-7.2);cap('Next',28.3,-7.2);
  fitCam([X(1),0,-3],X(30),12,[-.12,.32,1]);
  return `<div class="grp"><label>Pull apart <input type="range" id="explode" min="0" max="1" step=".02" value="${S.explode}"></label><span style="font-size:15px;color:#4d5666">Click any part to open it up. Click one attention slab to open that head.</span></div>`;
};

BUILD.embed=()=>{
  const g=root,e=.75+S.explode*.6,X=v=>v*e;
  const cw=.38,ch=.2,wallW=D*cw,top=(CHARS.length-1)*ch/2;
  const m=.55*Math.max(maxAbs(TOKTABLE),maxAbs(R.pos),maxAbs(R.x0));   // one colour scale for the whole scene
  heatBlock(g,TOKTABLE,X(-3)-wallW/2,top,0,cw,ch,.2,'embed',(id)=>({name:'Row for “'+show(CHARS[Math.floor(id/D)])+'”',short:'id '+Math.floor(id/D)+', one of '+CHARS.length+' rows, '+D+' of its 384 numbers shown'}),m);
  label(g,'Token table: '+CHARS.length+' rows × 384 numbers ('+D+' shown)',[X(-3),top+1.4,0],.9);
  const rowY=i=>((T-1)/2-i)*.9;
  tokens.forEach((k,i)=>{
    box(g,.8,.8,.8,COL.tok,[X(-12),rowY(i),0],'tokens');label(g,show(k.ch),[X(-12),rowY(i),.7],.7);
    edge(g,[X(-11.6),rowY(i),0],[X(-3)-wallW/2-.3,top-k.id*ch,0],.012,0x8a93a6,.7);
    edge(g,[X(-3)+wallW/2+.2,top-k.id*ch,0],[X(5)-.3,rowY(i),0],.012,0x8a93a6,.7);
  });
  label(g,'Tokens',[X(-12),((T-1)/2)*.9+1.1,0],.9);
  const blk=(M,x,t)=>{heatBlock(g,M,X(x),((T-1)/2)*.9,0,cw,.9,.4,'embed',null,m);label(g,t,[X(x)+(D-1)*cw/2,((T-1)/2)*.9+1.1,0],.9)};
  blk(R.tok,5,'Token vectors');
  label(g,'+',[X(5)+D*cw+.9,0,0],1.6);
  blk(R.pos,5+D*cw+2.4,'Position vectors');
  label(g,'=',[X(5+D*cw+2.4)+D*cw+.9,0,0],1.6);
  blk(R.x0,5+(D*cw+2.4)*2,'Input to the first block');
  fitCam([X(5)+3,0,0],X(17)+5,8.5,[0,.18,1]);
  return `<span style="font-size:15px;color:#4d5666">Real numbers from the trained model, first ${D} of 384 per token. Hover the wall to read a row.</span>`;
};

BUILD.norm=()=>{
  const g=root,e=.8+S.explode*.5,sx=.62,sz=1.05;
  const before=R.xin,after=R.n1,dim=i=>i!==S.sel,oz=-((T-1)*sz)/2;
  const sb=4.5/maxAbs(before),sa=4.5/maxAbs(after);
  barGrid(g,before,-17*e,0,oz,sx,sz,sb,{dim,hover:(id)=>({name:'Token “'+show(tokens[Math.floor(id/D)].ch)+'”',short:'value '+before[Math.floor(id/D)][id%D].toFixed(3)})});
  barGrid(g,after,3*e,0,oz,sx,sz,sa,{dim,hover:(id)=>({name:'Token “'+show(tokens[Math.floor(id/D)].ch)+'”',short:'normalised '+after[Math.floor(id/D)][id%D].toFixed(3)})});
  label(g,'Before',[-12*e,5.6,0],1);label(g,'After: mean 0, spread 1',[8*e,5.6,0],1);
  arrow(g,[-5*e,2.2,0],[-1*e,2.2,0]);label(g,'subtract mean, divide by spread',[-3*e,3.3,0],.7,{font:'xkcd Script'});
  tokens.forEach((k,i)=>{label(g,show(k.ch),[-18.1*e,.6,oz+i*sz],.7);label(g,show(k.ch),[1.9*e,.6,oz+i*sz],.7)});
  const st=R.stat[S.sel];
  fitCam([-2.5*e,.5,0],16*e,7,[0,.75,1]);
  return layerHtml()+chipsHtml()+`<span style="font-size:15px;color:#4d5666">Token “${show(tokens[S.sel].ch)}” in block ${S.layer+1}: mean ${st.m.toFixed(2)}, spread ${st.sd.toFixed(2)} → mean 0.00, spread 1.00. Each grid has its own height scale.</span>`;
};

BUILD.resid=()=>{
  const g=root,e=.8+S.explode*.5,sx=.5,sz=1;
  const [a,b,c,la,lb,lc]=S.resid===1?[R.xin,R.attn,R.x1,'Stream in','Attention output','Stream out']:[R.x1,R.ffn,R.x2,'Stream in','Feed-forward output','Stream out'];
  const dim=i=>i!==S.sel,oz=-((T-1)*sz)/2,w=D*sx,sc=4.5/Math.max(maxAbs(a),maxAbs(b),maxAbs(c));   // same height scale for all three
  barGrid(g,a,-20*e,0,oz,sx,sz,sc,{dim});barGrid(g,b,-8*e,0,oz,sx,sz,sc,{dim});barGrid(g,c,4*e,0,oz,sx,sz,sc,{dim});
  label(g,la,[-20*e+w/2,5.6,0],.9);label(g,lb,[-8*e+w/2,5.6,0],.9);label(g,lc,[4*e+w/2,5.6,0],.9);
  label(g,'+',[-8*e-1.6,.8,0],2.2);label(g,'=',[4*e-1.6,.8,0],2.2);
  tokens.forEach((k,i)=>{label(g,show(k.ch),[-21*e,.5,oz+i*sz],.7)});
  fitCam([-4.5*e,.5,0],17*e,7,[0,.75,1]);
  return `<div class="grp"><button class="btn" data-resid="1" aria-pressed="${S.resid===1}">After attention</button><button class="btn" data-resid="2" aria-pressed="${S.resid===2}">After feed forward</button></div>`+layerHtml()+chipsHtml()+
    `<span style="font-size:15px;color:#4d5666">Same height scale on all three grids, so you can see how small the sub-layer’s correction is.</span>`;
};

BUILD.ffn=()=>{
  const g=root,e=.7+S.explode*.7,i=S.sel,fv=R.fv;
  const x=R.n2[i],h=fv.hidden[i],o=R.ffn[i],w1=fv.w1[i],w2=fv.w2[i];
  const hmax=Math.max(...h,1e-6),omax=maxAbs([o]),xmax=maxAbs([x]),m1=maxAbs(w1),m2=maxAbs(w2);
  const inPos=k=>[-9*e,((k>>2)-1.5)*1.1,((k&3)-1.5)*1.1];
  const hPos=k=>[0,((k>>3)-3.5)*.9,((k&7)-3.5)*.9];
  const oPos=k=>[9*e,((k>>2)-1.5)*1.1,((k&3)-1.5)*1.1];
  const seg=(pairs)=>{const pos=[],col=[];pairs.forEach(([a,b,v])=>{pos.push(...a,...b);const c=WHITE.clone().lerp(v>=0?POS:NEG,clamp(Math.abs(v)/1.1,0,1)*.95);col.push(c.r,c.g,c.b,c.r,c.g,c.b)});
    const gm=new THREE.BufferGeometry();gm.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));gm.setAttribute('color',new THREE.Float32BufferAttribute(col,3));
    return new THREE.LineSegments(gm,new THREE.LineBasicMaterial({vertexColors:true}))};
  const p1=[],p2=[];
  for(let a=0;a<D;a++)for(let b=0;b<FF;b++)if(h[b]>0)p1.push([inPos(a),hPos(b),w1[a][b]/m1*1.6]);
  for(let b=0;b<FF;b++)if(h[b]>0)for(let c=0;c<D;c++)p2.push([hPos(b),oPos(c),w2[b][c]/m2*1.6]);
  g.add(seg(p1));g.add(seg(p2));
  x.forEach((v,k)=>ball(g,.14+.2*clamp(Math.abs(v)/xmax,0,1),v>=0?0x2a4fd6:0xd8602a,inPos(k),'ffn'));
  h.forEach((v,k)=>{if(v>0)ball(g,.1+.2*v/hmax,0xbfeaea,hPos(k),'hid');else ball(g,.07,0xe3e7ee,hPos(k),'hid')});
  o.forEach((v,k)=>ball(g,.14+.2*clamp(Math.abs(v)/omax,0,1),v>=0?0x2a4fd6:0xd8602a,oPos(k),'ffn'));
  label(g,'In: first '+D+' of 384',[-9*e,3.2,0],.9);label(g,'Hidden: first '+FF+' of '+fv.total,[0,4.5,0],.9);label(g,'Out: first '+D+' of 384',[9*e,3.2,0],.9);
  label(g,'Linear ×4 wider',[-4.5*e,-4.3,0],.8,{font:'xkcd Script'});label(g,'ReLU',[0,-4.8,0],.8,{font:'xkcd Script'});label(g,'Linear back',[4.5*e,-4.3,0],.8,{font:'xkcd Script'});
  fitCam([0,0,0],11*e+2,6.5,[.55,.2,1]);
  return layerHtml()+chipsHtml()+`<span style="font-size:15px;color:#4d5666">“${show(tokens[i].ch)}” in block ${S.layer+1} fires ${fv.active[i]} of ${fv.total} neurons. Grey dots (among the 64 drawn) were switched off by ReLU.</span>`;
};

BUILD.output=()=>{
  const g=root,lg=R.logits[T-1],p=softmax(lg,S.temp),sx=.5,ox=-(CHARS.length-1)*sx/2;
  const top=p.indexOf(Math.max(...p));
  barGrid(g,[lg],ox,0,-4,sx,1.1,4.5/maxAbs([lg]),{hover:(id)=>({name:'“'+show(CHARS[id])+'” score',short:lg[id].toFixed(2)}),part:'linear'});
  barGrid(g,[p],ox,0,3,sx,1.1,7/Math.max(...p),{color:(i,j)=>j===top?0xe0a800:0x2e9d4a,hover:(id)=>({name:'“'+show(CHARS[id])+'”',short:(p[id]*100).toFixed(1)+'% chance'}),part:'softmax'});
  CHARS.forEach((c,j)=>label(g,show(c),[ox+j*sx,-.55,5.1],.6));
  label(g,'Scores (logits)',[ox+5,5.6,-4],.95);label(g,'Probabilities (softmax)',[ox+6.5,9.6,3],.95);
  arrow(g,[ox-1.2,1,-3],[ox-1.2,1,2.4]);label(g,'softmax',[ox-2.6,2.4,0],.85,{font:'xkcd Script'});
  fitCam([0,2.5,0],20,9,[0,.45,1]);
  return `<div class="grp"><label>Temperature <input type="range" id="temp" min="0.2" max="2" step="0.1" value="${S.temp}"><b id="tv">${S.temp.toFixed(1)}</b></label>
    <button class="btn" id="sample">Pick the next letter</button><span style="font-size:15px;color:#4d5666">Most likely: “${show(CHARS[top])}” (${(p[top]*100).toFixed(0)}%)</span></div>`;
};

/* live view: what the model does for each letter it writes */
BUILD.live=()=>{
  const g=root,L=S.live,ev=liveEvent();
  // the layout always has MAXT slots, so it never changes size while the reply grows and the camera stays framed
  const ctx=ev?[...ev.ctx]:[...S.valid.slice(-MAXT)],n=ctx.length,sp=1.45,slot=i=>(MAXT-n+i)*sp;
  const qx=(MAXT-1)*sp,qy=5;
  ctx.forEach((c,i)=>{box(g,1.2,1.2,1.2,COL.tok,[slot(i),0,0],null);label(g,show(c),[slot(i),0,1],.9)});
  label(g,'The last '+n+' letters it can see',[qx/2,-2.1,0],1);
  ball(g,.55,COL.q,[qx,qy,0],null);label(g,'the newest letter asks',[qx,qy+1.2,0],.85,{font:'xkcd Script'});
  let att=null;
  if(ev){
    att=liveAttention(ev);
    att.forEach((w,i)=>{
      edge(g,[qx,qy,0],[slot(i),.7,0],.03+.28*Math.sqrt(w),0xd33a3a,.35+.65*Math.min(1,w*3));
      if(w>.1)label(g,Math.round(w*100)+'%',[slot(i),1.9,0],.8,{font:'xkcd Script',color:'#d33a3a'});
    });
  }
  const bx=qx+4.5,gap=1.05,endB=bx+(NLAYER-1)*gap;
  for(let b=0;b<NLAYER;b++)box(g,.6,6.5,3.4,b===S.liveLayer?0xf5d98a:COL.attn,[bx+b*gap,0,0],null);
  label(g,'Block '+(S.liveLayer+1)+' of '+NLAYER+' is shown',[(bx+endB)/2,4.4,0],.9);label(g,'the model',[(bx+endB)/2,-4.3,0],1);
  arrow(g,[qx+1.1,0,0],[bx-.8,0,0]);arrow(g,[endB+.8,0,0],[endB+3,0,0]);
  const xb=endB+4,w=1.5;
  if(ev){
    ev.top.forEach(([c,p],k)=>{
      const h=Math.max(.2,p*13),chosen=c===ev.c;
      box(g,1,h,1,chosen?0xe0a800:COL.soft,[xb+k*w,h/2-3,0],null);
      label(g,show(c),[xb+k*w,-3.9,0],1);
      label(g,(p*100).toFixed(p<.1?1:0)+'%',[xb+k*w,h-2.1,0],.75,{font:'xkcd Script'});
    });
    label(g,'Most likely next letters',[xb+7*w/2,5.7,0],1);
    label(g,'it drew “'+show(ev.c)+'” ('+(ev.p*100).toFixed(1)+'%)',[xb+7*w/2,-5.4,0],1);
  }else label(g,'Waiting for the first letter…',[xb+3*w,0,0],1);
  const xmin=-1.2,xmax=xb+7*w+1.2;
  if(!L.fitted){L.fitted=true;fitCam([(xmin+xmax)/2,.6,0],(xmax-xmin)/2,7.6,[0,.12,1])}   // frame once, so later letters do not move the camera
  const mode=liveMode(),label2={pause:'❚❚ Pause',replay:'↻ Replay',play:'▶ Play'}[mode];
  return `<div class="grp"><button class="btn" data-live="back" ${L.shown<=0?'disabled':''}>◀ Back</button><button class="btn" data-live="toggle" ${L.steps.length?'':'disabled'}>${label2}</button><button class="btn" data-live="next" ${L.shown>=L.steps.length-1?'disabled':''}>Next ▶</button></div>
   <div class="grp"><span style="font-size:15px;color:#4d5666">Block</span>${Array.from({length:NLAYER},(_,i)=>`<button class="chipbtn" data-lblock="${i}" aria-pressed="${S.liveLayer===i}">${i+1}</button>`).join('')}</div>
   <div class="grp"><span style="font-size:15px;color:#4d5666">Head</span><button class="chipbtn" data-lhead="avg" aria-pressed="${S.liveHead==='avg'}">avg</button>${Array.from({length:NH},(_,h)=>`<button class="chipbtn" data-lhead="${h}" aria-pressed="${S.liveHead===h}">${h+1}</button>`).join('')}</div>
   <span id="livecap" style="font-size:15px;color:#4d5666">${liveCaption()}</span>`;
};

/* attention: the main dissection */
let attnDyn=null,attnGeo=null;
BUILD.attn=()=>{
  const g=root,e=.65+S.explode*.8,sp=1.45;
  const y=i=>((T-1)/2-i)*sp,XT=-10*e,XM=0,XO=10*e,dz=3.6*e;
  attnGeo={y,XT,XM,XO,dz};
  tokens.forEach((k,i)=>{
    box(g,1,1,1,COL.tok,[XT,y(i),0],'tokens',{data:{tok:i}});label(g,show(k.ch),[XT,y(i),.85],.8);
    ball(g,.36,COL.q,[XM,y(i),dz],'q',{tok:i});ball(g,.36,COL.k,[XM,y(i),0],'k',{tok:i});ball(g,.36,COL.v,[XM,y(i),-dz],'v',{tok:i});
    ball(g,.42,COL.o,[XO,y(i),0],'o',{tok:i});
  });
  const yt=y(0)+1.9;
  label(g,'Tokens',[XT,yt,0],1);label(g,'Query',[XM,yt,dz],1);label(g,'Key',[XM,yt,0],1);label(g,'Value',[XM,yt,-dz],1);label(g,'Output',[XO,yt,0],1);
  label(g,'asks',[XM,yt-.9,dz],.65,{font:'xkcd Script',color:'#4d5666'});label(g,'offers',[XM,yt-.9,0],.65,{font:'xkcd Script',color:'#4d5666'});label(g,'carries',[XM,yt-.9,-dz],.65,{font:'xkcd Script',color:'#4d5666'});
  label(g,'weighted sum of values',[XO,yt-.9,0],.65,{font:'xkcd Script',color:'#4d5666'});
  attnDyn=new THREE.Group();g.add(attnDyn);
  drawAttnEdges();
  fitCam([0,-.5,0],12*e+1.5,(T*1.45)/2+3,[.55,.22,1]);
  return layerHtml()+`<div class="grp">${Array.from({length:NH},(_,h)=>`<button class="btn" data-head="${h}" aria-pressed="${S.head===h}">Head ${h+1}</button>`).join('')}</div>
   <label><input type="checkbox" id="mask" ${S.mask?'checked':''}> show blocked future</label>`+chipsHtml()+
   `<span class="legend"><span><i style="background:#f29a9a"></i>query</span><span><i style="background:#9ab9f2"></i>key</span><span><i style="background:#a9dc9c"></i>value</span></span>`;
};
function drawAttnEdges(){
  if(!attnDyn)return;
  while(attnDyn.children.length){const c=attnDyn.children.pop();c.geometry&&c.geometry.dispose();c.material&&c.material.dispose()}
  const {y,XT,XM,XO,dz}=attnGeo,A=R.heads[S.head].A,eff=S.hov!==null?S.hov:S.sel;
  for(let i=0;i<T;i++){
    const hl=i===eff;
    edge(attnDyn,[XT+.5,y(i),0],[XM,y(i),dz],.015,0x8a93a6,hl?.9:.16);edge(attnDyn,[XT+.5,y(i),0],[XM,y(i),0],.015,0x8a93a6,hl?.9:.16);edge(attnDyn,[XT+.5,y(i),0],[XM,y(i),-dz],.015,0x8a93a6,hl?.9:.16);
    for(let j=0;j<=i;j++){
      const w=A[i][j];if(w<.012&&!hl)continue;
      edge(attnDyn,[XM,y(i),dz],[XM,y(j),0],.02+.17*w,hl?0xd33a3a:0x7d8cc0,hl?.95:(eff===null?.5:.12));
      edge(attnDyn,[XM,y(j),-dz],[XO,y(i),0],.02+.15*w,hl?0x2e9d4a:0x8dbf95,hl?.95:(eff===null?.5:.1));
    }
    if(S.mask&&hl)for(let j=i+1;j<T;j++)edge(attnDyn,[XM,y(i),dz],[XM,y(j),0],.014,0xe05a5a,.35);
  }
}

/* ---------- teacher card ---------- */
const CREDIT='<div class="credit">After Karpathy’s “Let’s build GPT”.</div>';
function teachPart(id,extra=''){
  const p=PARTS[id];if(!p)return;
  $('#info').innerHTML=`<h2 class="sc">${p.name}</h2><span class="tag" style="--bg:${p.bg}">${p.tag}</span>
   <p>${p.desc}</p>${extra}<h3>Tensor shape</h3><div class="kv">${p.shape}</div><h3>In your code</h3><div class="kv">${p.code}</div>
   ${p.scene&&S.scene!==p.scene?`<div><button class="btn" data-open="${id}">Open it up →</button></div>`:''}${CREDIT}`;
}
function attnMatrix(){
  const A=R.heads[S.head].A,eff=S.hov!==null?S.hov:S.sel,cs=T>10?22:T>8?26:30;
  let h=`<div class="mat" style="grid-template-columns:20px repeat(${T},${cs}px)"><div></div>${tokens.map(k=>`<div class="h">${show(k.ch)}</div>`).join('')}`;
  for(let i=0;i<T;i++){h+=`<div class="r">${show(tokens[i].ch)}</div>`;for(let j=0;j<T;j++){
    const w=A[i][j],m=j>i;h+=`<div class="c ${m?'m':''} ${i===eff?'on':''}" data-i="${i}" style="${m?'':`background:color-mix(in srgb,#2a4fd6 ${Math.round(w*90)}%,#fff);color:${w>.5?'#fff':'#111'}`};height:${cs-6}px;width:${cs}px">${m?'':w.toFixed(2).slice(1)}</div>`}}
  return h+'</div>';
}
function teachScene(){
  const sc=S.scene;
  if(sc==='overview'){
    $('#info').innerHTML=`<h2 class="sc">The whole GPT</h2><span class="tag" style="--bg:#e4eaff">overview</span>
     <p>Text goes in on the left, the next letter comes out on the right. In the middle, the dashed box is one Transformer block. This model stacks ${NLAYER} of them, and each block does two things: <b>attention</b> lets tokens talk to each other, then a <b>feed-forward</b> network lets each token think.</p>
     <p>Everything you see is a real number from the trained model. Rotate it, pull it apart with the slider, hover any piece to see what it is, and click it to dissect it.</p>
     <h3>Size of this model</h3><div class="kv">layers ${NLAYER} · heads ${NH} · width 384\nvocabulary ${CHARS.length} · context 256\n${PARAMS?PARAMS.toLocaleString()+' parameters':''}</div>${CREDIT}`;return;
  }
  if(sc==='attn'){
    const eff=S.hov!==null?S.hov:S.sel,A=R.heads[S.head].A;
    const order=A[eff].map((v,j)=>[v,j]).slice(0,eff+1).sort((a,b)=>b[0]-a[0]).slice(0,3);
    $('#info').innerHTML=`<h2 class="sc">Attention, block ${S.layer+1}, head ${S.head+1}</h2><span class="tag" style="--bg:#f2d9b1">communication</span>
      <p>Token <b>“${show(tokens[eff].ch)}”</b> asks a question (red). Earlier tokens answer with their key (blue). The thicker the red line, the stronger the match. Then the token collects the values (green) in those proportions. Hover a token, or a row of the table, to change who is asking.</p>
      <p><b>Looks hardest at:</b> ${order.map(([v,j])=>`“${show(tokens[j].ch)}” ${(v*100).toFixed(0)}%`).join(', ')}</p>
      <h3>Attention weights (rows ask, columns answer)</h3>${attnMatrix()}
      <p style="font-size:14px;color:#4d5666">Striped cells are the future. The mask forces them to zero, so a token can’t peek at what comes next.</p>
      <h3>In your code</h3><div class="kv">gpt.py:85-91  wei = softmax(q@kᵀ) ; out = wei@v</div>${CREDIT}`;
    return;
  }
  if(sc==='live'){
    $('#info').innerHTML=`<h2 class="sc">Live view</h2><span class="tag" style="--bg:#f5d98a">one letter at a time</span>
     <p>For every letter it writes, the model reads the text so far and guesses what comes next. Watch it work on the left while the reply streams in.</p>
     <p>The row of cubes is the last 12 letters it can see. The <b>red lines</b> show how strongly the newest letter looked at each one, in the block and head you pick. The slabs are the ${NLAYER} blocks, and the gold one is the block you are watching. The bars on the right are the 8 most likely next letters, and the gold bar is the one it drew.</p>
     <p>Along the top, the flow chart lights up box by box each time a letter is made.</p>${CREDIT}`;return;
  }
  teachPart({embed:'embed',norm:'norm1',resid:'add'+S.resid,ffn:'ffn',output:'softmax'}[sc]);
}

/* ---------- flow chart ---------- */
const FLOW=['text','tokens','embed',{block:['norm1','attn','add1','norm2','ffn','add2']},'normf','linear','softmax','next'];
function nodeHtml(id){
  const p=PARTS[id],label=id==='text'?'“'+tokens.map(k=>show(k.ch)).join('')+'”':p.name;
  return `<button class="node" data-part="${id}" style="--bg:${p.bg}"><span class="bd"></span><span class="tx">${label}</span></button>`;
}
function renderFlow(){
  const out=[];
  FLOW.forEach((f,i)=>{
    if(i)out.push('<i class="ar"></i>');
    if(typeof f==='string')out.push(nodeHtml(f));
    else out.push(`<div class="bracket"><span class="rep">Block × ${NLAYER}</span>${f.block.map((b,k)=>(k?'<i class="ar"></i>':'')+nodeHtml(b)).join('')}</div>`);
  });
  $('#flow').innerHTML=out.join('');
  markFlow();
}
function markFlow(){
  document.querySelectorAll('.node').forEach(n=>{
    const p=PARTS[n.dataset.part];
    n.classList.toggle('on',S.scene!=='overview'&&p.scene===S.scene&&(S.scene!=='resid'||n.dataset.part==='add'+S.resid));
    n.classList.toggle('hl',n.dataset.part===S.hoverPart);
  });
}

/* ---------- build / navigate ---------- */
function build(keepCam,keepCtl){
  if(!R)return;
  clearRoot();attnDyn=null;
  const ctl=BUILD[S.scene]();
  if(!keepCtl){$('#ctl').innerHTML=ctl;$('#ctl').style.display=ctl?'flex':'none'}
  const nm={overview:null,embed:'Embeddings',norm:'Layer norm',attn:'Attention',resid:'Add',ffn:'Feed forward',output:'Output',live:'Live view'}[S.scene];
  const crumbs=S.scene==='overview'?'<span>The whole GPT</span>':`<button class="btn" data-home="1">← The whole GPT</button><span class="sep">›</span><span>${nm}</span>`;
  if($('#crumbs').dataset.html!==crumbs){$('#crumbs').innerHTML=crumbs;$('#crumbs').dataset.html=crumbs}   // same reason: do not replace a button that is being clicked
  $('#hintbar').style.display=S.scene==='overview'?'':'none';
  teachScene();markFlow();
  if(keepCam&&camGoal&&S.scene!=='live')tween=0;   // the live view keeps its first camera move going
}
function go(scene,o={}){
  S.scene=scene;if(o.head!==undefined)S.head=o.head;S.hov=null;
  try{history.replaceState(null,'','#'+scene)}catch(e){}
  build();
}
async function setLayer(n){S.layer=n;if(await load(S.valid.slice(-MAXT)))build(true)}

/* ---------- picking ---------- */
const ray=new THREE.Raycaster(),ptr=new THREE.Vector2(),tip=$('#tip');
let downAt=null;
function pick(e){const r=cvs.getBoundingClientRect();ptr.set(((e.clientX-r.left)/r.width)*2-1,-((e.clientY-r.top)/r.height)*2+1);ray.setFromCamera(ptr,camera);return ray.intersectObjects(interactives,false)[0]||null}
function glow(part,on){interactives.forEach(m=>{if(m.userData.part===part&&m.material.emissive)m.material.emissive.setHex(on?0x666666:0)})}
cvs.addEventListener('pointermove',e=>{
  if(e.buttons||!R)return;
  const hit=pick(e),r=cvs.getBoundingClientRect(),part=hit&&hit.object.userData.part;
  if(S.hoverPart&&S.hoverPart!==part)glow(S.hoverPart,false);
  const prevHov=S.hov;
  if(hit){
    const u=hit.object.userData;let nm,sh;
    if(u.hover&&hit.instanceId!==undefined){const h=u.hover(hit.instanceId);nm=h.name;sh=h.short}
    else{const p=PARTS[part];nm=p.name;sh=p.short}
    tip.innerHTML=`<b>${nm}</b>${sh}`;tip.style.display='block';
    tip.style.left=Math.min(e.clientX-r.left+14,r.width-270)+'px';tip.style.top=Math.max(e.clientY-r.top-10,50)+'px';
    cvs.style.cursor='pointer';
    if(S.hoverPart!==part){S.hoverPart=part;glow(part,true);markFlow();if(S.scene!=='attn'||u.tok===undefined)teachPart(part)}
    if(S.scene==='attn'&&u.tok!==undefined){S.hov=u.tok;if(S.hov!==prevHov){drawAttnEdges();teachScene()}}
  }else{
    tip.style.display='none';cvs.style.cursor='grab';
    if(S.hoverPart){S.hoverPart=null;markFlow();teachScene()}
    if(S.hov!==null&&S.scene==='attn'){S.hov=null;drawAttnEdges();teachScene()}
  }
});
cvs.addEventListener('pointerleave',()=>{tip.style.display='none';if(S.hoverPart){glow(S.hoverPart,false);S.hoverPart=null;markFlow();teachScene()}if(S.hov!==null){S.hov=null;if(S.scene==='attn'){drawAttnEdges();teachScene()}}});
cvs.addEventListener('pointerdown',e=>{downAt=[e.clientX,e.clientY];tip.style.display='none'});
cvs.addEventListener('pointerup',e=>{
  if(!downAt||!R||Math.hypot(e.clientX-downAt[0],e.clientY-downAt[1])>5)return;
  const hit=pick(e);if(!hit)return;const u=hit.object.userData,part=u.part;
  if(S.scene==='attn'&&u.tok!==undefined){S.sel=u.tok;S.hov=null;build(true);return}
  if(S.scene==='overview'){const p=PARTS[part];if(!p)return;
    if(part==='add2')S.resid=2;if(part==='add1')S.resid=1;
    go(p.scene,part==='attn'&&u.head!==undefined?{head:u.head}:{})}
});

/* ---------- ui events ---------- */
async function sampleNext(){
  const p=softmax(R.logits[T-1],S.temp);let r=Math.random(),k=0;for(;k<p.length-1;k++){r-=p[k];if(r<=0)break}
  await applyText(S.valid+CHARS[k]);
}
document.addEventListener('click',e=>{
  const t=e.target.closest('[data-cam],[data-home],[data-part],[data-open],[data-head],[data-sel],[data-resid],[data-layer],[data-lblock],[data-lhead],[data-live],#sample,#spin,#retry');if(!t)return;
  if(t.id==='retry'){applyText(S.valid)}
  else if(t.dataset.cam){
    const L=controls.target.clone(),d=viewDist,c=t.dataset.cam;
    const pos=c==='front'?[L.x,L.y,L.z+d]:c==='side'?[L.x+d,L.y,L.z]:c==='top'?[L.x,L.y+d,L.z+.01]:null;
    if(pos)camGoal={pos:new THREE.Vector3(...pos),look:L};else build();
    tween=60;
  }
  else if(t.id==='spin'){controls.autoRotate=!controls.autoRotate;t.setAttribute('aria-pressed',controls.autoRotate)}
  else if(t.dataset.home){go('overview')}
  else if(t.dataset.part||t.dataset.open){const id=t.dataset.part||t.dataset.open;if(id==='add2')S.resid=2;if(id==='add1')S.resid=1;go(PARTS[id].scene)}
  else if(t.dataset.head!==undefined){S.head=+t.dataset.head;build(true)}
  else if(t.dataset.sel!==undefined){S.sel=+t.dataset.sel;build(true)}
  else if(t.dataset.resid){S.resid=+t.dataset.resid;build(true)}
  else if(t.dataset.live){const a=t.dataset.live;if(a==='toggle'){liveMode()==='pause'?livePause():livePlay()}else liveStepBy(a==='next'?1:-1)}
  else if(t.dataset.lblock!==undefined){S.liveLayer=+t.dataset.lblock;build(true)}
  else if(t.dataset.lhead!==undefined){S.liveHead=t.dataset.lhead==='avg'?'avg':+t.dataset.lhead;build(true)}
  else if(t.dataset.layer!==undefined){setLayer(+t.dataset.layer)}
  else if(t.id==='sample'){sampleNext()}
});
document.addEventListener('input',e=>{
  if(e.target.id==='explode'){S.explode=+e.target.value;build(true,true)}
  else if(e.target.id==='temp'){S.temp=+e.target.value;$('#tv').textContent=S.temp.toFixed(1);build(true,true)}
  else if(e.target.id==='mask'){S.mask=e.target.checked;drawAttnEdges()}
});
document.addEventListener('mouseover',e=>{
  const n=e.target.closest('.node');
  if(n){const id=n.dataset.part;if(S.hoverPart!==id){if(S.hoverPart)glow(S.hoverPart,false);S.hoverPart=id;glow(id,true);markFlow();teachPart(id)}}
  const c=e.target.closest('.mat .c');
  if(c&&S.scene==='attn'){const i=+c.dataset.i;if(S.hov!==i){S.hov=i;drawAttnEdges();teachScene()}}
});
document.addEventListener('mouseout',e=>{
  const n=e.target.closest('.node');
  if(n&&!n.contains(e.relatedTarget)){glow(n.dataset.part,false);S.hoverPart=null;markFlow();teachScene()}
  const c=e.target.closest('.mat');
  if(c&&!c.contains(e.relatedTarget)&&S.scene==='attn'){S.hov=null;drawAttnEdges();teachScene()}
});

/* ---------- loop and start ---------- */
function frame(){
  requestAnimationFrame(frame);
  if(camGoal&&tween>0){camera.position.lerp(camGoal.pos,.1);controls.target.lerp(camGoal.look,.1);tween--}
  controls.update();renderer.render(scene,camera);
}
/* ---------- live view: watch the model work while it writes ---------- */
const FLOW_ORDER=['text','tokens','embed','norm1','attn','add1','norm2','ffn','add2','normf','linear','softmax','next'];
let sweeping=false;
function sweep(){   // lights up the flow chart boxes one after another; a new letter waits for the sweep to finish
  if(sweeping||(window.matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches))return;
  sweeping=true;
  FLOW_ORDER.forEach((id,i)=>setTimeout(()=>{
    const n=document.querySelector(`.node[data-part="${id}"]`);
    if(n){n.classList.add('live');setTimeout(()=>n.classList.remove('live'),260)}
    if(i===FLOW_ORDER.length-1)setTimeout(()=>{sweeping=false},260);
  },i*70));
}
const liveEvent=()=>S.live.shown>=0?S.live.steps[S.live.shown]:null;
function liveAttention(ev){   // how strongly the newest letter looked at each letter shown, for the chosen block and head
  const heads=ev.attn[S.liveLayer];
  return S.liveHead==='avg'?[...ev.ctx].map((_,i)=>heads.reduce((s,h)=>s+h[i],0)/heads.length):heads[S.liveHead].slice();
}
function liveCaption(){
  const L=S.live,ev=liveEvent();
  if(!ev)return 'Waiting for the first letter…';
  const ctx=[...ev.ctx],att=liveAttention(ev),best=att.indexOf(Math.max(...att)),pc=v=>Math.round(v*100);
  return `Letter ${L.shown+1} of ${L.steps.length}${L.done?'':' so far'}: it looked hardest at “${show(ctx[best])}” (${pc(att[best])}%). Top choice “${show(ev.top[0][0])}” (${pc(ev.top[0][1])}%), drew “${show(ev.c)}”.${L.done?' Finished.':!L.playing?' Paused: the reply keeps writing in the chat.':''}`;
}
/* The buttons are NOT rebuilt for every letter: a button replaced between mouse-down and mouse-up loses the click. */
function refreshLiveCtl(){
  if(S.scene!=='live')return;
  const L=S.live,t=document.querySelector('[data-live="toggle"]'),b=document.querySelector('[data-live="back"]'),n=document.querySelector('[data-live="next"]');
  if(!t||!b||!n)return;
  t.textContent={pause:'❚❚ Pause',replay:'↻ Replay',play:'▶ Play'}[liveMode()];t.disabled=!L.steps.length;
  b.disabled=L.shown<=0;n.disabled=L.shown>=L.steps.length-1;
  const cap=document.getElementById('livecap');if(cap)cap.textContent=liveCaption();
}
const liveAtEnd=()=>S.live.done&&S.live.shown===S.live.steps.length-1;
function liveMode(){   // what the play button does right now
  if(liveAtEnd())return 'replay';
  return S.live.playing?'pause':'play';
}
function liveShow(i){S.live.shown=i;sweep();if(S.scene==='live'){build(true,true);refreshLiveCtl()}}   // redraws the 3D scene, keeps the buttons
function liveStopTimer(){clearInterval(S.live.timer);S.live.timer=null}
function liveStart(){liveStopTimer();S.live=newLive();go('live')}
function liveStep(ev){   // a new letter arrived: follow it, unless the view is paused or still catching up
  const L=S.live;L.steps.push(ev);L.n=L.steps.length;
  if(L.playing&&!L.timer&&L.shown===L.steps.length-2)liveShow(L.steps.length-1);
  else refreshLiveCtl();   // only the counts changed
}
function liveEnd(){S.live.done=true;refreshLiveCtl()}
function livePause(){liveStopTimer();S.live.playing=false;refreshLiveCtl()}
function livePlay(){   // plays the letters you skipped at a readable pace, then follows the reply again
  const L=S.live;liveStopTimer();L.playing=true;
  if(liveAtEnd())L.shown=-1;   // replay from the first letter
  if(L.shown<L.steps.length-1)liveShow(L.shown+1);
  L.timer=setInterval(()=>{
    if(!L.playing||L.shown>=L.steps.length-1){liveStopTimer();refreshLiveCtl();return}
    liveShow(L.shown+1);
  },700);
  refreshLiveCtl();
}
function liveStepBy(d){   // one letter back or forward, which also pauses
  const L=S.live;liveStopTimer();L.playing=false;
  liveShow(clamp(L.shown+d,0,Math.max(0,L.steps.length-1)));
}
window.gptApp={applyText,previewText,go,state:S,liveStart,liveStep,liveEnd};   // used by chat.js
async function start(){
  const h0=(location.hash||'').slice(1);
  if(BUILD[h0])S.scene=h0;
  resize();
  try{
    const fonts=document.fonts&&document.fonts.load?Promise.all([document.fonts.load('700 40px "Humor Sans"'),document.fonts.load('40px "xkcd Script"')]).catch(()=>{}):Promise.resolve();
    const [vocab,health,table]=await Promise.all([api('/api/vocab'),api('/api/health'),api('/api/token-table?dims='+D),fonts]);
    CHARS=vocab.chars;PARAMS=health.parameters;TOKTABLE=table.rows;
  }catch(e){showError(e);return}
  const ok=await applyText('To be or');   // the text shown until the first chat message
  if(ok){camera.position.copy(camGoal.pos);controls.target.copy(camGoal.look)}
  frame();
}
window.addEventListener('load',start);
})();
