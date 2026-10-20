(() => {
'use strict';
const $ = id => document.getElementById(id);
const escape = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const finite = n => typeof n === 'number' && Number.isFinite(n);
const pct = n => finite(n) ? (100*n).toFixed(2) : '—';
const signed = n => `${n>=0?'+':''}${n.toFixed(2)}`;
const colors = ['#13294b','#4b9cd3','#6b8299','#8d6b52'];
const meta = {
  math: {name:'Math data selection',kind:'LONG-COT SFT · C = 3',title:'Validity-gated, progress-ranked selection',primary:'pass',k:32,onlineLabel:'Pass@8',heldout:'AIME25',model:'Qwen2.5-7B-Instruct',paper:39,
    description:'A coarse quality gate could not distinguish enough candidates. C3/P2 replaces it with task-aware validity checks and a ranking of directness, useful progress and low redundancy, while preserving length quotas and token exposure.',
    protocol:'384 of 3,840 OpenThoughts trajectories · Qwen2.5-7B-Instruct · AIME24 Pass@32, then Average@32 for ties.',
    limit:'The selected Plan’s downstream hypothesis was supported, but Judge evidence covered only 361 of 384 selected pairs. Subsequent experiments changed multiple properties; they do not isolate the cause of the gain.',
    insights:[['A sharper measurement','C3/P1’s coarse Judge assigned top scores broadly. C3/P2 introduced validity gates and a more discriminating progress ranking.','C3/P2'],['Stricter is not always better','Later repetition filters, shorter targets and exact-pair verification did not exceed the selected strategy.','C3/P4'],['Keep the evidence limits visible','C1/P5 had a score, but its proposed reference handoff was not realized. Its accepted hypothesis remained inconclusive.','C1/P5']]},
  code: {name:'Code data selection',kind:'LONG-COT SFT · C = 3',title:'Stratified Repetition Control',primary:'pass',k:3,onlineLabel:'Pass@1',heldout:'Later LiveCodeBench split',model:'Qwen2.5-7B-Instruct',paper:9,
    description:'C1/P3 inherits a code-focused subset and replaces highly repetitive records with low-repetition candidates matched by prompt subtype, cutoff status, length band and nearest token count.',
    protocol:'384 of 3,840 OpenThoughts trajectories · Qwen2.5-7B-Instruct · LiveCodeBench Pass@3, then Average@3 for ties.',
    limit:'The selected strategy has the best in-loop score, while its mechanism hypothesis remains inconclusive. Diagnostic knowledge links are distinct from inherited artifacts. Operator test is one held-out split, not the four-benchmark Macro Avg.',
    insights:[['Length is a control','Shortening C2/P1’s targets reduced both metrics. The selected strategy therefore matched length while testing repetition.','C2/P2'],['An observable selection signal','C1/P2’s Judge gate found no failing group. C1/P3 used a deterministic repetition measure to distinguish candidates.','C1/P3'],['Findings travel beyond parents','Image-marker findings from C1/P4 informed C1/P5, even though C1/P5 inherited the stronger C1/P3 strategy.','C1/P5']]},
  reward: {name:'Math reward design',kind:'REINFORCEMENT FINE-TUNING · C = 3',title:'Group-conditioned process credit',primary:'average',k:3,onlineLabel:'Average@1',heldout:'MATH-500',model:'Qwen2.5-0.5B',paper:44,
    description:'C1/P3 moves from relative process quality to absolute eligibility and then sibling corroboration. Credit depends on group correctness; in all-wrong groups, positive shaping requires corroborating process evidence.',
    protocol:'Qwen2.5-0.5B · 100 RFT steps · MATH-S Average@3, then Pass@3 for ties. One of the fifteen Plans failed before evaluation.',
    limit:'C1/P3’s direct candidate/source replay fell back because its Judge was not configured, so the complete hypothesis remains inconclusive. C2/P5 has a score but no accepted agent-authored conclusion. The failed C2/P4 is not assigned a zero score.',
    insights:[['From relative to corroborated credit','C1’s first two proposals regressed. Requiring sibling corroboration produced the selected 36.67% Average@3 strategy.','C1/P3'],['Useful parallel alternatives','C3 combined failed-progress discrimination with C2’s outcome anchoring. Its later refinements did not exceed C1/P3.','C3/P4'],['Failure is a distinct outcome','C2/P4’s negative-evidence veto failed during Builder validation. It has no evaluable score or rejected hypothesis.','C2/P4']]},
};
const cases = {math:window.ADE_ADDITIONAL_CASES.math,code:window.ADE_CASE,reward:window.ADE_ADDITIONAL_CASES.reward};
let slug = 'code', data = cases.code, info = meta.code, plans = new Map();
const state = {selected:data.selected,view:'ancestry',curve:'loss',count:15,timer:null,followups:false};
const title = id => slug === 'code' ? window.ADE_CODE_NARRATIVE[id][0] : plans.get(id).title;
const primaryLabel = () => `${info.primary==='pass'?'Pass':'Average'}@${info.k}`;
const secondaryLabel = () => `${info.primary==='pass'?'Average':'Pass'}@${info.k}`;
const value = p => p[info.primary];
const sourceRelations = id => data.relations.filter(r=>r.target===id);
const behavior = () => matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth';
function setPressed(selector,selected) {document.querySelectorAll(selector).forEach(b=>{const active=b.dataset.view===selected||b.dataset.curve===selected;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))})}
function stop(){clearInterval(state.timer);state.timer=null;$('play').textContent='▶ Replay';$('play').setAttribute('aria-label','Replay Plans')}
function selectCase(id,writeUrl=false) {
  stop(); slug=Object.hasOwn(cases,id)?id:'code';data=cases[slug];info=meta[slug];plans=new Map(data.plans.map(p=>[p.id,p]));
  Object.assign(state,{selected:data.selected,view:'ancestry',curve:'loss',count:data.plans.length,followups:false});
  $('compare').checked=true;$('chart-tooltip').hidden=true;
  setPressed('[data-view]','ancestry');setPressed('[data-curve]','loss');
  const knowledgeButton=document.querySelector('[data-view="knowledge"]');knowledgeButton.disabled=!data.knowledge.length;
  knowledgeButton.title=data.knowledge.length?'Curated diagnostic links':'Separate diagnostic links are curated only for the Code case; explicit Plan relations are shown here.';
  document.querySelector('[data-curve="loss"]').textContent=slug==='reward'?'Actor loss':'Training loss';
  $('case-tabs').innerHTML=Object.entries(meta).map(([key,m],i)=>`<button class="case-tab ${key===slug?'active':''}" data-case="${key}" aria-pressed="${key===slug}"><span class="case-index">0${i+1}</span><strong>${m.name}</strong><small>${m.title}</small><span class="case-tab-bottom">15 Plans · C = 3 <span aria-hidden="true">↗</span></span></button>`).join('');
  const winner=plans.get(data.selected);
  $('case-kind').textContent=info.kind;$('case-title').textContent=info.title;$('case-description').textContent=info.description;
  $('case-selected').textContent=`Selected strategy · ${data.selected}`;$('case-score').textContent=`${pct(value(winner))}%`;$('case-metric').textContent=`In-loop ${primaryLabel()}`;
  $('case-gain').textContent=`${signed((value(winner)-value(data.baseline))*100)} pp over baseline`;
  $('case-protocol').textContent=info.protocol;$('case-limit').textContent=info.limit;
  $('case-paper').href=`https://arxiv.org/pdf/2610.14238#page=${info.paper}`;
  $('case-permalink').href=`?case=${slug}#research`;
  $('progress-description').textContent=`${primaryLabel()} after each finished Plan. ${slug==='reward'?'The failed Plan advances the replay without adding a score.':'Replay follows result completion order.'}`;
  $('timeline').max=data.plans.length;
  $('insight-grid').innerHTML=info.insights.map(([heading,body,id])=>`<button class="insight-card" data-insight="${id}"><span>${id} ↗</span><h4>${escape(heading)}</h4><p>${escape(body)}</p></button>`).join('');
  if(writeUrl){const url=new URL(location.href);url.searchParams.set('case',slug);history.pushState(null,'',url)}
  render();
}
function selectPlan(id,scroll=false){if(!plans.has(id))return;stop();state.selected=id;state.followups=false;if(plans.get(id).completionOrder>state.count)state.count=data.plans.length;render();if(scroll)$('experiment').scrollIntoView({behavior:behavior(),block:'start'})}
function edges(){return state.view==='knowledge'?data.knowledge.flatMap(k=>k.sources.map(source=>({source,target:k.target,kind:'knowledge',label:k.id}))):data.relations}
function visibleEdges(){return edges().filter(r=>state.followups?r.source===data.selected:r.source===state.selected||r.target===state.selected)}
function renderBoard(){
  const linked=new Set([state.selected]);visibleEdges().forEach(r=>{linked.add(r.source);linked.add(r.target)});
  $('lanes').innerHTML=[1,2,3].map(lane=>`<div class="lane"><div class="lane-label"><b>C${lane}</b><span>Coordinator ${lane}</span><small>5 Plans</small></div>${data.plans.filter(p=>p.lane===lane).sort((a,b)=>a.plan-b.plan).map(p=>{
    const future=p.completionOrder>state.count, score=value(p),selected=p.id===state.selected;
    const parents=sourceRelations(p.id),ks=data.knowledge.filter(k=>k.target===p.id).map(k=>k.id);
    return `<button class="plan-card ${selected?'selected':''} ${linked.has(p.id)?'related':''} ${p.id===data.selected?'winner':''} ${future?'future':''} ${state.followups&&!linked.has(p.id)?'dimmed':''}" data-plan="${p.id}" aria-pressed="${selected}" ${future?'disabled':''} aria-label="${p.id}: ${escape(title(p.id))}, ${future?'awaiting result':finite(score)?pct(score)+'% '+primaryLabel():'failed, not evaluated'}"><span class="card-top">${p.id}<small>#${p.proposalOrder}${p.id===data.selected?' ★':''}</small></span><strong>${escape(title(p.id))}</strong><span class="plan-score">${future?'—':finite(score)?pct(score)+'<small>%</small>':'<small>Not evaluated</small>'}</span><span class="card-bottom">${parents.length?parents.map(r=>`${r.kind==='combine'?'C':r.kind==='contradiction'?'X':'R'}: ${r.source}`).join(' · '):'New direction'}${ks.length?`<b>${ks.join(' · ')}</b>`:''}</span></button>`;
  }).join('')}</div>`).join('');
  $('connection-legend').textContent=state.view==='knowledge'?'K: recorded diagnostic transfer · distinct from artifact inheritance':'R: revisit · C: combine · X: contradiction (dashed)';
  $('replay-count').textContent=`${state.count} / ${data.plans.length}`;$('timeline').value=state.count;
  $('timeline').setAttribute('aria-valuetext',`${state.count} finished Plans`);
  $('replay-status').textContent=state.followups?'Selected strategy and its direct follow-ups highlighted.':state.count===0?'Baseline established. Replay to reveal finished Plans.':state.count===data.plans.length?`All ${data.plans.length} Plans shown${slug==='reward'?' · 14 evaluated, 1 failed':''}.`:`${state.count} Plans finished · ${state.selected}: ${title(state.selected)}`;
  const knowledge=data.knowledge.filter(k=>(k.target===state.selected||k.sources.includes(state.selected))&&plans.get(k.target).completionOrder<=state.count);
  $('knowledge-notes').hidden=state.view!=='knowledge'||!knowledge.length;
  $('knowledge-notes').innerHTML=knowledge.map(k=>`<p><strong>${k.id} · ${escape(window.ADE_CODE_KNOWLEDGE[k.id][0])}</strong> ${escape(window.ADE_CODE_KNOWLEDGE[k.id][1])}<small>${k.sources.join(' + ')} → ${k.target}</small></p>`).join('');
  $('experiment').hidden=state.count===0;requestAnimationFrame(drawConnections);
}
function drawConnections(){
  const board=$('board'),box=board.getBoundingClientRect(),svg=$('connections');if(!box.width)return;
  svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
  svg.innerHTML='<defs><marker id="edge-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#8c9caa"/></marker></defs>'+visibleEdges().filter(r=>plans.get(r.source).completionOrder<=state.count&&plans.get(r.target).completionOrder<=state.count).map((e,i)=>{
    const a=board.querySelector(`[data-plan="${e.source}"]`).getBoundingClientRect(),b=board.querySelector(`[data-plan="${e.target}"]`).getBoundingClientRect();
    const ax=a.right-box.left,ay=a.top-box.top+a.height/2,bx=b.left-box.left,by=b.top-box.top+b.height/2;
    const d=Math.abs(ay-by)<5?`M${ax-12},${a.top-box.top} V${a.top-box.top-7-i*2} H${bx+12} V${b.top-box.top}`:`M${ax},${ay} C${ax+20},${ay} ${bx-20},${by} ${bx},${by}`;
    return `<path d="${d}" class="connection ${e.kind}" marker-end="url(#edge-arrow)"/>`;
  }).join('');
}
function tooltip(container,points,format,click){
  const tip=$('chart-tooltip');container.querySelectorAll('[data-point]').forEach(el=>{
    const p=points[Number(el.dataset.point)];const show=e=>{tip.hidden=false;tip.textContent=format(p);const r=el.getBoundingClientRect();tip.style.left=`${Math.max(8,Math.min(innerWidth-tip.offsetWidth-8,(e.clientX??r.x)+12))}px`;tip.style.top=`${Math.max(8,(e.clientY??r.y)-tip.offsetHeight-10)}px`};
    el.addEventListener('pointerenter',show);el.addEventListener('pointermove',show);el.addEventListener('pointerleave',()=>tip.hidden=true);el.addEventListener('focus',show);el.addEventListener('blur',()=>tip.hidden=true);
    if(click){el.addEventListener('click',()=>{tip.hidden=true;click(p)});el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();tip.hidden=true;click(p)}})}
  });
}
function range(values,includeZero=false){let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max((hi-lo)*.12,.005);lo=includeZero?Math.min(0,lo-pad):lo-pad;hi+=pad;return [lo,hi]}
function renderProgress(){
  const ordered=[...data.plans].sort((a,b)=>a.completionOrder-b.completionOrder),scores=[value(data.baseline),...ordered.map(value)].filter(finite),[lo,hi]=range(scores);
  const W=680,H=220,m={l:47,r:15,t:16,b:35},x=i=>m.l+i/data.plans.length*(W-m.l-m.r),y=v=>H-m.b-(v-lo)/(hi-lo)*(H-m.t-m.b);
  let best=value(data.baseline),d=`M${x(0)},${y(best)}`;
  const finished=ordered.filter(p=>p.completionOrder<=state.count),shown=finished.filter(p=>finite(value(p)));
  finished.forEach(p=>{d+=` H${x(p.completionOrder)}`;if(finite(value(p)))best=Math.max(best,value(p));d+=` V${y(best)}`});
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Best-so-far ${primaryLabel()} by finished Plan">`;
  for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;svg+=`<path class="grid-line" d="M${m.l},${y(v)}H${W-m.r}"/><text x="${m.l-8}" y="${y(v)+4}" text-anchor="end">${(v*100).toFixed(1)}%</text>`}
  svg+=`<path class="baseline-line" d="M${m.l},${y(value(data.baseline))}H${W-m.r}"/><path class="curve-line" d="${d}"/>`;
  shown.forEach((p,i)=>svg+=`<circle class="chart-point ${p.id===state.selected?'current':''}" cx="${x(p.completionOrder)}" cy="${y(value(p))}" r="${p.id===state.selected?5:4}" tabindex="0" role="button" data-point="${i}" aria-label="${p.id}: ${pct(value(p))}% ${primaryLabel()}"/>`);
  [0,3,6,9,12,15].forEach(i=>svg+=`<text x="${x(i)}" y="${H-17}" text-anchor="middle">${i}</text>`);
  svg+=`<text x="${W-m.r}" y="${H-1}" text-anchor="end">Finished Plans · ${primaryLabel()}</text></svg>`;
  $('progress-chart').innerHTML=svg;tooltip($('progress-chart'),shown,p=>`${p.id} · ${title(p.id)}\n${primaryLabel()} ${pct(value(p))}%`,p=>selectPlan(p.id,true));
}
function renderDetail(){
  const p=plans.get(state.selected),n=slug==='code'?window.ADE_CODE_NARRATIVE[p.id]:null;
  $('detail-id').textContent=p.id;$('detail-heading').textContent=title(p.id);$('winner-badge').hidden=p.id!==data.selected;
  $('detail-description').textContent=n?n[1]:p.conclusion;
  $('detail-description').hidden=!n;
  $('detail-question').parentElement.hidden=!n;
  $('question-label').textContent=n?'Research question':'Research direction';
  $('detail-question').textContent=n?n[2]:p.description;
  $('detail-change').textContent=n?n[3]:p.id===data.selected?info.description:p.description;
  $('detail-lesson').textContent=n?n[4]:p.conclusion+'.';
  const relations=sourceRelations(p.id);
  $('detail-parents').innerHTML=relations.length?relations.map(r=>`<button data-parent="${r.source}" title="${escape(r.label||r.kind)}">${r.kind} · ${r.source} ↗</button>`).join(''):'<span>New direction</span>';
  $('hypothesis-status').textContent=({supported:'Hypothesis supported',rejected:'Hypothesis rejected',inconclusive:'Hypothesis inconclusive',failed:'Failed before evaluation'})[p.hypothesis];
  $('hypothesis-status').dataset.status=p.hypothesis;
  $('primary-label').textContent=`Validation · ${primaryLabel()}`;$('secondary-label').textContent=`Validation · ${secondaryLabel()}`;
  $('detail-primary').textContent=finite(value(p))?pct(value(p))+'%':'Not evaluated';const secondary=p[info.primary==='pass'?'average':'pass'];
  $('detail-secondary').textContent=finite(secondary)?pct(secondary)+'%':'—';
  $('detail-delta').textContent=finite(value(p))?`${signed((value(p)-value(data.baseline))*100)} pp over baseline`:'No score recorded';
  $('heldout-label').textContent=`Operator test · ${info.heldout}`;
  $('detail-heldout').innerHTML=p.operator?`${pct(p.operator[info.primary])}%<small>${primaryLabel()} · ${pct(p.operator[info.primary==='pass'?'average':'pass'])}% ${secondaryLabel()}</small>`:'<small>Not recorded in this snapshot</small>';
  $('compare').disabled=!relations.length;renderTraining();
}
function renderTraining(){
  const p=plans.get(state.selected),loss=state.curve==='loss';
  const sources=sourceRelations(p.id).map(r=>plans.get(r.source));const all=[p,...($('compare').checked?sources:[])];
  const series=all.map((plan,i)=>({plan,color:colors[i%colors.length],rows:(loss?plan.training:plan.online).filter(r=>finite(loss?r.loss:r.score)).map(r=>({x:r.step,y:loss?r.loss:r.score,epoch:r.epoch}))}));
  const own=series[0].rows,values=series.flatMap(s=>s.rows.map(r=>r.y));
  const selectedStep=p.selectedStep??p.online.find(r=>r.epoch===p.selectedEpoch)?.step;
  $('curve-stop').textContent=p.training.length?`Last recorded training step · ${p.training.at(-1).step}`:'No training trace available';
  $('checkpoint-tag').textContent=selectedStep!=null?`Selected checkpoint · ${slug==='reward'?'step '+selectedStep:'epoch '+p.selectedEpoch+' / step '+selectedStep}`:'Selected checkpoint · not recorded';
  $('curve-caption').textContent=loss?(slug==='reward'?'Recorded VERL actor/loss. This optimization objective is not an SFT token loss.':'Recorded SFT training loss. The selected checkpoint can precede the end of training.'):`Online ${info.onlineLabel}; final in-loop ${primaryLabel()} is evaluated separately. Only recorded evaluation points are connected.`;
  $('curve-legend').innerHTML=series.map((s,i)=>`<span><i class="dot" style="background:${s.color}"></i>${s.plan.id} · ${i?'Source':'Selected Plan'}${s.rows.length?'':' (no trace)'}</span>`).join('');
  if(!own.length){$('training-chart').innerHTML=`<div class="empty-chart">${p.status==='failed'?'This Plan failed before training and evaluation.':`No ${loss?'training loss':'online validation'} trace is available for this Plan.`}</div>`;return}
  const W=600,H=235,m={l:48,r:16,t:18,b:35},xmax=Math.max(1,...series.flatMap(s=>s.rows.map(r=>r.x)),selectedStep??0),[lo,hi]=range(values,loss);
  const x=v=>m.l+v/xmax*(W-m.l-m.r),y=v=>H-m.b-(v-lo)/(hi-lo)*(H-m.t-m.b);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${loss?(slug==='reward'?'Actor loss':'Training loss'):info.onlineLabel} for ${p.id}">`;
  for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;svg+=`<path class="grid-line" d="M${m.l},${y(v)}H${W-m.r}"/><text x="${m.l-8}" y="${y(v)+4}" text-anchor="end">${loss?v.toFixed(3):(v*100).toFixed(1)+'%'}</text>`}
  if(selectedStep!=null)svg+=`<path class="checkpoint-line" d="M${x(selectedStep)},${m.t}V${H-m.b}"/><text x="${Math.min(x(selectedStep),W-100)}" y="12">★ checkpoint</text>`;
  const dots=[];
  [...series].reverse().forEach(s=>{if(!s.rows.length)return;const d=s.rows.map((r,i)=>`${i?'L':'M'}${x(r.x)},${y(r.y)}`).join(' ');svg+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.plan===p?1.8:1.3}" ${s.plan===p?'':'stroke-dasharray="5 3"'}/>`;
    s.rows.forEach((r,i)=>{const index=dots.length;dots.push({...r,id:s.plan.id});svg+=`<circle cx="${x(r.x)}" cy="${y(r.y)}" r="${loss?4:3}" fill="${s.color}" fill-opacity="${loss?0:1}" data-point="${index}" ${!loss||i===0||i===s.rows.length-1?'tabindex="0"':''} aria-label="${s.plan.id}, step ${r.x}, ${loss?r.y.toFixed(4):pct(r.y)+'%'}"/>`})});
  [...new Set([0,Math.round(xmax/4),Math.round(xmax/2),Math.round(xmax*.75),xmax])].forEach(v=>svg+=`<text x="${x(v)}" y="${H-17}" text-anchor="middle">${v}</text>`);
  svg+=`<text x="${W-m.r}" y="${H-1}" text-anchor="end">${slug==='reward'?'RL':'Training'} step</text></svg>`;
  $('training-chart').innerHTML=svg;tooltip($('training-chart'),dots,r=>`${r.id} · step ${r.x}\n${loss?'Loss '+r.y.toFixed(4):info.onlineLabel+' '+pct(r.y)+'%'}${finite(r.epoch)?' · epoch '+r.epoch:''}`);
}
function render(){renderBoard();renderProgress();renderDetail()}
function replayTo(n){state.count=n;state.followups=false;if(n)state.selected=data.plans.find(p=>p.completionOrder===n).id;render()}
function renderResults(id){
  const task=window.ADE_RESULTS.find(t=>t.id===id)||window.ADE_RESULTS[0];
  $('result-tabs').innerHTML=window.ADE_RESULTS.map(t=>`<button data-result="${t.id}" class="${t===task?'active':''}" aria-pressed="${t===task}">${t.title}</button>`).join('');
  const baseline=task.rows[1].values.at(-1),best=task.rows[3].values.at(-1);
  $('result-summary').innerHTML=`<span>Held-out Macro Avg. · C = 3 vs baseline</span><strong>${signed(best[0]-baseline[0])} pp <small>Pass@K</small></strong><strong>${signed(best[1]-baseline[1])} pp <small>Average@K</small></strong>`;
  $('results-table').innerHTML=`<table><caption>${task.title} · selected strategy results (%)</caption><thead><tr><th scope="col" rowspan="2">Method</th><th scope="col">In-loop</th><th scope="colgroup" colspan="5">Held-out</th></tr><tr>${task.columns.map(c=>`<th scope="col">${escape(c)}</th>`).join('')}</tr></thead><tbody>${task.rows.map((r,i)=>`<tr class="${i===3?'highlight-row':''}"><th scope="row">${r.name}</th>${r.values.map(v=>`<td>${v[0].toFixed(2)} <span>/ ${v[1].toFixed(2)}</span></td>`).join('')}</tr>`).join('')}</tbody></table>`;
  $('results-note').textContent=task.note;
}
$('case-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-case]');if(b)selectCase(b.dataset.case,true)});
$('result-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-result]');if(b)renderResults(b.dataset.result)});
$('lanes').addEventListener('click',e=>{const b=e.target.closest('[data-plan]');if(b)selectPlan(b.dataset.plan,true)});
$('detail-parents').addEventListener('click',e=>{const b=e.target.closest('[data-parent]');if(b)selectPlan(b.dataset.parent)});
$('insight-grid').addEventListener('click',e=>{const b=e.target.closest('[data-insight]');if(b)selectPlan(b.dataset.insight,true)});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;state.followups=false;setPressed('[data-view]',state.view);renderBoard()}));
document.querySelectorAll('[data-curve]').forEach(b=>b.addEventListener('click',()=>{state.curve=b.dataset.curve;setPressed('[data-curve]',state.curve);renderTraining()}));
$('compare').addEventListener('change',renderTraining);
$('play').addEventListener('click',()=>{if(state.timer){stop();return}if(state.count===data.plans.length)replayTo(0);state.timer=setInterval(()=>{replayTo(state.count+1);if(state.count===data.plans.length)stop()},1450);$('play').textContent='Ⅱ Pause';$('play').setAttribute('aria-label','Pause replay')});
$('timeline').addEventListener('input',e=>{stop();replayTo(Number(e.target.value))});
$('reset').addEventListener('click',()=>{stop();state.count=data.plans.length;selectPlan(data.selected)});
$('show-followups').addEventListener('click',()=>{stop();Object.assign(state,{count:data.plans.length,selected:data.selected,view:'ancestry',followups:true});setPressed('[data-view]','ancestry');render();$('board').scrollIntoView({behavior:behavior(),block:'center'})});
$('download').addEventListener('click',()=>{const payload={...data,task:info.name,primaryMetric:primaryLabel(),onlineMetric:info.onlineLabel,heldoutBenchmark:info.heldout,notes:info.limit,plans:data.plans.map(p=>({...p,title:title(p.id),...(slug==='code'?{narrative:window.ADE_CODE_NARRATIVE[p.id]}:{})}))};const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`ade-${slug}-research.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
$('copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('bibtex').textContent);$('copy-status').textContent='Copied.'}catch{const range=document.createRange();range.selectNodeContents($('bibtex'));getSelection().removeAllRanges();getSelection().addRange(range);$('copy-status').textContent='Citation selected. Press Ctrl+C / ⌘C to copy.'}});
window.addEventListener('popstate',()=>selectCase(new URLSearchParams(location.search).get('case')));
window.addEventListener('resize',()=>requestAnimationFrame(drawConnections));window.addEventListener('scroll',()=>$('chart-tooltip').hidden=true,{passive:true});
renderResults('math');selectCase(new URLSearchParams(location.search).get('case'));
})();
