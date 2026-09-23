(() => {
'use strict';
const data = window.ADE_CASE;
const $ = id => document.getElementById(id);
const plans = new Map(data.plans.map(p => [p.id, p]));
const state = { selected: data.selected, view: 'ancestry', curve: 'loss', count: 15, timer: null, followups: false };
const strings = {
  "skip": "Skip to research explorer",
  "navExplore": "The experiments",
  "navLearn": "What changed",
  "navMethod": "The setup",
  "heroLabel": "A FIELD STUDY IN AGENTIC DATA ENGINEERING",
  "heroTitle": "Better data.<br>Through <em>shared<br>discovery.</em>",
  "heroDescription": "Follow three research agents as they test, learn, and build on each other’s findings. One code data-selection run. Fifteen Plans that build on shared findings.",
  "explore": "Explore the research",
  "recorded": "An interactive, recorded case",
  "lengthMatch": "Match the length",
  "signal": "Make the signal observable",
  "selectedStrategy": "SELECTED STRATEGY",
  "winnerTitle": "Stratified<br>repetition control",
  "inloop": "In-loop validation",
  "pp": "pp",
  "overBaseline": "over the baseline",
  "sharedPool": "SAME POOL. SHARED KNOWLEDGE.",
  "statsAgents": "Coordinators<br>exploring in parallel",
  "statsPlans": "Plans<br>with shared findings",
  "statsData": "Selected trajectories<br>from one fixed pool",
  "statsHeldout": "Operator-test Pass@3<br>over the baseline",
  "sectionOne": "01 / THE RESEARCH EXPLORER",
  "researchTitle": "Every experiment leaves<br><em>something to learn.</em>",
  "researchDescription": "Trace the strategies that were inherited, and the ideas that travelled further. Select any experiment to look inside.",
  "ancestry": "Strategy ancestry",
  "knowledge": "Knowledge transfer",
  "replay": "Replay",
  "pause": "Pause",
  "ancestryLegend": "Recorded strategy inheritance · select a card to trace its connections",
  "knowledgeLegend": "Diagnostic knowledge links · distinct from artifact inheritance",
  "boardNote": "Columns: local Plan order. # marks proposal order, not completion order.",
  "progressLabel": "THE SEARCH, AS IT UNFOLDED",
  "progressTitle": "Progress is rarely a straight line.",
  "progressDescription": "Some experiments improve the score. Others improve the next question.",
  "bestSoFar": "Best so far",
  "trialScore": "Experiment",
  "baseline": "Baseline",
  "sectionTwo": "02 / INSIDE THE EXPERIMENT",
  "selectedByValidation": "✳ Selected by validation",
  "questionLabel": "The question",
  "changeLabel": "The intervention",
  "lessonLabel": "What it taught the search",
  "builtFrom": "BUILT FROM",
  "hypothesisNote": "Hypothesis assessment is separate from score-based selection.",
  "validationPass": "Validation · Pass@3",
  "validationAvg": "Validation · Average@3",
  "sameProtocol": "Same evaluation protocol",
  "loss": "Training loss",
  "online": "Online validation",
  "compareParent": "Compare parent",
  "heldoutLabel": "OPERATOR TEST · LATER LIVECODEBENCH SPLIT",
  "heldoutNote": "Held out from strategy selection.",
  "sectionThree": "03 / KNOWLEDGE THAT CROSSED BRANCHES",
  "insightTitle": "A better answer starts with<br><em>a better question.</em>",
  "insightDescription": "The selected strategy inherited one subset. Its reasoning drew on several experiments.",
  "insightBottom": "After C1/P3 became the leader, all three Coordinators revisited it—testing wrapper closure, example correctness, earlier code, and image-marker removal.",
  "followups": "See the follow-up experiments",
  "sectionFour": "04 / THE EXPERIMENTAL SETUP",
  "methodTitle": "Change the selection.<br><em>Hold the rest steady.</em>",
  "methodDescription": "Each strategy selects complete trajectories from the same mixed-domain OpenThoughts pool. The target model, training recipe, and evaluation protocol stay fixed.",
  "download": "Download the displayed data",
  "targetModel": "Target model",
  "trainingPool": "Training pool",
  "poolValue": "3,840 OpenThoughts trajectories",
  "selectionBudget": "Selection budget",
  "budgetValue": "384 whole trajectories per Trial",
  "selectionSignal": "Strategy selection",
  "signalValue": "In-loop Pass@3; Average@3 breaks ties",
  "trainingEnd": "Training",
  "endValue": "Recorded steps · early stopping",
  "footer": "Agentic Data Engineering · A real research trajectory, made explorable.",
  "backTop": "Back to top ↑",
  "newDirection": "New direction",
  "supported": "Hypothesis supported",
  "rejected": "Hypothesis rejected",
  "inconclusive": "Hypothesis inconclusive",
  "step": "Training step",
  "completed": "Completed Plans",
  "allRecorded": "All 15 Plans shown · results appear in completion order during replay.",
  "noResults": "Baseline established. Replay to reveal completed Plans.",
  "checkpoint": "Selected checkpoint",
  "earlyStop": "Early stop",
  "lossCaption": "Recorded loss at each training step. Curves end at early stopping; the baseline is shown only as a final evaluation reference.",
  "onlineCaption": "Online validation uses one response per problem (K=1). Final validation uses K=3 and is reported separately above.",
  "parent": "Parent",
  "selected": "Selected",
  "baselineFinal": "Final validation only",
  "offlinePoint": "Final validation",
  "notAvailable": "Not recorded",
  "followupNotice": "Highlighting the selected strategy and its four direct follow-ups.",
  "scrollHint": "Scroll to explore all Plans →"
};
const narrative = {
  "C1/P1": [
    "Complete targets",
    "Preserve complete supervision.",
    "Are cutoff-exceeding responses an avoidable source of supervision loss?",
    "Replace truncating baseline records with non-truncating candidates of similar length, keeping the 384-record budget and length allocation.",
    "Both validation metrics improved. The complete-target idea later crossed into another Coordinator’s combined strategy."
  ],
  "C1/P2": [
    "Judge validity",
    "Test semantic validity as a selection signal.",
    "Can a Judge identify invalid targets that should be replaced?",
    "Build on complete targets and seek same-family replacements using a semantic-validity rubric.",
    "The gate produced no failing group. Pass@3 rose slightly, but Average@3 fell; this informed the move to a deterministic signal."
  ],
  "C1/P3": [
    "Stratified repetition control",
    "Keep the exposure. Change the repetition.",
    "Does reducing repetition help when reasoning length and prompt type remain comparable?",
    "Replace highly repetitive parent records with low-repetition candidates matched by prompt type, truncation status and length band, then nearest token count.",
    "The best in-loop score, with the parent’s 142 / 107 / 135 length allocation preserved. Its hypothesis remained inconclusive; all three lanes reused the strategy."
  ],
  "C1/P4": [
    "Wrapper closure",
    "Check the whole executable wrapper.",
    "Is defining a requested function enough if the answer never calls it?",
    "Revisit C1/P3 and replace definite definition-and-call mismatches with matched wrapper-closed targets.",
    "The score fell below the parent. Inspection also exposed literal image markers, supplying a different question for C1/P5."
  ],
  "C1/P5": [
    "Image-marker removal",
    "Return to the stronger parent with a new clue.",
    "Do literal image placeholders weaken otherwise useful text-only supervision?",
    "Return to C1/P3 and replace marker-bearing records with matched marker-free candidates, retaining complete trajectories.",
    "The result did not exceed the leader. The diagnostic source and the inherited artifact came from different Plans."
  ],
  "C2/P1": [
    "Code-contract priority",
    "Select for the task the model will face.",
    "Can observable coding interfaces guide selection better than noisy domain labels?",
    "Prioritize prompts that request executable code through a named interface or a standard-input/output contract, with length and cutoff controls.",
    "The strongest initial direction. This subset became the parent of repetition control and a source for several combination Plans."
  ],
  "C2/P2": [
    "Moderate-length targets",
    "Separate long reasoning from wasted reasoning.",
    "Would less long-form supervision help a code-focused subset?",
    "Replace 43 long records with medium-length records within the same prompt-contract subtype.",
    "Both metrics regressed. C1/P3 learned to match length while targeting repetition instead of simply shortening reasoning."
  ],
  "C2/P3": [
    "Execution & coverage",
    "Combine executable targets with broader coverage.",
    "Can execution checks and prompt-family diversity improve the selection?",
    "Start a new direction: qualify targets with execution and prompt-example checks, then spread them across family/length cells.",
    "It did not beat C2/P1, but supplied the execution-and-coverage component for C2/P4."
  ],
  "C2/P4": [
    "Code-oriented execution mix",
    "Combine task alignment with executable coverage.",
    "Can the strongest code allocation benefit from execution qualification?",
    "Combine C2/P1’s allocation with C2/P3’s execution qualification and family coverage.",
    "Pass@3 improved over C2/P1 while Average@3 declined slightly. Better coverage did not improve both measures."
  ],
  "C2/P5": [
    "Earlier code onset",
    "Ask a complementary question.",
    "Can the final executable answer arrive earlier without changing other properties?",
    "Replace late-onset code targets with matched earlier-onset records, leaving wrapper and example defects to the other active branches.",
    "It did not beat C1/P3. The proposal explicitly divided the remaining questions with the other Coordinators."
  ],
  "C3/P1": [
    "Executable integrity",
    "Look for definite programming-target defects.",
    "Can deterministic checks improve machine-checkable targets?",
    "Check extraction, parsing, requested interfaces and runnable examples; replace definite failures with matched passes.",
    "The direction supplied concrete interface checks to later experiments; parsing alone proved too broad a test."
  ],
  "C3/P2": [
    "Requested callable",
    "A parseable answer can still miss the interface.",
    "Does the code actually define the function requested by the prompt?",
    "Combine C2/P1’s code-oriented subset with C3/P1’s integrity direction, targeting exact requested-callable presence.",
    "It did not exceed the stronger parent. Callable defects informed the more specific definition-and-invocation experiment in C1/P4."
  ],
  "C3/P3": [
    "Code-oriented complete targets",
    "Join task alignment and complete supervision.",
    "Can the code-focused allocation benefit from complete-target eligibility?",
    "Combine C2/P1’s allocation with C1/P1’s replacement of cutoff-exceeding targets.",
    "It became the primary-metric leader available to C3/P4, before repetition control completed."
  ],
  "C3/P4": [
    "AST branch complexity",
    "Explore structural complexity in matched targets.",
    "Would greater code branching help with harder problems?",
    "Start from C3/P3 and apply a bounded preference for AST branch complexity among otherwise matched execution-qualified targets.",
    "Pass@3 fell and Average@3 rose. This branch used the strongest completed strategy available at proposal time."
  ],
  "C3/P5": [
    "Prompt-example correctness",
    "Test correctness where the prompt makes it checkable.",
    "Can example-failing targets explain remaining wrong answers?",
    "Revisit C1/P3 and replace failures on unambiguous prompt examples, leaving static wrapper mismatches to C1/P4.",
    "It did not exceed the leader. The follow-up reused a shared strategy while investigating a separate residual failure mode."
  ]
};
const knowledgeText = {
  "K1": [
    "Target repetition.",
    "Reviews across the initial branches flagged redundancy and meandering. Repetition became an observable property worth testing."
  ],
  "K2": [
    "Longer isn’t the problem.",
    "Shortening the parent’s targets hurt both metrics. The next intervention matched length instead of imposing another cap."
  ],
  "K3": [
    "Use a sharper signal.",
    "A Judge gate produced no failing group. The selected strategy used a deterministic repetition measure to distinguish candidates."
  ],
  "K4": [
    "Close the wrapper.",
    "Callable defects motivated checking both function definition and invocation."
  ],
  "K5": [
    "Follow a new clue.",
    "Image-marker findings supplied a new question, while the stronger C1/P3 remained the inherited strategy."
  ]
};
const tr = k => strings[k] || k;
const title = id => narrative[id][0];
const pct = n => (100*n).toFixed(2);
const colors = ['#234f40','#477ba4','#b08443'];
const htmlEscape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let svgCounter = 0;
function renderCopy() {
  document.documentElement.lang = 'en';
  document.querySelectorAll('[data-i18n]').forEach(el => el.innerHTML = tr(el.dataset.i18n));
  document.querySelectorAll('[data-i18n-html]').forEach(el => el.innerHTML = tr(el.dataset.i18nHtml));
  document.title = 'ADE — A study in shared discovery';
  $('baseline-summary').textContent = `${pct(data.baseline.pass)}% Pass@3 · ${pct(data.baseline.average)}% Avg@3`;
}
function linkedIds() {
  const ids = new Set([state.selected]);
  if(state.followups) data.relations.filter(r=>r.source===data.selected).forEach(r=>{ids.add(r.source);ids.add(r.target)});
  else if(state.view==='ancestry') data.relations.filter(r=>r.source===state.selected||r.target===state.selected).forEach(r=>{ids.add(r.source);ids.add(r.target)});
  else data.knowledge.filter(k=>k.target===state.selected||k.sources.includes(state.selected)).forEach(k=>{ids.add(k.target);k.sources.forEach(s=>ids.add(s))});
  return ids;
}
function renderBoard() {
  $('experiment').hidden = state.count === 0;
  const linked = linkedIds();
  $('lanes').innerHTML = [1,2,3].map(lane => `<div class="lane" data-lane="${lane}"><div class="lane-label"><span class="lane-avatar" aria-hidden="true">${['✳','✧','◈'][lane-1]}</span><b>COORD. ${lane}</b><small>5 Plans</small></div>${data.plans.filter(p=>p.lane===lane).sort((a,b)=>a.plan-b.plan).map(p => {
    const future=p.completionOrder>state.count, selected=p.id===state.selected;
    const ks=data.knowledge.filter(k=>k.target===p.id).map(k=>k.id).join(' · ');
    const parents=data.relations.filter(r=>r.target===p.id);
    return `<button class="plan-card ${selected?'selected':''} ${p.id===data.selected?'winner':''} ${linked.has(p.id)&&!selected?'related':''} ${future?'future':''} ${state.followups&&!linked.has(p.id)?'dimmed':''}" data-plan="${p.id}" aria-pressed="${selected}" ${future?'disabled':''} aria-label="${p.id}: ${htmlEscape(title(p.id))}, ${future?'awaiting result':pct(p.pass)+'% Pass@3'}"><span class="card-top"><span>${p.id}</span><span class="order">#${String(p.proposalOrder).padStart(2,'0')} ${p.id===data.selected?'<span class="card-star">✳</span>':''}</span></span><span class="plan-title">${title(p.id)}</span><span class="plan-score">${future?'—':pct(p.pass)}<small>${future?'':'%'}</small></span><span class="card-bottom"><span>${parents.length?parents.map(r=>r.source).join(' + '):tr('newDirection')}</span>${ks?`<span class="card-k">${ks}</span>`:'<span>↗</span>'}</span></button>`;
  }).join('')}</div>`).join('');
  $('connection-legend').textContent=tr(state.view==='ancestry'?'ancestryLegend':'knowledgeLegend');
  $('replay-count').textContent=`${String(state.count).padStart(2,'0')} / 15`;
  $('timeline').value=state.count;
  $('timeline').setAttribute('aria-valuetext',`${state.count} ${tr('completed')}`);
  $('replay-status').textContent=state.followups?tr('followupNotice'):state.count===15?tr('allRecorded'):state.count===0?tr('noResults'):`${tr('completed')}: ${state.count} · ${state.selected} — ${title(state.selected)}`;
  requestAnimationFrame(drawConnections);
}
function drawConnections() {
  const board=$('board'), box=board.getBoundingClientRect(), svg=$('connections');
  if(!box.width)return;
  const edges=state.view==='ancestry'?data.relations.filter(r=>state.followups?r.source===data.selected:r.source===state.selected||r.target===state.selected):data.knowledge.filter(k=>k.target===state.selected||k.sources.includes(state.selected)).flatMap(k=>k.sources.map(s=>({source:s,target:k.target,kind:'knowledge'})));
  svg.setAttribute('viewBox',`0 0 ${box.width} ${box.height}`);
  svg.innerHTML=`<defs><marker id="arrow-edge" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8" fill="${state.view==='knowledge'?'#ae5e3b':'#879c77'}"/></marker></defs>`+edges.filter(e=>plans.get(e.source).completionOrder<=state.count&&plans.get(e.target).completionOrder<=state.count).map((e,i)=>{
    const a=board.querySelector(`[data-plan="${e.source}"]`).getBoundingClientRect(),b=board.querySelector(`[data-plan="${e.target}"]`).getBoundingClientRect();
    const ax=a.right-box.left+1,ay=a.top-box.top+a.height*.52,bx=b.left-box.left-2,by=b.top-box.top+b.height*.52;
    let d;
    if(Math.abs(ay-by)<5) { const up=a.top-box.top-8-i*2;d=`M${ax-15},${a.top-box.top} V${up} H${bx+15} V${b.top-box.top-2}`; }
    else {const mx=ax+Math.max(10,(bx-ax)/2);d=`M${ax},${ay} C${mx},${ay} ${bx-16},${by} ${bx},${by}`;}
    return `<path class="connection ${state.view==='knowledge'?'knowledge':''}" d="${d}" marker-end="url(#arrow-edge)"/>`;
  }).join('');
}
function selectPlan(id, scroll=false) {
  const p=plans.get(id);if(!p)return;
  state.selected=id;state.followups=false;
  if(p.completionOrder>state.count)state.count=15;
  renderBoard();renderDetail();renderProgress();
  if(scroll)$('experiment').scrollIntoView({behavior:motion()});
}
function motion(){return window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'}
function bindTooltip(svg, points, formatter, onClick) {
  const tooltip=$('chart-tooltip');
  svg.querySelectorAll('[data-point]').forEach(el=>{
    const point=points[Number(el.dataset.point)];
    const show=(event)=>{tooltip.hidden=false;tooltip.textContent=formatter(point);const r=el.getBoundingClientRect();const x=event.clientX??r.x,y=event.clientY??r.y;tooltip.style.left=`${Math.min(window.innerWidth-tooltip.offsetWidth-12,Math.max(12,x+12))}px`;tooltip.style.top=`${Math.max(12,y-tooltip.offsetHeight-12)}px`};
    el.addEventListener('pointerenter',show);el.addEventListener('pointermove',show);el.addEventListener('pointerleave',()=>tooltip.hidden=true);el.addEventListener('focus',show);el.addEventListener('blur',()=>tooltip.hidden=true);
    if(onClick){el.addEventListener('click',()=>{tooltip.hidden=true;onClick(point)});el.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();tooltip.hidden=true;onClick(point)}})}
  });
}
function renderProgress() {
  const W=670,H=220,m={l:37,r:17,t:15,b:34},x=i=>m.l+(W-m.l-m.r)*i/15,y=v=>H-m.b-(v-.36)/.15*(H-m.t-m.b);
  const ordered=[...data.plans].sort((a,b)=>a.completionOrder-b.completionOrder);
  let best=data.baseline.pass,stepPath=`M${x(0)},${y(best)}`;
  const show=ordered.filter(p=>p.completionOrder<=state.count);
  show.forEach(p=>{stepPath+=` H${x(p.completionOrder)}`;best=Math.max(best,p.pass);stepPath+=` V${y(best)}`});
  let content=`<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tr('progressLabel')}"><defs><linearGradient id="area-green" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#afc696" stop-opacity=".7"/><stop offset="100%" stop-color="#afc696" stop-opacity="0"/></linearGradient></defs>`;
  [.4,.45,.5].forEach(v=>content+=`<path class="grid-line" d="M${m.l},${y(v)}H${W-m.r}"/><text class="chart-axis" x="${m.l-8}" y="${y(v)+3}" text-anchor="end">${Math.round(v*100)}%</text>`);
  content+=`<path class="baseline-line" d="M${m.l},${y(data.baseline.pass)}H${W-m.r}"/><path class="chart-area" d="${stepPath} V${H-m.b} H${m.l}Z"/><path class="curve-line" d="${stepPath}"/>`;
  show.forEach((p,i)=>content+=`<circle class="chart-point ${p.id===state.selected?'current':''}" cx="${x(p.completionOrder)}" cy="${y(p.pass)}" r="${p.id===state.selected?5.5:4.5}" data-point="${i}" tabindex="0" role="button" aria-label="${p.id} ${pct(p.pass)}% Pass@3"/>`);
  [0,3,6,9,12,15].forEach(i=>content+=`<text class="chart-axis" x="${x(i)}" y="${H-17}" text-anchor="middle">${i}</text>`);
  content+=`<text class="chart-axis" x="${W-m.r}" y="${H-1}" text-anchor="end">${tr('completed')} · Pass@3</text></svg>`;
  $('progress-chart').innerHTML=content;
  bindTooltip($('progress-chart'),show,p=>`${p.id} · ${title(p.id)}\nPass@3 ${pct(p.pass)}%`,p=>selectPlan(p.id,true));
  let b=data.baseline.pass, path='M10,35';
  ordered.forEach((p,i)=>{path+=` H${10+(i+1)*260/15}`;b=Math.max(b,p.pass);path+=` V${35-(b-data.baseline.pass)/(plans.get(data.selected).pass-data.baseline.pass)*28}`});
  $('hero-spark').setAttribute('d',path);
}
function renderDetail() {
  const p=plans.get(state.selected),n=narrative[p.id];
  $('detail-id').textContent=p.id;$('detail-heading').textContent=title(p.id);
  $('detail-description').textContent=n[1];$('detail-question').textContent=n[2];$('detail-change').textContent=n[3];$('detail-lesson').textContent=n[4];
  $('winner-badge').hidden=p.id!==data.selected;
  const parents=data.relations.filter(r=>r.target===p.id).map(r=>r.source);
  $('detail-parents').innerHTML=parents.length?parents.map(id=>`<button class="parent-link" data-parent="${id}">${id} ↗</button>`).join(''):`<span class="parent-none">${tr('newDirection')}</span>`;
  $('hypothesis-status').textContent=tr(p.hypothesis);
  $('hypothesis-dot').style.background=p.hypothesis==='supported'?'#548773':p.hypothesis==='rejected'?'#ae5e3b':'#b4945e';
  $('detail-pass').innerHTML=`${pct(p.pass)}<small>%</small>`;$('detail-average').innerHTML=`${pct(p.average)}<small>%</small>`;
  const delta=(p.pass-data.baseline.pass)*100;$('detail-delta').textContent=`${delta>=0?'+':''}${delta.toFixed(2)} ${tr('pp')} ${tr('overBaseline')}`;
  $('detail-heldout').innerHTML=p.operator?`${pct(p.operator.pass)}%<small>Pass@3 · ${pct(p.operator.average)}% Avg@3</small>`:tr('notAvailable');
  $('compare').disabled=!parents.length;
  renderTraining();
}
function renderTraining() {
  const p=plans.get(state.selected),parents=data.relations.filter(r=>r.target===p.id).map(r=>plans.get(r.source));
  const all=[p,...($('compare').checked?parents:[])],loss=state.curve==='loss',W=590,H=225,m={l:37,r:17,t:14,b:33};
  const series=all.map((plan,i)=>({plan,color:colors[i],rows:loss?plan.training.map(r=>({x:r.step,y:r.loss,epoch:r.epoch})):plan.online.map(r=>({x:r.step,y:r.score,epoch:r.epoch}))}));
  const xmax=Math.max(...all.map(p=>p.training.at(-1).step)),maxValue=Math.max(...series.flatMap(s=>s.rows.map(r=>r.y))),ymax=loss?Math.ceil(maxValue*5)/5:Math.ceil((maxValue+.01)*10)/10;
  const x=v=>m.l+v/xmax*(W-m.l-m.r),y=v=>H-m.b-v/ymax*(H-m.t-m.b),gradientId=`curve-fill-${++svgCounter}`;
  let content=`<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${tr(loss?'loss':'online')} · ${p.id}"><defs><linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#a9c298" stop-opacity=".2"/><stop offset="100%" stop-color="#a9c298" stop-opacity="0"/></linearGradient></defs>`;
  for(let i=0;i<=4;i++){const v=ymax*i/4;content+=`<path class="grid-line" d="M${m.l},${y(v)}H${W-m.r}"/><text class="chart-axis" x="${m.l-8}" y="${y(v)+3}" text-anchor="end">${loss?v.toFixed(1):Math.round(v*100)+'%'}</text>`}
  const selectedPoint=p.online.find(r=>r.epoch===p.selectedEpoch);
  if(selectedPoint)content+=`<path d="M${x(selectedPoint.step)},${m.t}V${H-m.b}" stroke="#c49b6b" stroke-width="1" stroke-dasharray="3 4"/><text x="${x(selectedPoint.step)+5}" y="${m.t+5}" fill="#a97947" font-size="8">E${p.selectedEpoch} ✳</text>`;
  const dots=[];
  [...series].reverse().forEach(s=>{
    const d=s.rows.map((r,i)=>`${i?'L':'M'}${x(r.x)},${y(r.y)}`).join(' ');
    if(s.plan===p&&loss)content+=`<path d="${d} V${H-m.b} H${x(s.rows[0].x)}Z" fill="url(#${gradientId})"/>`;
    content+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.plan===p?1.9:1.4}" stroke-opacity="${s.plan===p?1:.65}" ${s.plan===p?'':'stroke-dasharray="5 3"'} stroke-linejoin="round"/>`;
    s.rows.forEach((r,i)=>{const index=dots.length;dots.push({...r,id:s.plan.id});content+=`<circle cx="${x(r.x)}" cy="${y(r.y)}" r="${loss?4.5:4}" fill="${s.color}" fill-opacity="${loss?0:1}" data-point="${index}" ${!loss||i===s.rows.length-1?'tabindex="0"':''} aria-label="${s.plan.id}, ${tr('step')} ${r.x}, ${loss?r.y.toFixed(4):pct(r.y)+'%'}"/>`});
  });
  const ticks=[0,Math.round(xmax/4),Math.round(xmax/2),Math.round(xmax*.75),xmax];ticks.forEach(v=>content+=`<text class="chart-axis" x="${x(v)}" y="${H-17}" text-anchor="middle">${v}</text>`);
  content+=`<text class="chart-axis" x="${W-m.r}" y="${H-1}" text-anchor="end">${tr('step')}</text></svg>`;
  $('training-chart').innerHTML=content;
  bindTooltip($('training-chart'),dots,r=>`${r.id} · ${tr('step')} ${r.x}\n${loss?'Loss '+r.y.toFixed(4):'Pass@1 '+pct(r.y)+'%'} · Epoch ${Number(r.epoch.toFixed(2))}`);
  $('curve-stop').textContent=`${tr('earlyStop')} · ${p.training.at(-1).step} steps`;
  $('checkpoint-tag').textContent=`${tr('checkpoint')} · E${p.selectedEpoch}${selectedPoint?' / step '+selectedPoint.step:''}`;
  $('curve-caption').textContent=tr(loss?'lossCaption':'onlineCaption');
  $('curve-legend').innerHTML=series.map((s,i)=>`<span><i class="dot" style="background:${s.color}"></i>${s.plan.id} · ${i===0?tr('selected'):tr('parent')}</span>`).join('');
}
function renderInsights() {
  $('insight-grid').innerHTML=['K1','K2','K3'].map(id=>{const k=data.knowledge.find(k=>k.id===id);return `<button class="insight-card" data-knowledge="${id}" aria-label="${id}: ${htmlEscape(knowledgeText[id][0])}"><div class="insight-card-top"><span class="insight-id">${id} / THE FINDING</span><span>↗</span></div><h3>${knowledgeText[id][0]}</h3><p>${knowledgeText[id][1]}</p><div class="insight-card-bottom">${k.sources.join(' + ')} → ${k.target}</div></button>`}).join('');
}
function updatePlay(){ $('play-icon').textContent=state.timer?'Ⅱ':'▶';$('play-label').textContent=tr(state.timer?'pause':'replay');$('play').setAttribute('aria-label',tr(state.timer?'pause':'replay')); }
function stop(){if(state.timer)clearInterval(state.timer);state.timer=null;updatePlay()}
function replayTo(n) {
  state.count=n;state.followups=false;
  if(n>0)state.selected=data.plans.find(p=>p.completionOrder===n).id;
  renderBoard();renderProgress();
  if(n>0)renderDetail();
}
function renderAll(){renderCopy();renderBoard();renderDetail();renderProgress();renderInsights();updatePlay()}
$('lanes').addEventListener('click',e=>{const card=e.target.closest('[data-plan]');if(card){stop();selectPlan(card.dataset.plan,true)}});
$('detail-parents').addEventListener('click',e=>{const b=e.target.closest('[data-parent]');if(b)selectPlan(b.dataset.parent)});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;state.followups=false;document.querySelectorAll('[data-view]').forEach(el=>{el.classList.toggle('active',el===b);el.setAttribute('aria-pressed',el===b)});renderBoard()}));
document.querySelectorAll('[data-curve]').forEach(b=>b.addEventListener('click',()=>{state.curve=b.dataset.curve;document.querySelectorAll('[data-curve]').forEach(el=>{el.classList.toggle('active',el===b);el.setAttribute('aria-pressed',el===b)});renderTraining()}));
$('compare').addEventListener('change',renderTraining);
$('play').addEventListener('click',()=>{if(state.timer){stop();return}if(state.count===15)replayTo(0);state.timer=setInterval(()=>{replayTo(state.count+1);if(state.count===15)stop()},1450);updatePlay()});
$('timeline').addEventListener('input',e=>{stop();replayTo(Number(e.target.value))});
$('reset').addEventListener('click',()=>{stop();state.count=15;selectPlan(data.selected)});
$('insight-grid').addEventListener('click',e=>{const b=e.target.closest('[data-knowledge]');if(!b)return;stop();state.count=15;state.view='knowledge';document.querySelectorAll('[data-view]').forEach(el=>{el.classList.toggle('active',el.dataset.view===state.view);el.setAttribute('aria-pressed',el.dataset.view===state.view)});selectPlan(data.knowledge.find(k=>k.id===b.dataset.knowledge).target);$('research').scrollIntoView({behavior:motion()})});
$('show-followups').addEventListener('click',()=>{stop();state.count=15;state.selected=data.selected;state.view='ancestry';state.followups=true;document.querySelectorAll('[data-view]').forEach(el=>{el.classList.toggle('active',el.dataset.view===state.view);el.setAttribute('aria-pressed',el.dataset.view===state.view)});renderBoard();renderDetail();renderProgress();$('research').scrollIntoView({behavior:motion()})});
$('download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='ade-code-research.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
window.addEventListener('resize',()=>requestAnimationFrame(drawConnections));window.addEventListener('scroll',()=>$('chart-tooltip').hidden=true,{passive:true});
renderAll();
})();
