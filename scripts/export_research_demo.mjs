#!/usr/bin/env node
// Export only public display fields from the restored ADE records archive.
// Requires Node 22+ and unzip. Does not execute archived code or contact services.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

const [archive, output = 'examples/code-research-demo/additional-cases.js'] = process.argv.slice(2);
if (!archive) throw new Error('Usage: node scripts/export_research_demo.mjs <restored-archive-root> [output.js]');
const repo = path.join(archive, 'repositories/agentic-data-engineering');
const deployments = path.join(repo, 'runs/deployments');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const short = key => key.split('/').map(s => s[0].toUpperCase() + Number(s.slice(1))).join('/');
const numeric = value => {
  if (value == null) return null;
  assert.equal(typeof value, 'number');
  assert(Number.isFinite(value));
  return value;
};
const lines = text => text.trim().split('\n').filter(Boolean).map(JSON.parse);
function recordedEvaluation(runId, coordinator, plan, trial, phase) {
  const pattern = path.join(repo, 'runs/smoke', runId, 'evaluation/wandb-evaluations',
    `${runId}--${coordinator}--${plan}--${trial}--${phase === 'operator_test' ? 'operator-evaluation' : 'evaluation'}`,
    'records/*/evaluation-result.json');
  return fs.globSync(pattern).map(file=>({file,data:read(file)})).filter(r=>r.data.phase===phase);
}
function operator(runId, coordinator, plan, trial) {
  const records = recordedEvaluation(runId,coordinator,plan,trial,'operator_test');
  if (!records.length) return null;
  assert.equal(records.length,1);
  const metrics = records[0].data.results[0].metrics;
  return {pass:numeric(metrics['pass@k']),average:numeric(metrics['avg@k'])};
}
function packageReader(control, plan) {
  const dir = path.join(control, 'coordinators', plan.split('/')[0], 'plans', plan.split('/')[1], 'trials/t001/record/engine');
  const file = ['outcome-package.bin.compact.zip', 'outcome-package.bin'].map(n => path.join(dir,n)).find(fs.existsSync);
  if (!file) return null;
  const names = execFileSync('unzip', ['-Z1', file], {encoding:'utf8'}).trim().split('\n');
  return {
    names,
    text: name => execFileSync('unzip', ['-p', file, name], {encoding:'utf8', maxBuffer:64*1024*1024}),
  };
}

const result = {};
for (const [slug, suffix, reward] of [['math','e2649a',false], ['reward','71ae91',true]]) {
  const source = read(path.join(repo, 'analysis/timeline', suffix, 'case.json'));
  const runId = source.case.run_id;
  const matches = fs.readdirSync(deployments).map(d => path.join(deployments,d,'control',runId)).filter(fs.existsSync);
  assert.equal(matches.length, 1);
  const control = matches[0], deployment = path.dirname(path.dirname(control));
  const run = read(path.join(control,'run.json'));
  const baseline = run.trials.find(t => t.coordinator_id === 'c000');
  const primary = reward ? 'average' : 'pass';
  const order = field => new Map([...source.plans].sort((a,b)=>a[field]-b[field]).map((p,i)=>[p.plan_key,i+1]));
  const proposal = order('decision_revision'), completion = order('completion_revision');
  const data = {
    schema:2, selected:short(source.case.selected_plan_key),
    // The accepted imported baseline does not always retain its secondary score.
    baseline: {pass: numeric(reward ? baseline.offline_secondary_score : baseline.offline_score),
      average:numeric(reward ? baseline.offline_score : baseline.offline_secondary_score), operator:null},
    plans:[], relations:[], knowledge:[],
  };
  const baselineRun = run.bootstrap.reference_run_id;
  data.baseline.operator = operator(baselineRun,'c000','p000',baseline.trial_id);
  for (const p of source.plans) {
    const actual = read(path.join(control,p.source_plan_record));
    assert.equal(actual.best_score ?? null, p.validation_score, `Score drift: ${p.plan_key}`);
    assert.equal(actual.best_secondary_score ?? null, p.validation_secondary, `Secondary score drift: ${p.plan_key}`);
    if (p.selected) assert.equal(actual.best_artifact_ref_id, source.case.selected_artifact);
    const pack = packageReader(control,p.plan_key);
    const dir = path.join(deployment,'engine-work/trial_artifacts',runId,p.plan_key,'t001/attempt-001');
    const manifestFile = path.join(dir,'manifest.json');
    const manifest = fs.existsSync(manifestFile) ? read(manifestFile) : null;
    let training = [], online = [];
    if (pack) {
      if (reward) {
        const name = pack.names.find(n => n.endsWith('/training-telemetry.jsonl'));
        if (name) training = lines(pack.text(name)).filter(r=>r.metric_name==='actor/loss')
          .map(r=>({step:r.wandb_history_step,loss:numeric(r.metric_value)}));
      } else {
        const log = path.join(dir,'checkpoints/trainer_log.jsonl');
        if (fs.existsSync(log)) training = lines(fs.readFileSync(log,'utf8')).filter(r=>r.loss!=null)
          .map(r=>({step:r.current_steps,loss:numeric(r.loss),epoch:r.epoch}));
      }
      for (const name of pack.names.filter(n => /\/online-validation-(epoch|rl-step)-\d+\.json$/.test(n))) {
        const e = JSON.parse(pack.text(name));
        const position = e.artifact_position.value;
        const step = reward ? position : training.find(r=>Math.abs(r.epoch-position)<1e-6)?.step;
        if (step != null) online.push({step,epoch:reward?null:position,score:numeric(e.ranking_score)});
      }
    }
    if (!online.length) {
      const [coordinator, planId] = p.plan_key.split('/');
      for (const {file,data:e} of recordedEvaluation(runId,coordinator,planId,'t001','online_validation')) {
        const match = file.match(/online-validation-(?:epoch|rl-step)-(\d+)/);
        assert(match, 'Missing online evaluation position');
        const position = Number(match[1]);
        const step = reward ? position : training.find(r=>Math.abs(r.epoch-position)<1e-6)?.step;
        if (step != null && e.score != null) online.push({step,epoch:reward?null:position,score:numeric(e.score)});
      }
    }
    training.sort((a,b)=>a.step-b.step);
    online.sort((a,b)=>a.step-b.step);
    assert.equal(new Set(training.map(r=>r.step)).size, training.length, 'Duplicate training steps');
    assert.equal(new Set(online.map(r=>r.step)).size, online.length, 'Duplicate evaluation positions');
    const checkpoint = manifest?.selected_checkpoint_id;
    const position = checkpoint ? Number(checkpoint.split('-').at(-1)) : null;
    const selectedStep = reward ? position : training.find(r=>Math.abs(r.epoch-position)<1e-6)?.step ?? null;
    const plan = {
      id:short(p.plan_key),lane:Number(p.plan_key.slice(1,4)),plan:Number(p.plan_key.slice(-3)),
      proposalOrder:proposal.get(p.plan_key),completionOrder:completion.get(p.plan_key),
      pass:numeric(reward?p.validation_secondary:p.validation_score),
      average:numeric(reward?p.validation_score:p.validation_secondary),
      hypothesis: {Supported:'supported',Challenged:'rejected',Unresolved:'inconclusive',Failed:'failed'}[p.adjudication],
      status:p.status, title:p.strategy_short, description:p.strategy_description, conclusion:p.conclusion_short,
      operator:operator(runId,...p.plan_key.split('/'),'t001'), selectedEpoch:reward?null:position, selectedStep,
      training, online,
    };
    assert(plan.hypothesis);
    if (plan.status === 'failed') assert.equal(plan[primary],null);
    data.plans.push(plan);
  }
  for (const r of source.relations) {
    if (!r.source_plan_key || r.explicitness !== 'explicit' || !r.available_before_decision || !r.main_text_eligible) continue;
    const from = source.plans.find(p=>p.plan_key===r.source_plan_key);
    const to = source.plans.find(p=>p.plan_key===r.target_plan_key);
    assert(from.completion_revision < to.decision_revision);
    data.relations.push({source:short(r.source_plan_key),target:short(r.target_plan_key),kind:r.relation_type,label:r.edge_label});
  }
  assert.equal(data.plans.length,15);
  const selected = data.plans.find(p=>p.id===data.selected);
  assert.equal(selected[primary], Math.max(...data.plans.map(p=>p[primary]).filter(v=>v!=null)));
  result[slug] = data;
  console.log(`${slug}: ${data.plans.length} Plans, ${data.plans.reduce((n,p)=>n+p.training.length,0)} loss points, ${data.plans.reduce((n,p)=>n+p.online.length,0)} online points`);
}
fs.writeFileSync(output, '// Public display snapshot; no archive paths or runtime connection.\nwindow.ADE_ADDITIONAL_CASES = ' + JSON.stringify(result) + ';\n');
