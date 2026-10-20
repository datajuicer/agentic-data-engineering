#!/usr/bin/env node
// Read accepted scripts and decision evidence; never execute archived code.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const [archive, output = 'examples/code-research-demo/source-data.js'] = process.argv.slice(2);
assert(archive, 'Usage: node scripts/export_homepage_evidence.mjs <RESTORED_ARCHIVE_ROOT> [output.js]');
const repo = path.join(archive, 'repositories/agentic-data-engineering');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const curation = read(new URL('../examples/code-research-demo/knowledge-curation.json', import.meta.url));
const short = key => key.split('/').map(s => s[0].toUpperCase() + Number(s.slice(1))).join('/');
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
const percent = value => (100 * value).toFixed(2) + '%';
const result = {};
for (const [slug, suffix] of Object.entries({math:'e2649a', code:'b823bb', reward:'71ae91'})) {
  const source = read(path.join(repo, 'analysis/timeline', suffix, 'case.json'));
  const deployments = path.join(repo, 'runs/deployments');
  const matches = fs.readdirSync(deployments).map(d => path.join(deployments,d,'control',source.case.run_id)).filter(fs.existsSync);
  assert.equal(matches.length, 1);
  const control = matches[0];
  const artifact = ref => {
    const i = ref.lastIndexOf('-'), kind = ref.slice(0,i), prefix = ref.slice(i+1);
    const dir = path.join(control,'artifacts',kind);
    const files = fs.readdirSync(dir).filter(n => n.startsWith(prefix));
    assert.equal(files.length,1);
    const text = fs.readFileSync(path.join(dir,files[0]),'utf8');
    assert.equal(hash(text),files[0], 'Artifact digest mismatch');
    return {text,sha256:files[0]};
  };
  const plans = {};
  const originals = new Map(source.plans.map(p=>[short(p.plan_key),p]));
  for (const p of source.plans) {
    const id = short(p.plan_key);
    const actual = read(path.join(control,p.source_plan_record));
    assert.equal(actual.best_score ?? null,p.validation_score, 'Accepted primary score mismatch');
    assert.equal(actual.best_secondary_score ?? null,p.validation_secondary, 'Accepted secondary score mismatch');
    const trial = read(path.join(control,'coordinators',p.plan_key.replace('/','/plans/'),'trials/t001/trial.json'));
    const ref = actual.best_artifact_ref_id;
    let script = null;
    if (ref) {
      assert.equal(ref,trial.artifact_ref_id, 'Plan/Trial artifact mismatch');
      script = {...artifact(ref),filename:slug==='reward'?'reward.py':'selection.py'};
      // Public scripts must not embed local deployment paths or credential literals.
      assert(!/(?:\/mnt\/workspace\/|\/home\/ruomeng\/|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})/.test(script.text));
    } else {
      assert.equal(p.status,'failed');
    }
    let failure = null;
    let finding;
    if (!ref) {
      const record = JSON.parse(artifact(trial.analysis_failure_ref_id).text);
      assert.equal(trial.failure_kind,'builder_validation_failed');
      const messages = [...new Set(record.violations.map(v=>v.message))];
      failure = {
        reason:record.reason,
        messages,
        explanation:'The last submitted code read conclusion_support and derivation_soundness through intermediate variables. The validator still rejected it; the record does not establish why that access failed validation.'
      };
      finding = `Script validation rejected assign_group_credit: “${messages.join('; ')}”. The initial submission and three retries did not pass, exhausting the Builder retry budget. Training and evaluation never started, so this is a build failure, not an observed loss in reward quality.`;
    }
    if (p.validation_score != null) {
      const label = slug==='reward'?'Average@3':slug==='math'?'Pass@32':'Pass@3';
      const second = slug==='reward'?'Pass@3':slug==='math'?'Average@32':'Average@3';
      finding = `Final validation reached ${percent(p.validation_score)} ${label} and ${percent(p.validation_secondary)} ${second}.`;
      const comparator = actual.hypothesis_comparator;
      if (comparator?.primary?.value != null) {
        const parts = comparator.subject_id.split('/');
        const sourceId = short(parts.slice(-3,-1).join('/'));
        finding += ` The recorded comparison ${sourceId==='C0/P0'?'baseline':sourceId} scored ${percent(comparator.primary.value)} ${label}`;
        if (comparator.secondary?.value != null) finding += ` and ${percent(comparator.secondary.value)} ${second}`;
        finding += '.';
      }
      if (slug==='reward' && id==='C1/P3') finding += ' This was the highest Average@3 among the 14 evaluated Plans.';
      if (slug==='math' && id==='C3/P2') finding += ' This solved 17 of 30 AIME24 problems at least once, the best result in this Run.';
      if (slug==='math' && id==='C1/P5') finding += ' The proposed independent-reference handoff was not realized, so the score does not test that intended change.';
      if (slug==='reward' && id==='C2/P5') finding += ' The archive contains the evaluation but no accepted agent-written analysis of this result.';
    }
    plans[id] = {script,finding,...(failure?{failure}:{})};
  }
  const knowledge = (curation[slug] || []).map(k=>{
    const target = originals.get(k.target);
    assert(target);
    for (const id of k.sources) {
      assert(originals.has(id));
      assert(originals.get(id).completion_revision < target.decision_revision, 'Knowledge source was not available at decision time');
    }
    const actual = read(path.join(control,target.source_plan_record));
    const report = artifact(actual.decision_report_ref_id);
    assert(report.text.includes(k.quote), `Decision evidence missing: ${slug} ${k.id}`);
    return {...k,evidence:{record:`${k.target} planning decision`,sha256:report.sha256}};
  });
  result[slug] = {plans,knowledge};
  console.log(`${slug}: ${Object.values(plans).filter(p=>p.script).length} accepted scripts; ${knowledge.length} curated knowledge links`);
}
fs.writeFileSync(output, '// Accepted strategy scripts and checked decision excerpts. Display only; never executed.\nwindow.ADE_SOURCE_DATA = ' + JSON.stringify(result) + ';\n');
