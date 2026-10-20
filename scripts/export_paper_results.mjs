#!/usr/bin/env node
// Extract Table 3 values without changing precision or selecting new winners.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const [source, output = 'examples/code-research-demo/results-data.js'] = process.argv.slice(2);
if (!source) throw new Error('Usage: node scripts/export_paper_results.mjs <paper-source-directory> [output.js]');
const text = fs.readFileSync(path.join(source,'section/experiments.tex'),'utf8');
const table = text.slice(text.indexOf('\\label{tab:effectiveness-results}'),text.indexOf('\\label{sec:experimental-design}'));
const rows = table.split('\n').filter(line=>/^(Base model|Generic Baseline|ADE \()/u.test(line));
assert.equal(rows.length,16);
const tasks = [
  {id:'math',title:'Data selection · Math',columns:['AIME24','MATH-500','MATH-S','AIME25','HMMT','Macro avg.'],note:'K = 32 on AIME24, AIME25 and HMMT; K = 3 on MATH-500 and MATH-S.'},
  {id:'code',title:'Data selection · Code',columns:['LCB ≤ Mar 30','LCB > Mar 30','HumanEval+','Codeforces','CodeElo','Macro avg.'],note:'K = 3 throughout. LiveCodeBench is split at March 30, 2024.'},
  {id:'reward',title:'Reward design · Math',columns:['MATH-S','MATH-500','AIME24','AIME25','HMMT','Macro avg.'],note:'K = 3 on MATH-S and MATH-500; K = 32 on AIME24, AIME25 and HMMT.'},
  {id:'curriculum',title:'Curriculum learning · Math',columns:['MATH-S','MATH-500','AIME24','AIME25','HMMT','Macro avg.'],note:'K = 3 on MATH-S and MATH-500; K = 32 on AIME24, AIME25 and HMMT.'},
];
tasks.forEach((task,i)=>{
  task.rows = rows.slice(i*4,i*4+4).map((row,j)=>{
    const cells = row.split('&').slice(1);
    assert.equal(cells.length,6);
    return {name:['Base model','Generic baseline','ADE (C = 1)','ADE (C = 3)'][j],values:cells.map(cell=>{
      const values = [...cell.matchAll(/\d+\.\d+/g)].map(m=>Number(m[0]));
      assert.equal(values.length,2);
      return values;
    })};
  });
});
fs.writeFileSync(output,'// Paper Table 3: Best in-loop-selected Run per capacity; percentages.\nwindow.ADE_RESULTS = '+JSON.stringify(tasks)+';\n');
console.log('Exported 4 tasks × 4 methods × 6 benchmark columns.');
