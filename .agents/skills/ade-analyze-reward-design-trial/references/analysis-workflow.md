# RFT reward-design analysis workflow

Read the executed reward definition and resolved runtime without executing
them. Reconstruct score bounds, fixed evidence inputs, Artifact projection,
missing/non-finite behavior, and typed fallback. Read complete population statistics, normalized
reward/loss/KL/entropy telemetry, and offline aggregate evidence. Use the
catalog to inspect prompt groups at available artifact positions.

Use final direct-reference replay to establish which row-reward, assignment,
realized-advantage, sign, and Judge effects actually occurred. The realization
report contains observations rather than Coordinator targets or satisfaction
labels. Treat unavailable evidence as mechanism uncertainty without
invalidating a valid objective score. Leave the scientific verdict to Plan
Summarization.

Stage A hypotheses should test observable reward behavior. Group selectors are
useful for within-prompt discrimination; record selectors may add targeted
correctness/reward disagreements, high-reward errors, low-reward correct
responses, or other anomalies. Follow the catalog rather than applying one
coverage rule to both mechanisms: a legacy v1 training pool uses its single
global record fraction and permits incomplete groups; a Group Credit v2 pool
uses its single global group fraction and every selected group must include all
sibling response units. Neither mode adds a per-position quota. Select all
offline validation records.

In Stage B, compare response quality, authoritative outcome/correctness,
Engine-owned raw evidence, Artifact projection/pre-group reward, final training
reward, actual realized advantage, and the same-group outcome-only
counterfactual; inspect low/zero variance and misordering;
compare earlier and later positions with population statistics and telemetry;
and ask whether training reward gains transfer to complete offline behavior.
For enabled Group Credit, additionally compare assignment direction,
mode/reason, declared evidence sources and process dimensions, and matched
downstream validation. Disabled Group Credit still uses complete sibling groups
and must not regress to row-only context. Provider
reasoning is advisory and must be checked against source evidence.
