# Reward-design analysis background

A useful GRPO reward distinguishes responses to the same prompt in a direction
aligned with the fixed outcome target. Inspect component and effective reward
alongside actual response quality and available correctness. Important
patterns include zero or low within-group variance, incorrect high-reward
responses, correct low-reward responses, component disagreement, and a reward
signal dominated by formatting or length rather than solution quality.

Artifact positions make temporal claims testable. Compare group-level
discrimination and population distributions at earlier and later positions,
then connect them to reward, policy loss, KL, entropy, and other complete
training telemetry. Rising reward alone can reflect exploitation or collapse;
offline validation tests whether optimization transferred to the intended
behavior.

Recommendations should state a next `reward.py` hypothesis. Parser, grader,
TrainingOutcome, checkpoints, trainer settings, model, data, and infrastructure
remain fixed controls and may appear only as limitations or alternative
explanations.
