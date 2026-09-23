from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def compute_score(
    question_prompt: str,
    response_content: str,
    extracted_answer: str,
    outcome_score: float,
    response_length_tokens: int,
    max_response_length_tokens: int,
) -> dict:
    del question_prompt, response_content, response_length_tokens, max_response_length_tokens
    del extracted_answer
    outcome = float(outcome_score)
    return {
        "schema_version": "ade.reward_result.v2",
        "score": outcome,
        "outcome_score": outcome,
        "artifact_projection": None,
        "rule_evidence": {"status": "not_configured", "value": None},
    }


def compute_fallback_score(
    question_prompt: str,
    response_content: str,
    extracted_answer: str,
    outcome_score: float,
    response_length_tokens: int,
    max_response_length_tokens: int,
) -> dict:
    del question_prompt, response_content, response_length_tokens, max_response_length_tokens
    del extracted_answer
    outcome = float(outcome_score)
    return {
        "schema_version": "ade.reward_result.v2",
        "score": outcome,
        "outcome_score": outcome,
        "artifact_projection": None,
        "rule_evidence": {"status": "not_configured", "value": None},
    }


def assign_group_credit(group_input):
    return {
        "schema_version": "ade.group_credit_output.v2",
        "decision": {
            "mode": "identity",
            "evidence_sources": ["outcome"],
            "process_dimensions": [],
            "reason_code": "identity_baseline",
        },
        "assignments": [
            {
                "record_id": record["record_id"],
                "training_reward": float(record["outcome_score"]),
            }
            for record in group_input["records"]
        ],
    }
