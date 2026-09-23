"""Typed payloads referenced by Engine commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ENGINE_INPUT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvaluationInput:
    purpose: str
    request: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class SFTInput:
    artifact_ref: str
    sft: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class RFTInput:
    artifact_ref: str
    rft: dict[str, Any]
    raw: dict[str, Any]


def decode_evaluation_input(payload: dict[str, Any]) -> EvaluationInput:
    _schema(payload)
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("Evaluation input requires evaluation config")
    purpose = str(evaluation.get("purpose") or "")
    request = evaluation.get("request")
    if not isinstance(request, dict):
        raise ValueError("Evaluation input requires evaluation.request")
    return EvaluationInput(purpose=purpose, request=dict(request), raw=dict(payload))


def decode_sft_input(payload: dict[str, Any]) -> SFTInput:
    _schema(payload)
    sft = payload.get("sft")
    artifact_ref = str(payload.get("artifact_ref") or "")
    if not isinstance(sft, dict) or not artifact_ref:
        raise ValueError("SFT input requires artifact_ref and sft config")
    return SFTInput(artifact_ref=artifact_ref, sft=dict(sft), raw=dict(payload))


def decode_rft_input(payload: dict[str, Any]) -> RFTInput:
    _schema(payload)
    rft = payload.get("rft")
    artifact_ref = str(payload.get("artifact_ref") or "")
    if not isinstance(rft, dict) or not artifact_ref:
        raise ValueError("RFT input requires artifact_ref and rft config")
    total_steps = int(rft.get("total_training_steps") or 0)
    interval = int(rft.get("artifact_interval") or 0)
    if total_steps < 1 or interval < 1:
        raise ValueError("RFT total_training_steps and artifact_interval must be positive")
    group_credit = rft.get("group_credit")
    if group_credit is not None:
        if not isinstance(group_credit, dict) or type(group_credit.get("enabled")) is not bool:
            raise ValueError("RFT group_credit.enabled must be explicit boolean")
        if group_credit["enabled"] is False:
            if set(group_credit) != {"enabled"}:
                raise ValueError("disabled RFT group_credit accepts only enabled")
            if "semantic_evidence_interval" in rft or "semantic_evidence_steps" in rft:
                raise ValueError("disabled RFT group_credit cannot declare semantic evidence")
        else:
            if group_credit != {
                "enabled": True,
                "entrypoint": "assign_group_credit",
                "schema_version": "ade.group_credit.v1",
            }:
                raise ValueError("enabled RFT group_credit contract is invalid")
            semantic_interval = rft.get("semantic_evidence_interval")
            semantic_steps = rft.get("semantic_evidence_steps")
            if type(semantic_interval) is not int or semantic_interval < 1:
                raise ValueError("RFT semantic_evidence_interval must be positive")
            if not isinstance(semantic_steps, list) or any(
                type(step) is not int or step < 1 for step in semantic_steps
            ):
                raise ValueError("RFT semantic_evidence_steps must be positive integers")
            artifact_steps = list(range(interval, total_steps + 1, interval))
            if not artifact_steps or artifact_steps[-1] != total_steps:
                artifact_steps.append(total_steps)
            expected_steps = [
                step for step in artifact_steps if step % semantic_interval == 0
            ]
            if not expected_steps or semantic_steps != expected_steps:
                raise ValueError("RFT semantic_evidence_steps do not match artifact positions")
            verl_config = rft.get("verl_config")
            if not isinstance(verl_config, dict):
                raise ValueError("enabled RFT group credit requires verl_config")
            for section in (
                verl_config.get("rft"),
                (verl_config.get("train") or {}).get("rft")
                if isinstance(verl_config.get("train"), dict)
                else None,
            ):
                if not isinstance(section, dict):
                    raise ValueError("enabled RFT group credit requires mirrored runtime config")
                if (
                    section.get("group_credit") != group_credit
                    or section.get("semantic_evidence_interval") != semantic_interval
                    or section.get("semantic_evidence_steps") != semantic_steps
                ):
                    raise ValueError("RFT group credit runtime mirrors are inconsistent")
    return RFTInput(artifact_ref=artifact_ref, rft=dict(rft), raw=dict(payload))


def _schema(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != ENGINE_INPUT_SCHEMA_VERSION:
        raise ValueError(f"Engine input schema_version must be {ENGINE_INPUT_SCHEMA_VERSION}")
