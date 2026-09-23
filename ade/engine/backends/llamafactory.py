"""LlamaFactory implementation of the typed SFT backend."""

from __future__ import annotations

import json
import re
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ade.core.engine import TrainSFTCommand
from ade.core.ranking import score_pair_improves
from ade.engine.execution.ray import checkpoint_ready, list_checkpoints, run_train_task
from ade.engine.requests import SFTInput


class LlamaFactorySFTBackend:
    def train(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
    ) -> dict[str, Any]:
        request = config.sft.get("request")
        if not isinstance(request, dict):
            raise ValueError("production SFT input requires sft.request")
        request = dict(request)
        request["task_id"] = command.command_id
        request["agent_task_id"] = command.run_id
        request["coordinator_id"] = command.coordinator_id
        request["plan_id"] = command.plan_id
        request["trial_id"] = command.trial_id
        request["logical_command_id"] = command.logical_command_id
        request["attempt_id"] = command.attempt_id
        request["attempt_index"] = command.attempt_index
        if isinstance(config.raw.get("fork_lineage"), dict):
            request["fork_lineage"] = dict(config.raw["fork_lineage"])
        result = run_train_task(request)
        checkpoints = _normalize_remaining_checkpoints(
            Path(request["checkpoint_output_dir"])
        )
        model_path = checkpoints[-1] if checkpoints else Path(request["checkpoint_output_dir"])
        if result.get("status") != "completed" or not model_path.is_dir():
            raise RuntimeError("LlamaFactory SFT did not produce a model directory")
        return {
            "model_ref": str(model_path.resolve()),
            "checkpoints": tuple(str(path.resolve()) for path in checkpoints),
            "metrics": {
                "backend": "llamafactory",
                "train_count": int(result["train_count"]),
                "steps_per_epoch": int(result["steps_per_epoch"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "log_path": str(result["log_path"]),
            },
        }

    def train_with_checkpoints(
        self,
        command: TrainSFTCommand,
        config: SFTInput,
        evaluate_checkpoint: Callable[[int, str], dict[str, Any]],
    ) -> dict[str, Any]:
        request = config.sft.get("request")
        if not isinstance(request, dict):
            raise ValueError("production SFT input requires sft.request")
        request = dict(request)
        request["task_id"] = command.command_id
        request["agent_task_id"] = command.run_id
        request["coordinator_id"] = command.coordinator_id
        request["plan_id"] = command.plan_id
        request["trial_id"] = command.trial_id
        request["logical_command_id"] = command.logical_command_id
        request["attempt_id"] = command.attempt_id
        request["attempt_index"] = command.attempt_index
        if isinstance(config.raw.get("fork_lineage"), dict):
            request["fork_lineage"] = dict(config.raw["fork_lineage"])
        checkpoint_root = Path(request["checkpoint_output_dir"])
        stable_seconds = float(request.get("checkpoint_ready_stable_seconds", 8))
        evaluated: set[str] = set()
        online: list[dict[str, Any]] = []
        early_stopper = OnlineEarlyStopper(
            patience=int(request.get("early_stopping_patience", 0))
        )
        stop_file = Path(request.get("stop_file") or checkpoint_root / "STOP")
        stop_file.unlink(missing_ok=True)
        evaluations = []
        with (
            ThreadPoolExecutor(max_workers=1) as training_pool,
            ThreadPoolExecutor(max_workers=2) as online_pool,
        ):
            training_future = training_pool.submit(run_train_task, request)
            while not training_future.done():
                if not stop_file.exists():
                    self._evaluate_ready_checkpoints(
                        checkpoint_root,
                        stable_seconds=stable_seconds,
                        evaluated=evaluated,
                        evaluations=evaluations,
                        submit_evaluation=lambda epoch, checkpoint: online_pool.submit(
                            evaluate_checkpoint, epoch, checkpoint
                        ),
                        require_stable=True,
                    )
                self._collect_evaluations(
                    evaluations,
                    online=online,
                    early_stopper=early_stopper,
                    stop_file=stop_file,
                    wait=False,
                )
                time.sleep(1)
            result = training_future.result()
            if not stop_file.exists():
                self._evaluate_ready_checkpoints(
                    checkpoint_root,
                    stable_seconds=stable_seconds,
                    evaluated=evaluated,
                    evaluations=evaluations,
                    submit_evaluation=lambda epoch, checkpoint: online_pool.submit(
                        evaluate_checkpoint, epoch, checkpoint
                    ),
                    require_stable=False,
                )
            self._collect_evaluations(
                evaluations,
                online=online,
                early_stopper=early_stopper,
                stop_file=stop_file,
                wait=True,
            )
        checkpoints = _list_sft_checkpoints(checkpoint_root)
        model_path = checkpoints[-1] if checkpoints else checkpoint_root
        if result.get("status") not in {"completed", "stopped"} or not model_path.is_dir():
            raise RuntimeError("LlamaFactory SFT did not produce a model directory")
        return {
            "model_ref": str(model_path.resolve()),
            "checkpoints": tuple(str(path.resolve()) for path in checkpoints),
            "online_evaluations": online,
            "metrics": {
                "backend": "llamafactory",
                "train_count": int(result["train_count"]),
                "steps_per_epoch": int(result["steps_per_epoch"]),
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "log_path": str(result["log_path"]),
                "early_stopped": early_stopper.triggered_epoch is not None,
                "early_stop_best_score": early_stopper.best_score,
                "early_stop_best_secondary_score": early_stopper.best_secondary_score,
                "early_stop_no_improvement_count": early_stopper.no_improvement_count,
                "early_stop_epoch": early_stopper.triggered_epoch,
                "online_cancelled_count": sum(
                    item.get("status") == "cancelled" for item in online
                ),
            },
        }

    @staticmethod
    def _evaluate_ready_checkpoints(
        checkpoint_root: Path,
        *,
        stable_seconds: float,
        evaluated: set[str],
        evaluations: list[tuple[int, str, Any]],
        submit_evaluation: Callable[[int, str], Any],
        require_stable: bool,
    ) -> None:
        for checkpoint in list_checkpoints(checkpoint_root):
            source_key = str(checkpoint.resolve())
            if source_key in evaluated:
                continue
            if require_stable and not checkpoint_ready(
                checkpoint,
                stable_seconds=stable_seconds,
            ):
                continue
            normalized, epoch, _train_step = _normalize_sft_checkpoint(checkpoint)
            key = str(normalized.resolve())
            evaluations.append((epoch, key, submit_evaluation(epoch, key)))
            evaluated.update((source_key, key))

    @staticmethod
    def _collect_evaluations(
        evaluations: list[tuple[int, str, Any]],
        *,
        online: list[dict[str, Any]],
        early_stopper: "OnlineEarlyStopper",
        stop_file: Path,
        wait: bool,
    ) -> None:
        while evaluations:
            item = min(evaluations, key=lambda value: value[0])
            if not wait and not item[2].done():
                break
            evaluations.remove(item)
            value = item[2].result()
            online.append(value)
            score = value.get("ranking_score", value.get("score"))
            secondary_score = value.get("secondary_score")
            if (
                not stop_file.exists()
                and score is not None
                and early_stopper.observe(
                    float(score),
                    secondary_score=(
                        float(secondary_score)
                        if secondary_score is not None
                        else None
                    ),
                    epoch=item[0],
                )
            ):
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.write_text(
                    f"online validation did not improve for {early_stopper.patience} checkpoints\n",
                    encoding="utf-8",
                )
                cancelled = LlamaFactorySFTBackend._cancel_queued_evaluations(
                    evaluations
                )
                for epoch, checkpoint in cancelled:
                    online.append(
                        {
                            "step": epoch,
                            "checkpoint": checkpoint,
                            "status": "cancelled",
                            "score": None,
                            "ranking_score": None,
                            "payload": {},
                            "command_id": None,
                            "reason": "early_stopping",
                        }
                    )

    @staticmethod
    def _cancel_queued_evaluations(
        evaluations: list[tuple[int, str, Any]],
    ) -> list[tuple[int, str]]:
        """Drop eval futures that have not started after early-stop fires."""
        retained = []
        cancelled = []
        for item in evaluations:
            future = item[2]
            cancel = getattr(future, "cancel", None)
            if callable(cancel) and cancel():
                cancelled.append((item[0], item[1]))
                continue
            retained.append(item)
        evaluations[:] = retained
        return cancelled


@dataclass
class OnlineEarlyStopper:
    patience: int
    best_score: float | None = None
    best_secondary_score: float | None = None
    no_improvement_count: int = 0
    triggered_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("early stopping patience must be non-negative")

    def observe(
        self,
        score: float,
        *,
        secondary_score: float | None = None,
        epoch: int | None = None,
    ) -> bool:
        if self.best_score is None or score_pair_improves(
            (score, secondary_score),
            (self.best_score, self.best_secondary_score),
            direction="maximize",
        ):
            self.best_score = score
            self.best_secondary_score = secondary_score
            self.no_improvement_count = 0
            return False
        self.no_improvement_count += 1
        triggered = self.patience > 0 and self.no_improvement_count >= self.patience
        if triggered and self.triggered_epoch is None:
            self.triggered_epoch = epoch
        return triggered


_EPOCH_CHECKPOINT_RE = re.compile(r"^epoch-(\d+)$")
_NON_MODEL_CHECKPOINT_NAMES = {
    "optimizer.pt",
    "trainer_state.json",
    "training_args.bin",
    "scheduler.pt",
    "scaler.pt",
    "zero_to_fp32.py",
}


def _normalize_sft_checkpoint(checkpoint: Path) -> tuple[Path, int, int]:
    state_path = checkpoint / "trainer_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        raw_epoch = float(state["epoch"])
        raw_train_step = state.get("global_step")
        if raw_train_step is None:
            candidates = [
                int(match.group(1))
                for child in checkpoint.iterdir()
                if (match := re.fullmatch(r"global_step(\d+)", child.name))
            ]
            if candidates:
                raw_train_step = max(candidates)
        if raw_train_step is None:
            match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
            raw_train_step = int(match.group(1)) if match else None
        if raw_train_step is None:
            raise KeyError("global_step")
        train_step = int(raw_train_step)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"SFT checkpoint lacks a valid trainer epoch: {checkpoint}") from exc
    epoch = int(round(raw_epoch))
    if epoch < 1 or abs(raw_epoch - epoch) > 1.0e-6:
        raise ValueError(
            f"SFT artifact checkpoint must end on a complete epoch: {checkpoint}"
        )
    for path in checkpoint.iterdir():
        if path.is_dir() and re.fullmatch(r"global_step\d+", path.name):
            shutil.rmtree(path)
            continue
        if path.name in _NON_MODEL_CHECKPOINT_NAMES or re.fullmatch(
            r"rng_state(?:_\d+)?\.pth",
            path.name,
        ):
            if path.is_file():
                path.unlink()
    target = checkpoint.parent / f"epoch-{epoch:03d}"
    if checkpoint != target:
        if target.exists():
            raise FileExistsError(f"SFT epoch checkpoint already exists: {target}")
        checkpoint.rename(target)
    (target / "checkpoint_position.json").write_text(
        json.dumps({"epoch": epoch, "train_step": train_step}) + "\n",
        encoding="utf-8",
    )
    return target, epoch, train_step


def _normalize_remaining_checkpoints(root: Path) -> list[Path]:
    for checkpoint in list_checkpoints(root):
        _normalize_sft_checkpoint(checkpoint)
    return _list_sft_checkpoints(root)


def _list_sft_checkpoints(root: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for path in root.iterdir() if root.is_dir() else ():
        match = _EPOCH_CHECKPOINT_RE.fullmatch(path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    return [path for _, path in sorted(checkpoints)]
