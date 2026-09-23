# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
import time
from collections import defaultdict

import numpy as np
import transfer_queue as tq
from omegaconf import DictConfig
from transfer_queue import KVBatchMeta

from verl.utils.skip import SkipManager

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS = int(os.getenv("VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS", "60"))


def _accumulate_drop_metrics(acc: dict, new: dict, dropped: int) -> None:
    """Merge one poll iteration's drop metrics into ``acc`` in place."""
    count_key = next((k for k in new if k.endswith("/dropped_samples")), None)
    prev_total = acc.get(count_key, 0) if count_key else 0

    for key, value in new.items():
        if key.endswith("/dropped_samples"):
            acc[key] = acc.get(key, 0) + value
        elif key.endswith("/dropped_samples_staleness/mean"):
            denom = prev_total + dropped
            acc[key] = (acc.get(key, 0.0) * prev_total + value * dropped) / denom if denom else value
        elif key.endswith("/dropped_samples_staleness/max"):
            acc[key] = max(acc.get(key, value), value)
        elif key.endswith("/dropped_samples_staleness/min"):
            acc[key] = min(acc.get(key, value), value)


# TODO: Pass custom sampler to TransferQueue:
# https://github.com/Ascend/TransferQueue/blob/main/tutorial/05_custom_sampler.py


class ReplayBuffer:
    """ReplayBuffer is used by trainer to sample trajectories produced during rollout.

    We use [TransferQueue](https://github.com/Ascend/TransferQueue) as kv store to store trajectories.

    ### [Trajectories storage format]
    The key format is `{uid}_{session_id}_{index}`, where:
    - uid: Auto generated unique id when prompt is sampled from dataset.
    - session_id: Session id for GRPO group sampling: [0, n).
    - index: Index of output trajectory in a session.

    There're two types of data associated with each key: tag and value. The tag are arbitrary metadata:
    `{"status": "running", ...}` used to track the status of the trajectory.

    The value is a dictionary containing the following fields:
    - messages/datasource/reward_model/...: fields from dataset.
    - prompt_ids/response_ids/response_mask/...: fields from AgentLoopOutput.

    TransferQueue store tag and value separately, the tag are stored in meta server, while the value is stored
    in storage units.

    ### [GRPO group sampling control]
    Except trajectories, we also store raw prompts in TransferQueue with key `{uid}`, with `status` tag to track
    status of GRPO group sampling.
    - pending: the prompt is sampled from dataset but its sessions are not yet started.
    - running: all sessions of the prompt are running.
    - finished: all sessions of the prompt are finished without error.
    - failure: all sessions of the prompt are finished, but at least one session failed.
    Only prompt with status `finished` or `failure`, its trajectories can be sampled by replay buffer.

    Args:
        trainer_mode (str): Trainer mode.
        trainer_config (DictConfig): Trainer configuration.
        max_off_policy_threshold (int): Maximum number of model versions that trajectory can span.
        max_off_policy_strategy (str): How to handle trajectory that exceeds the maximum number of model versions.
        sampler_kwargs (dict): Additional kwargs for the custom sampler.
        poll_interval (float, optional): Poll interval in seconds. Defaults to 2.0.
        refill_fn (callable, optional): Function to submits fresh prompts for generation to replace stale groups dropped
            by the ``drop`` strategy (trainer-injected)
        group_filter_config (DictConfig | dict | None): Reasoning360-compatible constant-reward group filtering.
        rollout_n (int): Expected number of sampled trajectories for each prompt group.
        refill_batch_size (int): Number of fresh prompts submitted for each full generation refill.
    """

    def __init__(
        self,
        trainer_mode: str,
        trainer_config: DictConfig,
        max_off_policy_threshold: int,
        max_off_policy_strategy: str,
        sampler_kwargs: DictConfig,
        poll_interval: float = 2.0,
        refill_fn=None,
        group_filter_config: DictConfig | dict | None = None,
        rollout_n: int = 1,
        refill_batch_size: int = 1,
    ):
        self.trainer_mode = trainer_mode
        self.trainer_config = trainer_config
        self.max_off_policy_threshold = max_off_policy_threshold
        self.max_off_policy_strategy = max_off_policy_strategy
        self.sampler_kwargs = sampler_kwargs
        self.poll_interval = poll_interval
        self.refill_fn = refill_fn
        self.group_filter_config = group_filter_config or {}
        self.filter_groups = bool(self.group_filter_config.get("enable", False))
        self.filter_groups_metric = str(self.group_filter_config.get("metric") or "")
        self.max_num_gen_batches = int(self.group_filter_config.get("max_num_gen_batches", 0))
        self.rollout_n = int(rollout_n)
        self.refill_batch_size = int(refill_batch_size)

        assert isinstance(self.max_off_policy_threshold, int) and self.max_off_policy_threshold > 0, (
            f"Invalid max off policy threshold: {self.max_off_policy_threshold}, must be an integer greater than 0"
        )
        assert self.max_off_policy_strategy in ["drop", "wait"], (
            f"Invalid max off policy strategy: {self.max_off_policy_strategy}, must be one of ['drop', 'wait']"
        )
        if self.filter_groups:
            if self.trainer_mode != "sync":
                raise ValueError("filter_groups is currently supported only by the V1 sync trainer")
            if not self.filter_groups_metric:
                raise ValueError("filter_groups.metric must be set when filter_groups is enabled")
            if self.rollout_n < 1:
                raise ValueError("filter_groups requires rollout_n >= 1")
            if self.refill_batch_size < 1:
                raise ValueError("filter_groups requires refill_batch_size >= 1")
            if self.refill_fn is None:
                raise ValueError("filter_groups requires a refill_fn")

        # partition_id => {key: tag}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)
        self.pending_keys: dict[str, set] = defaultdict(set)
        self.running_keys: dict[str, set] = defaultdict(set)
        self.finished_keys: dict[str, set] = defaultdict(set)
        self.failure_keys: dict[str, set] = defaultdict(set)
        self.prompt_order: dict[str, list[str]] = defaultdict(list)
        # partition_id => {prompt_key: global_steps}, used to prioritize older samples.
        self.prompt_global_steps: dict[str, dict[str, int]] = defaultdict(dict)

    def _sync_metadata_from_transfer_queue(self):
        """Sync the metadata from TransferQueue."""
        self.partitions.clear()
        self.pending_keys.clear()
        self.running_keys.clear()
        self.finished_keys.clear()
        self.failure_keys.clear()
        self.prompt_order.clear()
        self.prompt_global_steps.clear()

        data = tq.kv_list()
        if data is None:
            return

        for partition_id, items in data.items():
            partition = self.partitions[partition_id]
            for key, tag in items.items():
                if tag.get("is_prompt", False):
                    # see: [GRPO group sampling control]
                    self.prompt_order[partition_id].append(key)
                    self.prompt_global_steps[partition_id][key] = tag["global_steps"]
                    match tag["status"]:
                        case "pending":
                            self.pending_keys[partition_id].add(key)
                        case "running":
                            self.running_keys[partition_id].add(key)
                        case "finished":
                            self.finished_keys[partition_id].add(key)
                        case "failure":
                            self.failure_keys[partition_id].add(key)
                        case _:
                            raise ValueError(f"Unknown status: {tag['status']}")
                else:
                    # see: [Trajectories storage format]
                    if key not in partition:
                        partition[key] = {}
                    partition[key].update(tag)

    def _sampleable_terminal_keys(self, global_steps: int, partition_id: str) -> set[str]:
        """Terminal prompts eligible for selection in the current metadata snapshot."""
        terminal_keys = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        if self.max_off_policy_strategy != "drop":
            return terminal_keys

        prompt_global_steps = self.prompt_global_steps[partition_id]
        return {
            uid
            for uid in terminal_keys
            if global_steps - prompt_global_steps.get(uid, global_steps) + 1 <= self.max_off_policy_threshold
        }

    def _has_enough_samples(self, global_steps: int, partition_id: str, batch_size: int) -> bool:
        # For wait strategy, we need to wait all trajectories that reach threshold to finish
        if self.max_off_policy_strategy == "wait":
            for key in self.pending_keys[partition_id] | self.running_keys[partition_id]:
                prompt_global_steps = self.prompt_global_steps[partition_id][key]
                if (global_steps - prompt_global_steps + 1) >= self.max_off_policy_threshold:
                    return False

        return len(self._sampleable_terminal_keys(global_steps, partition_id)) >= batch_size

    def _drop_stale_finished(self, global_steps: int, partition_id: str) -> tuple[int, dict]:
        """Drop terminal (finished/failure) prompts whose version span exceeds the threshold."""
        # TODO: drop strategy only takes effect after the whole group is finished, which may be too late.
        if self.max_off_policy_strategy != "drop":
            return 0, {}

        prompt_global_steps = self.prompt_global_steps[partition_id]
        terminal_uids = self.finished_keys[partition_id] | self.failure_keys[partition_id]
        stale_uids = terminal_uids - self._sampleable_terminal_keys(global_steps, partition_id)

        if not stale_uids:
            return 0, {}

        stale_spans = [global_steps - prompt_global_steps.get(uid, global_steps) + 1 for uid in stale_uids]

        # Clear prompt keys and their trajectory keys "{uid}_..." (kv_clear does not cascade).
        traj_keys = [key for key in self.partitions[partition_id] if key.split("_")[0] in stale_uids]
        tq.kv_clear(partition_id=partition_id, keys=list(stale_uids) + traj_keys)

        staleness = np.array(stale_spans, dtype=float)
        # TODO: use logger here
        print(
            f"[drop] partition={partition_id} global_steps={global_steps} "
            f"threshold={self.max_off_policy_threshold} num_dropped={len(stale_uids)} "
            f"span(min/mean/max)={staleness.min():.0f}/{staleness.mean():.2f}/{staleness.max():.0f}",
            flush=True,
        )

        prefix = "training" if partition_id == "train" else "validation"
        metrics = {
            f"{prefix}/off_policy/dropped_samples": len(stale_uids),
            f"{prefix}/off_policy/dropped_samples_staleness/mean": staleness.mean(),
            f"{prefix}/off_policy/dropped_samples_staleness/max": staleness.max(),
            f"{prefix}/off_policy/dropped_samples_staleness/min": staleness.min(),
        }
        return len(stale_uids), metrics

    @staticmethod
    def _trajectory_identity(key: str) -> tuple[str, int, int]:
        try:
            uid, session_id, index = key.rsplit("_", 2)
            return uid, int(session_id), int(index)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid trajectory key for group filtering: {key!r}") from exc

    def _clear_prompt_groups(self, partition_id: str, prompt_uids: set[str], partition: dict[str, dict]) -> None:
        if not prompt_uids:
            return
        trajectory_keys = [
            key for key in partition if self._trajectory_identity(key)[0] in prompt_uids
        ]
        tq.kv_clear(partition_id=partition_id, keys=[*prompt_uids, *trajectory_keys])

    def _classify_prompt_groups(
        self,
        partition_id: str,
        prompt_uids: set[str],
        partition: dict[str, dict],
    ) -> tuple[set[str], set[str], dict[str, int]]:
        final_keys_by_uid: dict[str, list[str]] = {}
        incomplete_uids: set[str] = set()
        keys_by_uid: dict[str, dict[int, tuple[int, str]]] = defaultdict(dict)
        for key in partition:
            uid, session_id, index = self._trajectory_identity(key)
            if uid not in prompt_uids:
                continue
            current = keys_by_uid[uid].get(session_id)
            if current is None or index > current[0]:
                keys_by_uid[uid][session_id] = (index, key)

        expected_sessions = set(range(self.rollout_n))
        for uid in prompt_uids:
            sessions = keys_by_uid.get(uid, {})
            if set(sessions) != expected_sessions:
                incomplete_uids.add(uid)
                continue
            final_keys_by_uid[uid] = [sessions[session_id][1] for session_id in range(self.rollout_n)]

        ordered_uids = sorted(final_keys_by_uid)
        final_keys = [key for uid in ordered_uids for key in final_keys_by_uid[uid]]
        metric_values: dict[str, list[float]] = defaultdict(list)
        if final_keys:
            try:
                data = tq.kv_batch_get(
                    keys=final_keys,
                    partition_id=partition_id,
                    select_fields=["extra_fields"],
                )
                extra_fields_value = data["extra_fields"]
                extra_fields = (
                    extra_fields_value.tolist()
                    if hasattr(extra_fields_value, "tolist")
                    else list(extra_fields_value)
                )
            except (KeyError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"filter_groups metric '{self.filter_groups_metric}' is missing from rollout trajectories"
                ) from exc

            for uid, extra_field in zip(
                (uid for uid in ordered_uids for _ in range(self.rollout_n)),
                extra_fields,
                strict=True,
            ):
                reward_extra_info = extra_field.get("reward_extra_info", {}) if isinstance(extra_field, dict) else {}
                value = reward_extra_info.get(self.filter_groups_metric)
                if value is None:
                    raise ValueError(
                        f"filter_groups metric '{self.filter_groups_metric}' is missing from rollout trajectories"
                    )
                try:
                    number = float(value.item() if hasattr(value, "item") else value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"filter_groups metric '{self.filter_groups_metric}' must be numeric, got {value!r}"
                    ) from exc
                if not np.isfinite(number):
                    raise ValueError(
                        f"filter_groups metric '{self.filter_groups_metric}' must be finite, got {number!r}"
                    )
                metric_values[uid].append(number)

        variable_uids: set[str] = set()
        constant_uids: set[str] = set(incomplete_uids)
        counts = {
            "incomplete_groups": len(incomplete_uids),
            "all_zero_groups": 0,
            "all_one_groups": 0,
            "other_constant_groups": 0,
        }
        for uid, values in metric_values.items():
            if np.std(values) > 0 or len(values) == 1:
                variable_uids.add(uid)
                continue
            constant_uids.add(uid)
            if all(value == 0 for value in values):
                counts["all_zero_groups"] += 1
            elif all(value == 1 for value in values):
                counts["all_one_groups"] += 1
            else:
                counts["other_constant_groups"] += 1
        return variable_uids, constant_uids, counts

    def _sample_with_group_filter(
        self,
        global_steps: int,
        partition_id: str,
        batch_size: int,
        last_debug_time: float,
    ) -> tuple[list[str], dict[str, dict], dict[str, int], dict]:
        prefix = "training" if partition_id == "train" else "validation"
        metric_prefix = f"{prefix}/filter_groups"
        generation_batches = 1
        examined_uids: set[str] = set()
        variable_uids: set[str] = set()
        variable_order: dict[str, int] = {}
        next_variable_order = 0
        filter_metrics: dict[str, float | int] = {
            f"{metric_prefix}/examined_prompt_groups": 0,
            f"{metric_prefix}/variable_prompt_groups": 0,
            f"{metric_prefix}/dropped_prompt_groups": 0,
            f"{metric_prefix}/incomplete_groups": 0,
            f"{metric_prefix}/all_zero_groups": 0,
            f"{metric_prefix}/all_one_groups": 0,
            f"{metric_prefix}/other_constant_groups": 0,
            f"{metric_prefix}/surplus_prompt_groups": 0,
        }
        drop_metrics: dict = {}

        while True:
            self._sync_metadata_from_transfer_queue()
            partition = dict(self.partitions[partition_id])

            dropped, metrics = self._drop_stale_finished(global_steps, partition_id)
            if dropped > 0:
                _accumulate_drop_metrics(drop_metrics, metrics, dropped)
                if self.max_num_gen_batches > 0 and generation_batches >= self.max_num_gen_batches:
                    raise RuntimeError(
                        f"filter_groups exhausted {self.max_num_gen_batches} generation batches "
                        "after stale-group refill"
                    )
                self.refill_fn(self.refill_batch_size)
                generation_batches += 1
                continue

            sampleable = self._sampleable_terminal_keys(global_steps, partition_id)
            new_uids = [
                uid for uid in self.prompt_order[partition_id] if uid in sampleable and uid not in examined_uids
            ]
            if new_uids:
                variable, constant, counts = self._classify_prompt_groups(partition_id, set(new_uids), partition)
                examined_uids.update(new_uids)
                variable_uids.update(variable)
                for uid in new_uids:
                    if uid in variable:
                        variable_order[uid] = next_variable_order
                        next_variable_order += 1
                self._clear_prompt_groups(partition_id, constant, partition)
                filter_metrics[f"{metric_prefix}/examined_prompt_groups"] += len(new_uids)
                filter_metrics[f"{metric_prefix}/variable_prompt_groups"] += len(variable)
                filter_metrics[f"{metric_prefix}/dropped_prompt_groups"] += len(constant)
                for name, value in counts.items():
                    filter_metrics[f"{metric_prefix}/{name}"] += value
                if constant:
                    continue

            inflight = self.pending_keys[partition_id] | self.running_keys[partition_id]
            available_uids = variable_uids & self._sampleable_terminal_keys(global_steps, partition_id)
            if not inflight and len(available_uids) >= batch_size:
                prompt_global_steps = dict(self.prompt_global_steps[partition_id])
                selected = sorted(
                    available_uids,
                    key=lambda uid: (prompt_global_steps.get(uid, 0), variable_order[uid]),
                )[:batch_size]
                surplus = available_uids - set(selected)
                self._clear_prompt_groups(partition_id, surplus, partition)
                filter_metrics[f"{metric_prefix}/surplus_prompt_groups"] = len(surplus)
                examined = int(filter_metrics[f"{metric_prefix}/examined_prompt_groups"])
                variable_count = int(filter_metrics[f"{metric_prefix}/variable_prompt_groups"])
                filter_metrics[f"{metric_prefix}/generated_batches"] = generation_batches
                filter_metrics[f"{metric_prefix}/refill_prompt_groups"] = (
                    generation_batches - 1
                ) * self.refill_batch_size
                filter_metrics[f"{metric_prefix}/keep_ratio"] = variable_count / examined if examined else 0.0
                return selected, partition, prompt_global_steps, {**drop_metrics, **filter_metrics}

            if not inflight:
                if self.max_num_gen_batches > 0 and generation_batches >= self.max_num_gen_batches:
                    raise RuntimeError(
                        f"filter_groups exhausted {self.max_num_gen_batches} generation batches: "
                        f"kept {len(available_uids)} of {batch_size} required prompt groups"
                    )
                self.refill_fn(self.refill_batch_size)
                generation_batches += 1
                continue

            time.sleep(self.poll_interval)
            if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                logger.info(
                    "filter_groups waiting: pending=%d running=%d variable=%d required=%d",
                    len(self.pending_keys[partition_id]),
                    len(self.running_keys[partition_id]),
                    len(available_uids),
                    batch_size,
                )
                last_debug_time = time.time()

    @SkipManager.annotate_tq(role="rollout_tq", phase="sample")
    def sample(self, global_steps: int, partition_id: str, batch_size: int) -> KVBatchMeta:
        """Sample a batch of data from the replay buffer.

        NOTE: user can customize sampling strategy by setting:
        ```bash
        trainer.v1.sampler.custom_sampler.path = "path/to/your/sampler.py"
        trainer.v1.sampler.custom_sampler.name = "UserCustomReplayBuffer"
        ```

        Args:
            global_steps (int): Global steps of the current training.
            partition_id (str): Partition of TransferQueue, e.g. "train" or "val".
            batch_size (int, optional): Batch size.

        Returns:
            KVBatchMeta: A batch of data.
            dict: Auxiliary metrics.
        """
        # TODO: failure samples dropping and refilling
        last_debug_time = time.time()
        drop_metrics: dict = {}
        selected_prompt_uids: list[str] = []
        partition_snapshot: dict[str, dict] = {}
        prompt_global_steps_snapshot: dict[str, int] = {}

        if self.filter_groups:
            (
                selected_prompt_uids,
                partition_snapshot,
                prompt_global_steps_snapshot,
                drop_metrics,
            ) = self._sample_with_group_filter(global_steps, partition_id, batch_size, last_debug_time)

        while not selected_prompt_uids:
            # Drop, gating, and selection below must all use this snapshot.
            self._sync_metadata_from_transfer_queue()

            dropped, metrics = self._drop_stale_finished(global_steps, partition_id)
            if dropped > 0:
                _accumulate_drop_metrics(drop_metrics, metrics, dropped)
                if self.refill_fn is not None:
                    self.refill_fn(dropped)
                continue

            if self._has_enough_samples(global_steps, partition_id, batch_size):
                prompt_global_steps_snapshot = dict(self.prompt_global_steps[partition_id])
                partition_snapshot = dict(self.partitions[partition_id])
                sampleable_keys = sorted(
                    self._sampleable_terminal_keys(global_steps, partition_id),
                    key=lambda key: prompt_global_steps_snapshot.get(key, 0),
                )
                selected_prompt_uids = sampleable_keys[:batch_size]
                break

            time.sleep(self.poll_interval)
            if time.time() - last_debug_time > VERL_REPLAY_BUFFER_DEBUG_INTERVAL_SECONDS:
                logger.info(
                    f"pending: {len(self.pending_keys[partition_id])}, "
                    f"running: {len(self.running_keys[partition_id])}, "
                    f"finished: {len(self.finished_keys[partition_id])}, "
                    f"failure: {len(self.failure_keys[partition_id])}"
                )
                last_debug_time = time.time()

        if self.max_off_policy_strategy == "drop":
            selected_spans = [
                global_steps - prompt_global_steps_snapshot.get(uid, global_steps) + 1 for uid in selected_prompt_uids
            ]
            assert all(span <= self.max_off_policy_threshold for span in selected_spans), (
                f"drop strategy selected stale prompts: spans={selected_spans}, "
                f"threshold={self.max_off_policy_threshold}"
            )

        tq.kv_clear(partition_id=partition_id, keys=selected_prompt_uids)

        keys, tags = [], []
        selected = set(selected_prompt_uids)
        for key, tag in partition_snapshot.items():
            uid = key.split("_")[0]
            if uid in selected:
                keys.append(key)
                tags.append(tag)

        batch = KVBatchMeta(partition_id=partition_id, keys=keys, tags=tags)
        return batch, drop_metrics
