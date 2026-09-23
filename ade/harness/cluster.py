"""Node-local Ray CLI operations bound to an ADE deployment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from ade.harness.yaml_config import load_yaml_mapping


def add_cluster_parser(commands) -> None:
    cluster = commands.add_parser("cluster", help="Manage Ray on the current node")
    actions = cluster.add_subparsers(dest="cluster_command", required=True)
    for name in ("head", "worker", "status", "stop"):
        action = actions.add_parser(name)
        action.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
        if name == "stop":
            action.add_argument("--local", action="store_true", required=True,
                                help="Stop Ray processes on this node, not the whole cluster")
        else:
            action.add_argument("--deployment", type=Path, required=True)
        if name == "worker":
            action.add_argument("--node-ip", required=True, help="Reachable IP of this worker")
        if name in {"head", "worker"}:
            action.add_argument("--gpus", required=True, type=int, help="GPUs allocated to Ray on this node (head may use 0)")
            action.add_argument("--cpus", type=int, help="Ray CPU capacity; defaults to Ray's detection")


def ray_command(args) -> list[str]:
    # Use Ray from the same environment as ADE, not an unrelated PATH entry.
    command = [str(Path(sys.executable).parent / "ray")]
    if args.cluster_command == "stop":
        return command + ["stop"]
    deployment = args.deployment
    if not deployment.is_absolute():
        deployment = args.project_root / deployment
    config = load_yaml_mapping(deployment)
    resources = config.get("run_resources", {})
    if not isinstance(resources, dict):
        raise ValueError("deployment.run_resources must be a mapping")
    ray = resources.get("ray_cluster", {})
    if not isinstance(ray, dict):
        raise ValueError("deployment.run_resources.ray_cluster must be a mapping")
    address = ray.get("address")
    if not isinstance(address, str):
        raise ValueError("deployment.run_resources.ray_cluster.address is required")
    host, separator, port = address.rpartition(":")
    if not separator or not host or "/" in host or not port.isdecimal() or not 1 <= int(port) <= 65535:
        raise ValueError("Ray address must be an explicit head host:port, not auto or a Ray Client URL")
    if args.cluster_command == "status":
        return command + ["status", f"--address={address}"]
    if args.cpus is not None and args.cpus < 1:
        raise ValueError("--cpus must be positive")
    if args.cluster_command == "head":
        if args.gpus < 0:
            raise ValueError("Ray head GPU capacity must be a nonnegative integer")
        command += ["start", "--head", f"--node-ip-address={host}", f"--port={port}",
                    f"--num-gpus={args.gpus}", "--include-dashboard=False",
                    '--system-config={"kill_child_processes_on_worker_exit_with_raylet_subreaper":true}']
    else:
        if args.gpus < 1:
            raise ValueError("--gpus must be positive")
        if args.node_ip == host:
            raise ValueError("Run the worker command on a worker node, not the configured head")
        command += ["start", f"--address={address}", f"--node-ip-address={args.node_ip}",
                    f"--num-gpus={args.gpus}"]
    if args.cpus is not None:
        command.append(f"--num-cpus={args.cpus}")
    return command


def run_cluster(args) -> int:
    try:
        command = ray_command(args)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({"action": args.cluster_command, "command": command,
                      "dry_run": args.dry_run, "execution": "current_node"}), flush=True)
    if args.dry_run:
        return 0
    if not Path(command[0]).is_file():
        raise SystemExit("Ray is not installed in this ADE environment. Activate the full runtime installed by scripts/recreate_unified_vllm_env.sh.")
    return subprocess.run(command, check=False).returncode
