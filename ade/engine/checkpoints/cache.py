from __future__ import annotations

import hashlib
import errno
import json
import os
import shutil
import time
import traceback
import re
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        entries = path.rglob("*")
        for p in entries:
            try:
                if p.is_file():
                    total += p.stat().st_size
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return 0
    return total


def _copy_checkpoint(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f"{dst.name}.tmp.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=True)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    try:
        os.replace(tmp, dst)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _lock_owner_alive(lock_path: Path) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        payload = {}
    pid = payload.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_incomplete_staging(cache_dir: Path, key: str) -> None:
    for candidate in cache_dir.glob(f"{key}.tmp.*"):
        shutil.rmtree(candidate, ignore_errors=True)


def _has_weights(path: Path) -> bool:
    return (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin").exists()
        or any(path.glob("model-*.safetensors"))
        or any(path.glob("pytorch_model-*.bin"))
    )


def _verify_checkpoint(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"checkpoint is not a directory: {path}")
    is_lora = (path / "adapter_config.json").exists()
    if is_lora:
        if not ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()):
            raise ValueError(f"LoRA checkpoint missing adapter weights: {path}")
        return
    if not (path / "config.json").exists():
        raise ValueError(f"checkpoint missing config.json: {path}")
    if not _has_weights(path):
        raise ValueError(f"checkpoint missing model weights: {path}")


def _has_active_users(entry: Path) -> bool:
    users = entry / ".users"
    return users.is_dir() and any(users.iterdir())


def _cleanup_cache(
    cache_dir: Path,
    target_bytes: int,
    keep_key: str,
    *,
    min_entry_age_seconds: float = 0.0,
) -> None:
    current = _dir_size_bytes(cache_dir)
    if current <= target_bytes:
        return
    entries = []
    try:
        children = list(cache_dir.iterdir())
    except FileNotFoundError:
        return
    for child in children:
        if not child.is_dir() or child.name == keep_key or child.name.endswith(".tmp"):
            continue
        if (cache_dir / f"{child.name}.lock").exists():
            continue
        if _has_active_users(child):
            continue
        complete = child / ".complete"
        if not complete.exists():
            continue
        metadata = child / ".metadata.json"
        try:
            mtime = metadata.stat().st_mtime if metadata.exists() else child.stat().st_mtime
        except FileNotFoundError:
            continue
        if time.time() - mtime < min_entry_age_seconds:
            continue
        entries.append((mtime, child))
    for _, child in sorted(entries):
        shutil.rmtree(child, ignore_errors=True)
        current = _dir_size_bytes(cache_dir)
        if current <= target_bytes:
            break


def _prepare_copy_capacity(
    *,
    src: Path,
    cache_dir: Path,
    max_bytes: int,
    keep_key: str,
) -> int:
    source_bytes = _dir_size_bytes(src)
    if source_bytes > max_bytes:
        raise OSError(
            errno.ENOSPC,
            f"checkpoint size {source_bytes} exceeds cache limit {max_bytes}: {src}",
        )
    _cleanup_cache(cache_dir, max(0, max_bytes - source_bytes), keep_key)
    current_bytes = _dir_size_bytes(cache_dir)
    if current_bytes + source_bytes > max_bytes:
        raise OSError(
            errno.ENOSPC,
            "active cache entries leave insufficient configured capacity before "
            f"checkpoint copy: current={current_bytes}, checkpoint={source_bytes}, "
            f"limit={max_bytes}, cache_dir={cache_dir}",
        )
    usage = shutil.disk_usage(cache_dir)
    shm_root = Path("/dev/shm").resolve()
    try:
        in_shared_memory = cache_dir == shm_root or cache_dir.is_relative_to(shm_root)
    except ValueError:
        in_shared_memory = False
    reserve_bytes = int(usage.total * 0.05) if in_shared_memory else 0
    required_free = source_bytes + reserve_bytes
    if usage.free < required_free:
        raise OSError(
            errno.ENOSPC,
            "insufficient cache filesystem capacity before checkpoint copy: "
            f"free={usage.free}, checkpoint={source_bytes}, reserve={reserve_bytes}, "
            f"cache_dir={cache_dir}",
        )
    return source_bytes


def _safe_identity(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())


def _write_user_marker(entry: Path, *, run_id: str, consumer_id: str) -> None:
    users = entry / ".users"
    users.mkdir(exist_ok=True)
    _write_json_atomic(
        users / _safe_identity(consumer_id),
        {"run_id": run_id, "consumer_id": consumer_id, "created_at": time.time()},
    )


def stage_checkpoint_if_enabled(request: dict[str, Any]) -> dict[str, Any]:
    if not _truthy(request.get("checkpoint_staging")):
        return request
    src = Path(request["checkpoint_path"]).resolve()
    if not src.exists():
        raise FileNotFoundError(f"checkpoint path not found: {src}")
    if not src.is_dir():
        return request
    cache_dir_value = request.get("checkpoint_cache_dir")
    if not cache_dir_value or not str(cache_dir_value).strip():
        raise ValueError(
            "checkpoint_cache_dir is required when checkpoint staging is enabled"
        )
    cache_dir = Path(str(cache_dir_value)).expanduser()
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_gb = request.get("checkpoint_cache_max_gb")
    if max_gb is None or float(max_gb) <= 0:
        raise ValueError(
            "checkpoint_cache_max_gb must be positive when checkpoint staging is enabled"
        )
    max_bytes = int(float(max_gb) * 1024**3)
    run_id = str(request.get("ade_run_id") or request.get("staging_run_id") or "").strip()
    consumer_id = str(request.get("staging_consumer_id") or "").strip()
    if not run_id or not consumer_id:
        raise ValueError(
            "ade_run_id and staging_consumer_id are required when checkpoint staging is enabled"
        )
    key = hashlib.sha1(str(src).encode("utf-8")).hexdigest()
    dst = cache_dir / key
    complete = dst / ".complete"
    lock_path = cache_dir / f"{key}.lock"
    error_path = cache_dir / f"{key}.error.json"
    lock_stale_seconds = int(float(request.get("checkpoint_cache_lock_stale_seconds") or 600))
    start = time.time()
    cache_hit = complete.exists()
    while True:
        if complete.exists():
            try:
                _write_user_marker(dst, run_id=run_id, consumer_id=consumer_id)
                _verify_checkpoint(dst)
                cache_hit = True
                break
            except Exception:
                marker = dst / ".users" / _safe_identity(consumer_id)
                marker.unlink(missing_ok=True)
                if _has_active_users(dst):
                    raise RuntimeError(
                        f"invalid checkpoint cache entry is still in active use: {dst}"
                    )
                shutil.rmtree(dst, ignore_errors=True)
                error_path.unlink(missing_ok=True)
                cache_hit = False
                continue
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age > lock_stale_seconds:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                _remove_incomplete_staging(cache_dir, key)
                continue
            if time.time() - start > 7200:
                raise TimeoutError(f"timed out waiting for checkpoint staging lock: {lock_path}")
            time.sleep(2)
            continue
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "source": str(src), "created_at": time.time()}).encode("utf-8"))
            if not complete.exists():
                cache_hit = False
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                _remove_incomplete_staging(cache_dir, key)
                source_bytes = _prepare_copy_capacity(
                    src=src,
                    cache_dir=cache_dir,
                    max_bytes=max_bytes,
                    keep_key=key,
                )
                _copy_checkpoint(src, dst)
                _verify_checkpoint(dst)
                _write_json_atomic(
                    dst / ".metadata.json",
                    {"source": str(src), "staged_at": time.time(), "last_used_at": time.time(), "size_bytes": source_bytes},
                )
                complete.write_text("", encoding="utf-8")
                error_path.unlink(missing_ok=True)
            _write_user_marker(dst, run_id=run_id, consumer_id=consumer_id)
            break
        except Exception as exc:
            _write_json_atomic(
                error_path,
                {
                    "source": str(src),
                    "destination": str(dst),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "pid": os.getpid(),
                    "at": time.time(),
                },
            )
            if dst.exists() and not complete.exists():
                shutil.rmtree(dst, ignore_errors=True)
            _remove_incomplete_staging(cache_dir, key)
            raise
        finally:
            os.close(fd)
            lock_path.unlink(missing_ok=True)
    metadata_path = dst / ".metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"source": str(src), "staged_at": time.time(), "size_bytes": _dir_size_bytes(dst)}
        metadata["last_used_at"] = time.time()
        _write_json_atomic(metadata_path, metadata)
    _cleanup_cache(cache_dir, max_bytes, key)
    staged = dict(request)
    staged["source_checkpoint_path"] = request["checkpoint_path"]
    staged["checkpoint_path"] = str(dst)
    staged["checkpoint_staged"] = True
    staged["checkpoint_staging_cache_hit"] = cache_hit
    staged["checkpoint_staging_elapsed_seconds"] = round(time.time() - start, 2)
    staged["checkpoint_cache_dir"] = str(cache_dir)
    staged["staging_consumer_id"] = consumer_id
    return staged


def _release_staging_markers(
    cache_dir: str | Path,
    *,
    marker_matches,
) -> dict[str, Any]:
    root = Path(cache_dir).expanduser().resolve()
    removed_entries: list[str] = []
    released_entries: list[str] = []
    if not root.is_dir():
        return {"cache_dir": str(root), "released": [], "removed": []}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not (entry / ".complete").is_file():
            continue
        users = entry / ".users"
        matched = False
        for marker in tuple(users.iterdir()) if users.is_dir() else ():
            if marker_matches(marker):
                marker.unlink(missing_ok=True)
                matched = True
        if not matched:
            continue
        released_entries.append(entry.name)
        remaining = list(users.iterdir()) if users.is_dir() else []
        if not remaining:
            shutil.rmtree(entry)
            removed_entries.append(entry.name)
    return {
        "cache_dir": str(root),
        "released": released_entries,
        "removed": removed_entries,
    }


def release_staging_for_consumer(
    cache_dir: str | Path, consumer_id: str
) -> dict[str, Any]:
    """Release one command's node-local cache references."""
    safe_consumer_id = _safe_identity(consumer_id)
    if not safe_consumer_id:
        return {"cache_dir": str(Path(cache_dir).resolve()), "released": [], "removed": []}
    return _release_staging_markers(
        cache_dir,
        marker_matches=lambda marker: marker.name == safe_consumer_id,
    )


def release_staging_for_run(cache_dir: str | Path, run_id: str) -> dict[str, Any]:
    """Release every command reference owned by one Run."""
    expected = str(run_id).strip()
    safe_run_id = _safe_identity(expected)

    def belongs_to_run(marker: Path) -> bool:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return marker.name == safe_run_id
        return isinstance(payload, dict) and str(payload.get("run_id") or "") == expected

    if not expected:
        return {"cache_dir": str(Path(cache_dir).resolve()), "released": [], "removed": []}
    return _release_staging_markers(cache_dir, marker_matches=belongs_to_run)
