#!/usr/bin/env bash
set -euo pipefail

CACHE_ROOT="/dev/shm/ade-artifact-cache"
CUTOFF_ISO="2026-09-09T00:00:00+00:00"

show_status() {
    date -u '+Timestamp: %Y-%m-%d %H:%M:%S UTC'
    echo
    free -h
    echo
    df -h /dev/shm
    echo
    if [[ -d "$CACHE_ROOT" ]]; then
        du -sh -- "$CACHE_ROOT" || true
    fi
}

if [[ ! -d "$CACHE_ROOT" ]]; then
    echo "Cache directory does not exist: $CACHE_ROOT"
    exit 0
fi

echo "================ BEFORE CLEANUP ================"
show_status

echo
echo "================ DRY RUN ================"
echo "Cutoff: $CUTOFF_ISO"
echo "Ignoring all lock and user markers."
echo

CACHE_ROOT="$CACHE_ROOT" CUTOFF_ISO="$CUTOFF_ISO" python3 - <<'PY'
from pathlib import Path
import datetime
import json
import math
import os

root = Path(os.environ["CACHE_ROOT"])
cutoff = datetime.datetime.fromisoformat(
    os.environ["CUTOFF_ISO"]
).timestamp()

count = 0

for entry in sorted(root.iterdir()):
    if entry.is_symlink() or not entry.is_dir():
        continue

    if not (entry / ".complete").exists():
        continue

    try:
        last_used = float(
            json.loads((entry / ".metadata.json").read_text())["last_used_at"]
        )
        if not math.isfinite(last_used):
            raise ValueError("invalid last_used_at")
    except Exception:
        continue

    if last_used < cutoff:
        last_used_utc = datetime.datetime.fromtimestamp(
            last_used,
            tz=datetime.timezone.utc,
        ).isoformat()

        print(f"REMOVE: {entry}")
        print(f"  last_used_at: {last_used_utc}")
        count += 1

print()
print(f"Total removal candidates: {count}")
PY

echo
echo "Warning: deletion does not check locks or user markers."
read -r -p "Type DELETE to continue; any other input cancels: " confirmation

if [[ "$confirmation" != "DELETE" ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo
echo "================ CLEANUP ================"

CACHE_ROOT="$CACHE_ROOT" CUTOFF_ISO="$CUTOFF_ISO" python3 - <<'PY'
from pathlib import Path
import datetime
import json
import math
import os
import shutil

root = Path(os.environ["CACHE_ROOT"])
cutoff = datetime.datetime.fromisoformat(
    os.environ["CUTOFF_ISO"]
).timestamp()

removed = 0
failed = 0

for entry in sorted(root.iterdir()):
    if entry.is_symlink() or not entry.is_dir():
        continue

    if not (entry / ".complete").exists():
        continue

    try:
        last_used = float(
            json.loads((entry / ".metadata.json").read_text())["last_used_at"]
        )
        if not math.isfinite(last_used):
            raise ValueError("invalid last_used_at")
    except Exception:
        continue

    if last_used >= cutoff:
        continue

    # Intentionally ignore sibling locks and .users markers
    print(f"Removing: {entry}")

    try:
        shutil.rmtree(entry)
        removed += 1
    except Exception as exc:
        print(f"FAILED: {entry}: {exc}")
        failed += 1

print()
print(f"Removed entries: {removed}")
print(f"Failed entries: {failed}")
PY

echo
echo "================ AFTER CLEANUP ================"
show_status