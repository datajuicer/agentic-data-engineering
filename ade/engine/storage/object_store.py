"""Engine-owned object storage."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any


class FileEngineObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def exists(self, uri: str) -> bool:
        return self._resolve(uri).is_file()

    def read_json(self, uri: str) -> dict[str, Any]:
        payload = json.loads(self.read_bytes(uri))
        if not isinstance(payload, dict):
            raise ValueError(f"Engine object must be a JSON object: {uri}")
        return payload

    def read_bytes(self, uri: str) -> bytes:
        return self.path_for(uri).read_bytes()

    def path_for(self, uri: str) -> Path:
        target = self._resolve(uri)
        if not target.is_file():
            raise FileNotFoundError(f"Engine object does not exist: {uri}")
        return target

    def write_json(self, uri: str, payload: dict[str, Any]) -> str:
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        return self.write_bytes(uri, encoded)

    def write_bytes(self, uri: str, content: bytes) -> str:
        target = self._resolve(uri)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return uri

    def put_json(self, uri: str, payload: dict[str, Any]) -> str:
        target = self._resolve(uri)
        if target.exists():
            current = self.read_json(uri)
            if current != payload:
                raise ValueError(f"Engine object is immutable: {uri}")
            return uri
        return self.write_json(uri, payload)

    def put_bytes(self, uri: str, content: bytes) -> str:
        target = self._resolve(uri)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"Engine object is immutable: {uri}")
            return uri
        return self.write_bytes(uri, content)

    def _resolve(self, uri: str) -> Path:
        prefix = "engine://"
        if not uri.startswith(prefix):
            raise ValueError(f"unsupported Engine object URI: {uri}")
        relative = PurePosixPath(uri.removeprefix(prefix))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"unsafe Engine object URI: {uri}")
        target = self.root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"Engine object URI escapes root: {uri}")
        return target
