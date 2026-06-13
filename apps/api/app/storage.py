import hashlib
import json
from pathlib import Path
from typing import Any

from .config import get_settings


class ContentStore:
    def __init__(self, root: Path | None = None):
        self.root = root or get_settings().artifact_root
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, content: bytes, suffix: str = "") -> tuple[str, Path]:
        digest = hashlib.sha256(content).hexdigest()
        directory = self.root / digest[:2] / digest[2:4]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}{suffix}"
        if not path.exists():
            path.write_bytes(content)
        return digest, path

    def put_text(self, content: str, suffix: str = ".txt") -> tuple[str, Path]:
        return self.put_bytes(content.encode("utf-8"), suffix)

    def put_json(self, content: Any) -> tuple[str, Path]:
        data = json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2)
        return self.put_text(data, ".json")


content_store = ContentStore()
