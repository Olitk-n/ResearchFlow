import hashlib
import json
from collections import Counter
from typing import Any
from urllib.parse import quote

import httpx

from ..models import DataPreparation, DatasetAsset, RunStatus
from .artifacts import project_directory, write_artifact_index

AUTO_USE_LICENSES = {
    "apache-2.0",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "cdla-permissive-2.0",
    "mit",
    "odc-by",
}


def can_auto_use(dataset: DatasetAsset) -> bool:
    return (dataset.license or "").casefold() in AUTO_USE_LICENSES


def choose_dataset(
    datasets: list[DatasetAsset],
    minimum_relevance: int = 1,
) -> DatasetAsset | None:
    candidates = [
        item
        for item in datasets
        if can_auto_use(item)
        and int(item.metadata_json.get("relevance_score") or 0) >= minimum_relevance
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            int(item.metadata_json.get("relevance_score") or 0),
            int(item.metadata_json.get("downloads") or 0),
        ),
    )


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_normalize_value(item) for item in value[:50]]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in list(value.items())[:50]}
    return str(value)


def profile_rows(rows: list[dict[str, Any]]) -> tuple[dict, dict]:
    keys = sorted({key for row in rows for key in row})
    schema = {}
    missing = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        types = Counter(type(value).__name__ for value in values if value is not None)
        schema[key] = {
            "types": dict(types),
            "examples": [_normalize_value(value) for value in values if value is not None][:3],
        }
        missing[key] = sum(value is None for value in values)
    profile = {
        "rows": len(rows),
        "columns": len(keys),
        "missing_by_column": missing,
    }
    return schema, profile


async def fetch_huggingface_rows(
    dataset_id: str,
    max_rows: int = 5000,
) -> tuple[str, str, list[dict[str, Any]], int]:
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        splits_response = await client.get(
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": dataset_id},
        )
        splits_response.raise_for_status()
        splits = splits_response.json().get("splits") or []
        if not splits:
            raise RuntimeError("dataset has no viewer-compatible splits")
        split = next(
            (item for item in splits if item.get("split") == "train"),
            splits[0],
        )
        config_name = split["config"]
        split_name = split["split"]
        total_rows = int(split.get("num_rows") or 0)
        rows: list[dict[str, Any]] = []
        target_rows = min(total_rows or max_rows, max_rows)
        for offset in range(0, target_rows, 100):
            rows_response = await client.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": dataset_id,
                    "config": config_name,
                    "split": split_name,
                    "offset": offset,
                    "length": min(100, target_rows - offset),
                },
            )
            rows_response.raise_for_status()
            source_rows = rows_response.json().get("rows") or []
            rows.extend(_normalize_value(item.get("row") or {}) for item in source_rows)
            if not source_rows:
                break
    return config_name, split_name, rows, total_rows or len(rows)


async def prepare_dataset(
    project_id,
    dataset: DatasetAsset,
) -> DataPreparation:
    preparation = DataPreparation(
        project_id=project_id,
        dataset_id=dataset.id,
        status=RunStatus.RUNNING,
    )
    if dataset.source != "huggingface":
        preparation.status = RunStatus.BLOCKED
        preparation.profile_json = {"reason": "首版自动处理仅支持 Hugging Face Dataset Viewer"}
        return preparation
    if not can_auto_use(dataset):
        preparation.status = RunStatus.BLOCKED
        preparation.profile_json = {"reason": "数据许可不在自动使用白名单"}
        return preparation

    config_name, split_name, rows, total_rows = await fetch_huggingface_rows(dataset.external_id)
    if not rows:
        preparation.status = RunStatus.BLOCKED
        preparation.profile_json = {"reason": "数据集未返回可处理样本"}
        return preparation

    root = (
        project_directory(project_id)
        / "data"
        / quote(
            dataset.external_id,
            safe="",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    normalized = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    normalized_bytes = normalized.encode("utf-8")
    digest = hashlib.sha256(normalized_bytes).hexdigest()
    (root / "prepared.jsonl").write_bytes(normalized_bytes)
    schema, profile = profile_rows(rows)
    complete_snapshot = len(rows) >= total_rows
    profile.update({
        "source_total_rows": total_rows,
        "complete_snapshot": complete_snapshot,
        "sampling_strategy": "complete split" if complete_snapshot else "deterministic first 5000 rows",
    })
    data_card = {
        "dataset": dataset.external_id,
        "source_url": dataset.url,
        "license": dataset.license,
        "config": config_name,
        "split": split_name,
        "prepared_rows": len(rows),
        "source_total_rows": total_rows,
        "complete_snapshot": complete_snapshot,
        "content_hash": digest,
        "schema": schema,
        "profile": profile,
        "transformations": [
            "Fetched through Hugging Face Dataset Viewer",
            "Normalized nested values to JSON-compatible structures",
            "Sorted object keys and serialized as UTF-8 JSONL",
            (
                "Stored the complete selected split"
                if complete_snapshot
                else "Stored a deterministic capped snapshot; submission requires a justified sampling plan"
            ),
        ],
    }
    (root / "data-card.json").write_text(
        json.dumps(data_card, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_artifact_index(root)
    preparation.status = RunStatus.COMPLETED
    preparation.config_name = config_name
    preparation.split_name = split_name
    preparation.row_count = len(rows)
    preparation.schema_json = schema
    preparation.profile_json = profile
    preparation.transformations = data_card["transformations"]
    preparation.content_hash = digest
    preparation.artifact_path = str(root)
    return preparation
