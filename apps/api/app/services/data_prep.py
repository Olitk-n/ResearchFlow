import asyncio
import hashlib
import io
import json
from collections import Counter
from typing import Any
from urllib.parse import quote

import httpx
import pyarrow.parquet as pq

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
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_DOWNLOAD_ATTEMPTS = 3
MAX_PARQUET_DOWNLOAD_BYTES = 64 * 1024 * 1024


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    failure_history: list[dict[str, Any]],
) -> httpx.Response:
    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            response = await client.get(url, params=params)
            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response
            error = f"HTTP {response.status_code}"
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            error = type(exc).__name__
        failure_history.append({
            "url": url,
            "attempt": attempt,
            "error": error,
            "params": params or {},
        })
        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            await asyncio.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"download failed after {MAX_DOWNLOAD_ATTEMPTS} attempts: {url}")


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
) -> tuple[str, str, list[dict[str, Any]], int, dict[str, Any]]:
    failure_history: list[dict[str, Any]] = []
    timeout = httpx.Timeout(20, connect=15)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        splits_response = await _get_with_retry(
            client,
            "https://datasets-server.huggingface.co/splits",
            params={"dataset": dataset_id},
            failure_history=failure_history,
        )
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
        try:
            for offset in range(0, target_rows, 100):
                rows_response = await _get_with_retry(
                    client,
                    "https://datasets-server.huggingface.co/rows",
                    params={
                        "dataset": dataset_id,
                        "config": config_name,
                        "split": split_name,
                        "offset": offset,
                        "length": min(100, target_rows - offset),
                    },
                    failure_history=failure_history,
                )
                source_rows = rows_response.json().get("rows") or []
                rows.extend(_normalize_value(item.get("row") or {}) for item in source_rows)
                if not source_rows:
                    break
            method = "dataset_viewer_rows"
        except RuntimeError as viewer_error:
            parquet_response = await _get_with_retry(
                client,
                "https://datasets-server.huggingface.co/parquet",
                params={"dataset": dataset_id},
                failure_history=failure_history,
            )
            parquet_files = parquet_response.json().get("parquet_files") or []
            matching = [
                item for item in parquet_files
                if item.get("config") == config_name and item.get("split") == split_name
            ] or parquet_files
            if not matching:
                raise RuntimeError("dataset has no downloadable parquet files") from viewer_error
            rows = []
            for item in matching:
                head_response = await client.head(item["url"])
                content_length = int(head_response.headers.get("content-length") or 0)
                if content_length > MAX_PARQUET_DOWNLOAD_BYTES:
                    failure_history.append({
                        "url": item["url"],
                        "attempt": 1,
                        "error": f"parquet file exceeds local limit ({content_length} bytes)",
                        "params": {},
                    })
                    continue
                parquet_bytes = bytearray()
                async with client.stream("GET", item["url"]) as file_response:
                    file_response.raise_for_status()
                    async for chunk in file_response.aiter_bytes():
                        parquet_bytes.extend(chunk)
                        if len(parquet_bytes) > MAX_PARQUET_DOWNLOAD_BYTES:
                            raise RuntimeError(
                                "parquet file exceeds local 64 MB limit"
                            ) from viewer_error
                table = pq.read_table(io.BytesIO(parquet_bytes))
                remaining = max_rows - len(rows)
                rows.extend(
                    _normalize_value(row)
                    for row in table.slice(0, remaining).to_pylist()
                )
                if len(rows) >= max_rows:
                    break
            if not rows:
                raise RuntimeError("no parquet shard fits the local download limit") from viewer_error
            method = "parquet_fallback"
    return config_name, split_name, rows, total_rows or len(rows), {
        "download_method": method,
        "retry_count": len(failure_history),
        "failure_history": failure_history,
    }


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

    fetched = await fetch_huggingface_rows(dataset.external_id)
    if len(fetched) == 4:
        config_name, split_name, rows, total_rows = fetched
        download_report = {
            "download_method": "dataset_viewer_rows",
            "retry_count": 0,
            "failure_history": [],
        }
    else:
        config_name, split_name, rows, total_rows, download_report = fetched
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
        **download_report,
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
            f"Fetched through {download_report['download_method']}",
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
