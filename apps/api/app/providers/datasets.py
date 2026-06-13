import asyncio
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import get_settings

GENERIC_TERMS = {
    "evaluation",
    "benchmark",
    "robustness",
    "reliability",
    "dataset",
    "data",
    "study",
    "analysis",
}


@dataclass(slots=True)
class DatasetResult:
    source: str
    external_id: str
    name: str
    url: str
    license: str | None
    size_hint: str | None
    quality_notes: str
    metadata: dict[str, Any]


def query_terms(query: str) -> list[str]:
    return [
        word.casefold()
        for word in query.replace("-", " ").split()
        if len(word) >= 3 and word.casefold() not in GENERIC_TERMS
    ]


def dataset_relevance_score(item: DatasetResult, query: str) -> tuple[int, list[str]]:
    text = " ".join(
        [
            item.name,
            item.external_id,
            " ".join(str(tag) for tag in item.metadata.get("tags", [])),
        ]
    ).casefold()
    matched = []
    for term in dict.fromkeys(query_terms(query)):
        variants = {
            "llm": ("llm", "large-language-model", "large language model"),
            "agent": ("agent", "agentic"),
            "agents": ("agent", "agentic"),
        }.get(term, (term,))
        if any(variant in text for variant in variants):
            matched.append(term)
    return len(matched), matched


async def search_huggingface(client: httpx.AsyncClient, query: str, limit: int) -> list[DatasetResult]:
    headers = {}
    if get_settings().hf_token:
        headers["Authorization"] = f"Bearer {get_settings().hf_token}"
    response = await client.get(
        "https://huggingface.co/api/datasets",
        params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
        headers=headers,
    )
    response.raise_for_status()
    results = []
    for item in response.json():
        tags = item.get("tags") or []
        licenses = [tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")]
        dataset_id = item.get("id")
        results.append(
            DatasetResult(
                source="huggingface",
                external_id=dataset_id,
                name=dataset_id,
                url=f"https://huggingface.co/datasets/{dataset_id}",
                license=licenses[0] if licenses else None,
                size_hint=None,
                quality_notes=f"{item.get('downloads', 0):,} downloads; {item.get('likes', 0)} likes",
                metadata={"tags": tags, "downloads": item.get("downloads"), "likes": item.get("likes")},
            )
        )
    return results


async def search_openml(client: httpx.AsyncClient, query: str, limit: int) -> list[DatasetResult]:
    response = await client.get(
        "https://www.openml.org/api/v1/json/data/list",
        params={"limit": 200, "offset": 0, "status": "active"},
    )
    response.raise_for_status()
    datasets = response.json().get("data", {}).get("dataset", [])
    words = {word.casefold() for word in query.split() if len(word) > 2}
    ranked = [
        item
        for item in sorted(
            datasets,
            key=lambda item: sum(word in (item.get("name") or "").casefold() for word in words),
            reverse=True,
        )
        if sum(word in (item.get("name") or "").casefold() for word in words) >= min(2, len(words))
    ][:limit]
    return [
        DatasetResult(
            source="openml",
            external_id=str(item.get("did")),
            name=item.get("name") or f"OpenML {item.get('did')}",
            url=f"https://www.openml.org/d/{item.get('did')}",
            license=item.get("licence"),
            size_hint=f"{item.get('NumberOfInstances', '?')} rows",
            quality_notes=f"{item.get('NumberOfFeatures', '?')} features",
            metadata=item,
        )
        for item in ranked
    ]


async def aggregate_datasets(query: str, limit: int = 5) -> tuple[list[DatasetResult], list[str]]:
    errors: list[str] = []
    terms = query_terms(query)
    focus = " ".join(terms[:2]) or query
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        jobs = {
            "huggingface-focus": search_huggingface(client, focus, limit),
            "huggingface-evaluation": search_huggingface(
                client,
                f"{focus} evaluation",
                limit,
            ),
            "huggingface-benchmark": search_huggingface(
                client,
                f"{focus} benchmark",
                limit,
            ),
            "openml": search_openml(client, query, limit),
        }
        responses = await asyncio.gather(*jobs.values(), return_exceptions=True)
    results: list[DatasetResult] = []
    for source, response in zip(jobs, responses, strict=True):
        if isinstance(response, Exception):
            detail = str(response).splitlines()[0][:160] or type(response).__name__
            errors.append(f"{source}: {detail}")
        else:
            results.extend(response)
    deduplicated: dict[tuple[str, str], DatasetResult] = {}
    for item in results:
        score, matched_terms = dataset_relevance_score(item, query)
        item.metadata = {
            **item.metadata,
            "relevance_score": score,
            "matched_query_terms": matched_terms,
        }
        deduplicated[(item.source, item.external_id)] = item
    ranked_results = sorted(
        (
            item
            for item in deduplicated.values()
            if int(item.metadata.get("relevance_score") or 0) >= min(2, len(set(terms)))
        ),
        key=lambda item: (
            int(item.metadata.get("relevance_score") or 0),
            item.license is not None,
            int(item.metadata.get("downloads") or 0),
        ),
        reverse=True,
    )
    return ranked_results[: limit * 2], errors
