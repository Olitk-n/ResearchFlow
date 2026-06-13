import asyncio
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime
from html import unescape
from typing import Any

import feedparser
import httpx

from ..config import get_settings


@dataclass(slots=True)
class NormalizedPaper:
    source: str
    external_id: str
    title: str
    abstract: str = ""
    authors: list[str] | None = None
    publication_date: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    open_access_url: str | None = None
    citation_count: int = 0
    raw_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["authors"] = data["authors"] or []
        data["raw_metadata"] = data["raw_metadata"] or {}
        return data


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.casefold())


def _query_terms(query: str) -> list[str]:
    stopwords = {
        "with",
        "from",
        "that",
        "this",
        "using",
        "based",
        "evaluation",
        "benchmark",
        "robustness",
    }
    return [
        term
        for term in re.findall(r"[a-z0-9]+", query.casefold())
        if len(term) >= 3 and term not in stopwords
    ]


def _is_relevant(paper: NormalizedPaper, query: str) -> bool:
    terms = set(_query_terms(query))
    if len(terms) < 2:
        return True
    text = f"{paper.title} {paper.abstract}".casefold()
    matches = 0
    for term in terms:
        variants = {
            "llm": ("llm", "large language model"),
            "agent": ("agent", "agentic"),
            "agents": ("agent", "agentic"),
        }.get(term, (term,))
        matches += any(variant in text for variant in variants)
    return matches >= min(2, len(terms))


def _has_valid_date(paper: NormalizedPaper) -> bool:
    if not paper.publication_date:
        return True
    try:
        if len(paper.publication_date) >= 10:
            return date.fromisoformat(paper.publication_date[:10]) <= date.today()
    except ValueError:
        pass
    match = re.match(r"(\d{4})", paper.publication_date)
    return not match or int(match.group(1)) <= datetime.now().year


def _crossref_open_pdf(item: dict) -> str | None:
    licenses = [
        str(license_item.get("URL") or "").casefold()
        for license_item in item.get("license", [])
    ]
    openly_licensed = any(
        "creativecommons.org/licenses/" in url or "creativecommons.org/publicdomain/" in url
        for url in licenses
    )
    if not openly_licensed:
        return None
    for link in item.get("link", []):
        content_type = str(link.get("content-type") or "").casefold()
        url = link.get("URL")
        if url and "pdf" in content_type:
            return url
    return None


def deduplicate_papers(
    papers: list[NormalizedPaper],
    query: str | None = None,
) -> list[NormalizedPaper]:
    seen: set[str] = set()
    result: list[NormalizedPaper] = []
    for paper in papers:
        key = (
            f"doi:{paper.doi.lower()}"
            if paper.doi
            else f"arxiv:{paper.arxiv_id.lower()}"
            if paper.arxiv_id
            else f"title:{_title_key(paper.title)}"
        )
        if (
            key in seen
            or len(_title_key(paper.title)) < 8
            or not _has_valid_date(paper)
            or (query is not None and not _is_relevant(paper, query))
        ):
            continue
        seen.add(key)
        result.append(paper)
    result.sort(key=lambda item: item.publication_date or "", reverse=True)
    return result


async def search_openalex(client: httpx.AsyncClient, query: str, limit: int) -> list[NormalizedPaper]:
    response = await client.get(
        "https://api.openalex.org/works",
        params={
            "search": query,
            "sort": "publication_date:desc",
            "per-page": min(limit, 50),
            "select": (
                "id,doi,title,publication_date,authorships,primary_location,"
                "open_access,cited_by_count,abstract_inverted_index,referenced_works"
            ),
        },
    )
    response.raise_for_status()
    papers = []
    for item in response.json().get("results", []):
        inverted = item.get("abstract_inverted_index") or {}
        tokens = sorted(
            ((position, word) for word, positions in inverted.items() for position in positions),
            key=lambda pair: pair[0],
        )
        doi = (item.get("doi") or "").replace("https://doi.org/", "") or None
        location = item.get("primary_location") or {}
        papers.append(
            NormalizedPaper(
                source="openalex",
                external_id=(item.get("id") or "").rsplit("/", 1)[-1],
                title=_clean(item.get("title")),
                abstract=" ".join(word for _, word in tokens),
                authors=[
                    auth.get("author", {}).get("display_name", "")
                    for auth in item.get("authorships", [])
                    if auth.get("author", {}).get("display_name")
                ],
                publication_date=item.get("publication_date"),
                doi=doi,
                url=location.get("landing_page_url"),
                open_access_url=location.get("pdf_url"),
                citation_count=item.get("cited_by_count") or 0,
                raw_metadata={
                    "open_access": item.get("open_access") or {},
                    "referenced_works": item.get("referenced_works") or [],
                },
            )
        )
    return papers


async def search_crossref(client: httpx.AsyncClient, query: str, limit: int) -> list[NormalizedPaper]:
    response = await client.get(
        "https://api.crossref.org/works",
        params={
            "query": query,
            "sort": "published",
            "order": "desc",
            "rows": min(limit, 50),
            "filter": (f"from-pub-date:{date.today().year - 5}-01-01,until-pub-date:{date.today().isoformat()}"),
        },
        headers={"User-Agent": "ResearchFlow/0.1 (mailto:local@researchflow.invalid)"},
    )
    response.raise_for_status()
    papers = []
    for item in response.json().get("message", {}).get("items", []):
        title = " ".join(item.get("title") or [])
        published = item.get("published-online") or item.get("published-print") or {}
        parts = (published.get("date-parts") or [[]])[0]
        publication_date = (
            "-".join(str(value).zfill(2) if index else str(value) for index, value in enumerate(parts))
            if parts
            else None
        )
        papers.append(
            NormalizedPaper(
                source="crossref",
                external_id=item.get("DOI") or item.get("URL") or title,
                title=_clean(title),
                abstract=_clean(item.get("abstract")),
                authors=[
                    " ".join(filter(None, [author.get("given"), author.get("family")]))
                    for author in item.get("author", [])
                ],
                publication_date=publication_date,
                doi=item.get("DOI"),
                url=item.get("URL"),
                open_access_url=_crossref_open_pdf(item),
                citation_count=item.get("is-referenced-by-count") or 0,
                raw_metadata={
                    "publisher": item.get("publisher"),
                    "type": item.get("type"),
                    "licenses": item.get("license") or [],
                    "references": [
                        reference.get("DOI")
                        for reference in item.get("reference", [])
                        if reference.get("DOI")
                    ][:200],
                },
            )
        )
    return papers


async def search_arxiv(client: httpx.AsyncClient, query: str, limit: int) -> list[NormalizedPaper]:
    terms = _query_terms(query)[:6]
    search_query = " AND ".join(f"all:{term}" for term in terms) if terms else f'all:"{query}"'
    response = await client.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit, 50),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
    )
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.rsplit("/", 1)[-1]
        papers.append(
            NormalizedPaper(
                source="arxiv",
                external_id=arxiv_id,
                arxiv_id=arxiv_id,
                title=_clean(entry.get("title")),
                abstract=_clean(entry.get("summary")),
                authors=[author.get("name", "") for author in entry.get("authors", [])],
                publication_date=(entry.get("published") or "")[:10] or None,
                doi=entry.get("arxiv_doi"),
                url=entry.id,
                open_access_url=f"https://arxiv.org/pdf/{arxiv_id}",
                raw_metadata={"categories": [tag.get("term") for tag in entry.get("tags", [])]},
            )
        )
    return papers


async def search_semantic_scholar(
    client: httpx.AsyncClient,
    query: str,
    limit: int,
) -> list[NormalizedPaper]:
    headers = {}
    api_key = get_settings().semantic_scholar_api_key
    if api_key:
        headers["x-api-key"] = api_key
    response = await client.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": min(limit, 50),
            "fields": ("title,abstract,authors,year,publicationDate,externalIds,url,openAccessPdf,citationCount"),
        },
        headers=headers,
    )
    if response.status_code == 429:
        raise RuntimeError("rate_limited; configure SEMANTIC_SCHOLAR_API_KEY")
    response.raise_for_status()
    papers = []
    for item in response.json().get("data", []):
        ids = item.get("externalIds") or {}
        open_pdf = item.get("openAccessPdf") or {}
        papers.append(
            NormalizedPaper(
                source="semantic_scholar",
                external_id=item.get("paperId"),
                title=_clean(item.get("title")),
                abstract=_clean(item.get("abstract")),
                authors=[author.get("name", "") for author in item.get("authors", [])],
                publication_date=item.get("publicationDate") or str(item.get("year") or "") or None,
                doi=ids.get("DOI"),
                arxiv_id=ids.get("ArXiv"),
                url=item.get("url"),
                open_access_url=open_pdf.get("url"),
                citation_count=item.get("citationCount") or 0,
                raw_metadata={"external_ids": ids},
            )
        )
    return papers


async def aggregate_literature(query: str, per_source: int = 8) -> tuple[list[NormalizedPaper], list[str]]:
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        jobs = {
            "openalex": search_openalex(client, query, per_source),
            "arxiv": search_arxiv(client, query, per_source),
            "crossref": search_crossref(client, query, per_source),
            "semantic_scholar": search_semantic_scholar(client, query, per_source),
        }
        responses = await asyncio.gather(*jobs.values(), return_exceptions=True)
    papers: list[NormalizedPaper] = []
    for source, response in zip(jobs, responses, strict=True):
        if isinstance(response, Exception):
            detail = str(response).splitlines()[0][:160] or type(response).__name__
            errors.append(f"{source}: {detail}")
        else:
            papers.extend(response)
    return deduplicate_papers(papers, query=query), errors
