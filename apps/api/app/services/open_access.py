import asyncio
import hashlib
import io
import ipaddress
import re
from urllib.parse import urlparse

import httpx
from pypdf import PdfReader

from ..config import get_settings
from ..models import EvidenceRecord, PaperRecord
from ..storage import content_store

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAPERS_PER_REFRESH = 4
MAX_PAGES = 30
EVIDENCE_MARKERS = re.compile(
    r"(future work|remains? (?:an )?open|little is known|however|limitation|"
    r"underexplored|not yet|challenge|we leave|fails? to|failure mode)",
    re.IGNORECASE,
)


def safe_public_https_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    hostname = parsed.hostname.casefold()
    if hostname == "localhost" or hostname.endswith((".localhost", ".local", ".internal")):
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
    )


async def _download_pdf(client: httpx.AsyncClient, url: str) -> bytes:
    if not safe_public_https_url(url):
        raise ValueError("unsafe open-access URL")
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        for prior in [*response.history, response]:
            if not safe_public_https_url(str(prior.url)):
                raise ValueError("open-access redirect left the public HTTPS boundary")
        declared = int(response.headers.get("content-length") or 0)
        if declared > MAX_PDF_BYTES:
            raise ValueError("open-access PDF exceeds size limit")
        chunks = []
        size = 0
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_PDF_BYTES:
                raise ValueError("open-access PDF exceeds size limit")
            chunks.append(chunk)
    content = b"".join(chunks)
    if not content.startswith(b"%PDF"):
        raise ValueError("open-access URL did not return a PDF")
    return content


def _extract_evidence(project_id, paper: PaperRecord, content: bytes) -> tuple[list[EvidenceRecord], int]:
    reader = PdfReader(io.BytesIO(content))
    records = []
    for page_index, page in enumerate(reader.pages[:MAX_PAGES], start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        selected = next((sentence for sentence in sentences if EVIDENCE_MARKERS.search(sentence)), "")
        if not selected:
            continue
        excerpt = selected[:1200]
        records.append(
            EvidenceRecord(
                project_id=project_id,
                paper_id=paper.id,
                evidence_type="open_fulltext_excerpt",
                claim=f"{paper.title} 的开放全文包含与局限或未来工作相关的证据。",
                excerpt=excerpt,
                locator=f"{paper.open_access_url}#page={page_index}",
                content_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            )
        )
        if len(records) >= 3:
            break
    return records, len(reader.pages)


async def ingest_open_access_papers(
    project_id,
    papers: list[PaperRecord],
) -> tuple[list[EvidenceRecord], list[str]]:
    if get_settings().app_env == "test":
        return [], []
    candidates = [
        paper
        for paper in papers
        if paper.open_access_url and safe_public_https_url(paper.open_access_url)
    ][:MAX_PAPERS_PER_REFRESH]
    semaphore = asyncio.Semaphore(2)
    evidence: list[EvidenceRecord] = []
    errors: list[str] = []

    async with httpx.AsyncClient(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": "ResearchFlow/0.1"},
    ) as client:
        async def ingest(paper: PaperRecord):
            async with semaphore:
                content = await _download_pdf(client, paper.open_access_url or "")
                digest, path = content_store.put_bytes(content, ".pdf")
                records, page_count = await asyncio.to_thread(
                    _extract_evidence,
                    project_id,
                    paper,
                    content,
                )
                paper.raw_metadata = {
                    **paper.raw_metadata,
                    "open_fulltext": {
                        "content_hash": digest,
                        "artifact_path": str(path),
                        "page_count": page_count,
                        "source_url": paper.open_access_url,
                    },
                }
                return paper, records

        responses = await asyncio.gather(
            *(ingest(paper) for paper in candidates),
            return_exceptions=True,
        )
    for paper, response in zip(candidates, responses, strict=True):
        if isinstance(response, Exception):
            errors.append(f"{paper.title[:80]}: {type(response).__name__}")
        else:
            _, records = response
            evidence.extend(records)
    return evidence, errors
