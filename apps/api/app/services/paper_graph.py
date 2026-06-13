import hashlib
import json
import re

from sqlmodel import Session

from ..models import CitationEdge, PaperRecord, PaperVersion


def _version_label(paper: PaperRecord) -> str:
    if paper.arxiv_id:
        match = re.search(r"v(\d+)$", paper.arxiv_id)
        if match:
            return f"arxiv-v{match.group(1)}"
        return "arxiv-version-unspecified"
    return f"snapshot-{paper.publication_date or 'undated'}"


def record_paper_graph(session: Session, papers: list[PaperRecord]) -> None:
    for paper in papers:
        serialized = json.dumps(
            paper.raw_metadata,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        session.add(
            PaperVersion(
                project_id=paper.project_id,
                paper_id=paper.id,
                version_label=_version_label(paper),
                metadata_hash=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            )
        )
        references = [
            *paper.raw_metadata.get("referenced_works", []),
            *paper.raw_metadata.get("references", []),
        ]
        for cited_id in dict.fromkeys(str(item) for item in references if item):
            session.add(
                CitationEdge(
                    project_id=paper.project_id,
                    citing_paper_id=paper.id,
                    cited_external_id=cited_id[:500],
                )
            )
    session.commit()
