import hashlib
import math
import re
from uuid import UUID

from sqlalchemy import text
from sqlmodel import Session, select

from ..models import PaperEmbedding, PaperRecord

DIMENSIONS = 384
MODEL_NAME = "local-hash-384-v1"
STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w-]{2,}", value.casefold())
        if token not in STOP_WORDS
    }


def embed_text(value: str) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for token in _tokens(value):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % DIMENSIONS
        sign = -1.0 if digest[4] & 1 else 1.0
        vector[index] += sign
    norm = math.sqrt(sum(item * item for item in vector))
    if norm:
        return [item / norm for item in vector]
    return vector


def index_papers(session: Session, papers: list[PaperRecord]) -> None:
    for paper in papers:
        session.add(
            PaperEmbedding(
                project_id=paper.project_id,
                paper_id=paper.id,
                embedding_model=MODEL_NAME,
                dimensions=DIMENSIONS,
                embedding=embed_text(f"{paper.title}\n{paper.abstract}"),
            )
        )
    session.commit()


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _lexical_score(query_tokens: set[str], paper: PaperRecord) -> float:
    if not query_tokens:
        return 0.0
    title_tokens = _tokens(paper.title)
    abstract_tokens = _tokens(paper.abstract)
    title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
    abstract_coverage = len(query_tokens & abstract_tokens) / len(query_tokens)
    return min(1.0, 0.7 * title_coverage + 0.3 * abstract_coverage)


def semantic_search(
    session: Session,
    project_id: UUID,
    query: str,
    limit: int = 12,
) -> list[dict]:
    query_vector = embed_text(query)
    candidate_limit = max(limit * 5, 50)
    if session.bind and session.bind.dialect.name == "postgresql":
        serialized = "[" + ",".join(f"{item:.10g}" for item in query_vector) + "]"
        rows = session.execute(
            text(
                """
                SELECT paper_id, 1 - (embedding <=> CAST(:query_vector AS vector)) AS score
                FROM paperembedding
                WHERE project_id = :project_id
                ORDER BY embedding <=> CAST(:query_vector AS vector)
                LIMIT :limit
                """
            ),
            {
                "query_vector": serialized,
                "project_id": project_id,
                "limit": candidate_limit,
            },
        ).all()
        ranked = [(row._mapping["paper_id"], float(row._mapping["score"])) for row in rows]
    else:
        embeddings = session.exec(
            select(PaperEmbedding).where(PaperEmbedding.project_id == project_id)
        ).all()
        ranked = sorted(
            (
                (item.paper_id, _cosine(query_vector, item.embedding))
                for item in embeddings
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_limit]
    query_tokens = _tokens(query)
    results = []
    for paper_id, score in ranked:
        paper = session.get(PaperRecord, paper_id)
        if paper:
            lexical_score = _lexical_score(query_tokens, paper)
            hybrid_score = 0.55 * max(0.0, score) + 0.45 * lexical_score
            results.append(
                {
                    **paper.model_dump(mode="json"),
                    "semantic_score": round(score, 6),
                    "lexical_score": round(lexical_score, 6),
                    "hybrid_score": round(hybrid_score, 6),
                    "embedding_model": MODEL_NAME,
                }
            )
    results.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return results[:limit]
