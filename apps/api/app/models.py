from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import JSON, Column, Text, UniqueConstraint
from sqlalchemy.types import UserDefinedType
from sqlmodel import Field, SQLModel


def now_utc() -> datetime:
    return datetime.now(UTC)


class PGVectorType(UserDefinedType):
    cache_ok = True

    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def get_col_spec(self, **_kwargs) -> str:
        return f"vector({self.dimensions})"

    def bind_processor(self, _dialect):
        def process(value):
            if value is None or isinstance(value, str):
                return value
            return "[" + ",".join(f"{float(item):.10g}" for item in value) + "]"

        return process

    def result_processor(self, _dialect, _column_type):
        def process(value):
            if value is None or isinstance(value, list):
                return value
            return [float(item) for item in value.strip("[]").split(",") if item]

        return process


class ProjectStatus(StrEnum):
    DRAFT = "draft"
    DISCOVERING = "discovering"
    AWAITING_TOPIC = "awaiting_topic"
    PLANNING = "planning"
    PAUSED = "paused"
    READY = "ready"
    FAILED = "failed"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=now_utc)


class ModelConfig(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", "name"),)
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    name: str
    provider: str
    model: str
    base_url: str | None = None
    encrypted_api_key: str
    api_key_hint: str | None = None
    capabilities: dict = Field(default_factory=dict, sa_column=Column(JSON))
    budget_limit_usd: float = 5.0
    spent_usd: float = 0.0
    input_price_per_million_usd: float | None = None
    output_price_per_million_usd: float | None = None
    is_default: bool = False
    created_at: datetime = Field(default_factory=now_utc)


class ModelCallRecord(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    model_config_id: UUID = Field(foreign_key="modelconfig.id", index=True)
    project_id: UUID | None = Field(default=None, foreign_key="researchproject.id", index=True)
    provider: str
    model: str
    purpose: str = Field(index=True)
    status: str = Field(default="completed", index=True)
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    error_type: str | None = None
    created_at: datetime = Field(default_factory=now_utc, index=True)


class ResearchProject(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", index=True)
    title: str
    direction: str
    status: ProjectStatus = Field(default=ProjectStatus.DRAFT, index=True)
    selected_gap_id: UUID | None = Field(default=None, foreign_key="gapcandidate.id")
    search_queries: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class WorkflowControl(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("project_id"),)
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    pause_requested: bool = False
    updated_at: datetime = Field(default_factory=now_utc)


class WorkflowCheckpoint(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    workflow_run_id: UUID = Field(index=True)
    workflow_type: str = Field(index=True)
    stage: str = Field(index=True)
    status: str = Field(index=True)
    state: dict = Field(default_factory=dict, sa_column=Column(JSON))
    requires_action: bool = False
    error: str | None = Field(default=None, sa_column=Column(Text))
    created_at: datetime = Field(default_factory=now_utc, index=True)
    updated_at: datetime = Field(default_factory=now_utc)


class PaperRecord(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    source: str = Field(index=True)
    external_id: str = Field(index=True)
    doi: str | None = Field(default=None, index=True)
    arxiv_id: str | None = Field(default=None, index=True)
    title: str
    abstract: str = Field(default="", sa_column=Column(Text))
    authors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    publication_date: str | None = None
    url: str | None = None
    open_access_url: str | None = None
    citation_count: int = 0
    raw_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    retrieved_at: datetime = Field(default_factory=now_utc)


class PaperEmbedding(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("paper_id"),)
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    paper_id: UUID = Field(foreign_key="paperrecord.id", index=True)
    embedding_model: str = "local-hash-384-v1"
    dimensions: int = 384
    embedding: list[float] = Field(
        sa_column=Column(
            JSON().with_variant(PGVectorType(384), "postgresql"),
            nullable=False,
        )
    )
    created_at: datetime = Field(default_factory=now_utc)


class PaperVersion(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("paper_id", "version_label"),)
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    paper_id: UUID = Field(foreign_key="paperrecord.id", index=True)
    version_label: str
    metadata_hash: str = Field(index=True)
    observed_at: datetime = Field(default_factory=now_utc)


class CitationEdge(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    citing_paper_id: UUID = Field(foreign_key="paperrecord.id", index=True)
    cited_external_id: str = Field(index=True)
    relation: str = "references"
    created_at: datetime = Field(default_factory=now_utc)


class EvidenceRecord(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    paper_id: UUID | None = Field(default=None, foreign_key="paperrecord.id", index=True)
    evidence_type: str
    claim: str = Field(sa_column=Column(Text))
    excerpt: str = Field(default="", sa_column=Column(Text))
    locator: str | None = None
    content_hash: str
    created_at: datetime = Field(default_factory=now_utc)


class GapCandidate(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    title: str
    hypothesis: str = Field(sa_column=Column(Text))
    rationale: str = Field(sa_column=Column(Text))
    confidence: float
    novelty_score: float
    feasibility_score: float
    estimated_cost: str
    risks: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    evidence_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    counter_queries: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now_utc)


class GapValidation(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("gap_id"),)
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    gap_id: UUID = Field(foreign_key="gapcandidate.id", index=True)
    status: str = Field(default="pending", index=True)
    initial_confidence: float
    validated_confidence: float
    reverse_query_results: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    new_result_count: int = 0
    counterevidence_count: int = 0
    search_errors: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    validated_at: datetime = Field(default_factory=now_utc)


class CoverageMatrix(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    dimensions: dict = Field(default_factory=dict, sa_column=Column(JSON))
    rows: list[dict] = Field(default_factory=list, sa_column=Column(JSON))
    summary: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now_utc)


class DatasetAsset(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    gap_id: UUID | None = Field(default=None, foreign_key="gapcandidate.id")
    source: str
    external_id: str
    name: str
    url: str
    license: str | None = None
    size_hint: str | None = None
    quality_notes: str = ""
    metadata_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    validity_audit: dict = Field(default_factory=dict, sa_column=Column(JSON))
    human_confirmed: bool = False
    created_at: datetime = Field(default_factory=now_utc)


class DataPreparation(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    dataset_id: UUID = Field(foreign_key="datasetasset.id", index=True)
    status: RunStatus = Field(default=RunStatus.PENDING)
    config_name: str | None = None
    split_name: str | None = None
    row_count: int = 0
    schema_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    profile_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    transformations: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    content_hash: str | None = Field(default=None, index=True)
    artifact_path: str | None = None
    created_at: datetime = Field(default_factory=now_utc)


class ExperimentSpec(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    gap_id: UUID = Field(foreign_key="gapcandidate.id")
    name: str
    objective: str = Field(sa_column=Column(Text))
    metrics: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    seed: int = 42
    resource_profile: dict = Field(default_factory=dict, sa_column=Column(JSON))
    scientific_plan: dict = Field(default_factory=dict, sa_column=Column(JSON))
    validity_audit: dict = Field(default_factory=dict, sa_column=Column(JSON))
    quality_level: str = "concept_draft"
    artifact_path: str | None = None
    created_at: datetime = Field(default_factory=now_utc)


class ExperimentRun(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    spec_id: UUID = Field(foreign_key="experimentspec.id", index=True)
    status: RunStatus = Field(default=RunStatus.PENDING)
    executor: str = "docker"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    logs_path: str | None = None
    results: dict = Field(default_factory=dict, sa_column=Column(JSON))
    validity_audit: dict = Field(default_factory=dict, sa_column=Column(JSON))
    quality_level: str = "concept_draft"


class ManuscriptBuild(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    target: str = "arxiv"
    status: RunStatus = Field(default=RunStatus.PENDING)
    artifact_path: str | None = None
    citation_keys: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    mode: str = "draft"
    quality_level: str = "concept_draft"
    validity_audit: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now_utc)


class TaskEvent(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    project_id: UUID = Field(foreign_key="researchproject.id", index=True)
    stage: str
    level: str = "info"
    message: str
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now_utc, index=True)
