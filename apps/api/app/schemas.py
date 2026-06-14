from typing import Literal
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class AuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ModelConfigCreate(BaseModel):
    name: str
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    budget_limit_usd: float = Field(default=5.0, ge=0, le=1000)
    input_price_per_million_usd: float | None = Field(default=None, ge=0)
    output_price_per_million_usd: float | None = Field(default=None, ge=0)
    is_default: bool = False


class ProjectCreate(BaseModel):
    title: str
    direction: str


class GapSelection(BaseModel):
    gap_id: UUID


class ManuscriptRequest(BaseModel):
    target: Literal["arxiv", "iclr", "icml", "neurips"] = "arxiv"
    mode: Literal["draft", "submission"] = "draft"


class ExperimentRunRequest(BaseModel):
    allow_network: bool = False
    timeout_seconds: int = Field(default=300, ge=10, le=3600)
    executor: Literal["docker", "cloud_disabled"] = "docker"


class DatasetValidityConfirmation(BaseModel):
    dataset_id: UUID
    confirmed: bool
    reason: str = Field(min_length=10, max_length=1000)
