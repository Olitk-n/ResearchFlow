from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, create_engine

from .config import get_settings

settings = get_settings()
if settings.database_url.startswith("sqlite"):
    Path("data").mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(settings.database_url, pool_pre_ping=True)


def create_db_and_tables() -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    SQLModel.metadata.create_all(engine)
    model_columns = {
        column["name"] for column in inspect(engine).get_columns("modelconfig")
    }
    with engine.begin() as connection:
        if "api_key_hint" not in model_columns:
            connection.execute(
                text("ALTER TABLE modelconfig ADD COLUMN api_key_hint VARCHAR")
            )
    additive_columns = {
        "datasetasset": {
            "validity_audit": "JSON NOT NULL DEFAULT '{}'",
            "human_confirmed": "BOOLEAN NOT NULL DEFAULT 0",
        },
        "experimentspec": {
            "scientific_plan": "JSON NOT NULL DEFAULT '{}'",
            "validity_audit": "JSON NOT NULL DEFAULT '{}'",
            "quality_level": "VARCHAR NOT NULL DEFAULT 'concept_draft'",
        },
        "experimentrun": {
            "validity_audit": "JSON NOT NULL DEFAULT '{}'",
            "quality_level": "VARCHAR NOT NULL DEFAULT 'concept_draft'",
        },
        "manuscriptbuild": {
            "mode": "VARCHAR NOT NULL DEFAULT 'draft'",
            "quality_level": "VARCHAR NOT NULL DEFAULT 'concept_draft'",
            "validity_audit": "JSON NOT NULL DEFAULT '{}'",
        },
    }
    with engine.begin() as connection:
        inspector = inspect(engine)
        for table, columns in additive_columns.items():
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, definition in columns.items():
                if name not in existing:
                    connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))
        if "spent_usd" not in model_columns:
            connection.execute(
                text(
                    "ALTER TABLE modelconfig ADD COLUMN spent_usd FLOAT "
                    "NOT NULL DEFAULT 0"
                )
            )
        if "input_price_per_million_usd" not in model_columns:
            connection.execute(
                text(
                    "ALTER TABLE modelconfig ADD COLUMN "
                    "input_price_per_million_usd FLOAT"
                )
            )
        if "output_price_per_million_usd" not in model_columns:
            connection.execute(
                text(
                    "ALTER TABLE modelconfig ADD COLUMN "
                    "output_price_per_million_usd FLOAT"
                )
            )
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS paperembedding_embedding_hnsw
                    ON paperembedding USING hnsw (embedding vector_cosine_ops)
                    """
                )
            )


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
