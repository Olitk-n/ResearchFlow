import hashlib
import inspect
import zipfile
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.db import create_db_and_tables, engine
from app.main import app, resume_project, validated_model_base_url
from app.models import (
    DataPreparation,
    DatasetAsset,
    ExperimentSpec,
    GapCandidate,
    GapValidation,
    ModelCallRecord,
    ModelConfig,
    PaperEmbedding,
    PaperRecord,
    PaperVersion,
    ProjectStatus,
    ResearchProject,
    User,
    WorkflowCheckpoint,
    WorkflowControl,
)
from app.providers.datasets import DatasetResult, dataset_relevance_score
from app.providers.literature import (
    NormalizedPaper,
    _crossref_open_pdf,
    _is_relevant,
    _query_terms,
    deduplicate_papers,
)
from app.providers.llm import LLMConfig, ModelBudgetExceeded, complete_json
from app.security import decrypt_secret, encrypt_secret, hash_password
from app.services.artifacts import build_manuscript
from app.services.data_prep import prepare_dataset
from app.services.embeddings import index_papers, semantic_search
from app.services.executors import execute_experiment
from app.services.experiment_agent import validate_generated_code
from app.services.gaps import evidence_from_papers, generate_gap_drafts
from app.services.open_access import safe_public_https_url
from app.services.venue_templates import ensure_official_template
from app.services.workflow import (
    discover_project,
    pause_if_requested,
    persist_model_usage,
    plan_selected_gap,
)


def test_secret_encryption_roundtrip():
    encrypted = encrypt_secret("sk-local-secret")
    assert encrypted != "sk-local-secret"
    assert decrypt_secret(encrypted) == "sk-local-secret"


def test_model_key_hint_is_separate_from_ciphertext():
    plaintext = "sk-test-12345678"
    encrypted = encrypt_secret(plaintext)
    model = ModelConfig(
        user_id=uuid4(),
        name="safe-display",
        provider="openai",
        model="test-model",
        encrypted_api_key=encrypted,
        api_key_hint=plaintext[-4:],
    )
    assert model.api_key_hint == "5678"
    assert plaintext not in model.encrypted_api_key
    assert model.api_key_hint != model.encrypted_api_key[-4:]


def test_resume_endpoint_runs_inside_async_event_loop():
    assert inspect.iscoroutinefunction(resume_project)


def test_literature_deduplicates_by_doi_arxiv_and_title():
    papers = [
        NormalizedPaper("openalex", "1", "An Agent Benchmark", doi="10.1/example"),
        NormalizedPaper("crossref", "2", "Different title", doi="10.1/example"),
        NormalizedPaper("arxiv", "3", "Planning With Agents", arxiv_id="2501.00001"),
        NormalizedPaper("semantic_scholar", "4", "Planning With Agents", arxiv_id="2501.00001"),
        NormalizedPaper("openalex", "5", "Evidence Grounded Agent Evaluation"),
        NormalizedPaper("crossref", "6", "Evidence-grounded agent evaluation"),
    ]
    result = deduplicate_papers(papers)
    assert len(result) == 3


def test_only_openly_licensed_crossref_pdf_is_treated_as_full_text():
    closed = {
        "resource": {"primary": {"URL": "https://publisher.example/paper"}},
        "link": [{"URL": "https://publisher.example/paper.pdf", "content-type": "application/pdf"}],
    }
    openly_licensed = {
        **closed,
        "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
    }
    assert _crossref_open_pdf(closed) is None
    assert _crossref_open_pdf(openly_licensed) == "https://publisher.example/paper.pdf"
    assert safe_public_https_url("https://arxiv.org/pdf/2501.00001")
    assert not safe_public_https_url("http://arxiv.org/pdf/2501.00001")
    assert not safe_public_https_url("https://127.0.0.1/private.pdf")
    assert not safe_public_https_url("https://service.internal/private.pdf")


def test_llm_agent_relevance_expands_acronym_and_ignores_generic_terms():
    paper = NormalizedPaper(
        source="arxiv",
        external_id="1",
        title="Large Language Model Agents Under Tool Failures",
        abstract="We study agentic recovery.",
    )
    assert _is_relevant(paper, "LLM agent evaluation robustness")
    assert _query_terms("LLM agent evaluation robustness") == ["llm", "agent"]


def test_gap_generation_has_evidence_and_reverse_queries():
    project_id = uuid4()
    papers = [
        PaperRecord(
            project_id=project_id,
            source="arxiv",
            external_id="2501.1",
            title="Agent Evaluation Under Tool Noise",
            abstract="However, robustness under changing tool environments remains an open challenge.",
        )
    ]
    evidence = evidence_from_papers(project_id, papers)
    drafts = generate_gap_drafts("LLM agent evaluation", papers, evidence)
    assert len(drafts) == 4
    assert all(draft.evidence_ids for draft in drafts)
    assert all(draft.counter_queries for draft in drafts)


def test_dataset_relevance_rejects_generic_robustness_domain():
    pathology = DatasetResult(
        source="huggingface",
        external_id="lab/staining-robustness-evaluation",
        name="lab/staining-robustness-evaluation",
        url="https://huggingface.co/datasets/lab/staining-robustness-evaluation",
        license="apache-2.0",
        size_hint=None,
        quality_notes="fixture",
        metadata={"tags": ["pathology", "image"]},
    )
    traces = DatasetResult(
        source="huggingface",
        external_id="research/agent-llm-traces",
        name="research/agent-llm-traces",
        url="https://huggingface.co/datasets/research/agent-llm-traces",
        license="apache-2.0",
        size_hint=None,
        quality_notes="fixture",
        metadata={"tags": ["agents", "tool-use"]},
    )
    assert dataset_relevance_score(pathology, "LLM agent evaluation robustness")[0] == 0
    assert dataset_relevance_score(traces, "LLM agent evaluation robustness")[0] == 2


def test_auth_and_project_api():
    create_db_and_tables()
    email = f"researcher-{uuid4()}@local.dev"
    with TestClient(app) as client:
        register = client.post(
            "/auth/register",
            json={"email": email, "password": "researchflow"},
        )
        assert register.status_code == 200
        token = register.json()["access_token"]
        project = client.post(
            "/projects",
            headers={"Authorization": f"Bearer {token}"},
            json={"title": "Agent evaluation", "direction": "LLM agent evaluation"},
        )
        assert project.status_code == 201
        project_id = project.json()["id"]
        detail = client.get(
            f"/projects/{project_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert detail.status_code == 200
        assert detail.json()["project"]["direction"] == "LLM agent evaluation"
        unauthenticated_events = client.get(f"/projects/{project_id}/events")
        assert unauthenticated_events.status_code == 401
        readiness = client.get("/readiness", headers={"Authorization": f"Bearer {token}"})
        assert readiness.status_code == 200
        assert readiness.json()["checks"]["database"] is True
        assert readiness.json()["checks"]["storage"] is True
        assert readiness.json()["checks"]["model_reachable"] is False
        assert readiness.json()["acceptance_ready"] is False


def test_model_api_never_returns_plaintext_or_ciphertext():
    create_db_and_tables()
    email = f"model-{uuid4()}@local.dev"
    secret = "sk-test-never-return-9876"
    with TestClient(app) as client:
        token = client.post(
            "/auth/register",
            json={"email": email, "password": "researchflow"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        created = client.post(
            "/models",
            headers=headers,
            json={
                "name": f"model-{uuid4()}",
                "provider": "openai",
                "model": "test-model",
                "api_key": secret,
                "base_url": "https://malicious.example/v1",
                "is_default": True,
            },
        )
        assert created.status_code == 201
        models = client.get("/models", headers=headers)
        payload = models.json()
        serialized = models.text
        assert payload[0]["base_url"] == "https://api.openai.com/v1"
        assert payload[0]["key_hint"] == "••••9876"
        assert secret not in serialized
        assert "encrypted_api_key" not in serialized
        with Session(engine) as session:
            stored = session.get(ModelConfig, UUID(created.json()["id"]))
            assert stored
            assert decrypt_secret(stored.encrypted_api_key) == secret
            assert secret not in stored.encrypted_api_key


def test_custom_model_base_url_rejects_insecure_remote_hosts():
    assert (
        validated_model_base_url(
            "openai_compatible",
            "http://127.0.0.1:11434/v1/",
        )
        == "http://127.0.0.1:11434/v1"
    )
    with pytest.raises(HTTPException):
        validated_model_base_url(
            "openai_compatible",
            "http://untrusted.example/v1",
        )


async def test_mocked_discovery_to_experiment_package(monkeypatch, tmp_path):
    create_db_and_tables()
    user = User(
        email=f"workflow-{uuid4()}@local.dev",
        password_hash=hash_password("researchflow"),
    )
    project = ResearchProject(
        user_id=user.id,
        title="Workflow test",
        direction="LLM agent evaluation",
    )
    with Session(engine) as session:
        session.add(user)
        session.add(project)
        session.commit()
        session.refresh(project)

    async def fake_literature(_query, per_source=5):
        return [
            NormalizedPaper(
                source="arxiv",
                external_id="2501.12345",
                arxiv_id="2501.12345",
                title="Reliable LLM Agent Evaluation",
                abstract="However, multilingual failure recovery remains an open challenge.",
                authors=["Ada Researcher"],
                publication_date="2026-01-20",
                url="https://arxiv.org/abs/2501.12345",
                open_access_url="https://arxiv.org/pdf/2501.12345",
            )
        ], []

    async def fake_datasets(_query, limit=4):
        return [
            DatasetResult(
                source="huggingface",
                external_id="research/agent-eval",
                name="research/agent-eval",
                url="https://huggingface.co/datasets/research/agent-eval",
                license="apache-2.0",
                size_hint="100 rows",
                quality_notes="test fixture",
                metadata={"relevance_score": 2, "matched_query_terms": ["llm", "agent"]},
            )
        ], []

    async def fake_prepare_dataset(project_id, dataset):
        data_root = tmp_path / "prepared-data"
        data_root.mkdir()
        sample = '{"prompt":"test","score":1}\n'
        (data_root / "prepared.jsonl").write_text(sample, encoding="utf-8")
        (data_root / "data-card.json").write_text(
            '{"dataset":"research/agent-eval","sample_rows":1}',
            encoding="utf-8",
        )
        return DataPreparation(
            project_id=project_id,
            dataset_id=dataset.id,
            status="completed",
            config_name="default",
            split_name="train",
            row_count=1,
            content_hash=("4f6bc3ab9b5f15bdf0f9fc919ad65f504a42d43f7e4f3fbfcb1fdbb657e3a689"),
            artifact_path=str(data_root),
        )

    monkeypatch.setattr("app.services.workflow.aggregate_literature", fake_literature)
    monkeypatch.setattr("app.services.workflow.aggregate_datasets", fake_datasets)
    monkeypatch.setattr("app.services.workflow.prepare_dataset", fake_prepare_dataset)

    await discover_project(project.id)
    with Session(engine) as session:
        gaps = session.exec(select(GapCandidate).where(GapCandidate.project_id == project.id)).all()
        validations = session.exec(
            select(GapValidation).where(GapValidation.project_id == project.id)
        ).all()
        papers = session.exec(select(PaperRecord).where(PaperRecord.project_id == project.id)).all()
        versions = session.exec(
            select(PaperVersion).where(PaperVersion.project_id == project.id)
        ).all()
        discovery_checkpoint = session.exec(
            select(WorkflowCheckpoint)
            .where(
                WorkflowCheckpoint.project_id == project.id,
                WorkflowCheckpoint.workflow_type == "discover",
            )
            .order_by(WorkflowCheckpoint.created_at.desc())
        ).first()
        assert len(gaps) == 4
        assert len(validations) == 4
        assert all(item.status == "low_coverage_supported" for item in validations)
        assert len(papers) == 1
        assert len(versions) == 1
        assert versions[0].metadata_hash
        assert discovery_checkpoint
        assert discovery_checkpoint.stage == "select_gap"
        assert discovery_checkpoint.status == "awaiting_human"
        assert discovery_checkpoint.requires_action
        gap_id = gaps[0].id

    await plan_selected_gap(project.id, gap_id)
    with Session(engine) as session:
        refreshed = session.get(ResearchProject, project.id)
        datasets = session.exec(select(DatasetAsset).where(DatasetAsset.project_id == project.id)).all()
        preparations = session.exec(select(DataPreparation).where(DataPreparation.project_id == project.id)).all()
        spec = session.exec(select(ExperimentSpec).where(ExperimentSpec.project_id == project.id)).first()
        gap = session.get(GapCandidate, gap_id)
        papers = session.exec(select(PaperRecord).where(PaperRecord.project_id == project.id)).all()
        plan_checkpoint = session.exec(
            select(WorkflowCheckpoint)
            .where(
                WorkflowCheckpoint.project_id == project.id,
                WorkflowCheckpoint.workflow_type == "plan",
            )
            .order_by(WorkflowCheckpoint.created_at.desc())
        ).first()
        assert refreshed.status == "ready"
        assert datasets[0].license == "apache-2.0"
        assert preparations[0].row_count == 1
        assert spec and spec.artifact_path
        assert spec.resource_profile["code_origin"] == "auditable_fallback"
        assert plan_checkpoint
        assert plan_checkpoint.stage == "completed"
        assert plan_checkpoint.status == "completed"
        assert (Path(spec.artifact_path) / "artifact-index.json").exists()
        assert (Path(spec.artifact_path) / "uv.lock").exists()
        status, result = execute_experiment(
            "cloud_disabled",
            Path(spec.artifact_path),
            300,
        )
        assert status == "blocked"
        assert result["billable_action"] is False
        manuscript_root, _, keys = build_manuscript(refreshed, gap, papers, "arxiv")
        assert (manuscript_root / "main.tex").exists()
        assert (manuscript_root / "references.bib").exists()
        assert (manuscript_root / "artifact-index.json").exists()
        assert keys


def test_generated_experiment_code_rejects_host_and_network_access():
    unsafe = """
import os
from pathlib import Path
Path("results.json").write_text(Path("data/prepared.jsonl").read_text())
os.system("whoami")
"""
    try:
        validate_generated_code(unsafe)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("unsafe generated code was accepted")


def test_cloud_executor_is_disabled_without_billable_action(tmp_path):
    status, result = execute_experiment("cloud_disabled", tmp_path, 300)
    assert status == "blocked"
    assert result["billable_action"] is False
    assert result["experiment_package"] == str(tmp_path)


async def test_zero_model_budget_blocks_before_api_call():
    with pytest.raises(ModelBudgetExceeded):
        await complete_json(
            LLMConfig(
                provider="openai",
                model="unused",
                api_key="unused",
                budget_limit_usd=0,
            ),
            system="unused",
            prompt="unused",
            schema_hint={"ok": True},
        )


async def test_model_call_records_usage_without_prompts_or_secrets(monkeypatch):
    async def fake_completion(**kwargs):
        assert kwargs["api_key"] == "fixture-secret"
        assert kwargs["messages"][0]["content"] == "private system instruction"
        assert "private research prompt" in kwargs["messages"][1]["content"]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"ok": true, "echo": "researchflow"}',
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=23, completion_tokens=7),
        )

    monkeypatch.setattr("app.providers.llm.acompletion", fake_completion)
    monkeypatch.setattr(
        "app.providers.llm.completion_cost",
        lambda completion_response: 0.00125,
    )
    create_db_and_tables()
    user = User(
        email=f"usage-{uuid4()}@local.dev",
        password_hash=hash_password("researchflow"),
    )
    model = ModelConfig(
        user_id=user.id,
        name="usage-model",
        provider="openai",
        model="fixture",
        encrypted_api_key=encrypt_secret("fixture-secret"),
        api_key_hint="cret",
        budget_limit_usd=1.0,
        is_default=True,
    )
    with Session(engine) as session:
        session.add(user)
        session.add(model)
        session.commit()
        session.refresh(model)
        config = LLMConfig(
            provider=model.provider,
            model=model.model,
            api_key="fixture-secret",
            base_url="https://api.openai.com/v1",
            budget_limit_usd=model.budget_limit_usd,
            config_id=model.id,
        )
        result = await complete_json(
            config,
            system="private system instruction",
            prompt="private research prompt",
            schema_hint={"ok": True, "echo": "researchflow"},
            purpose="acceptance_probe",
        )
        persist_model_usage(session, None, config)
        record = session.exec(
            select(ModelCallRecord).where(
                ModelCallRecord.model_config_id == model.id,
                ModelCallRecord.purpose == "acceptance_probe",
            )
        ).one()
        serialized = record.model_dump_json()
        assert result["ok"] is True
        assert record.input_tokens == 23
        assert record.output_tokens == 7
        assert record.cost_usd == pytest.approx(0.00125)
        stored_model = session.get(ModelConfig, model.id)
        assert stored_model
        assert stored_model.spent_usd == pytest.approx(0.00125)
        assert "fixture-secret" not in serialized
        assert "private system instruction" not in serialized
        assert "private research prompt" not in serialized


async def test_persisted_model_budget_blocks_later_calls(monkeypatch):
    called = False

    async def should_not_call(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("provider should not be called after budget exhaustion")

    monkeypatch.setattr("app.providers.llm.acompletion", should_not_call)
    config = LLMConfig(
        provider="openai",
        model="fixture",
        api_key="fixture",
        budget_limit_usd=0.01,
        spent_usd=0.01,
    )
    with pytest.raises(ModelBudgetExceeded):
        await complete_json(
            config,
            system="unused",
            prompt="unused",
            schema_hint={"ok": True},
        )
    assert called is False


async def test_unknown_litellm_price_uses_provider_token_rates(monkeypatch):
    async def fake_completion(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=500_000),
        )

    monkeypatch.setattr("app.providers.llm.acompletion", fake_completion)
    monkeypatch.setattr(
        "app.providers.llm.completion_cost",
        lambda completion_response: (_ for _ in ()).throw(ValueError("unknown model")),
    )
    config = LLMConfig(
        provider="mimo",
        model="mimo-v2.5-pro",
        api_key="fixture",
        budget_limit_usd=2,
    )
    await complete_json(
        config,
        system="fixture",
        prompt="fixture",
        schema_hint={"ok": True},
    )
    assert config.last_call_cost_usd == pytest.approx(0.87)
    assert config.spent_usd == pytest.approx(0.87)


def test_pause_checkpoint_and_model_usage_are_persisted():
    create_db_and_tables()
    user = User(
        email=f"audit-{uuid4()}@local.dev",
        password_hash=hash_password("researchflow"),
    )
    project = ResearchProject(
        user_id=user.id,
        title="Audit test",
        direction="agent evaluation",
        status=ProjectStatus.DISCOVERING,
    )
    model = ModelConfig(
        user_id=user.id,
        name="audit-model",
        provider="openai",
        model="fixture",
        encrypted_api_key=encrypt_secret("fixture"),
        is_default=True,
    )
    with Session(engine) as session:
        session.add(user)
        session.add(project)
        session.add(model)
        session.commit()
        session.refresh(project)
        session.refresh(model)
        session.add(WorkflowControl(project_id=project.id, pause_requested=True))
        session.commit()
        paused = pause_if_requested(
            session,
            project,
            uuid4(),
            "discover",
            "literature",
            {"search_queries": ["agent evaluation"]},
        )
        llm = LLMConfig(
            provider=model.provider,
            model=model.model,
            api_key="fixture",
            config_id=model.id,
            usage_records=[
                {
                    "purpose": "query_expansion",
                    "status": "completed",
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "cost_usd": 0.0012,
                }
            ],
        )
        persist_model_usage(session, project.id, llm)
        checkpoint = session.exec(
            select(WorkflowCheckpoint).where(WorkflowCheckpoint.project_id == project.id)
        ).first()
        call = session.exec(
            select(ModelCallRecord).where(ModelCallRecord.project_id == project.id)
        ).first()
        session.refresh(project)
        assert paused
        assert project.status == ProjectStatus.PAUSED
        assert checkpoint
        assert checkpoint.stage == "literature"
        assert checkpoint.status == "paused"
        assert checkpoint.requires_action
        assert call
        assert call.purpose == "query_expansion"
        assert call.input_tokens == 120
        assert call.cost_usd == pytest.approx(0.0012)
        assert llm.usage_records == []


def test_local_vector_index_and_semantic_search():
    create_db_and_tables()
    user = User(
        email=f"vector-{uuid4()}@local.dev",
        password_hash=hash_password("researchflow"),
    )
    project = ResearchProject(
        user_id=user.id,
        title="Vector test",
        direction="agent evaluation",
    )
    with Session(engine) as session:
        session.add(user)
        session.add(project)
        session.commit()
        papers = [
            PaperRecord(
                project_id=project.id,
                source="arxiv",
                external_id="vector-1",
                title="Multilingual Agent Failure Recovery",
                abstract="Agents recover from tool failures in Chinese and English.",
            ),
            PaperRecord(
                project_id=project.id,
                source="arxiv",
                external_id="vector-2",
                title="Image Classification with Convolutional Networks",
                abstract="A vision model for image recognition.",
            ),
        ]
        session.add_all(papers)
        session.commit()
        for paper in papers:
            session.refresh(paper)
        index_papers(session, papers)
        indexed = session.exec(
            select(PaperEmbedding).where(PaperEmbedding.project_id == project.id)
        ).all()
        results = semantic_search(
            session,
            project.id,
            "multilingual agents recovering from broken tools",
        )
        assert len(indexed) == 2
        assert len(indexed[0].embedding) == 384
        assert results[0]["id"] == str(papers[0].id)
        assert results[0]["semantic_score"] >= results[1]["semantic_score"]
        assert results[0]["lexical_score"] > results[1]["lexical_score"]
        assert results[0]["hybrid_score"] >= results[1]["hybrid_score"]


def test_official_venue_template_is_cached_and_hashed(tmp_path):
    from app.config import get_settings

    cache = get_settings().storage_root / "venue-templates" / "iclr-2026"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "official-template.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("template/iclr2026_conference.sty", "% official fixture")
        bundle.writestr("template/iclr2026_conference.bst", "ENTRY{}{}{}")
    extracted = cache / "extracted"
    if extracted.exists():
        import shutil

        shutil.rmtree(extracted)
    metadata = ensure_official_template("iclr", tmp_path)
    assert metadata["official_template"] is True
    assert len(metadata["archive_sha256"]) == 64
    assert (tmp_path / "iclr2026_conference.sty").exists()
    assert (tmp_path / "venue-template.json").exists()


async def test_prepared_dataset_hash_matches_exact_file_bytes(monkeypatch):
    async def fake_rows(_dataset_id, max_rows=100):
        return "default", "train", [{"text": "line one"}, {"text": "第二行"}]

    monkeypatch.setattr(
        "app.services.data_prep.fetch_huggingface_rows",
        fake_rows,
    )
    dataset = DatasetAsset(
        project_id=uuid4(),
        source="huggingface",
        external_id="research/hash-fixture",
        name="research/hash-fixture",
        url="https://huggingface.co/datasets/research/hash-fixture",
        license="apache-2.0",
    )
    preparation = await prepare_dataset(dataset.project_id, dataset)
    prepared_file = Path(preparation.artifact_path) / "prepared.jsonl"
    assert preparation.content_hash == hashlib.sha256(prepared_file.read_bytes()).hexdigest()
