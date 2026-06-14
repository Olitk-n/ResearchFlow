import asyncio
import json
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlmodel import Session, select

from ..db import engine
from ..models import (
    CitationEdge,
    CoverageMatrix,
    DataPreparation,
    DatasetAsset,
    EvidenceRecord,
    ExperimentRun,
    ExperimentSpec,
    GapCandidate,
    GapValidation,
    ManuscriptBuild,
    ModelCallRecord,
    ModelConfig,
    PaperEmbedding,
    PaperRecord,
    PaperVersion,
    ProjectStatus,
    ResearchProject,
    RunStatus,
    TaskEvent,
    WorkflowCheckpoint,
    WorkflowControl,
)
from ..providers.datasets import aggregate_datasets
from ..providers.literature import aggregate_literature, deduplicate_papers
from ..providers.llm import LLMConfig, complete_json
from ..security import decrypt_secret
from .artifacts import build_experiment_package
from .data_prep import choose_dataset, prepare_dataset
from .embeddings import index_papers
from .experiment_agent import baseline_path_diagnostics, generate_experiment
from .gaps import (
    GapDraft,
    build_coverage_matrix,
    evidence_from_papers,
    generate_gap_drafts,
)
from .open_access import ingest_open_access_papers
from .paper_graph import record_paper_graph
from .research_graph import research_phase_graph
from .scientific_validity import (
    assess_topic_submission_readiness,
    audit_dataset_fit,
    audit_experiment_code,
    recommend_submission_targets,
)


def emit(session: Session, project_id: UUID, stage: str, message: str, **payload) -> None:
    session.add(
        TaskEvent(
            project_id=project_id,
            stage=stage,
            message=message,
            payload=payload,
        )
    )
    session.commit()


def default_model_config(session: Session, user_id: UUID) -> LLMConfig | None:
    model = session.exec(
        select(ModelConfig).where(
            ModelConfig.user_id == user_id,
            ModelConfig.is_default.is_(True),
        )
    ).first()
    if not model:
        return None
    return LLMConfig(
        provider=model.provider,
        model=model.model,
        api_key=decrypt_secret(model.encrypted_api_key),
        base_url=model.base_url,
        budget_limit_usd=model.budget_limit_usd,
        spent_usd=model.spent_usd,
        input_price_per_million_usd=model.input_price_per_million_usd,
        output_price_per_million_usd=model.output_price_per_million_usd,
        config_id=model.id,
    )


def persist_model_usage(
    session: Session,
    project_id: UUID | None,
    config: LLMConfig | None,
) -> None:
    if config is None or config.config_id is None:
        return
    stored_config = session.get(ModelConfig, config.config_id)
    if stored_config is not None:
        stored_config.spent_usd = config.spent_usd
        session.add(stored_config)
    for usage in config.usage_records:
        session.add(
            ModelCallRecord(
                model_config_id=config.config_id,
                project_id=project_id,
                provider=config.provider,
                model=config.model,
                purpose=usage["purpose"],
                status=usage["status"],
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cost_usd=usage.get("cost_usd"),
                error_type=usage.get("error_type"),
            )
        )
    config.usage_records.clear()
    session.commit()


async def continue_confirmed_dataset(
    session: Session,
    project: ResearchProject,
    gap: GapCandidate,
    dataset: DatasetAsset,
    preparation: DataPreparation,
) -> ExperimentSpec:
    model_config = default_model_config(session, project.user_id)
    try:
        experiment = await generate_experiment(project, gap, dataset, preparation, model_config)
    except Exception:
        experiment = await generate_experiment(project, gap, dataset, preparation, None)
    persist_model_usage(session, project.id, model_config)
    code_audit = audit_experiment_code(experiment.code, experiment.scientific_plan)
    if not (dataset.validity_audit or {}).get("passed"):
        code_audit.passed = False
        code_audit.findings.append(
            "Dataset mismatch was accepted by a human; scientific validity remains unresolved."
        )
        if code_audit.level != "synthetic_demonstration":
            code_audit.level = "concept_draft"
    spec = ExperimentSpec(
        project_id=project.id,
        gap_id=gap.id,
        name=experiment.name,
        objective=experiment.objective,
        metrics=experiment.metrics,
        scientific_plan=experiment.scientific_plan,
        validity_audit=code_audit.as_dict(),
        quality_level=code_audit.level,
        resource_profile={
            "cpu": 2,
            "memory_gb": 4,
            "gpu": False,
            "network": False,
            "methodology": experiment.methodology,
            "expected_outputs": experiment.expected_outputs,
            "code_origin": experiment.code_origin,
            "dataset_preparation_id": str(preparation.id),
            "dataset_id": str(dataset.id),
        },
    )
    session.add(spec)
    session.commit()
    session.refresh(spec)
    spec.artifact_path = str(
        build_experiment_package(project, gap, dataset, preparation, experiment)
    )
    project.status = ProjectStatus.READY
    session.add(spec)
    session.add(project)
    session.commit()
    session.refresh(spec)
    return spec


def set_checkpoint(
    session: Session,
    project_id: UUID,
    workflow_run_id: UUID,
    workflow_type: str,
    stage: str,
    status: str,
    *,
    state: dict | None = None,
    requires_action: bool = False,
    error: str | None = None,
) -> WorkflowCheckpoint:
    checkpoint = WorkflowCheckpoint(
        project_id=project_id,
        workflow_run_id=workflow_run_id,
        workflow_type=workflow_type,
        stage=stage,
        status=status,
        state=state or {},
        requires_action=requires_action,
        error=error,
    )
    session.add(checkpoint)
    session.commit()
    return checkpoint


def pause_if_requested(
    session: Session,
    project: ResearchProject,
    workflow_run_id: UUID,
    workflow_type: str,
    next_stage: str,
    state: dict | None = None,
) -> bool:
    control = session.exec(
        select(WorkflowControl).where(WorkflowControl.project_id == project.id)
    ).first()
    if not control or not control.pause_requested:
        return False
    project.status = ProjectStatus.PAUSED
    project.updated_at = datetime.now(UTC)
    session.add(project)
    session.commit()
    set_checkpoint(
        session,
        project.id,
        workflow_run_id,
        workflow_type,
        next_stage,
        "paused",
        state=state,
        requires_action=True,
    )
    emit(
        session,
        project.id,
        "workflow",
        "流程已在安全阶段暂停，可从工作台恢复",
        workflow_type=workflow_type,
        next_stage=next_stage,
    )
    return True


async def expand_queries(
    project: ResearchProject,
    model: LLMConfig | None,
) -> list[str]:
    direction = project.direction.strip()
    lowered = direction.casefold()
    fallback = [direction]
    if "llm" in lowered and "agent" in lowered:
        fallback.extend(
            [
                "LLM agent benchmark",
                "large language model agent evaluation",
                "LLM agents robustness failure modes",
                "agent benchmark tool use reliability",
            ]
        )
    else:
        fallback.extend(
            [
                f"{direction} benchmark",
                f"{direction} limitations future work",
                f"{direction} failure modes",
            ]
        )
    fallback = list(dict.fromkeys(fallback))[:6]
    if model is None:
        return fallback
    response = await complete_json(
        model,
        system=(
            "Expand an AI/ML research direction into precise scholarly search queries. "
            "Cover the core topic, recent benchmarks, limitations, and likely novelty "
            "counter-searches. Return 3-6 English queries and no prose."
        ),
        prompt=f"Research direction: {project.direction}",
        schema_hint={"queries": ["string"]},
        max_tokens=800,
        purpose="query_expansion",
    )
    queries = [str(query).strip() for query in response.get("queries", []) if str(query).strip()]
    return list(dict.fromkeys([project.direction, *queries]))[:6] or fallback


async def model_gap_drafts(
    project: ResearchProject,
    papers: list[PaperRecord],
    evidence: list[EvidenceRecord],
    model: LLMConfig | None,
) -> list[GapDraft] | None:
    if model is None:
        return None
    literature = [
        {
            "title": paper.title,
            "date": paper.publication_date,
            "abstract": paper.abstract[:900],
        }
        for paper in papers[:16]
    ]
    schema = {
        "candidates": [
            {
                "title": "string",
                "hypothesis": "string",
                "rationale": "string",
                "confidence": 0.0,
                "novelty_score": 0.0,
                "feasibility_score": 0.0,
                "estimated_cost": "string",
                "risks": ["string"],
                "counter_queries": ["string"],
            }
        ]
    }
    response = await complete_json(
        model,
        system=(
            "You are a skeptical AI research planner. Propose only low-coverage candidates, "
            "never claim global novelty, and include falsifying reverse searches."
        ),
        prompt=(
            f"Direction: {project.direction}\n"
            "Based only on this retrieved literature snapshot, propose 3-5 technically "
            "testable research candidates. Do not invent papers or results.\n"
            f"{json.dumps(literature, ensure_ascii=False)}"
        ),
        schema_hint=schema,
        purpose="gap_generation",
    )
    candidates = response.get("candidates") or []
    if not 3 <= len(candidates) <= 5:
        return None
    evidence_ids = [str(item.id) for item in evidence[:8]]
    drafts = []
    for item in candidates:
        drafts.append(
            GapDraft(
                title=str(item["title"])[:240],
                hypothesis=str(item["hypothesis"])[:1200],
                rationale=str(item["rationale"])[:1800],
                confidence=max(0.0, min(float(item["confidence"]), 1.0)),
                novelty_score=max(0.0, min(float(item["novelty_score"]), 1.0)),
                feasibility_score=max(
                    0.0,
                    min(float(item["feasibility_score"]), 1.0),
                ),
                estimated_cost=str(item["estimated_cost"])[:240],
                risks=[str(value)[:500] for value in item.get("risks", [])][:5],
                evidence_ids=evidence_ids,
                counter_queries=[str(value)[:500] for value in item.get("counter_queries", [])][:5],
            )
        )
    return drafts


def _paper_identity(paper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.casefold()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.casefold()}"
    title = "".join(character for character in paper.title.casefold() if character.isalnum())
    return f"title:{title}"


async def validate_gap_candidate(
    gap: GapCandidate,
    known_papers: list[PaperRecord],
) -> GapValidation:
    queries = gap.counter_queries[:2]
    responses = await asyncio.gather(
        *(aggregate_literature(query, per_source=2) for query in queries),
        return_exceptions=True,
    )
    known = {_paper_identity(paper) for paper in known_papers}
    seen = set(known)
    results: list[dict] = []
    errors: list[str] = []
    for query, response in zip(queries, responses, strict=True):
        if isinstance(response, Exception):
            errors.append(f"{query}: aggregate: {type(response).__name__}")
            continue
        papers, query_errors = response
        errors.extend(f"{query}: {error}" for error in query_errors)
        for paper in papers:
            identity = _paper_identity(paper)
            if identity in seen:
                continue
            seen.add(identity)
            results.append(
                {
                    "query": query,
                    "title": paper.title,
                    "source": paper.source,
                    "publication_date": paper.publication_date,
                    "doi": paper.doi,
                    "arxiv_id": paper.arxiv_id,
                    "url": paper.open_access_url or paper.url,
                }
            )
    result_count = len(results)
    if not results and errors and len(errors) >= len(queries) * 3:
        status = "inconclusive"
    elif result_count == 0:
        status = "low_coverage_supported"
    elif result_count <= 3:
        status = "low_coverage_with_counterevidence"
    else:
        status = "contested"
    confidence_delta = 0.02 if status == "low_coverage_supported" else -(min(result_count, 8) * 0.035)
    if status == "inconclusive":
        confidence_delta = -0.12
    validated_confidence = round(max(0.2, min(0.95, gap.confidence + confidence_delta)), 2)
    return GapValidation(
        project_id=gap.project_id,
        gap_id=gap.id,
        status=status,
        initial_confidence=gap.confidence,
        validated_confidence=validated_confidence,
        reverse_query_results=results[:12],
        new_result_count=result_count,
        counterevidence_count=result_count,
        search_errors=errors,
    )


async def discover_project(project_id: UUID) -> None:
    with Session(engine) as session:
        project = session.get(ResearchProject, project_id)
        if not project:
            return
        workflow_run_id = uuid4()
        model_config: LLMConfig | None = None
        try:
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "discover",
                "query_expansion",
                "running",
            )
            phase_plan = await research_phase_graph.ainvoke({"mode": "discover", "stages": []})
            project.status = ProjectStatus.DISCOVERING
            project.updated_at = datetime.now(UTC)
            model_config = default_model_config(session, project.user_id)
            try:
                project.search_queries = await expand_queries(
                    project,
                    model_config,
                )
            except Exception as query_error:
                project.search_queries = await expand_queries(project, None)
                emit(
                    session,
                    project.id,
                    "model",
                    "模型检索式扩展失败，已使用可审计默认检索式",
                    error=type(query_error).__name__,
                )
            session.add(project)
            session.commit()
            persist_model_usage(session, project.id, model_config)
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "discover",
                "query_expansion",
                "completed",
                state={"search_queries": project.search_queries},
            )
            if pause_if_requested(
                session,
                project,
                workflow_run_id,
                "discover",
                "literature",
                {"search_queries": project.search_queries},
            ):
                return
            emit(
                session,
                project.id,
                "literature",
                "正在聚合最新论文",
                phase_plan=phase_plan["stages"],
            )

            normalized = []
            errors = []
            query_results = await asyncio.gather(
                *(aggregate_literature(query, per_source=5) for query in project.search_queries),
                return_exceptions=True,
            )
            for query, result in zip(
                project.search_queries,
                query_results,
                strict=True,
            ):
                if isinstance(result, Exception):
                    errors.append(f"{query}: aggregate: {type(result).__name__}")
                    continue
                query_papers, query_errors = result
                normalized.extend(query_papers)
                errors.extend(f"{query}: {error}" for error in query_errors)
            normalized = deduplicate_papers(
                normalized,
                query=project.direction,
            )
            if not normalized:
                emit(
                    session,
                    project.id,
                    "literature",
                    "所有检索式均未得到相关论文",
                    source_errors=errors,
                    search_queries=project.search_queries,
                )
                raise RuntimeError(
                    "all literature queries failed or returned no relevant records; "
                    + "; ".join(errors[:12])
                )

            # Preserve the last valid snapshot until a replacement has been retrieved.
            project.selected_gap_id = None
            session.add(project)
            session.commit()
            specs = session.exec(select(ExperimentSpec).where(ExperimentSpec.project_id == project.id)).all()
            for spec in specs:
                runs = session.exec(select(ExperimentRun).where(ExperimentRun.spec_id == spec.id)).all()
                for run in runs:
                    session.delete(run)
                session.delete(spec)
            for model, condition in (
                (ManuscriptBuild, ManuscriptBuild.project_id == project.id),
                (DataPreparation, DataPreparation.project_id == project.id),
                (DatasetAsset, DatasetAsset.project_id == project.id),
                (GapValidation, GapValidation.project_id == project.id),
                (CoverageMatrix, CoverageMatrix.project_id == project.id),
                (EvidenceRecord, EvidenceRecord.project_id == project.id),
                (GapCandidate, GapCandidate.project_id == project.id),
                (CitationEdge, CitationEdge.project_id == project.id),
                (PaperVersion, PaperVersion.project_id == project.id),
                (PaperEmbedding, PaperEmbedding.project_id == project.id),
                (PaperRecord, PaperRecord.project_id == project.id),
            ):
                for item in session.exec(select(model).where(condition)).all():
                    session.delete(item)
            session.commit()

            for item in normalized:
                session.add(PaperRecord(project_id=project.id, **item.to_dict()))
            session.commit()
            papers = session.exec(
                select(PaperRecord)
                .where(PaperRecord.project_id == project.id)
                .order_by(PaperRecord.publication_date.desc())
            ).all()
            index_papers(session, papers)
            record_paper_graph(session, papers)
            emit(
                session,
                project.id,
                "literature",
                f"已获得 {len(papers)} 篇去重论文",
                source_errors=errors,
            )
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "discover",
                "literature",
                "completed",
                state={"paper_count": len(papers), "source_errors": errors},
            )
            if pause_if_requested(
                session,
                project,
                workflow_run_id,
                "discover",
                "evidence",
                {"paper_count": len(papers)},
            ):
                return

            fulltext_evidence, fulltext_errors = await ingest_open_access_papers(
                project.id,
                papers,
            )
            for paper in papers:
                session.add(paper)
            evidence = [
                *fulltext_evidence,
                *evidence_from_papers(project.id, papers),
            ]
            session.add_all(evidence)
            session.add(build_coverage_matrix(project.id, papers))
            session.commit()
            for item in evidence:
                session.refresh(item)
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "discover",
                "evidence",
                "completed",
                state={"evidence_count": len(evidence)},
            )
            emit(
                session,
                project.id,
                "fulltext",
                f"已提取 {len(fulltext_evidence)} 条开放全文页级证据",
                source_errors=fulltext_errors,
            )
            if pause_if_requested(
                session,
                project,
                workflow_run_id,
                "discover",
                "gap_generation",
                {"paper_count": len(papers), "evidence_count": len(evidence)},
            ):
                return

            try:
                drafts = await model_gap_drafts(
                    project,
                    papers,
                    evidence,
                    model_config,
                )
            except Exception as model_error:
                drafts = None
                emit(
                    session,
                    project.id,
                    "model",
                    "模型候选生成失败，已回退到可审计规则",
                    error=type(model_error).__name__,
                )
            drafts = drafts or generate_gap_drafts(project.direction, papers, evidence)
            persist_model_usage(session, project.id, model_config)
            if model_config is not None:
                emit(
                    session,
                    project.id,
                    "budget",
                    "模型任务预算已更新",
                    spent_usd=round(model_config.spent_usd, 8),
                    budget_limit_usd=model_config.budget_limit_usd,
                )
            for draft in drafts:
                session.add(GapCandidate(project_id=project.id, **asdict(draft)))
            session.commit()
            gaps = session.exec(
                select(GapCandidate).where(GapCandidate.project_id == project.id)
            ).all()
            emit(
                session,
                project.id,
                "gap_validation",
                "正在执行候选课题的反向检索验证",
                candidate_count=len(gaps),
            )
            validations = await asyncio.gather(
                *(validate_gap_candidate(gap, papers) for gap in gaps)
            )
            for gap, validation in zip(gaps, validations, strict=True):
                gap.confidence = validation.validated_confidence
                session.add(gap)
                session.add(validation)
            project.status = ProjectStatus.AWAITING_TOPIC
            project.updated_at = datetime.now(UTC)
            session.add(project)
            session.commit()
            emit(
                session,
                project.id,
                "gaps",
                f"生成并反向验证 {len(drafts)} 个低覆盖研究空白候选",
                validation_statuses=[item.status for item in validations],
            )
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "discover",
                "select_gap",
                "awaiting_human",
                state={
                    "paper_count": len(papers),
                    "evidence_count": len(evidence),
                    "candidate_count": len(drafts),
                },
                requires_action=True,
            )
        except Exception as exc:
            persist_model_usage(session, project.id, model_config)
            project.status = ProjectStatus.FAILED
            session.add(project)
            session.commit()
            emit(
                session,
                project.id,
                "error",
                "研究发现流程失败",
                error=type(exc).__name__,
                detail=str(exc)[:500],
                traceback=traceback.format_exc(limit=8)[-3000:],
            )
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "discover",
                "failed",
                "failed",
                error=str(exc)[:1000],
            )


async def plan_selected_gap(project_id: UUID, gap_id: UUID) -> None:
    with Session(engine) as session:
        project = session.get(ResearchProject, project_id)
        gap = session.get(GapCandidate, gap_id)
        if not project or not gap or gap.project_id != project.id:
            return
        workflow_run_id = uuid4()
        model_config: LLMConfig | None = None
        try:
            awaiting = session.exec(
                select(WorkflowCheckpoint)
                .where(
                    WorkflowCheckpoint.project_id == project.id,
                    WorkflowCheckpoint.workflow_type == "discover",
                    WorkflowCheckpoint.status == "awaiting_human",
                )
                .order_by(WorkflowCheckpoint.created_at.desc())
            ).first()
            if awaiting:
                awaiting.status = "completed"
                awaiting.requires_action = False
                awaiting.updated_at = datetime.now(UTC)
                awaiting.state = {**awaiting.state, "selected_gap_id": str(gap.id)}
                session.add(awaiting)
                session.commit()
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "plan",
                "datasets",
                "running",
                state={"gap_id": str(gap.id)},
            )
            phase_plan = await research_phase_graph.ainvoke({"mode": "plan", "stages": []})
            project.status = ProjectStatus.PLANNING
            project.selected_gap_id = gap.id
            session.add(project)
            session.commit()
            existing_specs = session.exec(
                select(ExperimentSpec).where(ExperimentSpec.project_id == project.id)
            ).all()
            for spec in existing_specs:
                for run in session.exec(
                    select(ExperimentRun).where(ExperimentRun.spec_id == spec.id)
                ).all():
                    session.delete(run)
                session.delete(spec)
            for model, condition in (
                (ManuscriptBuild, ManuscriptBuild.project_id == project.id),
                (DataPreparation, DataPreparation.project_id == project.id),
                (DatasetAsset, DatasetAsset.project_id == project.id),
            ):
                for item in session.exec(select(model).where(condition)).all():
                    session.delete(item)
            stale_stages = ("datasets", "data_processing", "experiment", "manuscript")
            for event in session.exec(
                select(TaskEvent).where(
                    TaskEvent.project_id == project.id,
                    TaskEvent.stage.in_(stale_stages),
                )
            ).all():
                session.delete(event)
            session.commit()
            emit(
                session,
                project.id,
                "datasets",
                "正在检索许可明确的公开数据集",
                phase_plan=phase_plan["stages"],
            )

            dataset_query = f"{project.direction} {gap.title} {gap.hypothesis}"
            datasets, errors = await aggregate_datasets(dataset_query, limit=4)
            for item in datasets:
                session.add(
                    DatasetAsset(
                        project_id=project.id,
                        gap_id=gap.id,
                        source=item.source,
                        external_id=item.external_id,
                        name=item.name,
                        url=item.url,
                        license=item.license,
                        size_hint=item.size_hint,
                        quality_notes=item.quality_notes,
                        metadata_json=item.metadata,
                    )
                )
            session.commit()
            assets = session.exec(select(DatasetAsset).where(DatasetAsset.project_id == project.id)).all()
            emit(
                session,
                project.id,
                "datasets",
                f"找到 {len(assets)} 个候选数据集",
                source_errors=errors,
            )
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "plan",
                "datasets",
                "completed",
                state={"gap_id": str(gap.id), "dataset_count": len(assets)},
            )
            if pause_if_requested(
                session,
                project,
                workflow_run_id,
                "plan",
                "data_processing",
                {"gap_id": str(gap.id)},
            ):
                return

            selected_dataset = choose_dataset(assets, minimum_relevance=2)
            if not selected_dataset:
                readiness = assess_topic_submission_readiness(project, gap, assets)
                gap.submission_readiness = readiness.as_dict()
                gap.alternative_topics = readiness.details["alternatives"]
                session.add(gap)
                session.commit()
                emit(
                    session,
                    project.id,
                    "topic_readiness",
                    "当前选题未找到足以支撑投稿实验的数据；已提供相似可行选题。",
                    audit=readiness.as_dict(),
                )
                raise RuntimeError("没有许可明确且可自动使用的数据集")
            emit(
                session,
                project.id,
                "data_processing",
                f"正在处理数据集 {selected_dataset.name}",
                license=selected_dataset.license,
            )
            preparation = await prepare_dataset(project.id, selected_dataset)
            session.add(preparation)
            session.commit()
            session.refresh(preparation)
            dataset_audit = audit_dataset_fit(project, gap, selected_dataset, preparation)
            audit_payload = dataset_audit.as_dict()
            audit_payload["details"] = {
                **audit_payload.get("details", {}),
                "baseline_paths": baseline_path_diagnostics(preparation),
            }
            selected_dataset.validity_audit = audit_payload
            readiness = assess_topic_submission_readiness(
                project, gap, assets, preparation,
            )
            gap.submission_readiness = readiness.as_dict()
            gap.alternative_topics = readiness.details["alternatives"]
            session.add(selected_dataset)
            session.add(gap)
            session.commit()
            emit(
                session,
                project.id,
                "topic_readiness",
                (
                    "当前选题具备继续构建投稿级实验的基础。"
                    if readiness.passed
                    else "当前选题暂不具备投稿实验条件；请修正数据或选择相似可行选题。"
                ),
                audit=readiness.as_dict(),
            )
            if (
                preparation.status == RunStatus.COMPLETED
                and not dataset_audit.passed
                and not selected_dataset.human_confirmed
            ):
                project.status = ProjectStatus.PAUSED
                session.add(project)
                session.commit()
                set_checkpoint(
                    session,
                    project.id,
                    workflow_run_id,
                    "plan",
                    "dataset_validity",
                    "awaiting_human",
                    state={
                        "gap_id": str(gap.id),
                        "dataset_id": str(selected_dataset.id),
                        "preparation_id": str(preparation.id),
                        "audit": dataset_audit.as_dict(),
                    },
                    requires_action=True,
                )
                emit(
                    session,
                    project.id,
                    "dataset_validity",
                    "Dataset-topic fit failed. Human confirmation is required.",
                    audit=dataset_audit.as_dict(),
                )
                return
            if preparation.status != RunStatus.COMPLETED:
                raise RuntimeError(preparation.profile_json.get("reason", "数据处理失败"))
            emit(
                session,
                project.id,
                "data_processing",
                f"已处理 {preparation.row_count} 条样本并生成数据指纹",
                content_hash=preparation.content_hash,
            )
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "plan",
                "data_processing",
                "completed",
                state={
                    "gap_id": str(gap.id),
                    "dataset_id": str(selected_dataset.id),
                    "preparation_id": str(preparation.id),
                },
            )
            if pause_if_requested(
                session,
                project,
                workflow_run_id,
                "plan",
                "experiment",
                {"gap_id": str(gap.id)},
            ):
                return

            model_config = default_model_config(session, project.user_id)
            try:
                experiment = await generate_experiment(
                    project,
                    gap,
                    selected_dataset,
                    preparation,
                    model_config,
                )
                if model_config is not None:
                    emit(
                        session,
                        project.id,
                        "budget",
                        "模型任务预算已更新",
                        spent_usd=round(model_config.spent_usd, 8),
                        budget_limit_usd=model_config.budget_limit_usd,
                    )
            except Exception as model_error:
                emit(
                    session,
                    project.id,
                    "model",
                    "模型实验代码未通过生成或安全审查，已回退到离线基线",
                    error=type(model_error).__name__,
                )
                experiment = await generate_experiment(
                    project,
                    gap,
                    selected_dataset,
                    preparation,
                    None,
                )
            persist_model_usage(session, project.id, model_config)
            code_audit = audit_experiment_code(experiment.code, experiment.scientific_plan)
            readiness.details["recommended_targets"] = recommend_submission_targets(
                gap,
                preparation.row_count,
                len(experiment.scientific_plan.get("baselines") or []),
                experiment.scientific_plan.get("evidence_class") == "real_task",
            )
            gap.submission_readiness = readiness.as_dict()
            session.add(gap)
            session.commit()
            if not dataset_audit.passed:
                code_audit.passed = False
                code_audit.findings.append(
                    "Dataset-topic fit did not pass the scientific audit."
                )
                if code_audit.level != "synthetic_demonstration":
                    code_audit.level = "concept_draft"
            spec = ExperimentSpec(
                project_id=project.id,
                gap_id=gap.id,
                name=experiment.name,
                objective=experiment.objective,
                metrics=experiment.metrics,
                scientific_plan=experiment.scientific_plan,
                validity_audit=code_audit.as_dict(),
                quality_level=code_audit.level,
                resource_profile={
                    "cpu": 2,
                    "memory_gb": 4,
                    "gpu": False,
                    "network": False,
                    "methodology": experiment.methodology,
                    "expected_outputs": experiment.expected_outputs,
                    "code_origin": experiment.code_origin,
                    "baseline_paths": baseline_path_diagnostics(preparation),
                    "dataset_preparation_id": str(preparation.id),
                    "dataset_id": str(selected_dataset.id),
                },
            )
            session.add(spec)
            session.commit()
            session.refresh(spec)
            package = build_experiment_package(
                project,
                gap,
                selected_dataset,
                preparation,
                experiment,
            )
            spec.artifact_path = str(package)
            project.status = ProjectStatus.READY
            session.add(spec)
            session.add(project)
            session.commit()
            emit(session, project.id, "experiment", "实验包已生成，等待 Docker 安全执行")
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "plan",
                "completed",
                "completed",
                state={
                    "gap_id": str(gap.id),
                    "experiment_spec_id": str(spec.id),
                },
            )
        except Exception as exc:
            persist_model_usage(session, project.id, model_config)
            project.status = ProjectStatus.FAILED
            session.add(project)
            session.commit()
            emit(
                session,
                project.id,
                "error",
                "实验规划流程失败",
                error=type(exc).__name__,
                detail=str(exc)[:500],
                traceback=traceback.format_exc(limit=8)[-3000:],
            )
            set_checkpoint(
                session,
                project.id,
                workflow_run_id,
                "plan",
                "failed",
                "failed",
                state={"gap_id": str(gap.id)},
                error=str(exc)[:1000],
            )


def schedule(coroutine) -> None:
    asyncio.create_task(coroutine)
