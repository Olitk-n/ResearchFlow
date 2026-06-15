import asyncio
import io
import json
import shutil
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pypdf import PdfReader
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from .auth import CurrentUser
from .config import get_settings
from .db import SessionDep, create_db_and_tables, engine
from .models import (
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
    PaperRecord,
    PaperVersion,
    ResearchProject,
    RunStatus,
    TaskEvent,
    User,
    WorkflowCheckpoint,
    WorkflowControl,
)
from .providers.llm import PROVIDER_PRESETS, LLMConfig, probe_model
from .schemas import (
    AlternativeTopicSelection,
    AuthRequest,
    AuthResponse,
    DatasetValidityConfirmation,
    ExperimentRunRequest,
    GapSelection,
    ManuscriptRequest,
    ModelConfigCreate,
    ProjectCreate,
)
from .security import (
    create_access_token,
    decrypt_secret,
    encrypt_secret,
    hash_password,
    verify_password,
)
from .services.artifacts import (
    build_manuscript,
    find_executable,
)
from .services.embeddings import semantic_search
from .services.executors import execute_experiment
from .services.manuscript_agent import generate_manuscript
from .services.scientific_validity import (
    audit_completed_run,
    backfill_scientific_validity,
    build_similar_feasible_topics,
    submission_gate,
)
from .services.venue_templates import TemplateUnavailable
from .services.workflow import (
    continue_confirmed_dataset,
    default_model_config,
    discover_project,
    emit,
    persist_model_usage,
    plan_selected_gap,
    schedule,
)
from .storage import content_store

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_db_and_tables()
    from sqlmodel import Session

    with Session(engine) as session:
        backfill_scientific_validity(session)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Local-first, evidence-grounded AI research automation.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.web_origin, "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def own_project(session: SessionDep, user: User, project_id: UUID) -> ResearchProject:
    project = session.get(ResearchProject, project_id)
    if not project or project.user_id != user.id:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def validated_model_base_url(provider: str, supplied: str | None) -> str | None:
    preset = PROVIDER_PRESETS.get(provider)
    if provider != "openai_compatible":
        return preset
    if not supplied:
        raise HTTPException(status_code=422, detail="兼容接口必须填写 base_url")
    parsed = urlparse(supplied)
    if parsed.username or parsed.password or not parsed.hostname:
        raise HTTPException(status_code=422, detail="base_url 格式无效")
    loopback_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and parsed.hostname not in loopback_hosts:
        raise HTTPException(status_code=422, detail="远程兼容接口必须使用 HTTPS")
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=422, detail="base_url 仅支持 HTTP 或 HTTPS")
    return supplied.rstrip("/")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "task_mode": settings.task_mode,
        "storage_root": str(settings.storage_root),
        "docker_available": bool(find_executable("docker")),
        "latex_available": bool(find_executable("pdflatex")),
    }


@app.get("/readiness")
def readiness(user: CurrentUser, session: SessionDep):
    models = session.exec(
        select(ModelConfig).where(ModelConfig.user_id == user.id)
    ).all()
    projects = session.exec(
        select(ResearchProject).where(ResearchProject.user_id == user.id)
    ).all()
    project_ids = [project.id for project in projects]
    completed_runs = []
    completed_manuscripts = []
    if project_ids:
        completed_runs = session.exec(
            select(ExperimentRun)
            .join(ExperimentSpec)
            .where(
                ExperimentSpec.project_id.in_(project_ids),
                ExperimentRun.status == RunStatus.COMPLETED,
            )
        ).all()
        completed_manuscripts = session.exec(
            select(ManuscriptBuild).where(
                ManuscriptBuild.project_id.in_(project_ids),
                ManuscriptBuild.status == RunStatus.COMPLETED,
            )
        ).all()
    checks = {
        "database": True,
        "storage": settings.storage_root.exists(),
        "docker": bool(find_executable("docker")),
        "latex": bool(find_executable("pdflatex")),
        "model_configured": bool(models),
        "model_reachable": any(
            model.capabilities.get("reachable") is True for model in models
        ),
        "completed_experiment": bool(completed_runs),
        "completed_manuscript": bool(completed_manuscripts),
    }
    software_checks = ("database", "storage", "docker", "latex")
    return {
        "software_ready": all(checks[name] for name in software_checks),
        "acceptance_ready": all(checks.values()),
        "checks": checks,
        "counts": {
            "projects": len(projects),
            "models": len(models),
            "completed_experiments": len(completed_runs),
            "completed_manuscripts": len(completed_manuscripts),
        },
        "next_action": (
            None
            if checks["model_reachable"]
            else "请在本机配置自己的模型密钥并执行连接测试"
        ),
    }


@app.post("/auth/register", response_model=AuthResponse)
def register(body: AuthRequest, session: SessionDep):
    user = User(email=body.email.casefold(), password_hash=hash_password(body.password))
    session.add(user)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="该邮箱已注册") from exc
    session.refresh(user)
    return AuthResponse(access_token=create_access_token(user.id))


@app.post("/auth/login", response_model=AuthResponse)
def login(body: AuthRequest, session: SessionDep):
    user = session.exec(select(User).where(User.email == body.email.casefold())).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    return AuthResponse(access_token=create_access_token(user.id))


@app.get("/providers")
def providers(_: CurrentUser):
    return [{"id": provider, "base_url": base_url} for provider, base_url in PROVIDER_PRESETS.items()]


@app.get("/models")
def list_models(user: CurrentUser, session: SessionDep):
    configs = session.exec(select(ModelConfig).where(ModelConfig.user_id == user.id)).all()
    return [
        {
            "id": item.id,
            "name": item.name,
            "provider": item.provider,
            "model": item.model,
            "base_url": item.base_url,
            "budget_limit_usd": item.budget_limit_usd,
            "spent_usd": item.spent_usd,
            "remaining_budget_usd": max(
                0.0,
                item.budget_limit_usd - item.spent_usd,
            ),
            "input_price_per_million_usd": item.input_price_per_million_usd,
            "output_price_per_million_usd": item.output_price_per_million_usd,
            "is_default": item.is_default,
            "capabilities": item.capabilities,
            "key_hint": f"••••{item.api_key_hint}" if item.api_key_hint else "已加密保存",
        }
        for item in configs
    ]


@app.post("/models", status_code=201)
def create_model(body: ModelConfigCreate, user: CurrentUser, session: SessionDep):
    if body.provider not in PROVIDER_PRESETS:
        raise HTTPException(status_code=422, detail="不支持的模型厂商")
    base_url = validated_model_base_url(body.provider, body.base_url)
    if body.is_default:
        for item in session.exec(select(ModelConfig).where(ModelConfig.user_id == user.id)).all():
            item.is_default = False
            session.add(item)
    config = ModelConfig(
        user_id=user.id,
        name=body.name,
        provider=body.provider,
        model=body.model,
        base_url=base_url,
        encrypted_api_key=encrypt_secret(body.api_key),
        api_key_hint=body.api_key[-4:],
        capabilities={"structured_output": True, "tools": True, "long_context": True},
        budget_limit_usd=body.budget_limit_usd,
        input_price_per_million_usd=body.input_price_per_million_usd,
        output_price_per_million_usd=body.output_price_per_million_usd,
        is_default=body.is_default,
    )
    session.add(config)
    session.commit()
    session.refresh(config)
    return {"id": config.id, "message": "密钥已加密保存"}


@app.post("/models/{model_id}/test")
async def test_model(model_id: UUID, user: CurrentUser, session: SessionDep):
    config = session.get(ModelConfig, model_id)
    if not config or config.user_id != user.id:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    model = LLMConfig(
        provider=config.provider,
        model=config.model,
        api_key=decrypt_secret(config.encrypted_api_key),
        base_url=config.base_url,
        budget_limit_usd=config.budget_limit_usd,
        spent_usd=config.spent_usd,
        input_price_per_million_usd=config.input_price_per_million_usd,
        output_price_per_million_usd=config.output_price_per_million_usd,
        config_id=config.id,
    )
    try:
        capabilities = await probe_model(model)
    except Exception as exc:
        capabilities = {
            "reachable": False,
            "structured_output": False,
            "error": type(exc).__name__,
        }
    persist_model_usage(session, None, model)
    config.capabilities = capabilities
    session.add(config)
    session.commit()
    return capabilities


@app.get("/projects")
def list_projects(user: CurrentUser, session: SessionDep):
    return session.exec(
        select(ResearchProject).where(ResearchProject.user_id == user.id).order_by(ResearchProject.updated_at.desc())
    ).all()


@app.post("/projects", status_code=201)
def create_project(body: ProjectCreate, user: CurrentUser, session: SessionDep):
    project = ResearchProject(
        user_id=user.id,
        title=body.title,
        direction=body.direction,
        workflow_mode=body.workflow_mode,
        target_track=body.target_track,
        topic_flexibility=body.topic_flexibility,
        human_checkpoints=body.human_checkpoints,
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    emit(session, project.id, "project", "科研项目已创建")
    session.refresh(project)
    return project


@app.get("/projects/{project_id}")
def project_detail(project_id: UUID, user: CurrentUser, session: SessionDep):
    project = own_project(session, user, project_id)
    return {
        "project": project,
        "papers": session.exec(
            select(PaperRecord)
            .where(PaperRecord.project_id == project.id)
            .order_by(PaperRecord.publication_date.desc())
            .limit(60)
        ).all(),
        "paper_versions": session.exec(
            select(PaperVersion).where(PaperVersion.project_id == project.id)
        ).all(),
        "citation_edges": session.exec(
            select(CitationEdge).where(CitationEdge.project_id == project.id)
        ).all(),
        "evidence": session.exec(select(EvidenceRecord).where(EvidenceRecord.project_id == project.id)).all(),
        "gaps": session.exec(
            select(GapCandidate)
            .where(GapCandidate.project_id == project.id)
            .order_by(GapCandidate.novelty_score.desc())
        ).all(),
        "gap_validations": session.exec(
            select(GapValidation).where(GapValidation.project_id == project.id)
        ).all(),
        "coverage_matrix": session.exec(
            select(CoverageMatrix)
            .where(CoverageMatrix.project_id == project.id)
            .order_by(CoverageMatrix.created_at.desc())
        ).first(),
        "datasets": session.exec(select(DatasetAsset).where(DatasetAsset.project_id == project.id)).all(),
        "data_preparations": session.exec(
            select(DataPreparation)
            .where(DataPreparation.project_id == project.id)
            .order_by(DataPreparation.created_at.desc())
        ).all(),
        "experiments": session.exec(select(ExperimentSpec).where(ExperimentSpec.project_id == project.id)).all(),
        "experiment_runs": session.exec(
            select(ExperimentRun)
            .join(ExperimentSpec)
            .where(ExperimentSpec.project_id == project.id)
            .order_by(ExperimentRun.started_at.desc())
        ).all(),
        "manuscripts": session.exec(select(ManuscriptBuild).where(ManuscriptBuild.project_id == project.id)).all(),
        "workflow_checkpoints": session.exec(
            select(WorkflowCheckpoint)
            .where(WorkflowCheckpoint.project_id == project.id)
            .order_by(WorkflowCheckpoint.created_at.desc())
            .limit(30)
        ).all(),
        "model_calls": session.exec(
            select(ModelCallRecord)
            .where(ModelCallRecord.project_id == project.id)
            .order_by(ModelCallRecord.created_at.desc())
            .limit(50)
        ).all(),
        "events": session.exec(
            select(TaskEvent).where(TaskEvent.project_id == project.id).order_by(TaskEvent.created_at.desc()).limit(30)
        ).all(),
    }


@app.post("/projects/{project_id}/discover", status_code=202)
async def discover(project_id: UUID, user: CurrentUser, session: SessionDep):
    project = own_project(session, user, project_id)
    if settings.task_mode == "celery":
        from .worker import discover_project_task

        discover_project_task.delay(str(project.id))
    else:
        schedule(discover_project(project.id))
    return {"message": "论文检索与空白发现已启动"}


@app.post("/projects/{project_id}/pause")
def pause_project(project_id: UUID, user: CurrentUser, session: SessionDep):
    project = own_project(session, user, project_id)
    if project.status not in {"discovering", "planning"}:
        raise HTTPException(status_code=409, detail="当前项目没有可暂停的运行流程")
    control = session.exec(
        select(WorkflowControl).where(WorkflowControl.project_id == project.id)
    ).first()
    if control is None:
        control = WorkflowControl(project_id=project.id)
    control.pause_requested = True
    control.updated_at = datetime.now(UTC)
    session.add(control)
    session.commit()
    return {"message": "已请求暂停，流程会在当前安全阶段结束后停下"}


@app.post("/projects/{project_id}/resume", status_code=202)
async def resume_project(project_id: UUID, user: CurrentUser, session: SessionDep):
    project = own_project(session, user, project_id)
    checkpoint = session.exec(
        select(WorkflowCheckpoint)
        .where(
            WorkflowCheckpoint.project_id == project.id,
            WorkflowCheckpoint.status == "paused",
        )
        .order_by(WorkflowCheckpoint.created_at.desc())
    ).first()
    if checkpoint is None:
        raise HTTPException(status_code=409, detail="当前项目没有可恢复的暂停流程")
    control = session.exec(
        select(WorkflowControl).where(WorkflowControl.project_id == project.id)
    ).first()
    if control is None:
        control = WorkflowControl(project_id=project.id)
    control.pause_requested = False
    control.updated_at = datetime.now(UTC)
    checkpoint.status = "resuming"
    checkpoint.requires_action = False
    checkpoint.updated_at = datetime.now(UTC)
    session.add(control)
    session.add(checkpoint)
    session.commit()
    if checkpoint.workflow_type == "discover":
        if settings.task_mode == "celery":
            from .worker import discover_project_task

            discover_project_task.delay(str(project.id))
        else:
            schedule(discover_project(project.id))
    else:
        selected_gap_id = checkpoint.state.get("gap_id") or project.selected_gap_id
        if selected_gap_id is None:
            raise HTTPException(status_code=409, detail="暂停记录缺少已选课题")
        gap_id = UUID(str(selected_gap_id))
        if settings.task_mode == "celery":
            from .worker import plan_selected_gap_task

            plan_selected_gap_task.delay(str(project.id), str(gap_id))
        else:
            schedule(plan_selected_gap(project.id, gap_id))
    return {"message": "流程已恢复，将从当前阶段重新执行"}


@app.post("/projects/{project_id}/select-gap", status_code=202)
async def select_gap(
    project_id: UUID,
    body: GapSelection,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    gap = session.get(GapCandidate, body.gap_id)
    if not gap or gap.project_id != project.id:
        raise HTTPException(status_code=404, detail="候选课题不存在")
    if settings.task_mode == "celery":
        from .worker import plan_selected_gap_task

        plan_selected_gap_task.delay(str(project.id), str(gap.id))
    else:
        schedule(plan_selected_gap(project.id, gap.id))
    return {"message": "已选择课题，正在寻找数据集并生成实验"}


@app.post("/projects/{project_id}/adopt-alternative-topic", status_code=202)
async def adopt_alternative_topic(
    project_id: UUID,
    body: AlternativeTopicSelection,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    source = session.get(GapCandidate, body.source_gap_id)
    if not source or source.project_id != project.id:
        raise HTTPException(status_code=404, detail="原候选课题不存在")
    alternatives = source.alternative_topics or []
    if body.alternative_index >= len(alternatives):
        raise HTTPException(status_code=404, detail="相似选题不存在")
    alternative = alternatives[body.alternative_index]
    title = str(alternative.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail="相似选题缺少标题")
    gap = GapCandidate(
        project_id=project.id,
        title=title[:240],
        hypothesis=str(
            alternative.get("why_feasible")
            or source.hypothesis
        )[:2000],
        rationale=(
            "This candidate was adopted from a ResearchFlow alternative topic. "
            f"Minimum experiment: {alternative.get('minimum_experiment', 'not specified')}. "
            f"Source candidate: {source.title}."
        ),
        confidence=max(0.45, min(source.confidence, 0.72)),
        novelty_score=max(0.45, source.novelty_score - 0.05),
        feasibility_score=max(source.feasibility_score, 0.82),
        estimated_cost=source.estimated_cost,
        risks=[
            *source.risks[:4],
            "Adopted alternative still requires fresh dataset fit and reverse-search review.",
        ],
        evidence_ids=source.evidence_ids,
        counter_queries=[
            f'"{title}" dataset benchmark',
            f'"{title}" reproducible baseline',
        ],
        submission_readiness={},
        alternative_topics=[],
    )
    session.add(gap)
    session.commit()
    session.refresh(gap)
    if settings.task_mode == "celery":
        from .worker import plan_selected_gap_task

        plan_selected_gap_task.delay(str(project.id), str(gap.id))
    else:
        schedule(plan_selected_gap(project.id, gap.id))
    return {
        "message": "已采用相似选题，正在重新寻找数据集并生成实验",
        "gap_id": str(gap.id),
    }


@app.post("/projects/{project_id}/papers/upload")
async def upload_paper(
    project_id: UUID,
    user: CurrentUser,
    session: SessionDep,
    rights_confirmed: bool = Form(...),
    file: UploadFile = File(...),
):
    project = own_project(session, user, project_id)
    if not rights_confirmed:
        raise HTTPException(status_code=422, detail="必须确认拥有合法使用权限")
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="仅支持 PDF")
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF 不得超过 50MB")
    digest, path = content_store.put_bytes(content, ".pdf")
    try:
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:30])
    except Exception:
        text = ""
    evidence = EvidenceRecord(
        project_id=project.id,
        evidence_type="uploaded_pdf",
        claim=f"用户合法上传：{file.filename}",
        excerpt=text[:2000],
        locator=str(path),
        content_hash=digest,
    )
    session.add(evidence)
    session.commit()
    return {"id": evidence.id, "content_hash": digest, "text_extracted": bool(text)}


@app.get("/projects/{project_id}/papers/semantic-search")
def search_project_papers(
    project_id: UUID,
    user: CurrentUser,
    session: SessionDep,
    q: str = Query(min_length=2, max_length=500),
    limit: int = Query(default=12, ge=1, le=50),
):
    project = own_project(session, user, project_id)
    return semantic_search(session, project.id, q, limit)


@app.post("/projects/{project_id}/run-experiment")
async def run_experiment(
    project_id: UUID,
    body: ExperimentRunRequest,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    spec = session.exec(
        select(ExperimentSpec).where(ExperimentSpec.project_id == project.id).order_by(ExperimentSpec.created_at.desc())
    ).first()
    if not spec or not spec.artifact_path:
        raise HTTPException(status_code=409, detail="尚未生成实验包")
    if body.allow_network:
        raise HTTPException(status_code=422, detail="首版沙箱不允许联网执行")
    run = ExperimentRun(
        spec_id=spec.id,
        status=RunStatus.RUNNING,
        executor=body.executor,
    )
    run.started_at = datetime.now(UTC)
    session.add(run)
    session.commit()
    session.refresh(run)
    status, results = await asyncio.to_thread(
        execute_experiment,
        body.executor,
        Path(spec.artifact_path),
        body.timeout_seconds,
    )
    run.status = RunStatus(status)
    run.results = results
    if status == "completed":
        run_audit = audit_completed_run(spec, results)
        run.validity_audit = run_audit.as_dict()
        run.quality_level = run_audit.level
        results["validity_audit"] = run_audit.as_dict()
    else:
        run.validity_audit = {
            "passed": False,
            "level": "concept_draft",
            "findings": ["Experiment did not complete successfully."],
        }
    run.finished_at = datetime.now(UTC)
    run.logs_path = str(Path(spec.artifact_path) / "runtime")
    session.add(run)
    session.commit()
    emit(session, project.id, "experiment", f"实验状态：{status}", **results)
    session.refresh(run)
    return run


@app.post("/projects/{project_id}/confirm-dataset-validity")
async def confirm_dataset_validity(
    project_id: UUID,
    body: DatasetValidityConfirmation,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    dataset = session.get(DatasetAsset, body.dataset_id)
    if not dataset or dataset.project_id != project.id:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not body.confirmed:
        raise HTTPException(status_code=422, detail="Confirmation must be explicit.")
    preparation = session.exec(
        select(DataPreparation)
        .where(
            DataPreparation.project_id == project.id,
            DataPreparation.dataset_id == dataset.id,
            DataPreparation.status == RunStatus.COMPLETED,
        )
        .order_by(DataPreparation.created_at.desc())
    ).first()
    gap = session.get(GapCandidate, project.selected_gap_id) if project.selected_gap_id else None
    if not preparation or not gap:
        raise HTTPException(status_code=409, detail="Prepared data or selected gap is missing.")
    dataset.human_confirmed = True
    dataset.validity_audit = {
        **dataset.validity_audit,
        "human_confirmation": {
            "confirmed": True,
            "reason": body.reason,
            "confirmed_at": datetime.now(UTC).isoformat(),
        },
    }
    session.add(dataset)
    session.commit()
    spec = await continue_confirmed_dataset(session, project, gap, dataset, preparation)
    checkpoint = session.exec(
        select(WorkflowCheckpoint)
        .where(
            WorkflowCheckpoint.project_id == project.id,
            WorkflowCheckpoint.stage == "dataset_validity",
            WorkflowCheckpoint.status == "awaiting_human",
        )
        .order_by(WorkflowCheckpoint.created_at.desc())
    ).first()
    if checkpoint:
        checkpoint.status = "completed"
        checkpoint.requires_action = False
        checkpoint.updated_at = datetime.now(UTC)
        session.add(checkpoint)
        session.commit()
    emit(
        session,
        project.id,
        "dataset_validity",
        "Human accepted the dataset mismatch risk; experiment is labeled accordingly.",
        reason=body.reason,
    )
    return spec


@app.post("/projects/{project_id}/manuscript")
async def manuscript(
    project_id: UUID,
    body: ManuscriptRequest,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    if not project.selected_gap_id:
        raise HTTPException(status_code=409, detail="请先选择研究空白候选")
    gap = session.get(GapCandidate, project.selected_gap_id)
    papers = session.exec(
        select(PaperRecord).where(PaperRecord.project_id == project.id).order_by(PaperRecord.citation_count.desc())
    ).all()
    if not gap or not papers:
        raise HTTPException(status_code=409, detail="证据不足，无法生成稿件")
    latest_spec = session.exec(
        select(ExperimentSpec)
        .where(ExperimentSpec.project_id == project.id)
        .order_by(ExperimentSpec.created_at.desc())
    ).first()
    completed_run = (
        session.exec(
            select(ExperimentRun)
            .where(
                ExperimentRun.spec_id == latest_spec.id,
                ExperimentRun.status == RunStatus.COMPLETED,
            )
            .order_by(ExperimentRun.finished_at.desc())
        ).first()
        if latest_spec
        else None
    )
    if body.mode == "submission" and not completed_run:
        raise HTTPException(
            status_code=409,
            detail="结果稿必须基于最新实验规格的已完成运行",
        )
    gate_audit = None
    if body.mode == "submission":
        if not latest_spec or not completed_run:
            raise HTTPException(status_code=409, detail="A completed audited experiment is required.")
        gate_audit = submission_gate(
            latest_spec,
            completed_run.validity_audit,
            len(papers),
        )
        if not gate_audit.passed:
            datasets = session.exec(
                select(DatasetAsset).where(DatasetAsset.project_id == project.id)
            ).all()
            gap.alternative_topics = build_similar_feasible_topics(
                project.direction,
                gap,
                datasets,
                gate_audit.findings,
            )
            gap.submission_readiness = {
                **(gap.submission_readiness or {}),
                "passed": False,
                "level": "submission_blocked",
                "findings": gate_audit.findings,
                "details": {
                    **((gap.submission_readiness or {}).get("details") or {}),
                    "alternatives": gap.alternative_topics,
                },
            }
            session.add(gap)
            session.commit()
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Submission manuscript is blocked by scientific validity gates.",
                    "quality_level": gate_audit.level,
                    "findings": gate_audit.findings,
                    "alternative_topics": gap.alternative_topics,
                },
            )
        if body.target in {"ieee_conference", "elsevier_journal"}:
            if not body.publication_name or not body.author_guide_url:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "A specific publication name and its HTTPS author guide "
                            "are required for a submission package."
                        ),
                        "findings": [
                            "Generic publisher templates cannot prove compliance with a specific venue.",
                            "Select the exact conference or journal before generating submission mode.",
                        ],
                    },
                )
            if not body.author_guide_url.startswith("https://"):
                raise HTTPException(
                    status_code=409,
                    detail="Author guide URL must use HTTPS.",
                )
            if (
                not body.venue_human_verified
                or not body.venue_evidence_url
                or not body.venue_evidence_url.startswith("https://")
                or not body.venue_claim
                or not body.venue_verified_on
            ):
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Submission target verification is incomplete.",
                        "findings": [
                            "Record an HTTPS indexing or quartile evidence URL.",
                            "Select the claimed SCI/EI category and verification date.",
                            "Confirm that a human checked the current source and subject category.",
                        ],
                    },
                )
    experiment_results = (
        {
            **completed_run.results,
            "_scientific_plan": latest_spec.scientific_plan if latest_spec else {},
            "_experiment_name": latest_spec.name if latest_spec else None,
        }
        if completed_run
        else None
    )
    model_config = default_model_config(session, project.user_id)
    try:
        draft = await generate_manuscript(
            project,
            gap,
            papers,
            experiment_results,
            body.target,
            body.mode,
            model_config,
        )
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
    except Exception as model_error:
        persist_model_usage(session, project.id, model_config)
        emit(
            session,
            project.id,
            "model",
            "模型稿件未通过事实约束，已回退到证据模板",
            error=type(model_error).__name__,
        )
        draft = await generate_manuscript(
            project,
            gap,
            papers,
            experiment_results,
            body.target,
            body.mode,
            None,
        )
    try:
        root, compiled, citation_keys = build_manuscript(
            project,
            gap,
            papers,
            body.target,
            draft=draft,
            experiment_results=experiment_results,
            experiment_root=Path(latest_spec.artifact_path) if latest_spec and latest_spec.artifact_path else None,
            quality_level=(
                gate_audit.level
                if gate_audit
                else (completed_run.quality_level if completed_run else "concept_draft")
            ),
            publication_name=body.publication_name,
            author_guide_url=body.author_guide_url,
            venue_profile={
                "publication_name": body.publication_name,
                "claim": body.venue_claim,
                "evidence_url": body.venue_evidence_url,
                "verified_on": body.venue_verified_on,
                "human_verified": body.venue_human_verified,
                "warning": (
                    "Quartile and indexing status are time-, database-, and category-dependent; "
                    "this record is a human attestation, not an acceptance guarantee."
                ),
            },
        )
    except TemplateUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    pre_submission_review = json.loads(
        (root / "pre-submission-review.json").read_text(encoding="utf-8")
    )
    base_validity_audit = gate_audit.as_dict() if gate_audit else (
        completed_run.validity_audit if completed_run else {}
    )
    validity_audit = {
        **base_validity_audit,
        "pre_submission_review": pre_submission_review,
        "manuscript_compilation": {
            "passed": compiled,
            "status": "pdf_completed" if compiled else "pdf_not_compiled",
            "message": (
                "LaTeX, BibTeX, and PDF were generated."
                if compiled
                else "LaTeX and BibTeX were generated, but PDF compilation was unavailable or failed."
            ),
        },
    }
    review_passed = body.mode != "submission" or pre_submission_review["passed"]
    build_quality_level = (
        gate_audit.level
        if gate_audit
        else (completed_run.quality_level if completed_run else "concept_draft")
    )
    if body.mode == "submission" and not review_passed:
        build_quality_level = "reproducible_research"
        review_findings = [
            item["message"] for item in pre_submission_review["findings"]
        ]
        datasets = session.exec(
            select(DatasetAsset).where(DatasetAsset.project_id == project.id)
        ).all()
        gap.alternative_topics = build_similar_feasible_topics(
            project.direction,
            gap,
            datasets,
            review_findings,
        )
        gap.submission_readiness = {
            **(gap.submission_readiness or {}),
            "passed": False,
            "level": "submission_review_failed",
            "findings": review_findings,
            "details": {
                **((gap.submission_readiness or {}).get("details") or {}),
                "alternatives": gap.alternative_topics,
            },
        }
        session.add(gap)
    build = ManuscriptBuild(
        project_id=project.id,
        target=body.target,
        status=RunStatus.COMPLETED if compiled and review_passed else RunStatus.BLOCKED,
        artifact_path=str(root),
        citation_keys=citation_keys,
        mode=body.mode,
        quality_level=build_quality_level,
        validity_audit=validity_audit,
    )
    session.add(build)
    session.commit()
    session.refresh(build)
    emit(
        session,
        project.id,
        "manuscript",
        "LaTeX 与 BibTeX 已生成" + ("，PDF 编译完成" if compiled else "，本机缺少 LaTeX 编译器"),
    )
    session.refresh(build)
    return build


@app.get("/projects/{project_id}/artifacts/{kind}")
def download_artifact(
    project_id: UUID,
    kind: str,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    if kind == "experiment":
        spec = session.exec(
            select(ExperimentSpec)
            .where(ExperimentSpec.project_id == project.id)
            .order_by(ExperimentSpec.created_at.desc())
        ).first()
        root = Path(spec.artifact_path) if spec and spec.artifact_path else None
    elif kind == "manuscript":
        build = session.exec(
            select(ManuscriptBuild)
            .where(ManuscriptBuild.project_id == project.id)
            .order_by(ManuscriptBuild.created_at.desc())
        ).first()
        root = Path(build.artifact_path) if build and build.artifact_path else None
    else:
        raise HTTPException(status_code=404, detail="未知产物类型")
    if not root or not root.exists():
        raise HTTPException(status_code=404, detail="产物尚未生成")
    archive_base = settings.storage_root / "exports" / f"{project.id}-{kind}"
    archive_base.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(archive_base), "zip", root)
    return FileResponse(archive, filename=Path(archive).name)


@app.get("/projects/{project_id}/events")
async def event_stream(
    project_id: UUID,
    user: CurrentUser,
    session: SessionDep,
):
    project = own_project(session, user, project_id)
    user_id = project.user_id

    async def stream():
        from sqlmodel import Session

        from .db import engine

        sent_ids: set[str] = set()
        initialized = False
        for _ in range(180):
            with Session(engine) as session:
                project = session.get(ResearchProject, project_id)
                if not project or project.user_id != user_id:
                    yield "event: error\ndata: unauthorized\n\n"
                    return
                events = session.exec(
                    select(TaskEvent).where(TaskEvent.project_id == project_id).order_by(TaskEvent.created_at)
                ).all()
                if not initialized:
                    sent_ids.update(str(event.id) for event in events)
                    initialized = True
                for event in events:
                    event_id = str(event.id)
                    if event_id in sent_ids:
                        continue
                    payload = event.model_dump(mode="json")
                    yield f"id: {event_id}\n"
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    sent_ids.add(event_id)
            yield ": heartbeat\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(stream(), media_type="text/event-stream")
