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
from .db import SessionDep, create_db_and_tables
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
    AuthRequest,
    AuthResponse,
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
from .services.venue_templates import TemplateUnavailable
from .services.workflow import (
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
    run.finished_at = datetime.now(UTC)
    run.logs_path = str(Path(spec.artifact_path) / "runtime")
    session.add(run)
    session.commit()
    emit(session, project.id, "experiment", f"实验状态：{status}", **results)
    session.refresh(run)
    return run


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
    completed_run = session.exec(
        select(ExperimentRun)
        .join(ExperimentSpec)
        .where(
            ExperimentSpec.project_id == project.id,
            ExperimentRun.status == RunStatus.COMPLETED,
        )
        .order_by(ExperimentRun.finished_at.desc())
    ).first()
    if body.mode == "submission" and not completed_run:
        raise HTTPException(
            status_code=409,
            detail="结果稿必须基于已完成的实验运行",
        )
    experiment_results = completed_run.results if completed_run else None
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
        )
    except TemplateUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    build = ManuscriptBuild(
        project_id=project.id,
        target=body.target,
        status=RunStatus.COMPLETED if compiled else RunStatus.BLOCKED,
        artifact_path=str(root),
        citation_keys=citation_keys,
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
