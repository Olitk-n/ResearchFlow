import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlmodel import Session, select

from ..models import (
    DataPreparation,
    DatasetAsset,
    ExperimentRun,
    ExperimentSpec,
    GapCandidate,
    ManuscriptBuild,
    ResearchProject,
)

SYNTHETIC_PATTERNS = (
    r"random\.(uniform|gauss|normalvariate|random)",
    r"simulate true",
    r"simulated? (label|target|truth|value)",
    r"synthetic (label|target|truth|value)",
)
MIN_SUBMISSION_ROWS = 200
REQUIRED_SUBMISSION_RESULT_FIELDS = (
    "primary_metric",
    "per_seed_metrics",
    "baseline_metrics",
    "uncertainty",
    "effect_size",
    "statistical_test",
    "ablation_results",
)


@dataclass(slots=True)
class AuditResult:
    passed: bool
    score: float
    level: str
    findings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 3),
            "level": self.level,
            "findings": self.findings,
            "details": self.details,
        }


def _terms(text: str) -> set[str]:
    stop = {
        "using", "based", "study", "evaluation", "evaluating", "analysis",
        "model", "models", "data", "dataset", "machine", "learning",
        "can", "for", "with", "from", "into", "identified",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) >= 3 and token not in stop
    }


def audit_dataset_fit(
    project: ResearchProject,
    gap: GapCandidate,
    dataset: DatasetAsset,
    preparation: DataPreparation,
) -> AuditResult:
    research_terms = _terms(f"{project.direction} {gap.title} {gap.hypothesis}")
    schema_text = json.dumps(preparation.schema_json, ensure_ascii=False)
    dataset_text = " ".join(
        [
            dataset.name,
            dataset.external_id,
            dataset.quality_notes,
            schema_text[:20_000],
        ]
    ).casefold()
    matched = sorted(term for term in research_terms if term in dataset_text)
    score = len(matched) / max(4, min(len(research_terms), 12))
    findings = []
    if not dataset.license:
        findings.append("Dataset license is missing.")
    if not matched:
        findings.append("Prepared data fields and examples do not match the research topic.")
    if preparation.row_count < MIN_SUBMISSION_ROWS:
        findings.append(
            f"Prepared data has fewer than {MIN_SUBMISSION_ROWS} rows; "
            "it is insufficient for the default submission track."
        )
    complete_snapshot = preparation.profile_json.get("complete_snapshot")
    if complete_snapshot is False:
        findings.append(
            "Prepared data is a capped snapshot rather than the complete split; "
            "a preregistered sampling design is required."
        )
    target_candidates = [
        name
        for name, descriptor in preparation.schema_json.items()
        if any(kind in (descriptor.get("types") or {}) for kind in ("int", "float"))
    ]
    if not target_candidates:
        findings.append("No numeric target candidate was found in the prepared schema.")
    passed = bool(
        dataset.license
        and matched
        and target_candidates
        and preparation.row_count >= MIN_SUBMISSION_ROWS
        and complete_snapshot is not False
    )
    return AuditResult(
        passed=passed,
        score=score,
        level="initial_experiment" if passed else "concept_draft",
        findings=findings,
        details={
            "matched_topic_terms": matched,
            "target_candidates": target_candidates,
            "sample_unit": "one prepared JSONL row",
            "prepared_rows": preparation.row_count,
            "source_total_rows": preparation.profile_json.get("source_total_rows"),
            "complete_snapshot": preparation.profile_json.get("complete_snapshot", False),
            "human_confirmation_required": not passed,
        },
    )


def assess_topic_submission_readiness(
    project: ResearchProject,
    gap: GapCandidate,
    datasets: list[DatasetAsset],
    preparation: DataPreparation | None = None,
) -> AuditResult:
    usable = [
        item for item in datasets
        if item.license and int(item.metadata_json.get("relevance_score") or 0) >= 2
    ]
    findings: list[str] = []
    if not usable:
        findings.append("No clearly licensed, topic-matched dataset was found.")
    if preparation is None:
        findings.append("No dataset snapshot has been prepared and inspected.")
    elif preparation.row_count < MIN_SUBMISSION_ROWS:
        findings.append(
            f"Only {preparation.row_count} usable rows were prepared; "
            f"the default submission track requires at least {MIN_SUBMISSION_ROWS}."
        )
    elif preparation.profile_json.get("complete_snapshot") is False:
        findings.append(
            "The prepared data is only a capped snapshot; use the complete split "
            "or register a defensible sampling design."
        )
    if gap.feasibility_score < 0.65:
        findings.append("The candidate feasibility score is below the submission planning threshold.")
    passed = not findings
    alternatives = build_similar_feasible_topics(project.direction, gap, usable)
    return AuditResult(
        passed=passed,
        score=min(1.0, gap.feasibility_score * (1.0 if usable else 0.45)),
        level="submission_plannable" if passed else "topic_revision_required",
        findings=findings,
        details={
            "usable_dataset_count": len(usable),
            "prepared_rows": preparation.row_count if preparation else 0,
            "alternatives": alternatives,
            "recommended_targets": recommend_submission_targets(
                gap,
                preparation.row_count if preparation else 0,
                baseline_count=0,
                real_task=False,
            ),
            "meaning": (
                "This is a planning assessment, not a guarantee of acceptance or publication."
            ),
        },
    )


def build_similar_feasible_topics(
    direction: str,
    gap: GapCandidate,
    usable_datasets: list[DatasetAsset],
    blockers: list[str] | None = None,
) -> list[dict[str, Any]]:
    dataset_name = usable_datasets[0].name if usable_datasets else "a licensed public benchmark"
    blocker_text = "；".join(blockers or []) or "当前选题缺少足够的投稿级实验条件"
    return [
        {
            "title": f"{direction}的可复现基线与失败模式复核",
            "why_feasible": (
                f"当前阻碍是：{blocker_text}。可围绕 {dataset_name} 复现公开基线，"
                "把贡献收缩为可靠复核、误差分析和可复现性证据。"
            ),
            "minimum_experiment": "至少2个可信基线、3个随机种子、独立测试集、95%置信区间和分层失败分析。",
            "suggested_track": "EI应用型会议；证据充分后可评估SCI四区应用期刊",
            "addresses": blockers or [],
        },
        {
            "title": f"{direction}在预算约束下的性能、成本与延迟权衡",
            "why_feasible": "无需声称发明全新模型，真实运行日志可直接支持多目标实证结论。",
            "minimum_experiment": "至少3种方法或配置，在统一预算下报告性能、成本、延迟、显著性和效应量。",
            "suggested_track": "EI工程会议或SCI三区/四区应用计算期刊",
            "addresses": blockers or [],
        },
        {
            "title": f"{direction}的数据质量敏感性与稳健性分析",
            "why_feasible": (
                f"可使用 {dataset_name} 的真实字段构造缺失、噪声或分布切片，"
                "不生成虚假真值，并通过预注册扰动验证稳健性。"
            ),
            "minimum_experiment": "真实标签、至少3档预注册扰动、2个基线、多种子和分层误差分析。",
            "suggested_track": "EI数据工程会议或SCI四区实证研究期刊",
            "addresses": blockers or [],
        },
    ]


def recommend_submission_targets(
    gap: GapCandidate,
    prepared_rows: int,
    baseline_count: int,
    real_task: bool,
) -> list[dict[str, Any]]:
    targets = []
    if real_task and prepared_rows >= MIN_SUBMISSION_ROWS and baseline_count >= 1:
        targets.append({
            "track": "EI conference / applied computing conference",
            "fit": "reasonable",
            "requirements": [
                "clear engineering contribution",
                "at least one credible baseline",
                "independent test results with uncertainty",
                "reproducible code and data statement",
            ],
            "warning": "Indexing and acceptance depend on the exact venue and year; verify before submission.",
        })
    if (
        real_task
        and prepared_rows >= 500
        and baseline_count >= 2
        and gap.novelty_score >= 0.65
    ):
        targets.append({
            "track": "SCI Q3/Q4 applied AI journal",
            "fit": "possible after expert review",
            "requirements": [
                "multiple competitive baselines",
                "ablation or sensitivity analysis",
                "statistical significance and effect size",
                "strong domain interpretation and limitations",
            ],
            "warning": (
                "A quartile is category- and year-dependent; the system does not "
                "guarantee indexing or acceptance."
            ),
        })
    if not targets:
        targets.append({
            "track": "workshop / preprint / pilot study",
            "fit": "current evidence level",
            "requirements": [
                "increase task sample size",
                "add competitive baselines",
                "complete multi-seed uncertainty analysis",
            ],
            "warning": "Current evidence is not sufficient for a default SCI/EI recommendation.",
        })
    return targets


def audit_experiment_code(code: str, scientific_plan: dict[str, Any]) -> AuditResult:
    findings = []
    lowered = code.casefold()
    synthetic = any(re.search(pattern, lowered) for pattern in SYNTHETIC_PATTERNS)
    if synthetic:
        findings.append("Code generates simulated or random ground-truth values.")
    required = {
        "field_mapping": scientific_plan.get("field_mapping"),
        "target_variable": scientific_plan.get("target_variable"),
        "model": scientific_plan.get("model"),
        "split_strategy": scientific_plan.get("split_strategy"),
        "baselines": scientific_plan.get("baselines"),
        "metric_definitions": scientific_plan.get("metric_definitions"),
        "statistical_analysis": scientific_plan.get("statistical_analysis"),
        "seeds": scientific_plan.get("seeds"),
    }
    missing = sorted(key for key, value in required.items() if not value)
    if missing:
        findings.append(f"Scientific plan is incomplete: {', '.join(missing)}.")
    try:
        ast.parse(code)
    except SyntaxError:
        findings.append("Experiment code is not valid Python.")
    level = "synthetic_demonstration" if synthetic else "initial_experiment"
    return AuditResult(
        passed=not missing,
        score=0.25 if synthetic else (1.0 if not missing else 0.4),
        level=level,
        findings=findings,
        details={"synthetic": synthetic, "missing_plan_fields": missing},
    )


def audit_completed_run(spec: ExperimentSpec, results: dict[str, Any]) -> AuditResult:
    plan = spec.scientific_plan or {}
    findings = []
    payload = results
    if not isinstance(payload.get("metrics"), dict) and isinstance(results.get("stdout"), str):
        try:
            payload = {**results, **json.loads(results["stdout"])}
        except json.JSONDecodeError:
            pass
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        findings.append("Run did not emit a non-empty metrics object.")
    expected_rows = plan.get("expected_sample_count")
    observed_rows = payload.get("num_samples", payload.get("sample_rows"))
    if expected_rows and observed_rows != expected_rows:
        findings.append(f"Sample count mismatch: expected {expected_rows}, observed {observed_rows}.")
    expected_k = plan.get("parameters", {}).get("k")
    observed_k = payload.get("parameters", {}).get("k")
    if expected_k is not None and observed_k != expected_k:
        findings.append(f"Parameter k mismatch: expected {expected_k}, observed {observed_k}.")
    seeds = plan.get("seeds") or []
    result_seeds = payload.get("seeds") or ([payload["seed"]] if "seed" in payload else [])
    if seeds and sorted(seeds) != sorted(result_seeds):
        findings.append("Run seeds do not match the registered scientific plan.")
    per_seed = payload.get("per_seed_metrics")
    if len(seeds) >= 3 and (not isinstance(per_seed, list) or len(per_seed) < len(seeds)):
        findings.append("Run did not emit one metric record per registered seed.")
    uncertainty = payload.get("uncertainty")
    if plan.get("statistical_analysis") and not isinstance(uncertainty, dict):
        findings.append("Run did not emit structured uncertainty estimates.")
    baseline_metrics = payload.get("baseline_metrics")
    if plan.get("baselines") and not isinstance(baseline_metrics, dict):
        findings.append("Run did not emit baseline metrics.")
    primary_metric = payload.get("primary_metric")
    if plan.get("evidence_class") == "real_task":
        missing_protocol = [
            field for field in REQUIRED_SUBMISSION_RESULT_FIELDS
            if not payload.get(field)
        ]
        if missing_protocol:
            findings.append(
                "Submission result protocol is incomplete: "
                + ", ".join(missing_protocol)
                + "."
            )
        if isinstance(primary_metric, dict):
            if not {"name", "value", "direction"} <= set(primary_metric):
                findings.append(
                    "Primary metric must include name, value, and direction."
                )
        else:
            findings.append("Primary metric must be a structured object.")
        statistical_test = payload.get("statistical_test")
        if isinstance(statistical_test, dict):
            if not {"name", "p_value"} <= set(statistical_test):
                findings.append(
                    "Statistical test must include name and p_value."
                )
            else:
                p_value = statistical_test.get("p_value")
                if (
                    not isinstance(p_value, (int, float))
                    or isinstance(p_value, bool)
                    or not 0 <= p_value <= 1
                ):
                    findings.append("Statistical-test p_value must be between 0 and 1.")
        effect_size = payload.get("effect_size")
        if isinstance(effect_size, dict) and not {"name", "value"} <= set(effect_size):
            findings.append("Effect size must include name and value.")
        ablations = payload.get("ablation_results")
        if not isinstance(ablations, list) or not ablations:
            findings.append("At least one structured ablation or sensitivity result is required.")
        elif any(
            not isinstance(item, dict)
            or not {"name", "metric", "value", "interpretation"} <= set(item)
            for item in ablations
        ):
            findings.append(
                "Each ablation must include name, metric, value, and interpretation."
            )
        if isinstance(uncertainty, dict):
            lower = uncertainty.get("lower")
            upper = uncertainty.get("upper")
            confidence = uncertainty.get("confidence")
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (lower, upper, confidence)
            ):
                findings.append(
                    "Uncertainty must include numeric lower, upper, and confidence."
                )
            elif lower > upper or not 0 < confidence < 1:
                findings.append("Uncertainty interval or confidence level is invalid.")
    synthetic = bool((spec.validity_audit or {}).get("details", {}).get("synthetic"))
    level = "synthetic_demonstration" if synthetic else "initial_experiment"
    passed = not findings
    if passed and len(result_seeds) >= 3 and uncertainty and baseline_metrics:
        level = "reproducible_research"
    return AuditResult(
        passed=passed,
        score=1.0 if passed else 0.3,
        level=level,
        findings=findings,
        details={
            "observed_sample_count": observed_rows,
            "observed_seeds": result_seeds,
            "synthetic": synthetic,
            "submission_protocol_fields": {
                field: bool(payload.get(field))
                for field in REQUIRED_SUBMISSION_RESULT_FIELDS
            },
        },
    )


def submission_gate(
    spec: ExperimentSpec,
    run_audit: dict[str, Any],
    citation_count: int,
) -> AuditResult:
    plan = spec.scientific_plan or {}
    findings = []
    if not run_audit.get("passed"):
        findings.append("Completed run failed the post-run consistency audit.")
    if run_audit.get("level") == "synthetic_demonstration":
        findings.append("Synthetic demonstrations cannot be promoted to submission manuscripts.")
    if not plan.get("baselines"):
        findings.append("At least one baseline is required.")
    if len(plan.get("seeds") or []) < 3:
        findings.append("At least three seeds are required.")
    if not plan.get("statistical_analysis"):
        findings.append("A statistical uncertainty analysis is required.")
    if plan.get("evidence_class") != "real_task":
        findings.append("Submission requires a real task with measured inputs and targets.")
    if not plan.get("target_variable") or str(plan.get("target_variable")).casefold().startswith("none"):
        findings.append("A measured target variable is required.")
    split_strategy = str(plan.get("split_strategy") or "").casefold()
    if not any(term in split_strategy for term in ("test", "held-out", "held out", "cross-validation")):
        findings.append("An independent test or cross-validation strategy is required.")
    expected_rows = int(plan.get("expected_sample_count") or 0)
    if expected_rows < MIN_SUBMISSION_ROWS:
        findings.append(f"At least {MIN_SUBMISSION_ROWS} prepared task rows are required by the default gate.")
    if citation_count < 3:
        findings.append("At least three resolved literature citations are required.")
    passed = not findings
    return AuditResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        level="submission_candidate" if passed else run_audit.get("level", "concept_draft"),
        findings=findings,
        details={"citation_count": citation_count},
    )


def build_claim_provenance(
    sections: dict[str, str],
    citation_keys: list[str],
    experiment_results: dict[str, Any] | None,
    evidence_ids: list[str],
) -> list[dict[str, Any]]:
    claims = []
    result_path = (experiment_results or {}).get("artifact_path")
    result_hash = (experiment_results or {}).get("artifact_sha256")
    for section, text in sections.items():
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
            if not sentence:
                continue
            cited = [
                citation_keys[int(index) - 1]
                for index in re.findall(r"\[paper(\d+)\]", sentence)
                if 0 < int(index) <= len(citation_keys)
            ]
            numbers = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", sentence)
            source = {"type": "unresolved"}
            if cited:
                source = {"type": "literature", "citation_keys": cited, "evidence_ids": evidence_ids}
            elif result_path and (numbers or section in {"method", "results"}):
                source = {
                    "type": "experiment",
                    "artifact_path": result_path,
                    "artifact_sha256": result_hash,
                    "reported_values": numbers,
                }
            claims.append({"section": section, "claim": sentence, "source": source})
    return claims


def copy_reproducibility_bundle(manuscript_root: Path, experiment_root: Path | None) -> None:
    if not experiment_root or not experiment_root.exists():
        return
    target = manuscript_root / "reproducibility"
    target.mkdir(parents=True, exist_ok=True)
    for relative in (
        "run.py", "manifest.json", "pyproject.toml", "uv.lock", "Dockerfile",
        "data/data-card.json", "artifact-index.json", "runtime/results.json",
        "runtime/stdout.log", "runtime/stderr.log", "runtime/artifact-index.json",
    ):
        source = experiment_root / relative
        if source.is_file():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())


def backfill_scientific_validity(session: Session) -> None:
    for gap in session.exec(select(GapCandidate)).all():
        project = session.get(ResearchProject, gap.project_id)
        datasets = session.exec(
            select(DatasetAsset).where(
                DatasetAsset.project_id == gap.project_id,
                DatasetAsset.gap_id == gap.id,
            )
        ).all()
        preparation = None
        if datasets:
            preparation = session.exec(
                select(DataPreparation)
                .where(DataPreparation.dataset_id.in_([item.id for item in datasets]))
                .order_by(DataPreparation.created_at.desc())
            ).first()
        if project:
            readiness = assess_topic_submission_readiness(
                project, gap, datasets, preparation,
            )
            gap.submission_readiness = readiness.as_dict()
            gap.alternative_topics = readiness.details["alternatives"]
            session.add(gap)

    specs = session.exec(select(ExperimentSpec)).all()
    for spec in specs:
        dataset_id = (spec.resource_profile or {}).get("dataset_id")
        preparation_id = (spec.resource_profile or {}).get("dataset_preparation_id")
        project = session.get(ResearchProject, spec.project_id)
        gap = session.get(GapCandidate, spec.gap_id)
        preparation = (
            session.get(DataPreparation, UUID(str(preparation_id)))
            if preparation_id
            else None
        )
        dataset = (
            session.get(DatasetAsset, UUID(str(dataset_id)))
            if dataset_id
            else None
        )
        if dataset is None and preparation is not None:
            dataset = session.get(DatasetAsset, preparation.dataset_id)
        if project and gap and dataset and preparation:
            dataset.validity_audit = audit_dataset_fit(
                project, gap, dataset, preparation,
            ).as_dict()
            session.add(dataset)
        root = Path(spec.artifact_path or "")
        code_path = root / "run.py"
        code = code_path.read_text(encoding="utf-8") if code_path.is_file() else ""
        plan = spec.scientific_plan or {}
        if not plan:
            manifest_path = root / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                plan = manifest.get("experiment", {}).get("scientific_plan") or {}
        audit = audit_experiment_code(code, plan)
        spec.scientific_plan = plan
        spec.validity_audit = audit.as_dict()
        spec.quality_level = audit.level
        session.add(spec)
        for run in session.exec(select(ExperimentRun).where(ExperimentRun.spec_id == spec.id)).all():
            if str(run.status) == "completed" or getattr(run.status, "value", None) == "completed":
                run_audit = audit_completed_run(spec, run.results)
                run.validity_audit = run_audit.as_dict()
                run.quality_level = run_audit.level
                session.add(run)
        latest_level = audit.level
        completed = session.exec(
            select(ExperimentRun)
            .where(ExperimentRun.spec_id == spec.id)
            .order_by(ExperimentRun.finished_at.desc())
        ).first()
        if completed:
            latest_level = completed.quality_level
        for build in session.exec(
            select(ManuscriptBuild).where(ManuscriptBuild.project_id == spec.project_id)
        ).all():
            build.quality_level = latest_level
            build.validity_audit = completed.validity_audit if completed else audit.as_dict()
            session.add(build)
    session.commit()
