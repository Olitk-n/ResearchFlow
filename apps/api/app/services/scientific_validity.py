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
    if preparation.row_count < 20:
        findings.append("Prepared sample has fewer than 20 rows.")
    target_candidates = [
        name
        for name, descriptor in preparation.schema_json.items()
        if any(kind in (descriptor.get("types") or {}) for kind in ("int", "float"))
    ]
    if not target_candidates:
        findings.append("No numeric target candidate was found in the prepared schema.")
    passed = bool(dataset.license and matched and target_candidates)
    return AuditResult(
        passed=passed,
        score=score,
        level="initial_experiment" if passed else "concept_draft",
        findings=findings,
        details={
            "matched_topic_terms": matched,
            "target_candidates": target_candidates,
            "sample_unit": "one prepared JSONL row",
            "human_confirmation_required": not passed,
        },
    )


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
    synthetic = bool((spec.validity_audit or {}).get("details", {}).get("synthetic"))
    level = "synthetic_demonstration" if synthetic else "initial_experiment"
    passed = not findings
    if passed and len(result_seeds) >= 3 and payload.get("uncertainty") and plan.get("baselines"):
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
