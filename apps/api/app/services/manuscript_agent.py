import json
import re
from dataclasses import dataclass
from typing import Any

from ..models import GapCandidate, PaperRecord, ResearchProject
from ..providers.llm import LLMConfig, complete_json


@dataclass(slots=True)
class ManuscriptDraft:
    title: str
    abstract: str
    introduction: str
    related_work: str
    method: str
    results: str
    limitations: str
    conclusion: str
    mode: str


def format_verified_results(experiment_results: dict[str, Any]) -> str:
    primary = experiment_results.get("primary_metric") or {}
    uncertainty = experiment_results.get("uncertainty") or {}
    effect = experiment_results.get("effect_size") or {}
    test = experiment_results.get("statistical_test") or {}
    baselines = experiment_results.get("baseline_metrics") or {}
    ablations = experiment_results.get("ablation_results") or []
    seeds = experiment_results.get("seeds") or []
    samples = experiment_results.get(
        "num_samples", experiment_results.get("sample_rows"),
    )
    if not primary:
        return (
            "The completed run did not satisfy the structured submission-result "
            "protocol, so no publication-level performance claim is reported."
        )
    return (
        f"The primary metric was {primary.get('name')}="
        f"{primary.get('value')} ({primary.get('direction')}) over {samples} samples "
        f"and seeds {seeds}. The {uncertainty.get('confidence', 0.95):.0%} "
        f"interval was [{uncertainty.get('lower')}, {uncertainty.get('upper')}] "
        f"using {uncertainty.get('method')}. Baseline results were "
        f"{json.dumps(baselines, ensure_ascii=False, sort_keys=True)}. "
        f"The reported effect size was {effect.get('name')}={effect.get('value')}; "
        f"{test.get('name')} yielded statistic={test.get('statistic')} and "
        f"p={test.get('p_value')}. Ablation and sensitivity results were "
        f"{json.dumps(ablations, ensure_ascii=False, sort_keys=True)}. "
        "These values are copied from the immutable "
        "sandbox result artifact and require domain-expert interpretation."
    )


def fallback_manuscript(
    project: ResearchProject,
    gap: GapCandidate,
    papers: list[PaperRecord],
    experiment_results: dict[str, Any] | None,
    mode: str,
) -> ManuscriptDraft:
    paper_names = "; ".join(f"{paper.title} [paper{index}]" for index, paper in enumerate(papers[:5], start=1))
    synthetic = bool(
        (experiment_results or {})
        .get("validity_audit", {})
        .get("details", {})
        .get("synthetic")
    )
    if experiment_results:
        results = (
            ("This is a synthetic demonstration, not a domain experiment. " if synthetic else "")
            + format_verified_results(experiment_results)
        )
    else:
        results = (
            "No completed experiment is available. This section contains no empirical "
            "claim and records the planned evaluation only."
        )
    return ManuscriptDraft(
        title=f"Cross-Benchmark Robustness and Failure Modes in {project.direction}",
        abstract=(
            f"We study a low-coverage research candidate in {project.direction}. "
            "The workflow binds literature evidence, licensed data, "
            "generated code, and experiment artifacts into a reproducible record."
        ),
        introduction=(
            "We test whether conclusions from a single benchmark remain stable under "
            "changes in task distribution, tool noise, and model family. The candidate "
            "is described as low coverage within a dated retrieval snapshot, not as "
            "proof of global novelty."
        ),
        related_work=(
            f"The evidence snapshot includes {len(papers)} deduplicated records. "
            f"Representative retrieved works are: {paper_names}."
        ),
        method=(
            "The method records search queries, source failures, dataset license, "
            "normalized sample hash, generated source code, fixed seeds, resource "
            "limits, logs, and output hashes. Generated code executes without network "
            "access in a non-root container."
        ),
        results=results,
        limitations=(
            "The literature search is not exhaustive, provider APIs may omit records, "
            "the prepared dataset is a preview sample, and model-generated research "
            "decisions require expert review. No unobserved result is inferred. "
            + (
                "Synthetic outputs cannot support real-world domain-performance claims."
                if synthetic
                else ""
            )
        ),
        conclusion=(
            "The artifact establishes an auditable research workflow. Strong scientific "
            "claims require the completed experiment, robustness checks, and external "
            "replication."
        ),
        mode=mode,
    )


async def generate_manuscript(
    project: ResearchProject,
    gap: GapCandidate,
    papers: list[PaperRecord],
    experiment_results: dict[str, Any] | None,
    target: str,
    mode: str,
    model: LLMConfig | None,
) -> ManuscriptDraft:
    if model is None:
        return fallback_manuscript(
            project,
            gap,
            papers,
            experiment_results,
            mode,
        )
    citations = [
        {
            "key": f"paper{index}",
            "title": paper.title,
            "abstract": paper.abstract[:700],
        }
        for index, paper in enumerate(papers[:12], start=1)
    ]
    synthetic = bool(
        (experiment_results or {})
        .get("validity_audit", {})
        .get("details", {})
        .get("synthetic")
    )
    response = await complete_json(
        model,
        system=(
            "Write a conservative academic manuscript in English. Use only supplied "
            "literature and experiment results. Never invent a citation, dataset fact, "
            "metric, number, result, or novelty claim. Refer to literature only with the "
            "provided citation keys such as [paper1]. If results are absent, explicitly "
            "state that the section is a plan without empirical claims. If the experiment "
            "is synthetic, repeatedly and explicitly call it a synthetic demonstration and "
            "do not make domain-performance claims. In submission mode, report the "
            "primary metric, confidence interval, baselines, effect size, statistical "
            "test, sample count, and seeds exactly as supplied. Do not dump raw JSON "
            "into prose and do not describe statistical significance unless the supplied "
            "p-value and test support that wording. For submission mode, produce a "
            "substantive manuscript rather than a synopsis: abstract 150-250 words, "
            "introduction at least 400 words, related work at least 300 words, method at "
            "least 500 words, results at least 450 words, limitations at least 200 words, "
            "and conclusion at least 150 words. Cite at least three distinct supplied "
            "papers in the manuscript text when three are available. Explain the task, "
            "data split, baselines, metric definition, seeds, uncertainty, effect size, "
            "statistical test, ablations, failure cases, and threats to validity. If the "
            "supplied evidence cannot support a requested detail, state that limitation "
            "instead of inventing content."
        ),
        prompt=(
            f"Target: {target}; mode: {mode}\nProject: {project.title}\n"
            f"Direction: {project.direction}\nCandidate: {gap.title}\n"
            f"Hypothesis: {gap.hypothesis}\nRationale: {gap.rationale}\n"
            f"Literature: {json.dumps(citations, ensure_ascii=False)}\n"
            f"Experiment results: "
            f"{json.dumps(experiment_results, ensure_ascii=False) if experiment_results else 'NONE'}\n"
            f"Synthetic demonstration: {synthetic}"
        ),
        schema_hint={
            "title": "string",
            "abstract": "string",
            "introduction": "string with [paperN] citations",
            "related_work": "string with [paperN] citations",
            "method": "string",
            "results": "string",
            "limitations": "string",
            "conclusion": "string",
        },
        max_tokens=6000,
        purpose="manuscript_generation",
    )
    serialized = json.dumps(response, ensure_ascii=False)
    for match in re.findall(r"\[paper(\d+)\]", serialized):
        if int(match) > len(citations):
            raise ValueError("model referenced an unknown citation key")
    if not experiment_results and any(
        token in str(response.get("results", "")).casefold()
        for token in ("we achieved", "our accuracy", "outperformed", "significant")
    ):
        raise ValueError("model produced an empirical claim without a completed run")
    if synthetic:
        joined = " ".join(
            str(response.get(key, ""))
            for key in ("abstract", "method", "results", "limitations", "conclusion")
        ).casefold()
        if "synthetic" not in joined:
            raise ValueError("synthetic experiment was not explicitly disclosed")
    return ManuscriptDraft(
        title=str(response["title"])[:400],
        abstract=str(response["abstract"])[:4000],
        introduction=str(response["introduction"])[:12_000],
        related_work=str(response["related_work"])[:12_000],
        method=str(response["method"])[:12_000],
        results=str(response["results"])[:12_000],
        limitations=str(response["limitations"])[:8000],
        conclusion=str(response["conclusion"])[:8000],
        mode=mode,
    )
