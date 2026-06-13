import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from ..models import CoverageMatrix, EvidenceRecord, PaperRecord


@dataclass(slots=True)
class GapDraft:
    title: str
    hypothesis: str
    rationale: str
    confidence: float
    novelty_score: float
    feasibility_score: float
    estimated_cost: str
    risks: list[str]
    evidence_ids: list[str]
    counter_queries: list[str]


TAXONOMY = {
    "tasks": {
        "evaluation": ("evaluation", "benchmark", "assessment"),
        "planning": ("planning", "plan generation"),
        "tool_use": ("tool use", "tool-use", "tool calling"),
        "reasoning": ("reasoning", "inference"),
        "multilingual": ("multilingual", "cross-lingual", "chinese"),
        "safety": ("safety", "alignment", "harm"),
        "robustness": ("robustness", "failure", "perturbation"),
        "retrieval": ("retrieval", "search"),
    },
    "methods": {
        "benchmark_suite": ("benchmark", "test suite"),
        "llm_judge": ("llm-as-a-judge", "llm judge", "model-based evaluator"),
        "human_evaluation": ("human evaluation", "human annotation"),
        "simulation": ("simulation", "simulated"),
        "ablation": ("ablation",),
        "red_teaming": ("red team", "adversarial"),
        "trace_analysis": ("trace", "trajectory", "provenance"),
    },
    "datasets": {
        "public_dataset": ("dataset", "corpus"),
        "synthetic_data": ("synthetic data", "generated data"),
        "interaction_logs": ("interaction log", "trajectory", "trace"),
        "multilingual_data": ("multilingual dataset", "cross-lingual dataset"),
    },
    "metrics": {
        "accuracy": ("accuracy", "exact match"),
        "success_rate": ("success rate", "task success"),
        "cost": ("cost", "token usage"),
        "latency": ("latency", "runtime"),
        "robustness": ("robustness", "failure rate"),
        "calibration": ("calibration", "confidence"),
        "reproducibility": ("reproducibility", "replication"),
    },
}


def _tags(text: str, vocabulary: dict[str, tuple[str, ...]]) -> list[str]:
    lowered = text.casefold()
    return [name for name, markers in vocabulary.items() if any(marker in lowered for marker in markers)]


def build_coverage_matrix(project_id, papers: list[PaperRecord]) -> CoverageMatrix:
    rows = []
    counters = {dimension: Counter() for dimension in TAXONOMY}
    for paper in papers:
        text = f"{paper.title} {paper.abstract}"
        row = {
            "paper_id": str(paper.id),
            "title": paper.title,
            "publication_date": paper.publication_date,
        }
        for dimension, vocabulary in TAXONOMY.items():
            values = _tags(text, vocabulary)
            row[dimension] = values
            counters[dimension].update(values)
        rows.append(row)
    summary = {
        dimension: dict(counter.most_common())
        for dimension, counter in counters.items()
    }
    return CoverageMatrix(
        project_id=project_id,
        dimensions={dimension: list(vocabulary) for dimension, vocabulary in TAXONOMY.items()},
        rows=rows,
        summary=summary,
    )


def evidence_from_papers(project_id, papers: list[PaperRecord]) -> list[EvidenceRecord]:
    evidence = []
    markers = re.compile(
        r"(future work|remains? (?:an )?open|little is known|however|limitation|"
        r"underexplored|not yet|challenge)",
        re.IGNORECASE,
    )
    for paper in papers:
        sentences = re.split(r"(?<=[.!?])\s+", paper.abstract)
        selected = next((sentence for sentence in sentences if markers.search(sentence)), "")
        if not selected and sentences:
            selected = max(sentences, key=len)
        selected = selected[:800]
        if not selected:
            continue
        evidence.append(
            EvidenceRecord(
                project_id=project_id,
                paper_id=paper.id,
                evidence_type="paper_excerpt",
                claim=f"{paper.title} 提供了与研究空白判断相关的证据。",
                excerpt=selected,
                locator=paper.url,
                content_hash=hashlib.sha256(selected.encode()).hexdigest(),
            )
        )
    return evidence


def generate_gap_drafts(
    direction: str,
    papers: list[PaperRecord],
    evidence: list[EvidenceRecord],
) -> list[GapDraft]:
    recent_titles = [paper.title for paper in papers[:6]]
    evidence_ids = [str(item.id) for item in evidence[:6]]
    context = "；".join(recent_titles[:3]) or direction
    templates = [
        {
            "title": f"{direction} 的跨基准稳健性与失效模式评测",
            "hypothesis": "现有结论在任务分布、工具噪声或模型家族变化后会显著改变。",
            "rationale": (
                f"近期工作集中于单一设置，代表性文献包括：{context}。候选课题将统一复现实验条件并刻画失效边界。"
            ),
            "novelty": 0.78,
            "feasibility": 0.90,
            "cost": "低：公开结果与轻量推理实验",
            "risks": ["可能已有未检索到的并行评测", "不同论文的指标定义可能不兼容"],
            "counter": [f'"{direction}" robustness benchmark', f'"{direction}" failure modes evaluation'],
        },
        {
            "title": f"{direction} 中成本、可靠性与延迟的帕累托前沿",
            "hypothesis": "更强的任务成功率并不稳定对应更高成本，存在可复现的效率甜点区。",
            "rationale": "现有论文通常单独报告质量或成本，缺少统一预算约束下的多目标比较。",
            "novelty": 0.82,
            "feasibility": 0.84,
            "cost": "中：需要多模型 API 调用并设置硬预算",
            "risks": ["API 模型版本会变化", "价格随厂商调整而变化"],
            "counter": [f'"{direction}" cost latency reliability', f'"{direction}" pareto efficiency'],
        },
        {
            "title": f"面向{direction}的证据可追溯评测协议",
            "hypothesis": "将每项结论绑定到可验证证据，可显著降低自动评测中的不可复现结论。",
            "rationale": "许多自动评测只输出分数，缺乏从结论到输入、轨迹和版本的完整证据链。",
            "novelty": 0.76,
            "feasibility": 0.92,
            "cost": "低：协议设计与小规模验证",
            "risks": ["协议开销可能影响吞吐", "证据完整性需要人工抽检"],
            "counter": [f'"{direction}" provenance benchmark', f'"{direction}" traceable evaluation protocol'],
        },
        {
            "title": f"{direction} 在中英文任务迁移中的评价偏差",
            "hypothesis": "英文基准上的排序无法可靠迁移到中文任务与中文工具环境。",
            "rationale": "跨语言代理评测覆盖仍较稀疏，且提示、工具描述和判分器会共同引入偏差。",
            "novelty": 0.80,
            "feasibility": 0.79,
            "cost": "中：需要构建或翻译小型双语任务集",
            "risks": ["翻译可能改变任务难度", "中文数据许可需要逐项确认"],
            "counter": [f'"{direction}" Chinese benchmark', f'"{direction}" multilingual evaluation bias'],
        },
    ]
    coverage = min(len(papers) / 25, 1)
    return [
        GapDraft(
            title=item["title"],
            hypothesis=item["hypothesis"],
            rationale=item["rationale"],
            confidence=round(0.55 + coverage * 0.25, 2),
            novelty_score=item["novelty"],
            feasibility_score=item["feasibility"],
            estimated_cost=item["cost"],
            risks=item["risks"],
            evidence_ids=evidence_ids,
            counter_queries=item["counter"],
        )
        for item in templates
    ]
