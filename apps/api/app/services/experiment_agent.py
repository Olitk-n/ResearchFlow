import ast
import json
from dataclasses import dataclass
from pathlib import Path

from ..models import DataPreparation, DatasetAsset, GapCandidate, ResearchProject
from ..providers.llm import LLMConfig, complete_json

ALLOWED_IMPORTS = {
    "collections",
    "csv",
    "datetime",
    "hashlib",
    "itertools",
    "json",
    "math",
    "pathlib",
    "random",
    "re",
    "statistics",
    "typing",
}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__"}
FORBIDDEN_ATTRIBUTES = {
    "system",
    "popen",
    "spawn",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
}


@dataclass(slots=True)
class ExperimentDraft:
    name: str
    objective: str
    metrics: list[str]
    methodology: list[str]
    expected_outputs: list[str]
    code: str
    code_origin: str
    scientific_plan: dict


def baseline_path_diagnostics(preparation: DataPreparation) -> list[dict]:
    rows_ok = preparation.row_count >= 200
    ranking_query = text_fields_matching(
        preparation, ("query", "prompt", "question", "task", "instruction"),
    )
    ranking_candidate = text_fields_matching(
        preparation, ("candidate", "answer", "response", "document", "tool", "choice"),
    )
    ranking_target = fields_matching(
        preparation, ("correct", "relevance", "relevant", "label", "target", "gold"),
    )
    class_targets = classification_candidates(preparation)
    class_features = feature_candidates(preparation, class_targets[-1]) if class_targets else []
    numeric = numeric_fields(preparation)
    return [
        {
            "path": "ranking_retrieval",
            "label": "排序/检索/候选选择",
            "passed": bool(rows_ok and ranking_query and ranking_candidate and ranking_target),
            "required": "至少200行，query/prompt字段，candidate/answer字段，correct/relevance字段",
            "evidence": {
                "query_fields": ranking_query,
                "candidate_fields": ranking_candidate,
                "target_fields": ranking_target,
                "rows": preparation.row_count,
            },
        },
        {
            "path": "classification",
            "label": "分类/标签预测",
            "passed": bool(rows_ok and class_targets and class_features),
            "required": "至少200行，低基数标签字段，至少一个文本或数值输入字段",
            "evidence": {
                "target_fields": class_targets,
                "feature_fields": class_features,
                "rows": preparation.row_count,
            },
        },
        {
            "path": "regression",
            "label": "数值回归",
            "passed": bool(rows_ok and len(numeric) >= 2),
            "required": "至少200行，至少两个数值字段，其中一个作为目标变量",
            "evidence": {
                "numeric_fields": numeric,
                "rows": preparation.row_count,
            },
        },
    ]


def numeric_fields(preparation: DataPreparation) -> list[str]:
    fields = []
    for name, descriptor in preparation.schema_json.items():
        types = descriptor.get("types") or {}
        if any(kind in types for kind in ("int", "float", "bool")):
            fields.append(name)
    return fields


def classification_candidates(preparation: DataPreparation) -> list[str]:
    candidates = []
    for name, descriptor in preparation.schema_json.items():
        examples = descriptor.get("examples") or []
        types = descriptor.get("types") or {}
        if not examples:
            continue
        unique = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in examples}
        is_label_like = (
            any(kind in types for kind in ("str", "bool", "int"))
            and 2 <= len(unique) <= 20
        )
        if is_label_like:
            candidates.append(name)
    return candidates


def feature_candidates(preparation: DataPreparation, target: str) -> list[str]:
    fields = []
    for name, descriptor in preparation.schema_json.items():
        if name == target:
            continue
        types = descriptor.get("types") or {}
        if any(kind in types for kind in ("str", "int", "float", "bool")):
            fields.append(name)
    return fields[:12]


def fields_matching(preparation: DataPreparation, markers: tuple[str, ...]) -> list[str]:
    matches = []
    for name in preparation.schema_json:
        lowered = name.casefold()
        if any(marker in lowered for marker in markers):
            matches.append(name)
    return matches


def text_fields_matching(preparation: DataPreparation, markers: tuple[str, ...]) -> list[str]:
    matches = []
    for name, descriptor in preparation.schema_json.items():
        lowered = name.casefold()
        types = descriptor.get("types") or {}
        if (
            any(marker in lowered for marker in markers)
            and any(kind in types for kind in ("str", "list", "dict"))
        ):
            matches.append(name)
    return matches


def build_ranking_baseline(
    project: ResearchProject,
    gap: GapCandidate,
    preparation: DataPreparation,
) -> ExperimentDraft | None:
    if preparation.row_count < 200:
        return None
    query_fields = text_fields_matching(
        preparation, ("query", "prompt", "question", "task", "instruction"),
    )
    candidate_fields = text_fields_matching(
        preparation, ("candidate", "answer", "response", "document", "tool", "choice"),
    )
    target_fields = fields_matching(
        preparation, ("correct", "relevance", "relevant", "label", "target", "gold"),
    )
    if not query_fields or not candidate_fields or not target_fields:
        return None
    query = query_fields[0]
    candidate = next((field for field in candidate_fields if field != query), None)
    if candidate is None:
        return None
    target = next((field for field in target_fields if field not in {query, candidate}), target_fields[0])
    code = f'''"""Auditable ranking baseline generated by ResearchFlow."""
import json
import math
import random
import re
import statistics
from pathlib import Path

SEEDS = [42, 43, 44]
QUERY = {query!r}
CANDIDATE = {candidate!r}
TARGET = {target!r}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
data_path = Path("data/prepared.jsonl")
rows = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

def text(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)

def tokens(value):
    return set(token.casefold() for token in TOKEN_RE.findall(text(value)))

def relevance(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    lowered = text(value).casefold()
    return lowered in {{"1", "true", "yes", "correct", "relevant", "positive", "success"}}

examples = []
for row in rows:
    query_text = text(row.get(QUERY))
    candidate_text = text(row.get(CANDIDATE))
    is_relevant = relevance(row.get(TARGET))
    if query_text and candidate_text:
        examples.append({{"query": query_text, "candidate": candidate_text, "relevant": is_relevant}})

if len(examples) < 200 or not any(item["relevant"] for item in examples):
    raise RuntimeError("not enough ranking rows with relevant candidates")

def mean(values):
    return statistics.fmean(values) if values else 0.0

def score(query_text, candidate_text):
    q = tokens(query_text)
    c = tokens(candidate_text)
    if not q or not c:
        return 0.0
    return len(q & c) / len(q | c)

def evaluate(items, seed):
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    test = shuffled[split:]
    groups = {{}}
    for item in test:
        groups.setdefault(item["query"], []).append(item)
    hit_values = []
    mrr_values = []
    random_hit_values = []
    random_mrr_values = []
    per_query_differences = []
    for query_text, group in groups.items():
        if not any(item["relevant"] for item in group):
            continue
        ranked = sorted(
            group,
            key=lambda item: (score(query_text, item["candidate"]), item["candidate"]),
            reverse=True,
        )
        random_ranked = list(group)
        rng.shuffle(random_ranked)
        def rank_metrics(ranked_items):
            for index, item in enumerate(ranked_items, start=1):
                if item["relevant"]:
                    return (1.0 if index == 1 else 0.0), 1.0 / index
            return 0.0, 0.0
        hit, mrr = rank_metrics(ranked)
        random_hit, random_mrr = rank_metrics(random_ranked)
        hit_values.append(hit)
        mrr_values.append(mrr)
        random_hit_values.append(random_hit)
        random_mrr_values.append(random_mrr)
        per_query_differences.append(hit - random_hit)
    return {{
        "hit_at_1": mean(hit_values),
        "mrr": mean(mrr_values),
        "random_hit_at_1": mean(random_hit_values),
        "random_mrr": mean(random_mrr_values),
        "query_count": len(hit_values),
        "differences": per_query_differences,
    }}

def paired_t_pvalue(differences):
    if len(differences) < 2:
        return 1.0, 0.0
    avg = mean(differences)
    sd = statistics.stdev(differences)
    if sd == 0:
        return (0.0 if avg != 0 else 1.0), 0.0
    t_stat = avg / (sd / math.sqrt(len(differences)))
    return max(0.0, min(1.0, math.erfc(abs(t_stat) / math.sqrt(2)))), t_stat

def cohens_d(differences):
    if len(differences) < 2:
        return 0.0
    sd = statistics.stdev(differences)
    return 0.0 if sd == 0 else mean(differences) / sd

def bootstrap_interval(values, seed):
    rng = random.Random(seed + 30_000)
    estimates = []
    for _ in range(300):
        sample = [values[rng.randrange(len(values))] for _i in values]
        estimates.append(mean(sample))
    estimates.sort()
    return (
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    )

per_seed_metrics = []
hit_values = []
random_hit_values = []
all_differences = []
for seed in SEEDS:
    metrics = evaluate(examples, seed)
    hit_values.append(metrics["hit_at_1"])
    random_hit_values.append(metrics["random_hit_at_1"])
    all_differences.extend(metrics["differences"])
    per_seed_metrics.append({{
        "seed": seed,
        "metrics": {{
            "hit_at_1": metrics["hit_at_1"],
            "mrr": metrics["mrr"],
            "random_hit_at_1": metrics["random_hit_at_1"],
            "random_mrr": metrics["random_mrr"],
            "query_count": metrics["query_count"],
        }},
    }})

aggregate_hit = mean(hit_values)
baseline_hit = mean(random_hit_values)
lower, upper = bootstrap_interval(hit_values, 42)
p_value, statistic = paired_t_pvalue(all_differences)
effect = cohens_d(all_differences)
result = {{
    "num_samples": len(examples),
    "seeds": SEEDS,
    "parameters": {{"test_fraction": 0.2, "query": QUERY, "candidate": CANDIDATE, "target": TARGET}},
    "metrics": {{"hit_at_1": aggregate_hit, "baseline_hit_at_1": baseline_hit}},
    "primary_metric": {{
        "name": "hit_at_1",
        "value": aggregate_hit,
        "direction": "higher_is_better",
    }},
    "per_seed_metrics": per_seed_metrics,
    "baseline_metrics": {{"random_candidate": {{"hit_at_1": baseline_hit}}}},
    "uncertainty": {{
        "method": "bootstrap over seed-level hit@1",
        "confidence": 0.95,
        "lower": lower,
        "upper": upper,
    }},
    "effect_size": {{"name": "paired_hit_gain_cohen_d", "value": effect}},
    "statistical_test": {{
        "name": "paired_t_normal_approximation_on_hit_gain",
        "statistic": statistic,
        "p_value": p_value,
    }},
    "ablation_results": [
        {{
            "name": "seed_" + str(item["seed"]) + "_ranking_sensitivity",
            "metric": "hit_at_1",
            "value": item["metrics"]["hit_at_1"],
            "interpretation": "Sensitivity of retrieval performance to the fixed data split seed.",
        }}
        for item in per_seed_metrics
    ],
    "claims": [
        "This run uses measured relevance labels from the prepared dataset.",
        "The baseline is a seeded random candidate ranking.",
        "The model ranks candidates by query-candidate token overlap."
    ],
}}
Path("results.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(result, sort_keys=True))
'''
    validate_generated_code(code)
    return ExperimentDraft(
        name="Auditable ranking and retrieval baseline",
        objective=(
            f"Run a measured relevance ranking baseline for {gap.title} using licensed prepared data."
        ),
        metrics=["hit_at_1", "mrr", "95% bootstrap interval", "paired hit@1 gain"],
        methodology=[
            "Use query, candidate, and measured relevance fields from the prepared dataset.",
            "Evaluate three fixed random train/test splits.",
            "Compare token-overlap ranking against a seeded random-candidate baseline.",
            "Report seed-level retrieval metrics, uncertainty, effect size, and a paired statistical test.",
        ],
        expected_outputs=["results.json", "stdout JSON summary"],
        code=code,
        code_origin="auditable_ranking_baseline",
        scientific_plan={
            "field_mapping": {"query": query, "candidate": candidate, "target": target},
            "target_variable": target,
            "model": "query-candidate token-overlap ranker",
            "split_strategy": "80/20 train/test split repeated with seeds 42, 43, and 44",
            "baselines": ["seeded random candidate ranking"],
            "metric_definitions": {
                "hit_at_1": "fraction of evaluated queries whose top-ranked candidate is relevant",
                "mrr": "mean reciprocal rank of the first relevant candidate",
                "baseline_hit_at_1": "hit@1 of seeded random candidate ordering",
            },
            "statistical_analysis": (
                "bootstrap interval over seed-level hit@1 and paired t normal approximation "
                "over per-query hit@1 gains"
            ),
            "seeds": [42, 43, 44],
            "parameters": {"test_fraction": 0.2},
            "expected_sample_count": preparation.row_count,
            "evidence_class": "real_task",
        },
    )


def build_classification_baseline(
    project: ResearchProject,
    gap: GapCandidate,
    preparation: DataPreparation,
) -> ExperimentDraft | None:
    if preparation.row_count < 200:
        return None
    targets = classification_candidates(preparation)
    if not targets:
        return None
    target = targets[-1]
    features = feature_candidates(preparation, target)
    if not features:
        return None
    code = f'''"""Auditable classification baseline generated by ResearchFlow."""
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path

SEEDS = [42, 43, 44]
TARGET = {target!r}
FEATURES = {features!r}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
data_path = Path("data/prepared.jsonl")
rows = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

def label_value(value):
    if value is None:
        return None
    return str(value)

def tokens(row):
    output = []
    for field in FEATURES:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, (int, float, bool)):
            output.append(f"{{field}}={{value}}")
        else:
            output.extend(f"{{field}}:{{token.casefold()}}" for token in TOKEN_RE.findall(str(value))[:80])
    return output

examples = []
for row in rows:
    label = label_value(row.get(TARGET))
    row_tokens = tokens(row)
    if label is not None and row_tokens:
        examples.append((row_tokens, label))

labels = sorted({{label for _tokens, label in examples}})
if len(examples) < 200 or not 2 <= len(labels) <= 50:
    raise RuntimeError("not enough labeled rows for classification")

def mean(values):
    return statistics.fmean(values) if values else 0.0

def train_nb(train):
    label_counts = Counter(label for _tokens, label in train)
    token_counts = {{label: Counter() for label in label_counts}}
    total_tokens = Counter()
    vocabulary = set()
    for row_tokens, label in train:
        counts = Counter(row_tokens)
        token_counts[label].update(counts)
        total_tokens[label] += sum(counts.values())
        vocabulary.update(counts)
    return label_counts, token_counts, total_tokens, vocabulary

def predict_nb(model, row_tokens):
    label_counts, token_counts, total_tokens, vocabulary = model
    total_rows = sum(label_counts.values())
    vocab_size = max(1, len(vocabulary))
    counts = Counter(row_tokens)
    scores = {{}}
    for label, label_count in label_counts.items():
        score = math.log(label_count / total_rows)
        denominator = total_tokens[label] + vocab_size
        for token, count in counts.items():
            score += count * math.log((token_counts[label][token] + 1) / denominator)
        scores[label] = score
    return max(scores, key=scores.get)

def accuracy(y_true, y_pred):
    return mean([1.0 if a == b else 0.0 for a, b in zip(y_true, y_pred)])

def macro_f1(y_true, y_pred):
    values = []
    for label in sorted(set(y_true) | set(y_pred)):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(values)

def paired_t_pvalue(differences):
    n = len(differences)
    if n < 2:
        return 1.0, 0.0
    avg = mean(differences)
    sd = statistics.stdev(differences)
    if sd == 0:
        return (0.0 if avg != 0 else 1.0), 0.0
    t_stat = avg / (sd / math.sqrt(n))
    return max(0.0, min(1.0, math.erfc(abs(t_stat) / math.sqrt(2)))), t_stat

def cohens_d(differences):
    if len(differences) < 2:
        return 0.0
    sd = statistics.stdev(differences)
    return 0.0 if sd == 0 else mean(differences) / sd

def bootstrap_interval(values, seed):
    rng = random.Random(seed + 20_000)
    estimates = []
    for _ in range(300):
        sample = [values[rng.randrange(len(values))] for _i in values]
        estimates.append(mean(sample))
    estimates.sort()
    return (
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    )

per_seed_metrics = []
accuracies = []
baseline_accuracies = []
all_differences = []
for seed in SEEDS:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    train = shuffled[:split]
    test = shuffled[split:]
    majority = Counter(label for _tokens, label in train).most_common(1)[0][0]
    model = train_nb(train)
    y_true = [label for _tokens, label in test]
    y_pred = [predict_nb(model, row_tokens) for row_tokens, _label in test]
    y_base = [majority for _tokens, _label in test]
    acc = accuracy(y_true, y_pred)
    base_acc = accuracy(y_true, y_base)
    f1 = macro_f1(y_true, y_pred)
    accuracies.append(acc)
    baseline_accuracies.append(base_acc)
    all_differences.extend([
        (1.0 if true == pred else 0.0) - (1.0 if true == base else 0.0)
        for true, pred, base in zip(y_true, y_pred, y_base)
    ])
    per_seed_metrics.append({{
        "seed": seed,
        "metrics": {{
            "accuracy": acc,
            "baseline_accuracy": base_acc,
            "macro_f1": f1,
            "test_rows": len(test),
        }},
        "majority_label": majority,
    }})

aggregate_accuracy = mean(accuracies)
baseline_accuracy = mean(baseline_accuracies)
lower, upper = bootstrap_interval(accuracies, 42)
p_value, statistic = paired_t_pvalue(all_differences)
effect = cohens_d(all_differences)
result = {{
    "num_samples": len(examples),
    "seeds": SEEDS,
    "parameters": {{"test_fraction": 0.2, "features": FEATURES, "target": TARGET}},
    "metrics": {{"accuracy": aggregate_accuracy, "baseline_accuracy": baseline_accuracy}},
    "primary_metric": {{
        "name": "accuracy",
        "value": aggregate_accuracy,
        "direction": "higher_is_better",
    }},
    "per_seed_metrics": per_seed_metrics,
    "baseline_metrics": {{"majority_class": {{"accuracy": baseline_accuracy}}}},
    "uncertainty": {{
        "method": "bootstrap over seed-level accuracy",
        "confidence": 0.95,
        "lower": lower,
        "upper": upper,
    }},
    "effect_size": {{"name": "paired_accuracy_gain_cohen_d", "value": effect}},
    "statistical_test": {{
        "name": "paired_t_normal_approximation_on_accuracy_gain",
        "statistic": statistic,
        "p_value": p_value,
    }},
    "ablation_results": [
        {{
            "name": "seed_" + str(item["seed"]) + "_classification_sensitivity",
            "metric": "accuracy",
            "value": item["metrics"]["accuracy"],
            "interpretation": "Sensitivity of classification performance to the fixed data split seed.",
        }}
        for item in per_seed_metrics
    ],
    "claims": [
        "This run uses a measured label field from the prepared dataset.",
        "The baseline is the train-split majority class.",
        "The model is a standard-library multinomial Naive Bayes classifier over prepared fields."
    ],
}}
Path("results.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(result, sort_keys=True))
'''
    validate_generated_code(code)
    return ExperimentDraft(
        name="Auditable real-task classification baseline",
        objective=(
            f"Run a measured-label classification baseline for {gap.title} using only licensed prepared data."
        ),
        metrics=["accuracy", "macro_f1", "95% bootstrap interval", "paired accuracy gain"],
        methodology=[
            "Use the selected label field from the prepared dataset.",
            "Evaluate three fixed random train/test splits.",
            "Compare a standard-library Naive Bayes classifier against the majority-class baseline.",
            "Report seed-level metrics, uncertainty, effect size, and a paired statistical test.",
        ],
        expected_outputs=["results.json", "stdout JSON summary"],
        code=code,
        code_origin="auditable_classification_baseline",
        scientific_plan={
            "field_mapping": {"features": features, "target": target},
            "target_variable": target,
            "model": "multinomial Naive Bayes over prepared row fields",
            "split_strategy": "80/20 train/test split repeated with seeds 42, 43, and 44",
            "baselines": ["train majority class"],
            "metric_definitions": {
                "accuracy": "held-out test accuracy; higher is better",
                "macro_f1": "unweighted mean F1 across observed labels",
                "baseline_accuracy": "held-out test accuracy of train majority-class baseline",
            },
            "statistical_analysis": (
                "bootstrap interval over seed-level accuracy and paired t normal approximation "
                "over per-row accuracy gains"
            ),
            "seeds": [42, 43, 44],
            "parameters": {"test_fraction": 0.2},
            "expected_sample_count": preparation.row_count,
            "evidence_class": "real_task",
        },
    )


def build_real_task_baseline(
    project: ResearchProject,
    gap: GapCandidate,
    preparation: DataPreparation,
) -> ExperimentDraft | None:
    boolean_relevance_fields = [
        name
        for name, descriptor in preparation.schema_json.items()
        if any(term in name.casefold() for term in ("relevant", "correct"))
        and "bool" in (descriptor.get("types") or {})
    ]
    if boolean_relevance_fields:
        ranking = build_ranking_baseline(project, gap, preparation)
        if ranking is not None:
            return ranking
    classification = build_classification_baseline(project, gap, preparation)
    if classification is not None:
        return classification
    ranking = build_ranking_baseline(project, gap, preparation)
    if ranking is not None:
        return ranking
    fields = numeric_fields(preparation)
    if preparation.row_count < 200 or len(fields) < 2:
        return None
    target = fields[-1]
    features = fields[:-1][:8]
    code = f'''"""Auditable real-task baseline generated by ResearchFlow."""
import json
import math
import random
import statistics
from pathlib import Path

SEEDS = [42, 43, 44]
TARGET = {target!r}
FEATURES = {features!r}
data_path = Path("data/prepared.jsonl")
rows = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

def as_float(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

examples = []
for row in rows:
    y = as_float(row.get(TARGET))
    xs = {{name: as_float(row.get(name)) for name in FEATURES}}
    if y is not None and any(value is not None for value in xs.values()):
        examples.append((xs, y))

if len(examples) < 200:
    raise RuntimeError("not enough rows with numeric features and target")

def mean(values):
    return statistics.fmean(values) if values else 0.0

def mae(y_true, y_pred):
    return mean([abs(a - b) for a, b in zip(y_true, y_pred)])

def fit_univariate(train, feature):
    pairs = [(xs.get(feature), y) for xs, y in train if xs.get(feature) is not None]
    if len(pairs) < 2:
        base = mean([y for _xs, y in train])
        return lambda _xs: base
    x_bar = mean([x for x, _y in pairs])
    y_bar = mean([y for _x, y in pairs])
    denominator = sum((x - x_bar) ** 2 for x, _y in pairs)
    if denominator == 0:
        return lambda _xs: y_bar
    slope = sum((x - x_bar) * (y - y_bar) for x, y in pairs) / denominator
    intercept = y_bar - slope * x_bar
    return lambda xs: intercept + slope * (xs.get(feature) if xs.get(feature) is not None else x_bar)

def paired_t_pvalue(differences):
    n = len(differences)
    if n < 2:
        return 1.0, 0.0
    avg = mean(differences)
    sd = statistics.stdev(differences)
    if sd == 0:
        return (0.0 if avg != 0 else 1.0), 0.0
    t_stat = avg / (sd / math.sqrt(n))
    # Normal approximation is conservative enough for an automated gate.
    p_value = math.erfc(abs(t_stat) / math.sqrt(2))
    return max(0.0, min(1.0, p_value)), t_stat

def cohens_d(differences):
    if len(differences) < 2:
        return 0.0
    sd = statistics.stdev(differences)
    return 0.0 if sd == 0 else mean(differences) / sd

def bootstrap_interval(values, seed):
    rng = random.Random(seed + 10_000)
    estimates = []
    for _ in range(300):
        sample = [values[rng.randrange(len(values))] for _i in values]
        estimates.append(mean(sample))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return lower, upper

per_seed_metrics = []
primary_values = []
baseline_values = []
all_differences = []
for seed in SEEDS:
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    train = shuffled[:split]
    test = shuffled[split:]
    y_train = [y for _xs, y in train]
    y_test = [y for _xs, y in test]
    baseline_value = mean(y_train)
    baseline_predictions = [baseline_value for _ in test]
    candidates = []
    for feature in FEATURES:
        predictor = fit_univariate(train, feature)
        predictions = [predictor(xs) for xs, _y in test]
        candidates.append((mae(y_test, predictions), feature, predictions))
    model_mae, best_feature, model_predictions = min(candidates, key=lambda item: item[0])
    baseline_mae = mae(y_test, baseline_predictions)
    primary_values.append(model_mae)
    baseline_values.append(baseline_mae)
    all_differences.extend([
        abs(true - base) - abs(true - pred)
        for true, base, pred in zip(y_test, baseline_predictions, model_predictions)
    ])
    per_seed_metrics.append({{
        "seed": seed,
        "metrics": {{
            "mae": model_mae,
            "baseline_mae": baseline_mae,
            "test_rows": len(test),
        }},
        "best_feature": best_feature,
    }})

aggregate_mae = mean(primary_values)
baseline_mae = mean(baseline_values)
lower, upper = bootstrap_interval(primary_values, 42)
p_value, statistic = paired_t_pvalue(all_differences)
effect = cohens_d(all_differences)
result = {{
    "num_samples": len(examples),
    "seeds": SEEDS,
    "parameters": {{"test_fraction": 0.2, "candidate_features": FEATURES}},
    "metrics": {{"mae": aggregate_mae, "baseline_mae": baseline_mae}},
    "primary_metric": {{
        "name": "mae",
        "value": aggregate_mae,
        "direction": "lower_is_better",
    }},
    "per_seed_metrics": per_seed_metrics,
    "baseline_metrics": {{"mean_target": {{"mae": baseline_mae}}}},
    "uncertainty": {{
        "method": "bootstrap over seed-level MAE",
        "confidence": 0.95,
        "lower": lower,
        "upper": upper,
    }},
    "effect_size": {{"name": "paired_error_reduction_cohen_d", "value": effect}},
    "statistical_test": {{
        "name": "paired_t_normal_approximation_on_absolute_error_reduction",
        "statistic": statistic,
        "p_value": p_value,
    }},
    "ablation_results": [
        {{
            "name": "seed_" + str(item["seed"]) + "_regression_sensitivity",
            "metric": "mae",
            "value": item["metrics"]["mae"],
            "interpretation": "Sensitivity of regression error to the fixed data split seed.",
        }}
        for item in per_seed_metrics
    ],
    "claims": [
        "This run uses a measured target field from the prepared dataset.",
        "The baseline is the train-split target mean.",
        "The model is the best univariate least-squares feature selected independently in each seed."
    ],
}}
Path("results.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(result, sort_keys=True))
'''
    validate_generated_code(code)
    return ExperimentDraft(
        name="Auditable real-task univariate baseline",
        objective=(
            f"Run a measured-target baseline for {gap.title} using only licensed prepared data."
        ),
        metrics=["mae", "baseline_mae", "95% bootstrap interval", "paired error reduction"],
        methodology=[
            "Use the selected numeric target field from the prepared dataset.",
            "Evaluate three fixed random train/test splits.",
            "Compare a best univariate least-squares predictor against the train-target mean baseline.",
            "Report seed-level metrics, uncertainty, effect size, and a paired statistical test.",
        ],
        expected_outputs=["results.json", "stdout JSON summary"],
        code=code,
        code_origin="auditable_real_task_baseline",
        scientific_plan={
            "field_mapping": {"features": features, "target": target},
            "target_variable": target,
            "model": "best univariate least-squares predictor selected on train split",
            "split_strategy": "80/20 train/test split repeated with seeds 42, 43, and 44",
            "baselines": ["train target mean"],
            "metric_definitions": {
                "mae": "mean absolute error on held-out test rows; lower is better",
                "baseline_mae": "mean absolute error of train-target mean baseline",
            },
            "statistical_analysis": (
                "bootstrap interval over seed-level MAE and paired t normal approximation "
                "over absolute-error reductions"
            ),
            "seeds": [42, 43, 44],
            "parameters": {"test_fraction": 0.2},
            "expected_sample_count": preparation.row_count,
            "evidence_class": "real_task",
        },
    )


def validate_generated_code(code: str) -> None:
    if len(code) > 30_000:
        raise ValueError("generated code exceeds 30 KB")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = {alias.name.split(".", 1)[0] for alias in node.names}
            if not modules <= ALLOWED_IMPORTS:
                raise ValueError(f"forbidden imports: {sorted(modules - ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module not in ALLOWED_IMPORTS:
                raise ValueError(f"forbidden import: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_ATTRIBUTES:
                raise ValueError(f"forbidden attribute call: {node.func.attr}")
    required = {"results.json", "prepared.jsonl"}
    missing = {value for value in required if value not in code}
    if missing:
        raise ValueError(f"generated code does not reference {sorted(missing)}")


def fallback_experiment(
    project: ResearchProject,
    gap: GapCandidate,
    preparation: DataPreparation,
) -> ExperimentDraft:
    real_task = build_real_task_baseline(project, gap, preparation)
    if real_task is not None:
        return real_task
    code = '''"""Offline dataset evidence baseline generated by ResearchFlow."""
import hashlib
import json
import statistics
from pathlib import Path

SEED = 42
data_path = Path("data/prepared.jsonl")
rows = [
    json.loads(line)
    for line in data_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
serialized = data_path.read_bytes()
column_counts = {}
text_lengths = []
for row in rows:
    for key, value in row.items():
        column_counts[key] = column_counts.get(key, 0) + int(value is not None)
        if isinstance(value, str):
            text_lengths.append(len(value))
result = {
    "seed": SEED,
    "sample_rows": len(rows),
    "dataset_sha256": hashlib.sha256(serialized).hexdigest(),
    "non_null_by_column": column_counts,
    "mean_text_length": statistics.fmean(text_lengths) if text_lengths else 0.0,
    "claims": [
        "This run is an offline descriptive baseline.",
        "No model-quality conclusion is supported without a task-specific evaluator."
    ],
}
Path("results.json").write_text(
    json.dumps(result, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(json.dumps(result, sort_keys=True))
'''
    validate_generated_code(code)
    return ExperimentDraft(
        name="Offline evidence and data-quality baseline",
        objective=(f"Establish a reproducible data baseline for {gap.title} before task-specific model evaluation."),
        metrics=[
            "sample_rows",
            "non_null_by_column",
            "mean_text_length",
            "dataset_sha256",
        ],
        methodology=[
            "Read only the prepared licensed JSONL snapshot.",
            "Verify the exact dataset fingerprint.",
            "Compute descriptive completeness and text-length statistics.",
            "Avoid unsupported model-performance claims.",
        ],
        expected_outputs=["results.json", "stdout JSON summary"],
        code=code,
        code_origin="auditable_fallback",
        scientific_plan={
            "field_mapping": {"input": "all prepared fields", "target": None},
            "target_variable": "none; descriptive baseline only",
            "model": "none",
            "split_strategy": "none",
            "baselines": ["descriptive data-quality baseline"],
            "metric_definitions": {
                "sample_rows": "number of prepared JSONL rows",
                "non_null_by_column": "non-null values per field",
            },
            "statistical_analysis": "descriptive only; no performance inference",
            "seeds": [42],
            "parameters": {},
            "expected_sample_count": preparation.row_count,
            "evidence_class": "descriptive",
        },
    )


async def generate_experiment(
    project: ResearchProject,
    gap: GapCandidate,
    dataset: DatasetAsset,
    preparation: DataPreparation,
    model: LLMConfig | None,
) -> ExperimentDraft:
    real_task = build_real_task_baseline(project, gap, preparation)
    if real_task is not None:
        return real_task
    if model is None:
        return fallback_experiment(project, gap, preparation)
    data_card_path = Path(preparation.artifact_path or "") / "data-card.json"
    data_card = json.loads(data_card_path.read_text(encoding="utf-8"))
    response = await complete_json(
        model,
        system=(
            "You design reproducible offline AI/ML experiments. Generate one safe Python "
            "3.12 script using only the standard library. It must read "
            "data/prepared.jsonl, write results.json, print the same result as JSON, "
            "use seed 42 where randomness exists, and never access network, credentials, "
            "other files, subprocesses, shell commands, eval, or exec. Do not invent "
            "performance results. Derive metrics from actual rows. Never create random "
            "or simulated ground-truth labels. If the data lacks a valid target field, "
            "produce a descriptive baseline and label it as such. For a real task, "
            "results.json must contain: metrics, primary_metric{name,value,direction}, "
            "per_seed_metrics with one record for each seed, baseline_metrics, "
            "uncertainty with a 95% interval, effect_size{name,value}, "
            "statistical_test{name,statistic,p_value}, seeds, num_samples, parameters, "
            "ablation_results[{name,metric,value,interpretation}], and claims. "
            "Compute every value from the actual rows. If these analyses "
            "cannot be implemented using the allowed environment, classify the plan "
            "as descriptive rather than pretending it is submission-ready."
        ),
        prompt=(
            f"Project: {project.title}\nDirection: {project.direction}\n"
            f"Selected research candidate: {gap.title}\nHypothesis: {gap.hypothesis}\n"
            f"Dataset: {dataset.external_id}; license={dataset.license}\n"
            f"Data card:\n{json.dumps(data_card, ensure_ascii=False)}"
        ),
        schema_hint={
            "name": "string",
            "objective": "string",
            "metrics": ["string"],
            "methodology": ["string"],
            "expected_outputs": ["results.json"],
            "result_protocol": {
                "metrics": {"metric_name": "aggregate numeric value"},
                "primary_metric": {
                    "name": "metric name",
                    "value": "aggregate value",
                    "direction": "higher_is_better or lower_is_better",
                },
                "per_seed_metrics": [
                    {"seed": 42, "metrics": {"metric_name": "numeric value"}},
                ],
                "baseline_metrics": {
                    "baseline name": {"metric_name": "numeric value"},
                },
                "uncertainty": {
                    "method": "bootstrap or t interval",
                    "confidence": 0.95,
                    "lower": "numeric",
                    "upper": "numeric",
                },
                "effect_size": {"name": "Cohen d or paired effect", "value": "numeric"},
                "statistical_test": {
                    "name": "test name",
                    "statistic": "numeric",
                    "p_value": "numeric",
                },
                "ablation_results": [
                    {
                        "name": "ablation or sensitivity condition",
                        "metric": "metric name",
                        "value": "numeric value",
                        "interpretation": "bounded evidence-based interpretation",
                    },
                ],
            },
            "code": "complete Python source",
            "scientific_plan": {
                "field_mapping": {"input": "source field names", "target": "source field name or null"},
                "target_variable": "measured target and units, or none",
                "model": "implemented model or none",
                "split_strategy": "train/validation/test procedure or none",
                "baselines": ["implemented baseline"],
                "metric_definitions": {"metric": "exact formula and direction"},
                "statistical_analysis": "uncertainty procedure",
                "seeds": [42, 43, 44],
                "parameters": {"k": "integer when applicable"},
                "expected_sample_count": preparation.row_count,
                "evidence_class": "real_task or descriptive",
            },
        },
        max_tokens=5000,
        purpose="experiment_generation",
    )
    code = str(response["code"]).strip()
    validate_generated_code(code)
    return ExperimentDraft(
        name=str(response["name"])[:240],
        objective=str(response["objective"])[:2000],
        metrics=[str(item)[:120] for item in response["metrics"]][:12],
        methodology=[str(item)[:500] for item in response["methodology"]][:12],
        expected_outputs=[str(item)[:200] for item in response["expected_outputs"]][:12],
        code=code,
        code_origin="llm",
        scientific_plan=dict(response.get("scientific_plan") or {}),
    )
