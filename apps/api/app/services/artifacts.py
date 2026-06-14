import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from jinja2 import Template

from ..config import get_settings
from ..models import (
    DataPreparation,
    DatasetAsset,
    GapCandidate,
    PaperRecord,
    ResearchProject,
)
from .experiment_agent import ExperimentDraft
from .manuscript_agent import ManuscriptDraft
from .scientific_validity import build_claim_provenance, copy_reproducibility_bundle
from .venue_templates import ensure_official_template

ARXIV_TEMPLATE = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\title{ {{ title }} }
\author{Anonymous Authors}
\date{}
\begin{document}
\maketitle
\noindent\textbf{Evidence level: {{ quality_level }}.}
\begin{abstract}
{{ abstract }}
\end{abstract}
\section{Introduction}
{{ introduction }}
\section{Related Work}
{{ related_work }}
\section{Method}
{{ method }}
\section{Results}
{{ results }}
\input{results-tables.tex}
\section{Limitations}
{{ limitations }}
\section{Conclusion}
{{ conclusion }}
\section*{LLM Usage Disclosure}
Research ideation, literature triage, code drafting, and manuscript drafting
were substantially assisted by ResearchFlow and the user-configured language
model. The human authors selected the research direction and remain fully
responsible for verification, originality, and all scientific claims.
\bibliographystyle{plain}
\bibliography{references}
\end{document}
"""

ICLR_TEMPLATE = r"""
\documentclass{article}
\usepackage{iclr2026_conference,times}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\title{ {{ title }} }
\author{Anonymous Authors}
\begin{document}
\maketitle
\noindent\textbf{Evidence level: {{ quality_level }}.}
\begin{abstract}
{{ abstract }}
\end{abstract}
\section{Introduction}
{{ introduction }}
\section{Related Work}
{{ related_work }}
\section{Method}
{{ method }}
\section{Results}
{{ results }}
\input{results-tables.tex}
\section{Limitations}
{{ limitations }}
\section{Conclusion}
{{ conclusion }}
\section*{LLM Usage Disclosure}
Research ideation, literature triage, code drafting, and manuscript drafting
were substantially assisted by ResearchFlow and the user-configured language
model. The human authors selected the research direction and remain fully
responsible for verification, originality, and all scientific claims.
\bibliography{references}
\bibliographystyle{iclr2026_conference}
\end{document}
"""

ICML_TEMPLATE = r"""
\documentclass{article}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subfigure}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{icml2026}
\icmltitlerunning{ {{ short_title }} }
\begin{document}
\twocolumn[
\icmltitle{ {{ title }} }
\begin{icmlauthorlist}
\icmlauthor{Anonymous Author}{anon}
\end{icmlauthorlist}
\icmlaffiliation{anon}{Anonymous Institution}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
\noindent\textbf{Evidence level: {{ quality_level }}.}
\begin{abstract}
{{ abstract }}
\end{abstract}
\section{Introduction}
{{ introduction }}
\section{Related Work}
{{ related_work }}
\section{Method}
{{ method }}
\section{Results}
{{ results }}
\input{results-tables.tex}
\section{Limitations}
{{ limitations }}
\section{Conclusion}
{{ conclusion }}
\section*{LLM Usage Disclosure}
Research ideation, literature triage, code drafting, and manuscript drafting
were substantially assisted by ResearchFlow and the user-configured language
model. The human authors selected the research direction and remain fully
responsible for verification, originality, and all scientific claims.
\bibliography{references}
\bibliographystyle{icml2026}
\end{document}
"""

NEURIPS_TEMPLATE = r"""
\documentclass{article}
\usepackage{neurips_2026}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\title{ {{ title }} }
\author{Anonymous Authors}
\begin{document}
\maketitle
\noindent\textbf{Evidence level: {{ quality_level }}.}
\begin{abstract}
{{ abstract }}
\end{abstract}
\section{Introduction}
{{ introduction }}
\section{Related Work}
{{ related_work }}
\section{Method}
{{ method }}
\section{Results}
{{ results }}
\input{results-tables.tex}
\section{Limitations}
{{ limitations }}
\section{Conclusion}
{{ conclusion }}
\section*{LLM Usage Disclosure}
Research ideation, literature triage, code drafting, and manuscript drafting
were substantially assisted by ResearchFlow and the user-configured language
model. The human authors selected the research direction and remain fully
responsible for verification, originality, and all scientific claims.
\bibliographystyle{plainnat}
\bibliography{references}
\end{document}
"""

IEEE_CONFERENCE_TEMPLATE = r"""
\documentclass[conference]{IEEEtran}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\title{ {{ title }} }
\author{\IEEEauthorblockN{Anonymous Authors}}
\begin{document}
\maketitle
\noindent\textbf{Evidence level: {{ quality_level }}.}
\begin{abstract}
{{ abstract }}
\end{abstract}
\section{Introduction}
{{ introduction }}
\section{Related Work}
{{ related_work }}
\section{Method}
{{ method }}
\section{Results}
{{ results }}
\input{results-tables.tex}
\section{Limitations}
{{ limitations }}
\section{Conclusion}
{{ conclusion }}
\section*{Generative AI Disclosure}
Research ideation, literature triage, code drafting, and manuscript drafting
were assisted by ResearchFlow. Human authors verified all claims and results.
\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
"""

ELSEVIER_JOURNAL_TEMPLATE = r"""
\documentclass[preprint,12pt]{elsarticle}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\journal{ {{ publication_name }} }
\begin{document}
\begin{frontmatter}
\title{ {{ title }} }
\author{Anonymous Authors}
\begin{abstract}
{{ abstract }}
\end{abstract}
\end{frontmatter}
\noindent\textbf{Evidence level: {{ quality_level }}.}
\section{Introduction}
{{ introduction }}
\section{Related Work}
{{ related_work }}
\section{Method}
{{ method }}
\section{Results}
{{ results }}
\input{results-tables.tex}
\section{Limitations}
{{ limitations }}
\section{Conclusion}
{{ conclusion }}
\section*{Declaration of generative AI-assisted technologies}
Research ideation, literature triage, code drafting, and manuscript drafting
were assisted by ResearchFlow. Human authors verified all claims and results.
\bibliographystyle{elsarticle-num}
\bibliography{references}
\end{document}
"""

MANUSCRIPT_TEMPLATES = {
    "arxiv": ARXIV_TEMPLATE,
    "iclr": ICLR_TEMPLATE,
    "icml": ICML_TEMPLATE,
    "neurips": NEURIPS_TEMPLATE,
    "ieee_conference": IEEE_CONFERENCE_TEMPLATE,
    "elsevier_journal": ELSEVIER_JOURNAL_TEMPLATE,
}

TARGET_PROFILES = {
    "arxiv": {
        "venue": "arXiv preprint",
        "anonymous": False,
        "page_limit": None,
    },
    "iclr": {
        "venue": "International Conference on Learning Representations",
        "anonymous": True,
        "page_limit": 9,
    },
    "icml": {
        "venue": "International Conference on Machine Learning",
        "anonymous": True,
        "page_limit": 8,
    },
    "neurips": {
        "venue": "Conference on Neural Information Processing Systems",
        "anonymous": True,
        "page_limit": 9,
    },
    "ieee_conference": {
        "venue": "Specific IEEE conference required",
        "anonymous": True,
        "page_limit": None,
    },
    "elsevier_journal": {
        "venue": "Specific Elsevier journal required",
        "anonymous": False,
        "page_limit": None,
    },
}


def find_executable(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if name == "docker":
        candidate = Path("C:/Program Files") / "Docker" / "Docker" / "resources" / "bin" / "docker.exe"
        if candidate.exists():
            return str(candidate)
    if name in {"pdflatex", "bibtex"}:
        candidate = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / f"{name}.exe"
        if candidate.exists():
            return str(candidate)
    return None


def safe_key(title: str, index: int) -> str:
    words = re.findall(r"[A-Za-z0-9]+", title)
    return ("".join(words[:2]) or "paper")[:24] + str(index)


def latex_escape(text: str, citation_keys: list[str]) -> str:
    escaped = text
    placeholders = {}
    for index, key in enumerate(citation_keys, start=1):
        marker = f"RFCITATION{index}RF"
        escaped = escaped.replace(f"[paper{index}]", marker)
        placeholders[marker] = rf"\cite{{{key}}}"
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = "".join(replacements.get(char, char) for char in escaped)
    for marker, citation in placeholders.items():
        escaped = escaped.replace(marker, citation)
    return escaped


def latex_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_results_tables(experiment_results: dict | None) -> str:
    if not experiment_results:
        return "% No completed experiment results were available.\n"
    primary = experiment_results.get("primary_metric") or {}
    uncertainty = experiment_results.get("uncertainty") or {}
    baselines = experiment_results.get("baseline_metrics") or {}
    rows = [
        (
            str(primary.get("name") or "primary metric"),
            latex_value(primary.get("value", "n/a")),
            str(primary.get("direction") or "n/a"),
        )
    ]
    for baseline_name, values in baselines.items():
        if isinstance(values, dict):
            for metric, value in values.items():
                rows.append((f"{baseline_name}: {metric}", latex_value(value), "baseline"))
    table_rows = "\n".join(
        f"{latex_escape(name, [])} & {latex_escape(value, [])} & "
        f"{latex_escape(direction, [])} \\\\"
        for name, value, direction in rows
    )
    interval = (
        f"{latex_value(uncertainty.get('lower', 'n/a'))}--"
        f"{latex_value(uncertainty.get('upper', 'n/a'))}"
    )
    ablations = experiment_results.get("ablation_results") or []
    ablation_rows = "\n".join(
        f"{latex_escape(str(item.get('name', 'condition')), [])} & "
        f"{latex_escape(str(item.get('metric', 'metric')), [])} & "
        f"{latex_escape(latex_value(item.get('value', 'n/a')), [])} \\\\"
        for item in ablations
        if isinstance(item, dict)
    )
    if not ablation_rows:
        ablation_rows = "No verified ablation & n/a & n/a \\\\"
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Verified primary and baseline results. Values are copied from the "
        "sandbox result artifact.}\n"
        "\\label{tab:verified-results}\n"
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Method or metric & Value & Role/direction \\\\\n"
        "\\midrule\n"
        f"{table_rows}\n"
        "\\midrule\n"
        f"95\\% interval & {latex_escape(interval, [])} & "
        f"{latex_escape(str(uncertainty.get('method', 'n/a')), [])} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Verified ablation and sensitivity results.}\n"
        "\\label{tab:verified-ablations}\n"
        "\\begin{tabular}{lll}\n"
        "\\toprule\n"
        "Condition & Metric & Value \\\\\n"
        "\\midrule\n"
        f"{ablation_rows}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def project_directory(project_id) -> Path:
    path = get_settings().storage_root / "projects" / str(project_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_artifact_index(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact-index.json":
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
    (root / "artifact-index.json").write_text(
        json.dumps({"files": files}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_experiment_package(
    project: ResearchProject,
    gap: GapCandidate,
    dataset: DatasetAsset,
    preparation: DataPreparation,
    experiment: ExperimentDraft,
) -> Path:
    base_image = get_settings().sandbox_base_image
    if not re.fullmatch(r"[A-Za-z0-9._/:@-]+", base_image):
        raise ValueError("invalid sandbox base image reference")
    root = project_directory(project.id) / "experiment"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.py").write_text(experiment.code + "\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname="researchflow-experiment"\nversion="0.1.0"\nrequires-python=">=3.12"\ndependencies=[]\n',
        encoding="utf-8",
    )
    uv = shutil.which("uv")
    if uv:
        lock = subprocess.run(
            [uv, "lock", "--project", str(root), "--offline"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if lock.returncode:
            raise RuntimeError(f"uv lock failed: {lock.stderr[-1000:]}")
    else:
        (root / "uv.lock").write_text(
            'version = 1\nrevision = 3\nrequires-python = ">=3.12"\n\n'
            "[[package]]\n"
            'name = "researchflow-experiment"\n'
            'version = "0.1.0"\n'
            'source = { virtual = "." }\n',
            encoding="utf-8",
        )
    data_root = root / "data"
    data_root.mkdir()
    source_root = Path(preparation.artifact_path or "")
    shutil.copy2(source_root / "prepared.jsonl", data_root / "prepared.jsonl")
    shutil.copy2(source_root / "data-card.json", data_root / "data-card.json")
    manifest = {
        "researchflow_version": "0.1.0",
        "project": project.title,
        "direction": project.direction,
        "gap": gap.title,
        "hypothesis": gap.hypothesis,
        "seed": 42,
        "dataset": {
            "name": dataset.name,
            "url": dataset.url,
            "license": dataset.license,
            "content_hash": preparation.content_hash,
            "rows": preparation.row_count,
        },
        "experiment": {
            "name": experiment.name,
            "objective": experiment.objective,
            "metrics": experiment.metrics,
            "methodology": experiment.methodology,
            "expected_outputs": experiment.expected_outputs,
            "code_origin": experiment.code_origin,
            "code_sha256": hashlib.sha256(experiment.code.encode("utf-8")).hexdigest(),
            "scientific_plan": experiment.scientific_plan,
        },
        "limits": {"network": False, "cpus": 2, "memory": "4g", "timeout_seconds": 300},
        "base_image": base_image,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "Dockerfile").write_text(
        f"FROM {base_image}\nRUN useradd -m runner\n"
        "WORKDIR /app\nCOPY --chown=runner:runner run.py ./run.py\n"
        'USER runner\nWORKDIR /work\nCMD ["python", "/app/run.py"]\n',
        encoding="utf-8",
    )
    write_artifact_index(root)
    return root


def run_experiment_package(root: Path, timeout_seconds: int = 300) -> tuple[str, dict]:
    docker = find_executable("docker")
    if not docker:
        return "blocked", {"reason": "Docker 未安装，实验包已导出。"}
    docker_env = os.environ.copy()
    docker_bin = str(Path(docker).parent)
    docker_env["PATH"] = f"{docker_bin}{os.pathsep}{docker_env.get('PATH', '')}"
    runtime = root / "runtime"
    if runtime.exists():
        shutil.rmtree(runtime)
    (runtime / "data").mkdir(parents=True)
    shutil.copy2(root / "data" / "prepared.jsonl", runtime / "data" / "prepared.jsonl")
    shutil.copy2(root / "data" / "data-card.json", runtime / "data" / "data-card.json")
    image = f"researchflow-{root.parent.name}".lower()
    build = subprocess.run(
        [docker, "build", "-t", image, str(root)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        env=docker_env,
    )
    if build.returncode:
        return "failed", {"logs": build.stderr[-4000:]}
    container_name = f"researchflow-run-{uuid4().hex[:12]}"
    try:
        run = subprocess.run(
            [
                docker,
                "run",
                "--rm",
                "--name",
                container_name,
                "--network",
                "none",
                "--cpus",
                "2",
                "--memory",
                "4g",
                "--pids-limit",
                "128",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--mount",
                f"type=bind,source={runtime.resolve()},target=/work",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=256m",
                image,
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=docker_env,
        )
    except subprocess.TimeoutExpired:
        with suppress(subprocess.SubprocessError):
            subprocess.run(
                [docker, "rm", "-f", container_name],
                capture_output=True,
                timeout=20,
                check=False,
                env=docker_env,
            )
        return "failed", {
            "reason": "实验超过时间限制，容器已强制终止。",
            "timeout_seconds": timeout_seconds,
        }
    if run.returncode:
        return "failed", {"logs": run.stderr[-4000:], "exit_code": run.returncode}
    try:
        result = json.loads(run.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        result = {"stdout": run.stdout[-4000:]}
    result_path = runtime / "results.json"
    if result_path.exists():
        result["artifact_path"] = str(result_path)
        result["artifact_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (runtime / "stdout.log").write_text(run.stdout, encoding="utf-8")
    (runtime / "stderr.log").write_text(run.stderr, encoding="utf-8")
    write_artifact_index(runtime)
    return "completed", result


def build_manuscript(
    project: ResearchProject,
    gap: GapCandidate,
    papers: list[PaperRecord],
    target: str,
    draft: ManuscriptDraft | None = None,
    experiment_results: dict | None = None,
    experiment_root: Path | None = None,
    quality_level: str = "concept_draft",
    publication_name: str | None = None,
    author_guide_url: str | None = None,
) -> tuple[Path, bool, list[str]]:
    root = project_directory(project.id) / "manuscript" / target
    root.mkdir(parents=True, exist_ok=True)
    template_metadata = ensure_official_template(target, root)
    cited = []
    bib_entries = []
    for index, paper in enumerate(papers[:12], start=1):
        key = safe_key(paper.title, index)
        cited.append({"key": key, "title": paper.title})
        authors = " and ".join(paper.authors) or "Unknown"
        year = (paper.publication_date or "n.d.")[:4]
        bib_entries.append(
            f"@article{{{key},\n  title = {{{paper.title}}},\n"
            f"  author = {{{authors}}},\n  year = {{{year}}},\n"
            f"  url = {{{paper.url or paper.open_access_url or ''}}}\n}}"
        )
    if draft is None:
        from .manuscript_agent import fallback_manuscript

        draft = fallback_manuscript(
            project,
            gap,
            papers,
            experiment_results,
            "draft",
        )
    citation_keys = [item["key"] for item in cited]
    tex = Template(MANUSCRIPT_TEMPLATES[target]).render(
        title=latex_escape(draft.title, citation_keys),
        short_title=latex_escape(draft.title[:80], citation_keys),
        abstract=latex_escape(draft.abstract, citation_keys),
        introduction=latex_escape(draft.introduction, citation_keys),
        related_work=latex_escape(draft.related_work, citation_keys),
        method=latex_escape(draft.method, citation_keys),
        results=latex_escape(draft.results, citation_keys),
        limitations=latex_escape(draft.limitations, citation_keys),
        conclusion=latex_escape(draft.conclusion, citation_keys),
        quality_level=latex_escape(quality_level.replace("_", " "), citation_keys),
        publication_name=latex_escape(
            publication_name or "Specific publication not selected",
            citation_keys,
        ),
    )
    (root / "main.tex").write_text(tex.strip() + "\n", encoding="utf-8")
    (root / "results-tables.tex").write_text(
        build_results_tables(experiment_results),
        encoding="utf-8",
    )
    (root / "references.bib").write_text("\n\n".join(bib_entries) + "\n", encoding="utf-8")
    (root / "README.txt").write_text(
        "Compile with: pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex\n"
        "Do not treat placeholder text as a completed scientific claim.\n",
        encoding="utf-8",
    )
    (root / "target-profile.json").write_text(
        json.dumps(
            {
                **TARGET_PROFILES[target],
                "target": target,
                "template": template_metadata,
                "manuscript_mode": draft.mode,
                "experiment_results_present": experiment_results is not None,
                "quality_level": quality_level,
                "publication_name": publication_name,
                "author_guide_url": author_guide_url,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sections = {
        "abstract": draft.abstract,
        "introduction": draft.introduction,
        "related_work": draft.related_work,
        "method": draft.method,
        "results": draft.results,
        "limitations": draft.limitations,
        "conclusion": draft.conclusion,
    }
    claims = build_claim_provenance(
        sections,
        citation_keys,
        experiment_results,
        gap.evidence_ids,
    )
    unresolved = [
        claim
        for claim in claims
        if claim["source"]["type"] == "unresolved"
        and claim["section"] in {"abstract", "introduction", "related_work", "results"}
    ]
    if draft.mode == "submission" and unresolved:
        raise ValueError(
            f"submission manuscript contains {len(unresolved)} unresolved scientific claims"
        )
    (root / "claim-provenance.json").write_text(
        json.dumps(
            {
                "citation_keys": citation_keys,
                "experiment_results": experiment_results,
                "gap_evidence_ids": gap.evidence_ids,
                "quality_level": quality_level,
                "claims": claims,
                "warning": (
                    "Low coverage is not proof of global novelty. Only completed run outputs support numerical claims."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    copy_reproducibility_bundle(root, experiment_root)
    submission_items = [
        {"item": "Specific publication selected", "passed": bool(publication_name)},
        {
            "item": "Publication author guide recorded",
            "passed": bool(author_guide_url and author_guide_url.startswith("https://")),
        },
        {"item": "Scientific validity gate passed", "passed": quality_level == "submission_candidate"},
        {"item": "Claim-level provenance exported", "passed": True},
        {"item": "Reproducibility bundle exported", "passed": bool(experiment_root)},
        {"item": "Human author and affiliation details completed", "passed": False},
        {"item": "Conflict of interest and funding statements reviewed", "passed": False},
        {"item": "Publication indexing and current quartile verified manually", "passed": False},
    ]
    (root / "submission-checklist.json").write_text(
        json.dumps(
            {
                "publication_name": publication_name,
                "author_guide_url": author_guide_url,
                "ready_for_human_submission": all(
                    item["passed"] for item in submission_items
                ),
                "items": submission_items,
                "warning": (
                    "Indexing, quartile, scope fit, ethics declarations, and acceptance "
                    "must be verified by the human authors."
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "cover-letter.txt").write_text(
        (
            f"Dear Editor or Program Committee,\n\n"
            f"Please consider the enclosed manuscript for {publication_name or '[PUBLICATION NAME]'}.\n"
            f"The work studies: {gap.title}.\n"
            "All quantitative claims are linked to the enclosed reproducibility artifacts. "
            "The authors confirm that the manuscript is not simultaneously under review "
            "elsewhere and will complete the publication-specific declarations before submission.\n\n"
            "Sincerely,\n[CORRESPONDING AUTHOR]\n"
        ),
        encoding="utf-8",
    )
    compiled = False
    pdflatex = find_executable("pdflatex")
    bibtex = find_executable("bibtex")
    if pdflatex and bibtex and get_settings().app_env != "test":
        for command in (
            [pdflatex, "-interaction=nonstopmode", "main.tex"],
            [bibtex, "main"],
            [pdflatex, "-interaction=nonstopmode", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "main.tex"],
        ):
            result = subprocess.run(command, cwd=root, capture_output=True, timeout=120)
            if result.returncode:
                break
        else:
            compiled = True
    write_artifact_index(root)
    return root, compiled, citation_keys
