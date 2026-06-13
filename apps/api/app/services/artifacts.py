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

MANUSCRIPT_TEMPLATES = {
    "arxiv": ARXIV_TEMPLATE,
    "iclr": ICLR_TEMPLATE,
    "icml": ICML_TEMPLATE,
    "neurips": NEURIPS_TEMPLATE,
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
    )
    (root / "main.tex").write_text(tex.strip() + "\n", encoding="utf-8")
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "claim-provenance.json").write_text(
        json.dumps(
            {
                "citation_keys": citation_keys,
                "experiment_results": experiment_results,
                "gap_evidence_ids": gap.evidence_ids,
                "warning": (
                    "Low coverage is not proof of global novelty. Only completed run outputs support numerical claims."
                ),
            },
            ensure_ascii=False,
            indent=2,
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
