# ResearchFlow

ResearchFlow is a local-first AI research automation prototype for LLM and agent research. It
retrieves recent literature, records traceable evidence, proposes low-coverage research
candidates, finds licensed datasets, exports isolated experiments, and creates citation-bound
LaTeX manuscripts.

## What works

- Local registration and encrypted BYOK model profiles.
- OpenAlex, arXiv, Crossref, and Semantic Scholar aggregation with DOI/arXiv/title deduplication.
- Evidence extraction, coverage matrices, and ranked research-gap candidates with executed
  reverse searches, dated counterevidence, and recalibrated confidence.
- Hugging Face and OpenML dataset discovery, license gating, sample preparation, schema profiling,
  and byte-exact SHA-256 fingerprints.
- Model-generated or auditable fallback experiment code, AST safety checks, `uv.lock`, and
  non-root Docker execution with no network, zero Linux capabilities, and strict resource limits.
- Result-grounded arXiv projects plus official 2026 ICLR/ICML/NeurIPS template retrieval,
  LaTeX, BibTeX, PDF, template hashes, LLM disclosure, and claim provenance.
- Scientific-validity gates for dataset-topic fit, registered experiment plans, synthetic-label
  detection, post-run consistency, submission readiness, and sentence-level provenance.
- Evidence levels from concept draft through synthetic demonstration, initial experiment,
  reproducible research, and submission candidate.
- SSE progress updates, persistent workflow checkpoints, pause/resume, and model-call cost logs.
- SQLite/in-process mode plus PostgreSQL, Redis, Celery, pgvector semantic paper search, and the
  web application through Docker Compose.

Research-gap candidates mean low coverage within the retrieval snapshot. They never prove that
no related publication exists.

## Quick start

中文新手教程：[运行与使用指南](docs/USER_GUIDE.zh-CN.md) ·
[Git 与 GitHub 版本管理](docs/VERSION_CONTROL.zh-CN.md) ·
[科学有效性等级](docs/SCIENTIFIC_VALIDITY.zh-CN.md)

Requirements: Node.js 20+ and `uv`.

```powershell
Copy-Item .env.example .env
uv sync --directory apps/api
npm install --prefix apps/web
.\scripts\start-local.ps1
```

Open `http://localhost:3000`. The API documentation port follows `API_PORT` in `.env`.

Two visible process windows are opened. Close both windows to stop ResearchFlow.

The default storage root is `D:\ResearchFlow\data`. Before entering real API keys, replace
`SECRET_KEY` and `ENCRYPTION_KEY` in `.env` with long random values.

## Docker mode

Run the elevated prerequisite installer once:

```powershell
.\scripts\install-local-prerequisites.ps1
```

It enables WSL2 and installs Docker Desktop. This workstation stores Docker's WSL virtual disks
under `D:\ResearchFlow\docker\wsl`; project data remains under `D:\ResearchFlow\data`.
Start PostgreSQL, Redis, the API, Celery worker, and web application:

```powershell
docker compose up --build
```

Without Docker, ResearchFlow will not execute generated code. It safely exports the experiment
package instead.

In networks where Docker Hub is unavailable, set `SANDBOX_BASE_IMAGE`, `API_BASE_IMAGE`, and
`WEB_BASE_IMAGE` to trusted reachable mirrors. The current local `.env` uses the DaoCloud mirror
for the experiment sandbox; API and web mirrors can use the same registry prefix.

## Tests

```powershell
.\scripts\test-local.ps1
```
