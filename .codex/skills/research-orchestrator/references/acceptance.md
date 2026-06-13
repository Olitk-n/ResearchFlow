# Acceptance checks

- Literature results include retrieval timestamps and source identity.
- Duplicate DOI, arXiv ID, and normalized title records collapse deterministically.
- Every gap candidate contains evidence IDs, risks, executed counter-queries, dated reverse
  search results, source failures, and a recalibrated score.
- Open full text is demonstrably open-access or user-authorized, content-hashed, and cited with
  page-level locators.
- A task/method/dataset/metric coverage matrix is stored with the retrieval snapshot.
- Dataset license is visible or explicitly marked for manual review.
- Generated code cannot read unrelated host files or access the network.
- Experiment packages include a dependency specification, seed, manifest, and resource limits.
- Workflow checkpoints survive process restarts and expose pause, resume, failure, and human
  confirmation states.
- Model call records include provider, model, purpose, tokens, cost, and status but no key or
  prompt content.
- Manuscript citations resolve to stored paper records.
- Manuscript numerical claims resolve to completed experiment artifacts.
- ICLR, ICML, and NeurIPS submission projects contain the official current-year style file,
  source URL, archive SHA-256, and an LLM usage disclosure.
- PostgreSQL mode creates a pgvector column and executes a real vector-distance query; SQLite
  mode has a deterministic local fallback.
- Failed providers, blocked runs, and missing compilers remain visible to the user.
