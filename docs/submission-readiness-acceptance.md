# Submission Readiness Acceptance

ResearchFlow may label a manuscript build as `submission_candidate` only when all
of the following are true:

- The selected topic has licensed, relevant task data and a completed real-task run.
- The run includes multiple seeds, credible baselines, uncertainty, effect size,
  statistical testing, and at least two ablation or sensitivity results.
- Every numerical claim resolves to the immutable experiment artifact.
- The manuscript has sufficient section depth, uses at least three verified citations
  in its text, and explicitly reports the required result fields.
- A reproducibility bundle is attached and covered by the artifact hash index.
- Publisher submissions name a specific venue and record the official author guide.
- SCI/EI claims include an evidence URL, verification date, category, and human
  attestation.
- The deterministic pre-submission review reports zero critical and zero major findings.

If any scientific or manuscript condition fails, the build is blocked or downgraded,
the reasons are shown to the user, and three related feasible topics are generated with
minimum experiment requirements and suggested target levels.

Automated evidence is provided by
`test_complete_submission_candidate_exports_passing_package`, which builds a complete
fixture and verifies the manuscript, bibliography, result tables, claim provenance,
pre-submission review, target profile, checklist, cover letter, reproducibility files,
and artifact hash index.

This acceptance does not guarantee peer-review acceptance or current indexing status.
Authors remain responsible for scientific judgment, authorship, ethics, declarations,
venue scope, current quartile or indexing verification, and final submission.
