---
name: research-orchestrator
description: Design, implement, or audit evidence-grounded AI research workflows that retrieve literature, propose low-coverage research gaps, select licensed datasets, generate reproducible experiments, and produce citation-safe manuscripts. Use for ResearchFlow changes, autonomous research pipelines, paper-to-experiment workflows, research-gap validation, provenance review, or scientific manuscript automation.
---

# Research Orchestrator

Build the workflow as an auditable state machine, not a single unconstrained agent.

## Workflow

1. Record the user's direction, retrieval date, query expansion, source, and source failures.
2. Deduplicate papers by DOI, then arXiv ID, then normalized title.
3. Separate metadata, open full text, and user-uploaded licensed documents.
4. Bind every scientific claim to an evidence record or experiment artifact.
5. Describe novelty as a low-coverage candidate within the search snapshot. Never claim that
   no paper exists.
6. Create three to five candidates with confidence, feasibility, cost, risks, and reverse
   queries designed to falsify novelty. Execute the reverse queries, retain counterevidence and
   source failures, and recalibrate confidence before showing candidates.
7. Require a human choice before downloading data, spending meaningful budget, running
   generated code, or producing a submission-ready draft.
8. Accept only datasets with visible license information or mark them for manual review.
9. Lock dependencies, random seeds, data fingerprints, code versions, resource limits, logs,
   and artifact hashes.
10. Generate manuscript claims only from completed experiments and verified evidence. Label
    planned or placeholder results explicitly.
11. Persist workflow checkpoints, human gates, model/provider identity, token usage, and cost
    without storing secrets or prompt bodies.
12. Use official venue templates for submission projects. Record the template source, year,
    archive hash, and LLM-use disclosure; never present a generic article as an official format.

## Safety Gates

- Keep model keys encrypted and out of prompts, logs, traces, and client responses.
- Run generated code as non-root, without network access, with CPU, memory, process, filesystem,
  and time limits.
- Export experiments that exceed local capacity. Do not silently submit paid cloud jobs.
- Reject unauthorized full-text collection and require an explicit rights confirmation for
  uploaded PDFs.
- Download only metadata-identified open-access PDFs over public HTTPS. Store the PDF hash and
  page-level locators for extracted evidence.
- Preserve source failures and contradictory evidence rather than hiding them.

## Review

Read [references/acceptance.md](references/acceptance.md) when evaluating a complete workflow.
Reject a manuscript build if citations cannot be resolved to project records or quantitative
claims cannot be resolved to run artifacts.
