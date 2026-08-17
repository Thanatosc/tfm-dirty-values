# Public release scope

This directory is a whitelist-built public copy of the research software. It is
not a mirror of the private project workspace.

## Included

- source code and tests;
- frozen dependency and experiment specifications;
- experiment start, stop, resume, status, verification, and analysis scripts;
- Phase 1 and Phase 2 preregistration protocols;
- the registered ECE result-key compatibility amendment;
- dataset, split, task, model, and runtime identity manifests;
- compact machine-readable results, tables, and figures;
- citation metadata, licenses, and release documentation.

The two public YAML files differ from the formal-run bytes only in machine-local
path strings. Checkpoint paths are repository-relative and the source-project
path is represented as `external/tfm-missingness`. Scientific parameters are
unchanged. The original formal hashes remain recorded in `FROZEN_IDENTITIES.md`.
The public Phase 2 preflight JSON applies the same path-only normalization while
retaining the formal artifact hashes, gate state, model identities, and counts.

## Excluded

- manuscript prose, abstracts, outlines, review reports, revision logs, and writing audits;
- author submission files, biography text, postal address, portrait, and journal forms;
- internal pipeline state, project dashboards, authorization JSON, and process records;
- literature PDFs, extracted full text, search narratives, and reviewer-style validation prose;
- raw OpenML downloads and processed feature matrices;
- model checkpoints and all other third-party model artifacts;
- caches, bytecode, virtual environments, temporary files, logs, event streams,
  heartbeats, launchers, lock files, and transient controls;
- private machine paths, credentials, API tokens, and SSH material;
- the 4,575 per-task JSON files, because the companion public SQLite database
  preserves the same task identities and completed metrics in a compact form.

The companion data package is independently checksummed and licensed. No file is
selected by a recursive `git add` or workspace mirror operation.
