# Licensing, attribution, and distribution boundaries

This repository is mixed-license by design. File-level [REUSE](../REUSE.toml) annotations and the [provenance ledger](../compliance/provenance.json) are authoritative; the root license does not erase retained upstream terms.

## Project and retained source

- New original champion-publication code and documentation: Copyright 2026 dfi / 0xdfi, Apache-2.0.
- Original recipe tree through commit `d1aca44241a5100bf7af6a5388985cb42ffca4cd`: retained MIT.
- vLLM and B12X derivatives: retained Apache-2.0 with upstream identity and modification provenance.
- FlashAttention donor source: retained BSD-3-Clause.
- MiniMax MSA and DeepSeek FlashMLA donor source: retained MIT.
- Jordan Miller's `benchmarks/probe.py` contribution from commit `525b210dc100b75792ac5a0818d4eab0cdddc5a8`: retained MIT and authorship; Claude Fable 5 is credited as the recorded co-author.

Canonical texts are under [LICENSES](../LICENSES), broader technical lineage is in [CREDITS.md](../CREDITS.md), and distributed legal notices are in [NOTICE](../NOTICE).

## Model and sidecar

The GLM-5.2 checkpoint and W8 lm-head sidecar are separate external artifacts. They are not distributed here, their exact whole-artifact integrity manifests are incomplete, and their rights status is unknown. Apache-2.0 coverage of project source is not a model license and is not permission to obtain, use, modify, or redistribute those artifacts.

## OCI image

A local build combines project/upstream source with an official NVIDIA CUDA base, Ubuntu packages, Python/CUDA dependencies, and operator-supplied external model artifacts. Source releasability does not automatically clear that combined OCI image for redistribution.

[container-components.json](../compliance/container-components.json) currently blocks GHCR because exact image/SBOM reconciliation, NVIDIA/base-package conditions, model boundaries, and registry evidence are incomplete. The source build can therefore be useful while `bin/champion pull` and convenience-image claims remain blocked.

## Historical evidence baseline

[secret-history-baseline.json](../compliance/secret-history-baseline.json) acknowledges only six exact already-public RFC1918 documentation locations by scanner rule, Git blob, path, and line. It does not accept credentials, canaries, private paths, hostnames, new findings, wildcards, or stale entries. This is an auditable record of public history, not a weakened scanning rule.
