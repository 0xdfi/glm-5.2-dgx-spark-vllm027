# Release status

This file is generated from `publication/status.json`; edit the JSON only through the evidence gate tooling, then regenerate this page.

Machine status updated: `2026-08-23T00:00:00Z`.

A `pending` or `blocked` row is not a PASS. Historical measurements and source-build inputs do not promote any row.

| Claim | Profile | Status | Evidence / reason |
|---|---|---|---|
| source-complete | — | pending | Trusted evidence has not promoted this claim. |
| model-license | — | blocked | Checkpoint and W8 sidecar redistribution/use rights are not yet verified by trusted evidence. |
| clean-build | — | blocked | No trusted isolated ARM64 clean-build evidence exists. |
| bit-reproducible | — | blocked | Two trusted distinct clean builds have not been compared. |
| live-launch | fast-320k | blocked | Clean-build and model-license gates are not passed. |
| benchmark-reproduced | fast-320k | blocked | Trusted live-launch and three-valid-in-five battery evidence is absent. |
| live-launch | balanced-650k | blocked | Clean-build and model-license gates are not passed. |
| benchmark-reproduced | balanced-650k | blocked | Trusted live-launch and three-valid-in-five battery evidence is absent. |
| live-launch | large-1m | blocked | Clean-build and model-license gates are not passed. |
| benchmark-reproduced | large-1m | blocked | Trusted live-launch and three-valid-in-five battery evidence is absent. |
| three-profile-suite | — | blocked | All three profile live-launch and benchmark gates are not passed. |
| anonymous-readback | — | blocked | No trusted anonymous clean-checkout/readback evidence exists. |
| convenience-image-published | — | blocked | Registry redistribution clearance and publication attestation are absent. |
| convenience-image-readback | — | blocked | No convenience image has been published by immutable digest. |

See [Reproducibility](reproducibility.md) for the evidence ladder and [Licensing](licensing.md) for the source, model, and OCI boundaries.
