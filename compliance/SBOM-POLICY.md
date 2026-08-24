# Champion runtime SBOM policy

Copyright 2026 dfi / 0xdfi. SPDX-License-Identifier: Apache-2.0.

SBOM evidence is generated from every layer of an immutable built-image digest
in SPDX JSON format. The generator must itself be a versioned Syft OCI image
pinned by a full SHA-256 digest. Every discovered package must map to
`compliance/container-components.json`; `NOASSERTION`, an unmatched package, a
missing governing license, or a subject mismatch fails closed.

The current source release does not select an audited Syft image digest and has
no built-image receipt. Therefore SBOM, image-license, and GHCR publication
remain blocked. This does not prevent source-only publication or local source
builds from the official digest-pinned NVIDIA CUDA base. Evidence paths must be
relative, non-symlink files contained by their declared evidence bundle.
