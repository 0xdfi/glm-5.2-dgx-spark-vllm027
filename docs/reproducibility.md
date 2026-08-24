# Reproducibility and evidence ladder

The repository separates five claims that are often blurred together:

1. **Source complete:** a clean public checkout reproduces the locked source closure, licenses/provenance, component rights for source distribution, secret/history scan, and large-file/history scan. This is still pending trusted attestation.
2. **Clean build:** an isolated ARM64 builder produces an image from the accepted source subject and passes image verification, SBOM, component-license, and image-secret gates. No trusted clean build exists.
3. **Live launch:** that exact image plus exact externally supplied checkpoint/sidecar/preset passes four-node runtime, correctness, stability, and Fast pin-state gates. No public-source-build live launch exists.
4. **Benchmark reproduction:** one live subject produces exactly three valid complete batteries within five attempts and passes individual and median floors. No profile has reproduced its historical reference from this build.
5. **Anonymous readback:** a fresh unauthenticated checkout verifies the public release assets, builds, and completes the accepted readback chain. This is blocked.

[publication/status.json](../publication/status.json) is the machine authority; [release-status.md](release-status.md) is its deterministic rendering. A status changes to `passed` only from a complete issuer-verified receipt chain. Local PASS reports and historical measurements are provisional.

## Source identity

`source_release_id` hashes the distributed non-generated source projection. `build_subject_id` additionally binds every input consumed by the Dockerfile. Mutable status and generated evidence are excluded so evidence can be added without changing the source subject; runtime inputs, manifests, licenses, or included provenance do change it.

The public O14 source root is commit `9ba54db8f7c852eabb90372f7dcf47f87eafbe20`. Effective overlays, donor roots, build sources, native patch, and Build A v0 are locked under [runtime/manifests](../runtime/manifests). Build A v1 and historical release wheels are not build roots.

## Build and cache boundary

The Dockerfile begins at an immutable ARM64 CUDA digest and fetches immutable source/artifact identities. It is a public BuildKit path, but source availability is not evidence that a trusted clean ARM64 build completed.

CUDA graphs and JIT caches are generated locally from a fresh per-profile cache. They are not copied from a serving cluster, distributed, or compared as portable artifacts. Bit reproducibility is a separate blocked claim; source-equivalent images may still differ byte-for-byte.

## External model boundary

The checkpoint and W8 sidecar are `external-required`, absent from Git and release archives, and currently lack complete whole-artifact locks and verified rights. `verify-hosts` and `launch` fail closed until exact artifacts and site configuration are available. This repository does not grant permission to obtain them.

## Trusted workflow boundary

Static CI never promotes a claim. The manual protected source-evidence workflow has one fixed mode selector and accepts no arbitrary receipt path, URL, ref, or SHA. It checks out public `main`, runs the gates itself, sanitizes a fixed evidence bundle, and attests fixed paths. Build, live, and registry-readback modes remain blocked before any self-hosted runner could be scheduled.
