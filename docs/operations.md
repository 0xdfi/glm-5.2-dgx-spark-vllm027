# Operations — UMA memory hygiene, boot discipline, and the watchdog spec

Condensed from the campaign's operations handbook. This is the "run it for real, unattended,
for more than an afternoon" knowledge — the mistakes here cost real wall-clock time to learn.

---

## UMA memory hygiene

Each node has 121GB of unified (CPU+GPU) memory. Across a long-running deployment this
reliably gets dirtier over time through a few independent mechanisms — know all of them
before trusting a single `free -g` reading.

### Page cache growth fools MemAvailable, and the allocator guard reads it conservatively

Linux's page cache grows to fill idle RAM (normal, reclaimable), but the allocator guards
described in `docs/patches.md` read `MemAvailable` from `/proc/meminfo` to decide how much
headroom exists before a risky allocation — and that number **systematically underestimates**
what's actually allocatable on GB10 UMA. This was proven directly during the campaign: an
older, unguarded image ran `mnbt=2048` with a **larger** KV allocation than a newer, guarded
image refused on the same physical nodes, at the same moment. The guard's conservatism, not
physics, was the wall. Don't read a tight `MemAvailable` as "the node is nearly full" without
first trying a compaction pass.

**Pre-boot compaction** (run before any large-mnbt or DCP4 boot — required, not optional; a
`mnbt` step-up refusing cleanly with "available host memory does not exceed the protected
reserve" is the signature of skipping this):

```bash
ssh NODE1 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches'
# repeat identically on NODE2, NODE3, NODE4
```

Run this once per idle window (e.g. once daily, or immediately before any boot/reboot cycle) —
cheap but pointless to spam continuously, and dropping caches during active serving briefly
slows anything relying on warm page cache.

### Leaked processes from failed boots

Two failure signatures leave zombie state that inflates apparent usage on the *next* attempt:

- **Silent-preflight-death.** A retained-container launch dies at its preflight check under
  `set -e` with **zero log output** if containers aren't already stopped or if
  `nvidia-smi --query-compute-apps` isn't empty. The containers from the *previous* run are
  left `Up` with a dead process inside. Retrying blind stacks a failed launch on top of a
  live-but-broken container.
- **Up-with-dead-process.** A launcher's log-redirect line (`exec > $SERVE_LOG`) dying
  instantly because the log path isn't mounted inside the container: `docker ps` reports `Up`,
  but there is no server process inside and no log anywhere. `nvidia-smi` may still show GPU
  memory held by a process that never fully released it.

Before every boot, verify state is genuinely clean, not just "not crash-looping":

```bash
ssh NODE1 'sudo -n docker ps -a; nvidia-smi --query-compute-apps=pid,used_memory --format=csv'
# repeat per node — the compute-apps list MUST be empty on all 4 before a fresh launch
```

### Swap creep — know the normal band for your configuration

At the largest configs in this repo's matrix (e.g. DCP4 1M@4096), hosts sitting at
**118-119GB/121GB resident with 2-5GB swap is normal**, not a leak. Swap creep *beyond* that
band across a session (not the presence of swap itself) is the actionable signal:

```bash
ssh NODE1 'free -g'   # repeat per node; log Mem used/avail + Swap used over time
```

Flag: swap > 6GB, or `MemAvailable` in monotonic decline across 3+ consecutive idle checks.

### Container hygiene

```bash
ssh NODE1 'sudo -n docker stop -t 30 <name>'
ssh NODE1 'sudo -n docker ps -a --filter name=<name> --format "{{.Status}}"'   # confirm Exited
# ONLY for throwaway/boot-attempt containers — never a retained fleet you're keeping warm:
ssh NODE1 'sudo -n docker rm <throwaway-name>'
```

---

## Boot discipline — known silent-failure gotchas

1. **Usage-arg exit 0.** A bare invocation of a launcher script with no subcommand prints a
   usage banner and **exits 0**. Under `nohup ... &`, this looks *exactly* like a successful
   background launch in the log — clean exit, no error text. Always pass the subcommand
   explicitly (`render` or `launch`) and confirm containers actually appear (`docker ps`)
   before walking away.
2. **The serve log path must be container-visible.** A host-only log path that isn't
   bind-mounted into the container kills the log-redirect line instantly under `set -e`, with
   zero error anywhere. Diagnostic signature: container state `Up`, but no server process
   inside and no log on any mounted path. `recipes/launch.sh` mounts `$RUN_DIR` at `/run`
   inside the container specifically to avoid this class of failure — if you fork the
   launcher, keep the log path under a mounted directory.
3. **Stale-log false alarms.** After a failed boot, the evidence/run directory may still
   contain a log file from a *previous* attempt at the same path. Check the log's timestamp
   before reading it as the current attempt's output.
4. **Image-ID pin, always.** Launch with the full image tag/ID pinned, never a bare "current"
   assumption. `recipes/launch.sh`'s preflight resolves and compares the image ID across all 4
   nodes for exactly this reason — an unpinned/stale tag can silently reintroduce a bug that
   was already fixed in a newer image.
5. **Failed-boot containers block relaunch.** `docker run --name ...` fails if a stopped
   container from a prior failed attempt already holds that name. Remove it explicitly before
   retrying — never blind-`rm` a container you didn't create for this run.
6. **Preflight is there for a reason — don't work around it.** Every check in
   `recipes/launch.sh`'s preflight section (protected-container check, GPU-apps-empty check,
   image-ID match, port-free check, model-path/RDMA-device check) exists because a real boot
   failure taught the lesson. If a preflight check is inconvenient, fix the underlying state,
   don't bypass the check.

### `DCP_SIZE` override

DCP1/DCP2/DCP4 share the same launcher; the decode-context-parallel degree is set via
`DCP_SIZE` (1/2/4). Don't assume a default — pass it explicitly for every boot, especially
when switching profiles on a cluster that was last running a different DCP degree.

### Monitor patterns

Health 200 alone doesn't prove the right config came up — confirm the KV geometry banner
matches what you intended to boot:

```bash
ssh NODE1 'curl -s -o /dev/null -w "%{http_code}\n" http://FABRIC_IP_0:8211/health'
ssh NODE1 'curl -s http://FABRIC_IP_0:8211/v1/models'
ssh NODE1 'sudo -n docker logs --tail 100 <rank0-container> | grep -E "GPU KV cache size|max_model_len"'
```

Confirm a coherent temperature-0 completion round-trips before declaring the boot done — a
healthy `/health` with an empty or garbage completion is still a bad boot.

---

## The idle-endpoint rule for benchmarking

Every probe in `benchmarks/protocol.md` requires the server to be genuinely idle before it
starts, and the campaign learned — twice — that "idle" is harder to guarantee than it looks.

**Stopping a forwarder is not enough.** If you run any TCP proxy/forwarder in front of the
serving port (a `socat` bridge, an nginx passthrough, anything), stopping that forwarder's
unit does **not** guarantee the endpoint is quiet. **Long-lived `ssh -L` port-forward tunnels
from operator workstations are a second, independent traffic path** that a forwarder shutdown
does nothing to close — a tunnel opened days earlier, forgotten about, and still carrying
traffic will keep the server non-idle indefinitely while every "is the forwarder up?" check
reports the answer you wanted to hear. This happened during the campaign: a three-day-old SSH
session with a live `-L` tunnel into the serving port kept generating chat-completion traffic
well after the forwarder unit had been stopped, contaminating an in-progress battery.

**Server-side connection attribution is unreliable when client and server share a fabric IP.**
On a fabric where the operator's tunnel egress and the server's listen address are the same
IP, a server-side `ss` inspection of the listening socket's established connections will
resolve every peer back to the server process itself — it tells you nothing about who the
real caller is:

```bash
# MISLEADING on a shared-fabric-IP setup — resolves to the server, not the caller:
ssh NODE1 'sudo ss -tnpH state established "( sport = :8211 )"'
```

**Trace the client side instead** — follow the *destination* port from the connection's other
end:

```bash
# Useful: what is this node's OWN client-side socket connecting out to :8211?
ssh NODE1 'sudo ss -tnpH state established "( dport = :8211 )"'
```

This surfaces the actual forwarding process (an `sshd` session, a proxy, whatever it is) by
PID, which you can then trace back to its own remote peer and session age.

**Also check for old SSH sessions directly**, independent of the port-8211 trace — a session
carrying a `-L` forward doesn't always show up cleanly in a single `ss` query, and knowing
"how many operator SSH sessions are even open right now" is useful context on its own:

```bash
ssh NODE1 'sudo ss -tnpH state established "( sport = :22 )"'
```

**The complete quieting procedure before any benchmark run:**
1. Stop any known forwarder/proxy unit.
2. Trace `dport = :8211` client-side on the node that hosts the API port, per the recipe
   above, and identify every process still holding a connection.
3. Check for lingering `sport = :22` sessions that might be carrying an `-L` tunnel you didn't
   know about.
4. Confirm `/metrics`' `vllm:num_requests_running` reads 0 and stays 0 across two or three
   checks a few seconds apart — a single zero reading is not proof of idle, since a forwarded
   session's traffic can be bursty.
5. Only then start the probe battery.

Treat any unexplained token consumption on a server you believe is idle as a live-traffic
incident, not noise — investigate before discarding, and discard the contaminated result once
you've captured the evidence (connection trace, `/metrics` deltas at the time), not before.

---

## The 24-hour watchdog — design spec

Hand this to a monitoring agent or cron job. All checks are read-only except the two
explicitly whitelisted auto-fixes below.

### Daily checks

1. **Health 200.**
   ```bash
   ssh NODE1 'curl -s -o /dev/null -w "%{http_code}\n" http://FABRIC_IP_0:8211/health'
   ```
   Anything other than `200` → escalate.

2. **Real completion probe** — health alone doesn't prove inference works:
   ```bash
   ssh NODE1 'curl -s http://FABRIC_IP_0:8211/v1/completions \
     -H "Content-Type: application/json" \
     -d "{\"model\":\"glm-5.2\",\"prompt\":\"The capital of France is\",\"max_tokens\":32,\"temperature\":1}"'
   ```
   Flag: non-200, empty `choices`, or a response that takes noticeably longer than expected for
   32 tokens at your currently-serving profile's known decode speed (see the README matrix) —
   anything reading well under the profile's floor on an otherwise-idle server is a red flag.

3. **`/metrics` sanity** — `num_requests_running` should be 0 (or a known operator probe) at a
   randomly-sampled idle check, not chronically nonzero:
   ```bash
   ssh NODE1 'curl -s http://FABRIC_IP_0:8211/metrics | grep num_requests_running'
   ```
   Flag: nonzero on 2+ consecutive checks with no known operator activity.

4. **`free -g` trend per node**, logged to CSV:
   ```bash
   for n in NODE1 NODE2 NODE3 NODE4; do
     ssh "$n" "free -g | awk -v n=$n -v t=\$(date -u +%FT%TZ) '/Mem:/{print t\",\"n\",\"\$3\",\"\$7} /Swap:/{print t\",\"n\",\"\$3}'"
   done >> watchdog-mem-trend.csv
   ```
   Flags: `avail < 6G` at idle; `swap_used > 6G` (see the normal-band note above); a 3-day
   monotonic decline in `avail` at matched idle windows.

5. **`docker ps` drift vs. expected fleet.** Compare actual state per node against what you
   expect to be serving. Flag any unexpected container.

6. **Who-is-connected audit.** Run the client-side `dport = :8211` trace above. Flag any
   source that isn't a known operator probe or gateway.

7. **Disk > 90%** per node — flag it, but don't let an agent start deleting images
   unsupervised; image-layer dedup means the reclaimable-space estimate `docker system df`
   reports is often much larger than what a prune actually frees.

### Weekly checks

1. **Temp-0 spot-check triplet** — run the same fixed prompt 3x at `temperature=0` and diff
   the outputs. This tracks the nondeterminism anomaly documented in `docs/findings.md`; log
   the diverging fraction per profile per week to build a rate trend.
2. **Cold-prefill gate at ~190K tokens** — catches host-OOM-during-prefill on undersized
   configs, the historical killer failure class. Gate PASS = completes without OOM/timeout.
3. **Page-cache trim during a confirmed idle window**, logged so the memory trend data above
   isn't confused by the trim event itself.

### Escalation rules

**Auto-fix allowed (no human needed):**
- Page-cache trim (`sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`) — always safe, always
  reversible.
- Restarting a **dead process inside an already-running container that is part of a retained
  fleet you deliberately keep warm** — a restart by container ID, not a rebuild or profile
  change. Do not extend this to any throwaway/DCP-ladder-attempt container; those get full
  recreation, which needs a human unless an exact, pre-agreed recipe is being followed by a
  supervised agent.

**Escalate to a human (stop, do not auto-fix):**
- Node unreachable (SSH fails, not just the health check) — possible hardware/network issue.
- A reserve breach that isn't the expected clean-abort-and-log pattern, or any boot that OOMs
  the *host* rather than being cleanly refused by the allocator guard.
- An unknown client in the who-is-connected audit — treat as a potential live-traffic
  incident until identified, don't silently block it, surface it.
- Restart loops — more than 2 auto-restart attempts on the same container within an hour is a
  real fault, not a transient blip.
- Any disk >90% with no obvious reclaimable candidate — deletion decisions on a cluster like
  this should never be automated blindly, even under time pressure.
