"""The container's real CPU and memory limits.

`os.cpu_count()` reports the host's cores, not the container's quota (a Railway replica capped at
8 vCPU can see dozens), and one MediaPipe process per reported core would run out of memory.
"""

import os
from pathlib import Path

POSE_WORKER_MB = 900  # MediaPipe heavy + a decoded clip, per process, with headroom


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def available_cpus() -> int:
    quota = _read("/sys/fs/cgroup/cpu.max")  # cgroup v2: "<quota> <period>" or "max <period>"
    if quota and not quota.startswith("max"):
        q, p = quota.split()[:2]
        return max(1, int(int(q) / int(p)))
    q, p = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")  # v1
    if q and p and int(q) > 0:
        return max(1, int(int(q) / int(p)))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


def memory_limit_mb() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read(path)
        if v and v != "max" and int(v) < 1 << 50:
            return int(v) // (1 << 20)
    return None


def pose_workers() -> int:
    """Parallel pose-extraction processes: one per vCPU, capped so they fit in memory."""
    n = available_cpus()
    mem = memory_limit_mb()
    if mem:
        n = min(n, max(1, (mem - 1024) // POSE_WORKER_MB))  # keep ~1 GB for the main process
    return n
