"""The GolfDB pose pool survives a dead worker (OOM-kill) and doesn't hang on SIGTERM. No torch
needed, unlike test_training.py."""

import pytest

from app.training import golfdb


def _crash_once(args):
    """Stand-in for _extract_one: the first time clip 3 is seen, its worker is SIGKILLed (like the
    kernel OOM killer)."""
    import os
    import signal

    clip_id, marker = args
    if clip_id == 3 and not os.path.exists(marker):
        open(marker, "w").close()
        os.kill(os.getpid(), signal.SIGKILL)
    return clip_id, b"ok", None


def _always_crash(args):
    import os
    import signal

    os.kill(os.getpid(), signal.SIGKILL)


def _sigterm_handler_of_worker(args):
    import signal

    return args[0], None, str(signal.getsignal(signal.SIGTERM))


def test_pose_pool_retries_with_fewer_workers_after_a_worker_dies(tmp_path):
    marker = str(tmp_path / "crashed")
    got = {}
    golfdb.run_pool([(i, marker) for i in range(8)], 4, lambda r: got.__setitem__(r[0], r[1]), fn=_crash_once)
    assert sorted(got) == list(range(8)) and set(got.values()) == {b"ok"}


def test_pose_pool_gives_up_when_a_single_worker_dies():
    with pytest.raises(RuntimeError, match="single worker"):
        golfdb.run_pool([(1, "")], 2, lambda r: None, fn=_always_crash)


def test_pose_workers_restore_default_sigterm():
    """Workers inherit the job runner's "finish the current job" handler; they must not keep it,
    or a broken pool waits forever for workers that ignore SIGTERM."""
    import signal

    from app import worker

    old = signal.signal(signal.SIGTERM, worker._handle_signal)
    try:
        got = []
        golfdb.run_pool([(1, "")], 1, got.append, fn=_sigterm_handler_of_worker)
    finally:
        signal.signal(signal.SIGTERM, old)
    assert got[0][2] == str(signal.SIG_DFL)
