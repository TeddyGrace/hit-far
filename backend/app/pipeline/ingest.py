"""Video preprocessing: checksum, playback proxy, probe."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.config import get_settings


class IngestError(Exception):
    pass


@dataclass
class ProbeResult:
    fps: float
    width: int
    height: int
    num_frames: int
    duration_s: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffmpeg_bin() -> str:
    configured = get_settings().ffmpeg_bin
    if configured:
        return configured
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _source_fps(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        cap.release()
    return fps


def make_proxy(src: Path, dst: Path) -> None:
    """Transcode to H.264/yuv420p MP4 at the source's native frame rate, constant frame rate.

    * applies rotation metadata (phones), so pixels are upright for both the viewer and the model
    * keeps every captured frame (slow-mo stays e.g. 240 fps; playback speed is a viewer control)
    * caps the long side at PROXY_MAX_SIDE to keep pose inference and browser playback fast
    """
    fps = _source_fps(src)
    if not fps or fps <= 0 or fps > 1000:
        raise IngestError(f"could not determine frame rate (got {fps!r})")
    m = get_settings().proxy_max_side
    vf = f"scale='if(gt(iw,ih),min({m},iw),-2)':'if(gt(iw,ih),-2,min({m},ih))',format=yuv420p"
    cmd = [
        ffmpeg_bin(), "-y", "-v", "error", "-i", str(src),
        "-an", "-vf", vf, "-fps_mode", "cfr", "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "15",  # short GOP: snappy frame stepping in the viewer
        "-movflags", "+faststart", str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise IngestError(f"ffmpeg failed: {proc.stderr.strip()[-2000:]}")


def probe(path: Path) -> ProbeResult:
    """Probe by decoding: container frame counts are unreliable for phone video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IngestError("could not open video")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while cap.grab():
            n += 1
    finally:
        cap.release()
    if n == 0 or not fps:
        raise IngestError("video has no decodable frames")
    return ProbeResult(fps=float(fps), width=w, height=h, num_frames=n, duration_s=n / fps)
