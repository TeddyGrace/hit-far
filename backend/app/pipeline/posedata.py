"""In-memory pose sequence + its .npz serialization (the artifact stored in object storage)."""

import io
from dataclasses import dataclass

import numpy as np

from app.pipeline.landmarks import SCHEMA_VERSION


@dataclass
class PoseData:
    kp2d: np.ndarray  # (T, 33, 2) pixel coords in the proxy video; NaN where not detected
    visibility: np.ndarray  # (T, 33) in [0, 1]; 0 where not detected
    world: np.ndarray  # (T, 33, 3) metres, hip-centred (MediaPipe world landmarks); NaN where not detected
    fps: float
    width: int
    height: int
    schema: str = SCHEMA_VERSION

    @property
    def num_frames(self) -> int:
        return int(self.kp2d.shape[0])

    @property
    def detected(self) -> np.ndarray:
        return ~np.isnan(self.kp2d[:, 0, 0])

    def mean_confidence(self) -> float | None:
        d = self.detected
        return float(self.visibility[d].mean()) if d.any() else None

    def to_npz(self) -> bytes:
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            kp2d=self.kp2d.astype(np.float32),
            visibility=self.visibility.astype(np.float32),
            world=self.world.astype(np.float32),
            meta=np.array([self.fps, self.width, self.height], dtype=np.float64),
            schema=np.array(self.schema),
        )
        return buf.getvalue()

    @classmethod
    def from_npz(cls, data: bytes) -> "PoseData":
        z = np.load(io.BytesIO(data))
        fps, w, h = z["meta"]
        return cls(
            kp2d=z["kp2d"].astype(float),
            visibility=z["visibility"].astype(float),
            world=z["world"].astype(float),
            fps=float(fps),
            width=int(w),
            height=int(h),
            schema=str(z["schema"]),
        )
