"""Stage-2 club detector: a small heatmap CNN on a crop centred on the hands.

Input: a square crop around the pose grip (side = CROP_TORSOS x torso, so the whole shaft fits),
three channels: grayscale, grayscale minus the swing's static background (range clutter removed),
and frame-to-frame motion. Output: two heatmaps (grip end, clubhead) at stride 4.

The clubhead heatmap is turned into a score per shaft direction (probability mass in a narrow
wedge from the grip). Those scores go through the same temporal decoding as the stage-1 line
tracker, so the learned detector only replaces the per-frame evidence, and its confidence is the
probability the model puts on the chosen direction.
"""

import numpy as np

from app.pipeline.club import CONF_Z, N_BINS

CROP_VERSION = 1
CROP = 128  # crop side in pixels
STRIDE = 4
HM = CROP // STRIDE  # heatmap side
CROP_TORSOS = 4.2  # crop side in torso lengths (shaft ~1.9 torso from the grip)
MIN_RADIUS_TORSOS = 0.3  # heatmap cells closer to the grip than this don't count as a clubhead
# Direction wedges (Gaussian sigma, degrees): narrow for the training loss (precision), wider for
# the per-direction score, so confidence reads as "the shaft is within a few degrees of this".
LOSS_WEDGE_SIGMA_DEG = 2.0
SCORE_WEDGE_SIGMA_DEG = 5.0

N_KEYPOINTS = 2  # 0 = grip end, 1 = clubhead


# --- Crops -------------------------------------------------------------------------------------


def crop_matrix(g: np.ndarray, torso: float, rot: float = 0.0, scale: float = 1.0, mirror: bool = False) -> np.ndarray:
    """2x3 affine: image pixels -> crop pixels, centred on the grip. rot (radians) rotates the
    content; mirror flips it horizontally (both used for augmentation)."""
    s = CROP / (CROP_TORSOS * torso) * scale
    c, si = np.cos(rot), np.sin(rot)
    R = np.array([[c, -si], [si, c]]) * s
    if mirror:
        R = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ R
    t = np.array([CROP / 2, CROP / 2]) - R @ g
    return np.hstack([R, t[:, None]])


def apply_affine(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def transform_angle(angle: float, rot: float = 0.0, mirror: bool = False) -> float:
    a = angle + rot
    return (np.pi - a) % (2 * np.pi) if mirror else a % (2 * np.pi)


def make_crop(frames: list[np.ndarray], background: np.ndarray, i: int, g: np.ndarray, torso: float) -> np.ndarray:
    """(CROP, CROP, 3) uint8 input crop for window frame i (frames: blurred gray, from read_frames)."""
    import cv2

    gray = frames[i].astype(np.float32)
    fg = np.clip((gray - background) / 2 + 128, 0, 255)
    a, b = frames[max(0, i - 1)].astype(np.float32), frames[min(len(frames) - 1, i + 1)].astype(np.float32)
    motion = np.clip(np.abs(b - a), 0, 255)
    stack = np.stack([gray, fg, motion], axis=-1).astype(np.uint8)
    M = crop_matrix(g, torso)
    # Outside the frame: black, no foreground (128 = zero difference from the background), no motion.
    return cv2.warpAffine(stack, M, (CROP, CROP), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 128, 0))


def to_input(crops: np.ndarray) -> np.ndarray:
    """(N, CROP, CROP, 3) uint8 -> (N, 3, CROP, CROP) float32, roughly zero-centred."""
    x = crops.astype(np.float32).transpose(0, 3, 1, 2) / 255.0 - np.array([0.5, 0.5, 0.0], np.float32)[None, :, None, None]
    x[:, 2] *= 4.0  # motion is mostly near zero; spread it out
    return x


# --- Heatmap geometry --------------------------------------------------------------------------


def _cell_polar() -> tuple[np.ndarray, np.ndarray]:
    """Angle and radius (crop pixels) of each heatmap cell centre from the crop centre (the grip)."""
    c = (np.arange(HM) + 0.5) * STRIDE - CROP / 2
    xx, yy = np.meshgrid(c, c)
    return np.arctan2(yy, xx), np.hypot(xx, yy)


CELL_ANGLE, CELL_RADIUS = _cell_polar()
TORSO_CROP_PX = CROP / CROP_TORSOS


def wedge_weights(angle: np.ndarray, sigma_deg: float = LOSS_WEDGE_SIGMA_DEG) -> np.ndarray:
    """(N,) angles -> (N, HM, HM) soft wedge masks from the grip out to the crop edge."""
    d = np.angle(np.exp(1j * (CELL_ANGLE[None] - np.asarray(angle)[:, None, None])))
    w = np.exp(-0.5 * (np.degrees(d) / sigma_deg) ** 2)
    return (w * (CELL_RADIUS[None] >= MIN_RADIUS_TORSOS * TORSO_CROP_PX)).astype(np.float32)


BIN_WEDGES = wedge_weights(np.arange(N_BINS) * (2 * np.pi / N_BINS), SCORE_WEDGE_SIGMA_DEG)  # (N_BINS, HM, HM)


def gaussian_target(pt: np.ndarray, sigma_cells: float = 1.0) -> np.ndarray:
    """Normalized heatmap target for a point in crop pixels."""
    c = (np.arange(HM) + 0.5) * STRIDE
    xx, yy = np.meshgrid(c, c)
    h = np.exp(-0.5 * (((xx - pt[0]) ** 2 + (yy - pt[1]) ** 2) / (sigma_cells * STRIDE) ** 2))
    return (h / max(h.sum(), 1e-9)).astype(np.float32)


def direction_scores(prob: np.ndarray) -> np.ndarray:
    """(N, HM, HM) clubhead probabilities -> (N, N_BINS) z-scores for the shared decoder.

    The mass in each direction's wedge is a probability q; z = CONF_Z + 0.75 logit(q), so the
    decoder's confidence sigmoid((z - CONF_Z) / 0.75) is exactly q on the chosen direction."""
    q = np.einsum("nhw,bhw->nb", prob, BIN_WEDGES)
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return np.clip(CONF_Z + 0.75 * np.log(q / (1 - q)), -5, 25)


def clubhead_along(prob: np.ndarray, angle: float) -> tuple[np.ndarray | None, float]:
    """Soft-argmax of one frame's clubhead heatmap inside the wedge of `angle`: crop pixels and
    the wedge's share of the probability mass."""
    w = wedge_weights(np.array([angle]), SCORE_WEDGE_SIGMA_DEG)[0]
    p = prob * w
    m = float(p.sum())
    if m < 1e-4:
        return None, m
    c = (np.arange(HM) + 0.5) * STRIDE
    xx, yy = np.meshgrid(c, c)
    return np.array([(p * xx).sum() / m, (p * yy).sum() / m]), m


# --- Network -----------------------------------------------------------------------------------


def build_model(width: int = 24):
    import torch
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class ClubNet(nn.Module):
        """U-Net-ish encoder/decoder with coordinate channels (the grip is always the crop centre,
        so position relative to it is what matters)."""

        def __init__(self):
            super().__init__()
            w = width
            self.d1 = block(5, w, 2)  # 64
            self.d2 = block(w, 2 * w, 2)  # 32
            self.d3 = block(2 * w, 4 * w, 2)  # 16
            self.d4 = block(4 * w, 6 * w, 2)  # 8
            self.u3 = block(6 * w + 4 * w, 4 * w)  # 16
            self.u2 = block(4 * w + 2 * w, 2 * w)  # 32
            self.head = nn.Conv2d(2 * w, N_KEYPOINTS, 1)
            c = torch.linspace(-1, 1, CROP)
            self.register_buffer("coords", torch.stack(torch.meshgrid(c, c, indexing="xy"))[None], persistent=False)

        def forward(self, x):  # (B, 3, CROP, CROP) -> (B, K, HM, HM) logits
            x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], 1)
            e1 = self.d1(x)
            e2 = self.d2(e1)
            e3 = self.d3(e2)
            e4 = self.d4(e3)
            up = nn.functional.interpolate
            y = self.u3(torch.cat([up(e4, scale_factor=2), e3], 1))
            y = self.u2(torch.cat([up(y, scale_factor=2), e2], 1))
            return self.head(y)

    return ClubNet()


def heatmap_probs(model, x: np.ndarray, batch: int = 64) -> np.ndarray:
    """(N, 3, CROP, CROP) inputs -> (N, K, HM, HM) spatial-softmax probabilities."""
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            logits = model(torch.from_numpy(x[i:i + batch]))
            B, K = logits.shape[:2]
            out.append(torch.softmax(logits.reshape(B, K, -1), -1).reshape(B, K, HM, HM).numpy())
    return np.concatenate(out) if out else np.zeros((0, N_KEYPOINTS, HM, HM), np.float32)
