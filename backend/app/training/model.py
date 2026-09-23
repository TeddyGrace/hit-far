"""The learned event model (BiLSTM over pose features) + ordered decoding + inference.

Per-frame classification into 9 classes (8 events + background), like GolfDB's SwingNet but over
pose sequences instead of raw frames. Decoding picks one frame per event with the constraint that
events are strictly ordered, maximizing total log-probability (dynamic programming).
"""

from dataclasses import dataclass

import numpy as np

from app.training.features import N_FEATURES, resample, to_features

N_EVENTS = 8
N_CLASSES = N_EVENTS + 1  # last class = background
# Test-time scale search: candidate resampling factors (1/stride). Training covers the
# equivalent range via time-scale augmentation, so any capture fps maps onto a familiar tempo.
INFERENCE_SCALES = (1.0, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 8)


def build_model(hidden: int = 128, layers: int = 2, dropout: float = 0.2):
    import torch.nn as nn

    class EventNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                               dropout=dropout if layers > 1 else 0.0)
            self.out = nn.Linear(2 * hidden, N_CLASSES)

        def forward(self, x):  # (B, T, F) -> (B, T, C) logits
            h, _ = self.rnn(self.inp(x))
            return self.out(h)

    return EventNet()


def decode_ordered(log_probs: np.ndarray) -> list[int]:
    """log_probs: (T, N_EVENTS). Returns strictly increasing frame per event maximizing the sum."""
    T, E = log_probs.shape
    if T < E:
        return list(np.linspace(0, max(T - 1, 0), E).astype(int))
    score = log_probs[:, 0].copy()
    back = np.zeros((E, T), dtype=int)
    for e in range(1, E):
        # best previous position strictly before t
        best_prev = np.full(T, -np.inf)
        arg_prev = np.zeros(T, dtype=int)
        run_best, run_arg = -np.inf, 0
        for t in range(1, T):
            if score[t - 1] > run_best:
                run_best, run_arg = score[t - 1], t - 1
            best_prev[t], arg_prev[t] = run_best, run_arg
        score = best_prev + log_probs[:, e]
        back[e] = arg_prev
    frames = [int(np.argmax(score))]
    for e in range(E - 1, 0, -1):
        frames.append(int(back[e][frames[-1]]))
    return frames[::-1]


@dataclass
class Prediction:
    frames: list[int]  # in the original sequence's frame indices
    confidences: list[float]
    scale: float


def predict(model, xy: np.ndarray, vis: np.ndarray) -> Prediction:
    """Run at several time scales; keep the scale where the model is most confident."""
    import torch

    model.eval()
    T = len(xy)
    best: Prediction | None = None
    best_score = -np.inf
    with torch.no_grad():
        for s in INFERENCE_SCALES:
            n = max(N_EVENTS + 1, int(round(T * s)))
            if n > T:
                continue
            feats = to_features(resample(xy, n / T), resample(vis, n / T))
            logits = model(torch.from_numpy(feats)[None])[0]
            probs = torch.softmax(logits, dim=-1).numpy()
            frames = decode_ordered(np.log(probs[:, :N_EVENTS] + 1e-9))
            conf = [float(probs[f, e]) for e, f in enumerate(frames)]
            score = float(np.mean(np.log(np.array(conf) + 1e-9)))
            if score > best_score:
                scale = n / T
                orig = [int(min(T - 1, round(f / scale))) for f in frames]
                for i in range(1, len(orig)):  # keep strict order after rounding
                    orig[i] = max(orig[i], orig[i - 1] + 1) if orig[i - 1] + 1 < T else orig[i - 1]
                best, best_score = Prediction(orig, conf, scale), score
    assert best is not None
    return best


def golfdb_tolerance(events: list[int]) -> int:
    """GolfDB's PCE tolerance: max(1, round((impact - address) / 30)) frames."""
    return int(max(np.round((events[5] - events[0]) / 30), 1))


def pce(pred: list[int], true: list[int], tol: int | None = None) -> np.ndarray:
    """Per-event correctness (bool array of length 8) within tolerance."""
    tol = golfdb_tolerance(true) if tol is None else tol
    return np.abs(np.array(pred) - np.array(true)) <= tol
