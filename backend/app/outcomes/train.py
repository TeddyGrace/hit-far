"""Outcome models: which of your measured positions predict a bad shot?

Small-data discipline throughout. One binary problem at a time; two candidate models (a
regularized logistic regression and a very shallow gradient-boosted tree ensemble); repeated
stratified cross-validation; a bootstrap CI on AUC; a label-permutation null so a lucky model on
20 swings doesn't read as a finding. Factors are reported only when the model (permutation
importance across CV folds) and a model-free check (single-feature AUC with a bootstrap CI) agree.
"""

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOGREG, HGB = "logreg", "hgb"
KIND_TITLES = {LOGREG: "Logistic regression (L2)", HGB: "Gradient-boosted trees (depth 2)"}
PROTOCOL = "repeated stratified k-fold CV (k<=5); AUC CI = bootstrap over swings of out-of-fold probabilities"


@dataclass
class CVSettings:
    n_splits: int = 5
    n_repeats: int = 10
    n_boot: int = 1000
    n_perm: int = 20  # label permutations for the chance-level null
    perm_repeats: int = 2  # CV repeats per permutation
    importance_repeats: int = 3  # CV repeats used for permutation importance
    importance_shuffles: int = 3
    min_n: int = 20
    min_class: int = 6
    reliable_ci_low: float = 0.60
    hgb_margin: float = 0.03  # the tree model must beat logistic regression by this much AUC
    seed: int = 0


def make_estimator(kind: str, n: int):
    if kind == LOGREG:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
    if kind == HGB:
        return HistGradientBoostingClassifier(
            max_depth=2, max_iter=80, learning_rate=0.05, min_samples_leaf=max(4, n // 10),
            l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=0)
    raise ValueError(kind)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney statistic (ties count half)."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def bootstrap_auc_ci(y: np.ndarray, score: np.ndarray, n_boot: int, rng: np.random.Generator,
                     level: float = 0.95) -> tuple[float, float]:
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auc(y[idx], score[idx])
        if np.isfinite(a):
            stats.append(a)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def eligibility(y: np.ndarray, cfg: CVSettings) -> dict:
    n, pos = len(y), int((y == 1).sum())
    neg = n - pos
    need = {"swings": max(0, cfg.min_n - n), "positive": max(0, cfg.min_class - pos),
            "negative": max(0, cfg.min_class - neg)}
    return {"n": n, "n_pos": pos, "n_neg": neg, "eligible": not any(need.values()), "need": need}


def _splitter(y: np.ndarray, cfg: CVSettings, n_repeats: int, seed: int) -> tuple[RepeatedStratifiedKFold, int]:
    k = int(max(2, min(cfg.n_splits, np.bincount(y.astype(int)).min())))
    return RepeatedStratifiedKFold(n_splits=k, n_repeats=n_repeats, random_state=seed), k


@dataclass
class CVRun:
    repeat_aucs: list[float]
    oof: np.ndarray  # mean out-of-fold probability per swing across repeats
    folds: list[list[tuple[object, np.ndarray]]]  # per repeat: [(fitted model, test idx)] (kept for the first few)


def cross_validate(kind: str, X: np.ndarray, y: np.ndarray, cfg: CVSettings, n_repeats: int | None = None,
                   seed: int | None = None, keep_folds: int = 0) -> CVRun:
    n_repeats = n_repeats or cfg.n_repeats
    splitter, k = _splitter(y, cfg, n_repeats, cfg.seed if seed is None else seed)
    base = make_estimator(kind, len(y))
    oof_sum = np.zeros(len(y))
    repeat_aucs, folds = [], []
    cur = np.zeros(len(y))
    cur_folds: list[tuple[object, np.ndarray]] = []
    for i, (tr, te) in enumerate(splitter.split(X, y)):
        m = clone(base).fit(X[tr], y[tr])
        cur[te] = m.predict_proba(X[te])[:, 1]
        r = i // k
        if r < keep_folds:
            cur_folds.append((m, te))
        if i % k == k - 1:
            repeat_aucs.append(auc(y, cur))
            oof_sum += cur
            if r < keep_folds:
                folds.append(cur_folds)
            cur, cur_folds = np.zeros(len(y)), []
    return CVRun(repeat_aucs=repeat_aucs, oof=oof_sum / n_repeats, folds=folds)


def null_auc_95(kind: str, X: np.ndarray, y: np.ndarray, cfg: CVSettings) -> float:
    """95th percentile of CV AUC when the labels are shuffled: what chance looks like at this n."""
    rng = np.random.default_rng(cfg.seed + 1)
    vals = []
    for p in range(cfg.n_perm):
        yp = rng.permutation(y)
        vals.append(float(np.mean(cross_validate(kind, X, yp, cfg, n_repeats=cfg.perm_repeats, seed=p).repeat_aucs)))
    return float(np.percentile(vals, 95)) if vals else 0.5


def permutation_importance_cv(run: CVRun, X: np.ndarray, y: np.ndarray, cfg: CVSettings
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Drop in pooled out-of-fold AUC when one feature is shuffled within each test fold.
    Returns (mean drop, standard error) per feature."""
    rng = np.random.default_rng(cfg.seed + 2)
    drops = np.zeros((0, X.shape[1]))
    for folds in run.folds:
        base = np.zeros(len(y))
        for m, te in folds:
            base[te] = m.predict_proba(X[te])[:, 1]
        base_auc = auc(y, base)
        for _ in range(cfg.importance_shuffles):
            row = np.zeros(X.shape[1])
            for j in range(X.shape[1]):
                p = base.copy()
                for m, te in folds:
                    Xte = X[te].copy()
                    Xte[:, j] = rng.permutation(Xte[:, j])
                    p[te] = m.predict_proba(Xte)[:, 1]
                row[j] = base_auc - auc(y, p)
            drops = np.vstack([drops, row])
    if len(drops) == 0:
        return np.zeros(X.shape[1]), np.zeros(X.shape[1])
    se = drops.std(axis=0, ddof=1) / np.sqrt(len(drops)) if len(drops) > 1 else np.zeros(X.shape[1])
    return drops.mean(axis=0), se


def univariate(x: np.ndarray, y: np.ndarray, cfg: CVSettings, level: float = 0.95) -> dict:
    """Model-free check: does this metric alone separate your good and bad shots?
    AUC > 0.5 means higher values go with the problem."""
    ok = ~np.isnan(x)
    xs, ys = x[ok], y[ok]
    a = auc(ys, xs)
    lo, hi = bootstrap_auc_ci(ys, xs, max(400, cfg.n_boot), np.random.default_rng(cfg.seed + 3), level)
    good, bad = xs[ys == 0], xs[ys == 1]
    p = float(mannwhitneyu(bad, good).pvalue) if len(good) and len(bad) else 1.0
    return {
        "auc": a, "ci": [lo, hi], "p": p,
        "good_median": float(np.median(good)) if len(good) else None,
        "bad_median": float(np.median(bad)) if len(bad) else None,
        "direction": "higher" if a > 0.5 else "lower",
        "n": int(ok.sum()),
    }


def jsonsafe(obj):
    """NaN/inf -> None, numpy scalars -> Python (JSONB and HTTP JSON both reject NaN)."""
    if isinstance(obj, dict):
        return {k: jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonsafe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class TrainResult:
    kind: str
    estimator: object  # fitted on all data
    features: list[str]
    eval: dict  # JSON-safe evaluation + factors
    oof: dict  # swing id (str) -> out-of-fold probability


def _summ(run: CVRun, y: np.ndarray, cfg: CVSettings) -> dict:
    lo, hi = bootstrap_auc_ci(y, run.oof, cfg.n_boot, np.random.default_rng(cfg.seed))
    return {"cv_auc": float(np.mean(run.repeat_aucs)), "cv_auc_sd": float(np.std(run.repeat_aucs)),
            "cv_auc_ci": [lo, hi]}


def train_problem(X: np.ndarray, y: np.ndarray, features: list[str], swing_ids: list[str], cfg: CVSettings,
                  feature_info: dict[str, dict] | None = None) -> TrainResult:
    runs = {kind: cross_validate(kind, X, y, cfg, keep_folds=cfg.importance_repeats) for kind in (LOGREG, HGB)}
    cands = {kind: _summ(run, y, cfg) for kind, run in runs.items()}
    # The simpler model wins unless the tree model is clearly better.
    kind = HGB if cands[HGB]["cv_auc"] >= cands[LOGREG]["cv_auc"] + cfg.hgb_margin else LOGREG
    chosen, run = cands[kind], runs[kind]
    null95 = null_auc_95(kind, X, y, cfg)
    reliable = chosen["cv_auc_ci"][0] > cfg.reliable_ci_low and chosen["cv_auc"] > null95

    imp, imp_se = permutation_importance_cv(run, X, y, cfg)
    # Many metrics are tested at once, so the model-free check is an exact rank test at a
    # Bonferroni-corrected level (bootstrap CIs run narrow at these sample sizes), and the model's
    # reliance must clear two standard errors: chance correlations mostly drop out.
    alpha = 0.05 / max(1, len(features))
    factors = []
    for j, name in enumerate(features):
        u = univariate(X[:, j], y, cfg)
        uni_sig = u["p"] < alpha
        model_sig = imp[j] > 0.005 and imp[j] - 2 * imp_se[j] > 0
        factors.append({
            "feature": name, **(feature_info or {}).get(name, {}),
            "importance": float(imp[j]), "importance_se": float(imp_se[j]),
            "univariate_auc": u["auc"], "univariate_ci": u["ci"], "univariate_p": u["p"],
            "direction": u["direction"],
            "good_median": u["good_median"], "bad_median": u["bad_median"], "n": u["n"],
            "supported": bool(uni_sig and model_sig),
        })
    factors.sort(key=lambda f: (not f["supported"], -f["importance"]))

    pos = int(y.sum())
    evaluation = {
        "n": len(y), "n_pos": pos, "n_neg": len(y) - pos,
        "kind": kind, "kind_title": KIND_TITLES[kind],
        **chosen,
        "null_auc_95": null95,
        "baseline": {"auc": 0.5, "positive_rate": pos / len(y)},
        "reliable": bool(reliable),
        "reliable_rule": f"AUC CI lower bound > {cfg.reliable_ci_low:.2f} and AUC > shuffled-label 95th percentile",
        "candidates": cands,
        "protocol": PROTOCOL,
        "cv": asdict(cfg),
        "n_features": len(features),
        "factor_rule": f"model importance > 2 SE and Mann-Whitney p < {alpha:.4f} (0.05 / {len(features)} metrics)",
        "factors": factors[:12],
    }
    final = make_estimator(kind, len(y)).fit(X, y)
    return TrainResult(kind=kind, estimator=final, features=features, eval=jsonsafe(evaluation),
                       oof={sid: float(p) for sid, p in zip(swing_ids, run.oof)})


def recipe_auc(kind: str, X: np.ndarray, y: np.ndarray, cfg: CVSettings) -> float:
    """CV AUC of a (kind, feature set) recipe on this data - how a previous model is re-scored so
    old and new are compared on the same swings with the same protocol."""
    return float(np.mean(cross_validate(kind, X, y, cfg).repeat_aucs))


def what_if(estimator, x: np.ndarray, features: list[str], factors: list[dict]) -> list[dict]:
    """For each supported factor: move just that metric to your good-shot median and report the
    change in predicted probability (individual conditional expectation at that point)."""
    base = float(estimator.predict_proba(x[None, :])[0, 1])
    out = []
    for f in factors:
        if not f.get("supported") or f.get("good_median") is None or f["feature"] not in features:
            continue
        j = features.index(f["feature"])
        if np.isnan(x[j]):
            continue
        x2 = x.copy()
        x2[j] = f["good_median"]
        p = float(estimator.predict_proba(x2[None, :])[0, 1])
        out.append({"feature": f["feature"], "value": float(x[j]), "target": f["good_median"],
                    "probability": base, "probability_if": p, "delta": p - base})
    out.sort(key=lambda w: w["delta"])
    return out
