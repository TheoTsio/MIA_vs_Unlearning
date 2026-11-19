"""
Offline LiRA (Likelihood Ratio Attack; Carlini et al., "Membership Inference Attacks From First Principles",
IEEE S&P 2022) for the MIA-vs-unlearning experiments. Used as MIA-3 in lr_grid_runner.py.

Reference ("OUT") models
    For every target model, K reference models with the same architecture are trained on random subsets of
    the shadow data X_shadow (same size as the target's training set). X_shadow is disjoint from the target
    data, so every sample the attack is evaluated on (forget, retain and non-member samples) is OUT for every
    reference model: the offline variant of LiRA, which needs no models trained on the evaluated samples.

Score of a sample (x, y) for a model f
    phi = log( p_y / (1 - p_y) ) = z_y - logsumexp_{c != y} z_c       (logit-scaled true-class confidence)
    The reference models give phi_1 .. phi_K for the same (x, y): mean mu(x, y), std sigma(x, y).
    Membership score = (phi_target - mu) / sigma; higher = more likely a member. This ranks samples exactly
    like the one-sided Gaussian likelihood test Phi((phi - mu) / sigma) used for the offline attack.
    With few reference models (K < 64) Carlini et al. recommend a global ("fixed") variance: sigma is the
    pooled std over all samples. VARIANCE = "global" (default) or "per_example".

Files (models/lira/<exp>/): pool_keys.npy (one key per pool row), ref_<k>.npz (phi of reference model k on
all pool rows + its config and accuracy). Rows are matched by a hash of (features, label), so the statistics
are valid for any sample drawn from the pools (e.g. other seeds in run_seed_repeats.py).
"""
import hashlib
import json
import os

import numpy as np
import torch

import experiment_runner as er
from utils.data_split import get_retain_forget_split, get_target_shadow_split, get_train_test_split

LIRA_DIR = os.path.join("models", "lira")
N_REFERENCE = 16
VARIANCE = "global"
EVAL_BATCH = 128

# Training of the reference models: same procedure as the target models (train_target_and_perform_MIA_attack.py:
# SGD, momentum 0.9, cross-entropy, batch 32, no early stopping; images lr 1e-3 / 300 epochs, tabular lr 1e-2 /
# 100 epochs). stop_train_acc: stop once the training accuracy of an epoch reaches this value (the targets fit
# their training data to ~100%); keeps the image models affordable. Edit if a target used other settings.
_IMG = dict(lr=1e-3, epochs=300, batch_size=32, momentum=0.9, stop_train_acc=0.999)
_TAB = dict(lr=1e-2, epochs=100, batch_size=32, momentum=0.9, stop_train_acc=0.999)
TRAIN_CONFIG = {e: dict(_IMG if er.EXPERIMENTS[e]["dataset"] in ("cifar10", "mufac") else _TAB)
                for e in er.ALL_EXPERIMENTS}


# ---------------------------------------------------------------------------
# Pools, keys and phi
# ---------------------------------------------------------------------------

def row_keys(X, y):
    """One key per row: hash of the float32 features and the label (matches er._features tensors)."""
    X = np.ascontiguousarray(np.asarray(X.reshape(len(X), -1) if torch.is_tensor(X) else X, dtype=np.float32))
    y = np.asarray(y).astype(np.int64)
    return np.array([hashlib.blake2b(X[i].tobytes() + y[i].tobytes(), digest_size=16).hexdigest()
                     for i in range(len(X))])


@torch.no_grad()
def phi_scores(model, X, y):
    """Logit-scaled true-class confidence z_y - logsumexp_{c!=y} z_c (float64, numerically stable)."""
    model.eval()
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    out = []
    for i in range(0, len(X), EVAL_BATCH):
        z = model(X[i:i + EVAL_BATCH].to(er.device)).double()
        yb = y_t[i:i + EVAL_BATCH].to(er.device)
        z_y = z.gather(1, yb[:, None]).squeeze(1)
        z_other = z.scatter(1, yb[:, None], float("-inf"))
        out.append((z_y - torch.logsumexp(z_other, dim=1)).cpu())
    return torch.cat(out).numpy()


def rebuild_pools(exp_id, data_seed=42):
    """Same splits as experiment_runner.prepare_experiment. Returns shadow data, the evaluation pools and
    the size of the target's training set."""
    cfg = er.EXPERIMENTS[exp_id]
    shape = cfg["input_shape"]
    X, y, num_features, num_classes = er.load_dataset(cfg["dataset"], data_seed)
    X_target, X_shadow, y_target, y_shadow, X_mia, y_mia = get_target_shadow_split(X, y, seed=data_seed)
    del X, y
    X_train, _, y_train, _ = get_train_test_split(X_target, y_target, seed=data_seed)
    del X_target, y_target
    X_retain, X_forget, y_retain, y_forget = get_retain_forget_split(X_train, y_train, seed=data_seed,
                                                                     forget_class=None)
    n_train = len(X_train)
    del X_train, y_train
    pools = {name: (er._features(Xp, shape), er._labels(yp)) for name, Xp, yp in
             (("forget", X_forget, y_forget), ("retain", X_retain, y_retain), ("non_member", X_mia, y_mia))}
    shadow = (er._features(X_shadow, shape), er._labels(y_shadow))
    return dict(shadow=shadow, pools=pools, n_train=n_train, num_features=num_features,
                num_classes=num_classes)


# ---------------------------------------------------------------------------
# Reference statistics used at attack time
# ---------------------------------------------------------------------------

class LiraScorer:
    """Holds mu / sigma of every pool row; scores the samples of a prepared experiment."""

    def __init__(self, exp_id, variance=VARIANCE, min_refs=2):
        d = os.path.join(LIRA_DIR, exp_id)
        keys_path = os.path.join(d, "pool_keys.npy")
        refs = sorted(f for f in os.listdir(d) if f.startswith("ref_") and f.endswith(".npz")) \
            if os.path.isdir(d) else []
        if not os.path.exists(keys_path) or len(refs) < min_refs:
            raise FileNotFoundError(
                f"LiRA reference models for {exp_id} missing ({len(refs)} found in {d}). "
                f"Run: python train_lira_references.py --experiments {exp_id}")
        keys = np.load(keys_path)
        phis = np.stack([np.load(os.path.join(d, f))["phi"] for f in refs])  # (K, N)
        self.n_reference = len(refs)
        self.mu = phis.mean(0)
        self.sigma_example = phis.std(0, ddof=1)
        self.sigma_global = float(np.sqrt(np.mean(phis.var(0, ddof=1))))
        self.variance = variance
        self.index = {k: i for i, k in reversed(list(enumerate(keys)))}  # duplicates -> first row
        self.prepared = {}

    def prepare(self, name, X, y):
        """Look up mu / sigma for the rows of a sample (call once per sample set)."""
        rows = np.array([self.index[k] for k in row_keys(X, y)])
        sigma = (self.sigma_example[rows] if self.variance == "per_example"
                 else np.full(len(rows), self.sigma_global))
        self.prepared[name] = (self.mu[rows], np.maximum(sigma, 1e-6))

    def score(self, name, phi):
        mu, sigma = self.prepared[name]
        return (phi - mu) / sigma
