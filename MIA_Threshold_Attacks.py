"""
Additional membership inference attacks, evaluated alongside the shadow-model RandomForest attack.

Attacks (all black-box, using only the model's output on (x, y)):
    - loss        : Loss-threshold attack (Yeom et al., 2018). Member if CE loss <= t.
    - confidence  : Confidence-threshold attack. Member if p(y|x) >= t.
    - mentropy    : Modified-entropy threshold attack (Song & Mittal, 2021). Member if Mentr <= t.
    - label_only  : Label-only gap attack (Yeom et al., 2018). Member if the prediction is correct.
    - shadow_rf   : The existing shadow-model + RandomForest attack (Shokri et al., 2017), re-scored
                    here so it also gets AUC / TPR@low-FPR.

Thresholds are calibrated on shadow models trained on X_shadow only (same assumption as the
shadow-model attack: the attacker has data from the same distribution, never the target's
members). Calibrated thresholds are cached to JSON so they are computed once per experiment.

Every attack reports accuracy / F1 / precision / recall at its decision threshold, plus the
threshold-free AUC and TPR at 1% FPR.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

THRESHOLD_ATTACKS = ["loss", "confidence", "mentropy"]
ALL_ATTACKS = ["shadow_rf", "loss", "confidence", "mentropy", "label_only"]
ATTACK_NAMES = {
    "shadow_rf": "Shadow-model RF",
    "loss": "Loss threshold",
    "confidence": "Confidence threshold",
    "mentropy": "Modified entropy",
    "label_only": "Label-only (gap)",
}
METRICS = ["acc", "f1", "precision", "recall", "auc", "tpr_at_1fpr"]
EPS = 1e-12


# ---------------------------------------------------------------------------
# Per-sample scores
# ---------------------------------------------------------------------------

def _to_tensors(X, y, input_shape=None):
    X_t = torch.tensor(np.asarray(X, dtype=np.float32))
    if input_shape is not None:
        X_t = X_t.reshape(-1, *input_shape)
    y_t = torch.tensor(np.asarray(y).ravel().astype(np.int64))
    return X_t, y_t


@torch.no_grad()
def get_probabilities(model, X, input_shape=None, device="cuda", batch_size=512):
    """Softmax outputs of `model` on X (DataFrame or array), as a numpy array."""
    model.eval()
    model.to(device)
    X_t = torch.tensor(np.asarray(X, dtype=np.float32))
    if input_shape is not None:
        X_t = X_t.reshape(-1, *input_shape)
    probs = []
    for i in range(0, len(X_t), batch_size):
        probs.append(F.softmax(model(X_t[i:i + batch_size].to(device)), dim=1).cpu())
    return torch.cat(probs).numpy()


def membership_scores(probs, y):
    """Per-sample membership scores. Higher score = more likely a member, for every attack."""
    y = np.asarray(y).ravel().astype(np.int64)
    idx = np.arange(len(y))
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1 - EPS)  # float64: 1-EPS rounds to 1.0 in float32
    p_y = p[idx, y]

    loss = -np.log(p_y)

    # Modified entropy (Song & Mittal, 2021):
    # Mentr(p, y) = -(1 - p_y) log(p_y) - sum_{i != y} p_i log(1 - p_i)
    others = -p * np.log(1 - p)
    others[idx, y] = 0.0
    mentr = -(1 - p_y) * np.log(p_y) + others.sum(axis=1)

    correct = (probs.argmax(axis=1) == y).astype(float)

    return {
        "loss": -loss,          # low loss -> member
        "confidence": p_y,      # high confidence -> member
        "mentropy": -mentr,     # low modified entropy -> member
        "label_only": correct,  # correctly classified -> member
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def tpr_at_fpr(labels, scores, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    valid = fpr <= target_fpr
    return float(tpr[valid].max()) if valid.any() else 0.0


def attack_metrics(labels, predictions, scores):
    labels = np.asarray(labels).astype(int)
    predictions = np.asarray(predictions).astype(int)
    return {
        "acc": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "auc": roc_auc_score(labels, scores),
        "tpr_at_1fpr": tpr_at_fpr(labels, scores, 0.01),
    }


def best_threshold(labels, scores):
    """Threshold maximising balanced accuracy (member if score >= t)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    balanced_acc = (tpr + (1 - fpr)) / 2
    i = int(np.argmax(balanced_acc))
    return float(thresholds[i]), float(balanced_acc[i])


# ---------------------------------------------------------------------------
# Threshold calibration on shadow models
# ---------------------------------------------------------------------------

def _train_shadow(model, train_ds, epochs, lr, batch_size, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    for _ in tqdm(range(epochs), desc="Shadow model (threshold calibration)", leave=False):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
    return model


def calibrate_thresholds(shadow_model_architecture, X_shadow, y_shadow, num_features, num_classes,
                         num_shadow_models=1, learning_rate=0.01, num_epochs=100, batch_size=32,
                         input_shape=None, device="cuda", seed=42):
    """Train shadow models on halves of X_shadow and pick each attack's threshold on their outputs.

    Use the same training recipe as the target model (the attacker is assumed to know it,
    exactly as for the shadow-model attack).
    """
    pooled_scores = {a: [] for a in THRESHOLD_ATTACKS}
    pooled_labels = []

    for k in range(num_shadow_models):
        X_in, X_out, y_in, y_out = train_test_split(
            X_shadow, y_shadow, test_size=0.5, random_state=seed + k)
        X_in_t, y_in_t = _to_tensors(X_in, y_in, input_shape)

        torch.manual_seed(seed + k)
        shadow = shadow_model_architecture(input_size=num_features, output_size=num_classes)
        shadow = _train_shadow(shadow, TensorDataset(X_in_t, y_in_t), num_epochs, learning_rate,
                               batch_size, device)

        n = min(len(X_in), len(X_out))
        in_scores = membership_scores(get_probabilities(shadow, X_in[:n], input_shape, device), y_in[:n])
        out_scores = membership_scores(get_probabilities(shadow, X_out[:n], input_shape, device), y_out[:n])
        for a in THRESHOLD_ATTACKS:
            pooled_scores[a].append(np.concatenate([in_scores[a], out_scores[a]]))
        pooled_labels.append(np.concatenate([np.ones(n), np.zeros(n)]))

        del shadow
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    labels = np.concatenate(pooled_labels)
    thresholds = {}
    for a in THRESHOLD_ATTACKS:
        t, acc = best_threshold(labels, np.concatenate(pooled_scores[a]))
        thresholds[a] = t
        print(f"[Calibration] {ATTACK_NAMES[a]}: threshold={t:.6f}, shadow balanced acc={acc:.4f}")
    return thresholds


def load_or_calibrate_thresholds(cache_path, *args, recalibrate=False, **kwargs):
    """Load cached thresholds from `cache_path`, or calibrate and save them."""
    if os.path.exists(cache_path) and not recalibrate:
        with open(cache_path) as f:
            thresholds = json.load(f)
        print(f"Loaded MIA thresholds from {cache_path}: {thresholds}")
        return thresholds
    thresholds = calibrate_thresholds(*args, **kwargs)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Saved MIA thresholds to {cache_path}")
    return thresholds


# ---------------------------------------------------------------------------
# Attack suite
# ---------------------------------------------------------------------------

class AttackSuite:
    """Runs all attacks on (forget vs non-member) and (retain vs non-member) for a given model.

    Call `track(model)` once per unlearning epoch to build a history, and `evaluate(model)` for
    one-off reference points (original model, retrain-from-scratch baseline).
    """

    def __init__(self, retain_X, retain_y, forget_X, forget_y, non_member_X, non_member_y,
                 thresholds, attack_model=None, input_shape=None, device="cuda"):
        self.splits = {
            "forget": (forget_X, forget_y),
            "retain": (retain_X, retain_y),
        }
        self.non_member = (non_member_X, non_member_y)
        self.thresholds = thresholds
        self.attack_model = attack_model
        self.input_shape = input_shape
        self.device = device
        self.attacks = [a for a in ALL_ATTACKS if a != "shadow_rf" or attack_model is not None]
        self.history = {a: {s: {m: [] for m in METRICS} for s in self.splits} for a in self.attacks}

    def _evaluate_split(self, model, member_X, member_y):
        nm_X, nm_y = self.non_member
        p_in = get_probabilities(model, member_X, self.input_shape, self.device)
        p_out = get_probabilities(model, nm_X, self.input_shape, self.device)
        labels = np.concatenate([np.ones(len(p_in)), np.zeros(len(p_out))])

        s_in, s_out = membership_scores(p_in, member_y), membership_scores(p_out, nm_y)
        results = {}
        for a in THRESHOLD_ATTACKS:
            scores = np.concatenate([s_in[a], s_out[a]])
            results[a] = attack_metrics(labels, scores >= self.thresholds[a], scores)

        scores = np.concatenate([s_in["label_only"], s_out["label_only"]])
        results["label_only"] = attack_metrics(labels, scores >= 0.5, scores)

        if self.attack_model is not None:
            # Same features the RF was trained on: softmax vector with integer column names.
            feats = pd.DataFrame(np.concatenate([p_in, p_out]))
            member_col = list(self.attack_model.classes_).index(True)
            scores = self.attack_model.predict_proba(feats)[:, member_col]
            results["shadow_rf"] = attack_metrics(labels, self.attack_model.predict(feats), scores)
        return results

    def evaluate(self, model):
        """Returns {attack: {split: {metric: value}}} without touching the history."""
        per_split = {s: self._evaluate_split(model, X, y) for s, (X, y) in self.splits.items()}
        return {a: {s: per_split[s][a] for s in self.splits} for a in self.attacks}

    def track(self, model):
        """Evaluate and append to the per-epoch history."""
        results = self.evaluate(model)
        for a in self.attacks:
            for s in self.splits:
                for m in METRICS:
                    self.history[a][s][m].append(results[a][s][m])
        summary = ", ".join(f"{ATTACK_NAMES[a]} F={results[a]['forget']['acc']:.3f}/"
                            f"R={results[a]['retain']['acc']:.3f}" for a in self.attacks)
        print(f"\t[MIA acc forget/retain] {summary}")
        return results

    # -- reporting ----------------------------------------------------------

    def history_dataframe(self, references=None):
        """Per-epoch history as a DataFrame. `references` = {name: evaluate() output} are added as
        constant columns (e.g. {'Original': ..., 'Retrain Baseline': ...})."""
        n_epochs = len(self.history[self.attacks[0]]["forget"]["acc"])
        columns = {"Epoch": list(range(n_epochs))}
        for a in self.attacks:
            for s in self.splits:
                for m in METRICS:
                    columns[f"{a} {s} {m}"] = self.history[a][s][m]
        for ref_name, ref in (references or {}).items():
            for a in self.attacks:
                for s in self.splits:
                    for m in METRICS:
                        columns[f"{ref_name} {a} {s} {m}"] = [ref[a][s][m]] * n_epochs
        return pd.DataFrame(columns)

    def summary_table(self, references=None):
        """Final-epoch forget/retain metrics per attack next to the reference models."""
        rows = []
        for a in self.attacks:
            for s in self.splits:
                row = {"attack": ATTACK_NAMES[a], "split": s}
                for m in ["acc", "auc", "tpr_at_1fpr"]:
                    row[f"unlearned {m}"] = self.history[a][s][m][-1]
                    for ref_name, ref in (references or {}).items():
                        row[f"{ref_name} {m}"] = ref[a][s][m]
                rows.append(row)
        return pd.DataFrame(rows)

    def plot(self, metric, path, references=None, ylabel=None):
        """One subplot per attack: forget/retain curves over unlearning epochs + reference lines."""
        import matplotlib.pyplot as plt

        ref_styles = {"Retrain Baseline": "--", "Original": ":"}
        fig, axes = plt.subplots(1, len(self.attacks), figsize=(4 * len(self.attacks), 4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, a in zip(axes, self.attacks):
            for s, color in zip(self.splits, ["C0", "C1"]):
                ax.plot(self.history[a][s][metric], color=color, label=f"{s.capitalize()}")
                for ref_name, ref in (references or {}).items():
                    ax.axhline(ref[a][s][metric], color=color, linestyle=ref_styles.get(ref_name, "-."),
                               linewidth=1.5, label=f"{ref_name}: {s}")
            ax.set_title(ATTACK_NAMES[a])
            ax.set_xlabel("Number of Epochs")
            ax.grid(True, linestyle=":", linewidth=0.5)
        axes[0].set_ylabel(ylabel or metric)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(path)
        plt.close(fig)
