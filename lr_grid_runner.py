"""
Learning-rate grid with separated selection and evaluation (reviewer response).

For every target model (1a ... 3d) x unlearning method (NegGrad, SCRUB, SFTC) x learning rate in the fixed
grid LR_GRID, unlearn for 30 epochs (seed 42) and evaluate every epoch on two DISJOINT halves of the
evaluation data. Attacks: MIA-1 shadow-model RandomForest, MIA-2 loss threshold, MIA-3 offline LiRA
(lira.py; needs the reference models of train_lira_references.py).

    selection half   used only to choose the learning rate and the epoch
    evaluation half  used only for every reported number

How the halves are built: prepare_experiment() (experiment_runner.py, unchanged) draws n equal-sized
forget / retain / non-member samples. Each of these three samples and the test set is split 50/50 with
SPLIT_SEED, stratified by class label (unstratified if a class has fewer than 2 samples). The original and
retrained models are evaluated on both halves, so each half has its own reference values.

Selection rule (Eq. best_epoch of the paper, Section 5.4, applied to the whole grid):
    (lr*, epoch*) = argmin over lr in LR_GRID, epoch in 1..30 (finite weights only) of
                    |MIA-1 forget AUC - Retr. MIA-1 forget AUC| + |Test Acc - Retr. Test Acc|
    computed on the selection half only. Ties -> smaller learning rate, then earlier epoch.
Reported: the evaluation-half metrics of (lr*, epoch*) ("Sel."), and of the last epoch of the run with lr*
("Final", or div. if that run diverged). Per learning rate, the epoch is chosen the same way (selection half)
for the sensitivity table and figures.

Metrics per half (forget-vs-non-members and retain-vs-non-members): test accuracy, forget/retain
classification accuracy, MIA-1/2/3 AUC (raw and max(AUC, 1-AUC)), TPR@1% and @0.1% FPR, and the balanced accuracy of
MIA-1 using the RandomForest's own decision (attack_model.predict).

Usage:
    python lr_grid_runner.py                              # run everything missing, then report
    python lr_grid_runner.py --experiments 2c --methods neg_grad
    python lr_grid_runner.py --report-only

Outputs (results_lrgrid_lira/):
    runs/<exp>/<method>/lr_<lr>.json (+ _epochs.csv)   every run, both halves, every epoch (cache; resumable)
    grid_epochs.csv        every (exp, method, lr, epoch): selection- and evaluation-half metrics
    grid_per_lr.csv        every (exp, method, lr): selection-half best epoch, evaluation-half metrics
    selection.csv          every (exp, method): selected lr / epoch and evaluation-half metrics
    table_selected.tex     main table (evaluation half): Test Acc., MIA-1/2 AUC, MIA-3 (LiRA) AUC and TPR@1%FPR
    table_selected_mia1acc.tex   MIA-1 balanced accuracy, forget and retain, Start/Retr./Sel./Final
    table_lr_grid.tex      appendix: search range, per-lr results and the selected value of every configuration
    fig_sensitivity_<method>_<metric>_<state>.pdf      evaluation-half metric vs. learning rate
"""
import argparse
import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import experiment_runner as er
import lira
from preprocess_data import create_merged_loader
from utils.unlearning_utils import RandomDistributionGenerator

# ---------------------------------------------------------------------------
# Fixed protocol (identical for all methods and configurations; do not adapt to results)
# ---------------------------------------------------------------------------
LR_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SEED = 42          # unlearning seed and MIA sample draw (as in experiment_runner)
SPLIT_SEED = 42    # selection / evaluation split
EPOCHS, BATCH_SIZE, MOMENTUM = 30, 32, 0.9
OUT_DIR = "results_lrgrid_lira"   # results_lrgrid/ = first grid (MIA-3 = max confidence), kept unchanged
GRID_VERSION = 2   # bump when evaluation logic changes, to invalidate cached runs (2: MIA-3 = LiRA)
ATTACKS = ["mia1", "mia2", "lira"]   # MIA-1 shadow-model RF, MIA-2 loss, MIA-3 offline LiRA
ATTACK_LABELS = {"mia1": "MIA-1", "mia2": "MIA-2", "lira": "MIA-3 (LiRA)"}
HALVES = ("sel", "eval")
SETS = ("forget", "retain")
SELECTION_CRITERION = ("argmin over (lr, epoch) of |MIA-1 forget AUC - retr| + |test acc - retr| "
                       "on the selection half; ties -> smaller lr, earlier epoch")


# ---------------------------------------------------------------------------
# Selection / evaluation split and two-half evaluation
# ---------------------------------------------------------------------------

def split_half(labels, seed=SPLIT_SEED):
    """Disjoint 50/50 index split, stratified by label when every class has at least 2 samples."""
    idx = np.arange(len(labels))
    try:
        a, b = train_test_split(idx, test_size=0.5, random_state=seed, stratify=labels)
    except ValueError:
        a, b = train_test_split(idx, test_size=0.5, random_state=seed)
    return np.sort(a), np.sort(b)


def add_halves(prep):
    """prep.halves[half][pool] = indices into prep.X_test / prep.mia_sets[pool] / prep.non_member."""
    pools = {"test": prep.y_test.numpy(), "forget": prep.mia_sets["forget"][1],
             "retain": prep.mia_sets["retain"][1], "non_member": prep.non_member[1]}
    prep.halves = {"sel": {}, "eval": {}}
    for pool, labels in pools.items():
        prep.halves["sel"][pool], prep.halves["eval"][pool] = split_half(labels)
    prep.half_sizes = {h: {p: int(len(i)) for p, i in prep.halves[h].items()} for h in HALVES}
    prep.lira = lira.LiraScorer(prep.exp_id, min_refs=lira.N_REFERENCE)
    for name, (X, y) in list(prep.mia_sets.items()) + [("non_member", prep.non_member)]:
        prep.lira.prepare(name, X, y)
    return prep


_PREDICT_CHECKED = set()


@torch.no_grad()
def _forward(model, X, y):
    """Softmax probabilities exactly as experiment_runner._probabilities, and the LiRA phi from the same logits."""
    model.eval()
    probs, phis = [], []
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    for i in range(0, len(X), er.EVAL_BATCH_SIZE):
        z = model(X[i:i + er.EVAL_BATCH_SIZE].to(er.device))
        probs.append(torch.softmax(z, dim=1).double().cpu())
        zd, yb = z.double(), y_t[i:i + er.EVAL_BATCH_SIZE].to(er.device)
        z_y = zd.gather(1, yb[:, None]).squeeze(1)
        phis.append((z_y - torch.logsumexp(zd.scatter(1, yb[:, None], float("-inf")), dim=1)).cpu())
    return torch.cat(probs).numpy(), torch.cat(phis).numpy()


def _scores_and_decision(prep, pool, probs, phi, y):
    """MIA-1 (RF P(member)), MIA-2 (log p_y) and MIA-3 (LiRA) scores, and the RF's own member decision.

    The RF is run once: predict() of a RandomForestClassifier is classes_[argmax(predict_proba)], so the
    decision is taken from the same probabilities. This is checked against attack_model.predict once.
    """
    attack_model = prep.attack_model
    proba = attack_model.predict_proba(pd.DataFrame(probs))
    classes = np.asarray(attack_model.classes_)
    decision = classes.take(np.argmax(proba, axis=1)).astype(bool)
    if id(attack_model) not in _PREDICT_CHECKED:
        assert np.array_equal(decision, np.asarray(attack_model.predict(pd.DataFrame(probs))).astype(bool)), \
            "RF decision from predict_proba differs from attack_model.predict"
        _PREDICT_CHECKED.add(id(attack_model))
    p_y = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    scores = {"mia1": proba[:, list(classes).index(True)], "mia2": np.log(p_y),
              "lira": prep.lira.score(pool, phi)}
    return scores, decision


def evaluate_halves(prep, model):
    """Evaluate one model once and compute every metric separately on the selection and evaluation half."""
    p_test = er._probabilities(model, prep.X_test)
    test_correct = p_test.argmax(1) == prep.y_test.numpy()

    nm_X, nm_y = prep.non_member
    p_out, phi_out = _forward(model, nm_X, nm_y)
    s_out, member_out = _scores_and_decision(prep, "non_member", p_out, phi_out, nm_y)

    per_set = {}
    for split, (X_in, y_in) in prep.mia_sets.items():
        p_in, phi_in = _forward(model, X_in, y_in)
        scores, decision = _scores_and_decision(prep, split, p_in, phi_in, y_in)
        per_set[split] = dict(correct=p_in.argmax(1) == y_in, scores=scores, member_pred=decision)

    out = {}
    for h in HALVES:
        idx = prep.halves[h]
        nm = idx["non_member"]
        m = {"test_acc": float(test_correct[idx["test"]].mean())}
        for split in SETS:
            d, mem = per_set[split], idx[split]
            m[f"{split}_acc"] = float(d["correct"][mem].mean())
            labels = np.r_[np.ones(len(mem)), np.zeros(len(nm))]
            for a in ATTACKS:
                scores = np.r_[d["scores"][a][mem], s_out[a][nm]]
                auc = float(roc_auc_score(labels, scores))
                m[f"{split}_{a}_auc_raw"] = auc
                m[f"{split}_{a}_auc"] = max(auc, 1 - auc)
                m[f"{split}_{a}_tpr1"] = er._tpr_at_fpr(labels, scores, 0.01)
                m[f"{split}_{a}_tpr01"] = er._tpr_at_fpr(labels, scores, 0.001)
            predicted = np.r_[d["member_pred"][mem], member_out[nm]]
            m[f"{split}_mia1_bacc"] = float(balanced_accuracy_score(labels.astype(bool), predicted))
        out[h] = m
    return out


def distance(m, ref):
    """Eq. (best_epoch): the same criterion as experiment_runner.distance_to_retrain."""
    return er.distance_to_retrain(m, ref)


# ---------------------------------------------------------------------------
# One unlearning run (same procedure as experiment_runner.run_method, two-half evaluation)
# ---------------------------------------------------------------------------

def run_config(method, lr):
    return dict(method=method, lr=lr, epochs=EPOCHS, batch_size=BATCH_SIZE, momentum=MOMENTUM, seed=SEED,
                split_seed=SPLIT_SEED, grid_version=GRID_VERSION, runner_version=er.RUNNER_VERSION,
                lira_reference_models=lira.N_REFERENCE, lira_variance=lira.VARIANCE)


def run_path(exp_id, method, lr):
    return os.path.join(OUT_DIR, "runs", exp_id, method, f"lr_{lr:g}.json")


def load_run(exp_id, method, lr):
    path = run_path(exp_id, method, lr)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rec = json.load(f)
    return rec if rec.get("config") == run_config(method, lr) else None


def run_one(prep, method, lr):
    config = run_config(method, lr)
    er.set_random_seed(SEED)  # every run starts from the same RNG state, independent of run order

    model = copy.deepcopy(prep.target_model).to(er.device)
    teacher = copy.deepcopy(prep.target_model).to(er.device).eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
    retain_loader = DataLoader(TensorDataset(prep.X_retain, prep.y_retain), batch_size=BATCH_SIZE, shuffle=True)
    forget_loader = DataLoader(TensorDataset(prep.X_forget, prep.y_forget), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(prep.X_test, prep.y_test), batch_size=BATCH_SIZE, shuffle=False)

    kwargs = {}
    if method == "neg_grad":
        algorithm = er._neg_grad_module.neg_grad
    elif method == "scrub":
        algorithm = er._scrub_module.scrub
        kwargs = dict(teacher_model=teacher)
    elif method == "sftc":
        algorithm = er._sftc_module.sftc_unlearn
        kwargs = dict(teacher_model=teacher,
                      dummy_model=RandomDistributionGenerator(dist="normal", dimensions=prep.num_classes),
                      merged_loader=create_merged_loader(prep.X_retain, prep.y_retain, prep.X_forget,
                                                         prep.y_forget, batch_size=BATCH_SIZE, shuffle=True))
    else:
        raise ValueError(f"Unknown method: {method}")

    modules = [er._neg_grad_module, er._scrub_module, er._sftc_module]
    saved_tqdm = [m.tqdm for m in modules]
    for m in modules:
        m.tqdm = er._no_tqdm

    tag = f"[{prep.exp_id} {er.METHOD_NAMES[method]} lr={lr:g}]"
    history = [dict(epoch=0, **prep.start2)]
    diverged_at = None
    t0 = time.time()
    try:
        for epoch in range(1, EPOCHS + 1):
            with contextlib.redirect_stdout(io.StringIO()):
                algorithm(model, retain_loader, test_loader, test_loader, forget_loader, optimizer, None,
                          criterion, epochs=1, device=er.device, **kwargs)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                diverged_at = epoch
                print(f"  {tag} epoch {epoch:2d}/{EPOCHS} | diverged (inf/NaN weights); "
                      f"keeping epochs 1-{epoch - 1}", flush=True)
                break
            m = evaluate_halves(prep, model)
            history.append(dict(epoch=epoch, **m))
            s, e = m["sel"], m["eval"]
            print(f"  {tag} epoch {epoch:2d}/{EPOCHS} | test {e['test_acc']:.3f} | eval: forget AUC MIA-1 "
                  f"{e['forget_mia1_auc']:.3f} LiRA {e['forget_lira_auc']:.3f}, MIA-1 acc forget "
                  f"{e['forget_mia1_bacc']:.3f} retain "
                  f"{e['retain_mia1_bacc']:.3f} | sel distance {distance(s, prep.retrain2['sel']):.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        for mod, t in zip(modules, saved_tqdm):
            mod.tqdm = t

    record = dict(exp_id=prep.exp_id, dataset=prep.dataset, method=method, lr=lr, config=config,
                  diverged_at_epoch=diverged_at, history=history, start=prep.start2, retrain=prep.retrain2,
                  sizes=prep.sizes, half_sizes=prep.half_sizes, runtime_seconds=round(time.time() - t0, 1),
                  environment=er.environment_info())
    path = run_path(prep.exp_id, method, lr)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    flat_history(record).to_csv(path.replace(".json", "_epochs.csv"), index=False)

    del model, teacher, optimizer
    er.free_memory()
    print(f"  {tag} done in {record['runtime_seconds']:.0f}s"
          + (f", diverged at epoch {diverged_at}" if diverged_at else ""), flush=True)
    return record


def flat_history(rec):
    rows = []
    for h in rec["history"]:
        row = dict(exp_id=rec["exp_id"], method=rec["method"], lr=rec["lr"], epoch=h["epoch"])
        for half in HALVES:
            row.update({f"{half}_{k}": v for k, v in h[half].items()})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Running (one subprocess per experiment: data and models loaded once, memory freed afterwards)
# ---------------------------------------------------------------------------

def lrs_for(exp_id, method, learning_rates=None):
    """The learning rates to run / report: the whole grid, or the one fixed rate of LEARNING_RATES."""
    if learning_rates is None:
        return LR_GRID
    lr = learning_rates.get(exp_id, {}).get(method)
    return [] if lr is None else [lr]


def worker(exp_id, methods, learning_rates=None):
    todo = [(m, lr) for m in methods for lr in lrs_for(exp_id, m, learning_rates)
            if load_run(exp_id, m, lr) is None]
    if not todo:
        print(f"[{exp_id}] all runs cached", flush=True)
        return
    prep = add_halves(er.prepare_experiment(exp_id, seed=SEED))
    prep.start2 = evaluate_halves(prep, prep.target_model)
    prep.retrain2 = evaluate_halves(prep, prep.retrain_model)
    print(f"  LiRA: {prep.lira.n_reference} reference models, {prep.lira.variance} variance", flush=True)
    for h, name in (("sel", "selection"), ("eval", "evaluation")):
        s, r = prep.start2[h], prep.retrain2[h]
        print(f"  {name} half {prep.half_sizes[h]}: original forget AUC MIA-1 {s['forget_mia1_auc']:.3f} "
              f"MIA-2 {s['forget_mia2_auc']:.3f} LiRA {s['forget_lira_auc']:.3f} (TPR@1%FPR "
              f"{s['forget_lira_tpr1']:.3f}) | retrained MIA-1 {r['forget_mia1_auc']:.3f} MIA-2 "
              f"{r['forget_mia2_auc']:.3f} LiRA {r['forget_lira_auc']:.3f}", flush=True)
    failed = []
    for i, (method, lr) in enumerate(todo, 1):
        print(f"[{exp_id}] run {i}/{len(todo)}: {er.METHOD_NAMES[method]} lr={lr:g}", flush=True)
        try:
            run_one(prep, method, lr)
        except Exception:  # an error (not a divergence) must not stop the other runs
            import traceback
            traceback.print_exc()
            failed.append((method, lr))
            er.free_memory()
    if failed:
        print(f"[{exp_id}] FAILED runs: {failed}", flush=True)
        raise SystemExit(1)


def run(experiments, methods, learning_rates=None):
    """learning_rates=None: the whole LR_GRID. A dict {exp: {method: lr}}: only that lr per configuration."""
    failed = []
    for exp_id in experiments:
        job = dict(exp_id=exp_id, methods=methods, learning_rates=learning_rates)
        cmd = [sys.executable, "-u", "-W", "ignore", os.path.abspath(__file__), "--worker", json.dumps(job)]
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        code = proc.wait()
        if code != 0:
            reason = "killed: out of memory?" if code in (-9, 137) else f"exit code {code}"
            print(f"[{exp_id}] FAILED ({reason}); continuing with the next experiment", flush=True)
            failed.append(exp_id)
    if failed:
        print(f"\nFailed experiments: {failed}. Re-run; finished runs are cached and not repeated.")


# ---------------------------------------------------------------------------
# Selection and report
# ---------------------------------------------------------------------------

REPORT_KEYS = (["test_acc"] + [f"forget_{a}_auc" for a in ATTACKS] + [f"retain_{a}_auc" for a in ATTACKS]
               + [f"{s}_mia1_bacc" for s in SETS] + [f"forget_{a}_tpr1" for a in ATTACKS]
               + [f"forget_{a}_tpr01" for a in ATTACKS] + ["forget_acc", "retain_acc"])


def best_epoch_on_selection(rec):
    """Epoch (>= 1, finite weights) minimising the selection-half distance; ties -> earliest."""
    ref = rec["retrain"]["sel"]
    candidates = rec["history"][1:]
    if not candidates:
        return None
    return min(candidates, key=lambda h: (distance(h["sel"], ref), h["epoch"]))


def collect(experiments, methods, learning_rates=None):
    per_lr, selected, epochs, missing = [], [], [], []
    for exp_id in experiments:
        for method in methods:
            recs = {lr: load_run(exp_id, method, lr) for lr in lrs_for(exp_id, method, learning_rates)}
            missing += [(exp_id, method, lr) for lr, r in recs.items() if r is None]
            recs = {lr: r for lr, r in recs.items() if r is not None}
            if not recs:
                continue
            any_rec = next(iter(recs.values()))
            start, retr = any_rec["start"], any_rec["retrain"]
            base = dict(exp_id=exp_id, dataset=er.DATASET_NAMES[er.EXPERIMENTS[exp_id]["dataset"]],
                        method=method)
            ref_cols = {f"{state}_{half}_{k}": v[half][k] for state, v in (("start", start), ("retr", retr))
                        for half in HALVES for k in REPORT_KEYS}

            for lr, rec in recs.items():
                epochs.append(flat_history(rec))
                best = best_epoch_on_selection(rec)
                final = rec["history"][-1] if rec["diverged_at_epoch"] is None else None
                row = dict(base, lr=lr, diverged_at_epoch=rec["diverged_at_epoch"],
                           best_epoch=best["epoch"] if best else None,
                           sel_distance=distance(best["sel"], retr["sel"]) if best else np.nan,
                           eval_distance=distance(best["eval"], retr["eval"]) if best else np.nan)
                for k in REPORT_KEYS:
                    row[f"best_{k}"] = best["eval"][k] if best else np.nan
                    row[f"final_{k}"] = final["eval"][k] if final else np.nan
                per_lr.append(dict(row, **ref_cols))

            # Selection over the whole grid, on the selection half only.
            cands = [(distance(h["sel"], retr["sel"]), lr, h["epoch"], h, rec)
                     for lr, rec in recs.items() for h in rec["history"][1:]]
            row = dict(base, n_lrs_run=len(recs), grid=",".join(f"{lr:g}" for lr in LR_GRID),
                       criterion=SELECTION_CRITERION, **ref_cols)
            if cands:
                d, lr, ep, h, rec = min(cands, key=lambda c: (c[0], c[1], c[2]))
                final = rec["history"][-1] if rec["diverged_at_epoch"] is None else None
                row.update(selected_lr=lr, selected_epoch=ep, diverged_at_epoch=rec["diverged_at_epoch"],
                           sel_distance=d, eval_distance=distance(h["eval"], retr["eval"]),
                           sel_mia1_forget_auc=h["sel"]["forget_mia1_auc"])
                # How optimistic is reusing the selection data? Same model, other half.
                for k in REPORT_KEYS:
                    row[f"sel_{k}"] = h["eval"][k]
                    row[f"final_{k}"] = final["eval"][k] if final else np.nan
            selected.append(row)
    if missing:
        print(f"Missing runs ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    return (pd.DataFrame(per_lr), pd.DataFrame(selected),
            pd.concat(epochs, ignore_index=True) if epochs else pd.DataFrame())


def _f(x, d=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _lr_tex(lr):
    e = int(round(np.log10(lr)))
    return f"$10^{{{e}}}$" if np.isclose(lr, 10.0 ** e) else f"{lr:g}"


def _rows_by_dataset(df):
    order = [e for _, exps in er.TABLE_LAYOUT for e in exps]
    df = df.assign(_o=df["exp_id"].map(order.index), _m=df["method"].map(er.METHODS.index))
    return df.sort_values(["_o", "_m"])


def make_selected_table(sel, groups, caption, label):
    """groups: list of (header, key). Columns per group: Start, Retr., Sel., Final (evaluation half)."""
    n = 5 + 4 * len(groups)
    head = " & ".join(rf"\multicolumn{{4}}{{c}}{{{g}}}" for g, _ in groups)
    cmid = "".join(rf"\cmidrule(lr){{{6 + 4 * i}-{9 + 4 * i}}}" for i in range(len(groups)))
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l c l c c " + "cccc" * len(groups) + "}",
             r"\toprule", r" & & & & & " + head + r" \\", cmid,
             r"Dataset & Model & Method & lr & Ep. & "
             + " & ".join(["Start & Retr. & Sel. & Final"] * len(groups)) + r" \\", r"\midrule"]
    prev_ds, prev_exp = None, None
    for _, r in _rows_by_dataset(sel).iterrows():
        if prev_ds is not None and r["dataset"] != prev_ds:
            lines.append(r"\midrule")
        elif prev_exp is not None and r["exp_id"] != prev_exp:
            lines.append(rf"\cmidrule(lr){{2-{n}}}")
        new_exp = r["exp_id"] != prev_exp
        has_sel = pd.notna(r.get("selected_lr"))
        cells = [r["dataset"] if r["dataset"] != prev_ds else "", r["exp_id"] if new_exp else "",
                 er.METHOD_NAMES[r["method"]], _lr_tex(r["selected_lr"]) if has_sel else "--",
                 str(int(r["selected_epoch"])) if has_sel else "--"]
        for _, k in groups:
            final = "div." if has_sel and pd.notna(r["diverged_at_epoch"]) else _f(r.get(f"final_{k}", np.nan))
            cells += [_f(r[f"start_eval_{k}"]) if new_exp else "", _f(r[f"retr_eval_{k}"]) if new_exp else "",
                      _f(r.get(f"sel_{k}", np.nan)), final]
        lines.append(" & ".join(cells) + r" \\")
        prev_ds, prev_exp = r["dataset"], r["exp_id"]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def make_grid_table(per_lr, sel):
    """Appendix: search range, per-lr evaluation-half MIA-1 forget AUC (epoch chosen on the selection half),
    selected value in bold."""
    chosen = {(r["exp_id"], r["method"]): r.get("selected_lr") for _, r in sel.iterrows()}
    idx = per_lr.set_index(["exp_id", "method", "lr"])
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Learning-rate search. Every method and target model was run with the same grid "
        rf"({', '.join(_lr_tex(l) for l in LR_GRID)}; {EPOCHS} epochs, SGD with momentum {MOMENTUM}, batch "
        rf"size {BATCH_SIZE}, seed {SEED}). The evaluation data were split into disjoint selection and "
        r"evaluation halves. For every learning rate the epoch is chosen on the selection half "
        r"($\min |\mathrm{AUC}_{\text{MIA-1}} - \mathrm{AUC}^{\text{retr}}| + |\mathrm{Acc} - \mathrm{Acc}^{\text{retr}}|$); "
        r"each cell gives the MIA-1 forget AUC / test accuracy of that epoch on the evaluation half, with the "
        r"epoch in parentheses. Bold: learning rate selected on the selection half. "
        r"$^{\dagger}$: diverged later in the run; div.: diverged in the first epoch.}",
        r"\label{tab:lr_grid}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c l " + "c" * len(LR_GRID) + "}", r"\toprule",
        r"Dataset & Model & Method & " + " & ".join(_lr_tex(l) for l in LR_GRID) + r" \\", r"\midrule"]
    prev_ds, prev_exp = None, None
    for _, r in _rows_by_dataset(sel).iterrows():
        if prev_ds is not None and r["dataset"] != prev_ds:
            lines.append(r"\midrule")
        cells = [r["dataset"] if r["dataset"] != prev_ds else "", r["exp_id"] if r["exp_id"] != prev_exp else "",
                 er.METHOD_NAMES[r["method"]]]
        for lr in LR_GRID:
            key = (r["exp_id"], r["method"], lr)
            if key not in idx.index:
                cells.append("--")
                continue
            p = idx.loc[key]
            if pd.isna(p["best_epoch"]):
                cells.append("div.")
                continue
            txt = (f"{p['best_forget_mia1_auc']:.2f} / {p['best_test_acc']:.2f} ({int(p['best_epoch'])})"
                   + (r"$^{\dagger}$" if pd.notna(p["diverged_at_epoch"]) else ""))
            cells.append(rf"\textbf{{{txt}}}" if chosen.get((r["exp_id"], r["method"])) == lr else txt)
        lines.append(" & ".join(cells) + r" \\")
        prev_ds, prev_exp = r["dataset"], r["exp_id"]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def make_figures(per_lr, methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = {"mia1acc": ("MIA-1 balanced accuracy", "{s}_mia1_bacc", True),
               "mia1auc": ("MIA-1 AUC", "{s}_mia1_auc", True),
               "testacc": ("Test accuracy", "test_acc", False)}
    exps = [e for _, ex in er.TABLE_LAYOUT for e in ex if e in set(per_lr["exp_id"])]
    paths = []
    for method in methods:
        d_m = per_lr[per_lr["method"] == method]
        if d_m.empty:
            continue
        for key, (label, pattern, by_set) in metrics.items():
            for state, state_txt in (("final", "last epoch"), ("best", "epoch chosen on the selection half")):
                cols = 4
                rows = int(np.ceil(len(exps) / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
                for ax, exp_id in zip(axes.flat, exps):
                    g = d_m[d_m["exp_id"] == exp_id].sort_values("lr")
                    if g.empty:
                        ax.axis("off")
                        continue
                    sets = SETS if by_set else (None,)
                    for c, s in enumerate(sets):
                        k = pattern.format(s=s)
                        ls = "-" if s != "retain" else "--"
                        name = s or "test"
                        ax.plot(g["lr"], g[f"{state}_{k}"], marker="o", ls=ls, color=f"C{c}", ms=4, lw=1.3,
                                label=name)
                        div = g[g[f"{state}_{k}"].isna()]
                        if len(div):
                            ax.plot(div["lr"], np.full(len(div), 0.02 if key == "testacc" else 0.98), "x",
                                    color=f"C{c}", ms=7, mew=1.5)
                        ax.axhline(g[f"retr_eval_{k}"].iloc[0], color=f"C{c}", ls=":", lw=1)
                        ax.axhline(g[f"start_eval_{k}"].iloc[0], color=f"C{c}", ls="-.", lw=0.7, alpha=0.6)
                    ax.set_xscale("log")
                    ax.set_title(f"{exp_id} ({g['dataset'].iloc[0]})", fontsize=9)
                    ax.set_ylim((0.0, 1.02) if key == "testacc" else (0.4, 1.02))
                    ax.grid(alpha=0.3)
                for ax in list(axes.flat)[len(exps):]:
                    ax.axis("off")
                for ax in axes[-1]:
                    ax.set_xlabel("Learning rate")
                for ax in axes[:, 0]:
                    ax.set_ylabel(label)
                axes.flat[0].legend(fontsize=7, loc="best")
                fig.suptitle(f"{er.METHOD_NAMES[method]}: {label} vs. learning rate, evaluation half "
                             f"({state_txt}). Dotted: retrained, dash-dot: original, x: diverged", fontsize=9)
                fig.tight_layout()
                path = os.path.join(OUT_DIR, f"fig_sensitivity_{method}_{key}_{state}.pdf")
                fig.savefig(path)
                plt.close(fig)
                paths.append(path)
    return paths


COLLAPSE_ACC = 0.10  # Best / Final test accuracies below this are bolded (collapsed model)


def make_paper_table(sel, selection_note=None):
    """The paper table (tab:utility_mia_all) in its original layout, one block per target model with the three
    methods stacked: Test Acc., MIA-1, MIA-2 and MIA-3 (LiRA) AUC, each Start / Retr. / Best / Final.
    Evaluation half only; Best = epoch chosen on the selection half (in parentheses after the method)."""
    def num(v):
        return "--" if v is None or not np.isfinite(v) else f"{v:.2f}"

    def stack(values, align="c"):
        return rf"\begin{{tabular}}[c]{{@{{}}{align}@{{}}}}" + r" \\ ".join(values) + r"\end{tabular}"

    groups = [("Test Acc.", "test_acc"), ("MIA-1 AUC", "forget_mia1_auc"), ("MIA-2 AUC", "forget_mia2_auc"),
              ("MIA-3 AUC", "forget_lira_auc")]
    lines = [
        r"% Requires: \usepackage{booktabs,graphicx}. Generated by lr_grid_runner.make_paper_table.",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Test accuracy and forget-set MIA AUC, $\max(\mathrm{AUC}, 1-\mathrm{AUC})$, for all 12 target "
        r"models and the three unlearning methods. Start: original model; Retr.: model retrained from scratch on "
        r"the retain set; Best: epoch closest to the retrained model (epoch in parentheses); Final: last epoch "
        rf"({EPOCHS}). MIA-1: shadow-model attack; MIA-2: loss-threshold attack; MIA-3: offline LiRA "
        rf"\citep{{carlini2022membership}} with {lira.N_REFERENCE} reference models. The MIAs are evaluated after "
        r"every unlearning epoch. "
        + (selection_note or r"The learning rate (Table~\ref{tab:lr_grid}) and the Best epoch were selected "
           r"on a selection half of the evaluation data; all reported values are measured on the disjoint "
           r"evaluation half. ")
        + r"div.: the run diverged before the last epoch. Bold: test accuracy below "
        rf"{COLLAPSE_ACC:.2f} (collapsed model).}}",
        r"\label{tab:utility_mia_all}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c l cccc cccc cccc cccc}", r"\toprule",
        r" & & & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{g}}}" for g, _ in groups) + r" \\",
        "".join(rf"\cmidrule(lr){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(len(groups))),
        r"Dataset & Model & Method & " + " & ".join(["Start & Retr. & Best & Final"] * len(groups)) + r" \\",
        r"\midrule"]
    by_key = {(r["exp_id"], r["method"]): r for _, r in sel.iterrows()}
    for d_i, (dataset, exps) in enumerate(er.TABLE_LAYOUT):
        present = [e for e in exps if any((e, m) in by_key for m in er.METHODS)]
        for e_i, exp_id in enumerate(present):
            rows = [by_key.get((exp_id, m)) for m in er.METHODS]
            first = next(r for r in rows if r is not None)
            methods = [er.METHOD_NAMES[m] + (f" ({int(r['selected_epoch'])})" if r is not None
                       and pd.notna(r.get("selected_epoch")) else "") for m, r in zip(er.METHODS, rows)]
            name = er.DATASET_NAMES[er.EXPERIMENTS[exp_id]["dataset"]]
            cells = [name if e_i == len(present) // 2 else "", exp_id, stack(methods, "l")]
            for _, k in groups:
                best, final = [], []
                for r in rows:
                    if r is None or pd.isna(r.get("selected_epoch")):
                        best.append("--"); final.append("--" if r is None else "div.")
                        continue
                    b = r[f"sel_{k}"]
                    f = "div." if pd.notna(r["diverged_at_epoch"]) else r[f"final_{k}"]
                    bold = lambda v: (rf"\textbf{{{num(v)}}}" if k == "test_acc" and isinstance(v, float)
                                      and np.isfinite(v) and v < COLLAPSE_ACC else num(v) if not isinstance(v, str) else v)
                    best.append(bold(b)); final.append(bold(f))
                cells += [num(first[f"start_eval_{k}"]), num(first[f"retr_eval_{k}"]), stack(best), stack(final)]
            lines.append(" & ".join(cells) + r" \\")
            if e_i < len(present) - 1:
                lines.append(r"\cmidrule(lr){2-19}")
        if present:
            lines.append(r"\midrule" if d_i < len(er.TABLE_LAYOUT) - 1 else r"\bottomrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def report(experiments, methods, learning_rates=None):
    """learning_rates=None: grid mode (lr and epoch selected). Dict: the lr is fixed, only the epoch is selected."""
    per_lr, sel, epochs = collect(experiments, methods, learning_rates)
    if per_lr.empty:
        print("No finished runs yet.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    epochs.to_csv(os.path.join(OUT_DIR, "grid_epochs.csv"), index=False)
    per_lr.to_csv(os.path.join(OUT_DIR, "grid_per_lr.csv"), index=False)
    sel.to_csv(os.path.join(OUT_DIR, "selection.csv"), index=False)

    note = (r"Learning rate and epoch were selected on a disjoint selection half of the evaluation data "
            rf"(grid {', '.join(_lr_tex(l) for l in LR_GRID)}; criterion: closest to the retrained model in "
            r"MIA-1 forget AUC + test accuracy); all numbers are measured on the evaluation half. "
            r"The MIAs are evaluated after every unlearning epoch. "
            r"Start/Retr.: original/retrained model; Sel.: selected epoch; Final: last epoch of the selected "
            r"learning rate; div.: that run diverged.")
    with open(os.path.join(OUT_DIR, "table_selected.tex"), "w") as f:
        f.write(make_selected_table(
            sel, [("Test Acc.", "test_acc"), ("MIA-1 AUC", "forget_mia1_auc"),
                  ("MIA-2 AUC", "forget_mia2_auc"), ("MIA-3 (LiRA) AUC", "forget_lira_auc"),
                  ("MIA-3 (LiRA) TPR@1\\%FPR", "forget_lira_tpr1")],
            "Test accuracy and forget-set MIA AUC, max(AUC, 1-AUC); MIA-1: shadow-model attack, MIA-2: "
            "loss-threshold attack, MIA-3: offline LiRA (Carlini et al., 2022) with "
            f"{lira.N_REFERENCE} reference models, also reported as TPR at 1\\% FPR. " + note,
            "tab:utility_mia_heldout"))
    with open(os.path.join(OUT_DIR, "table_selected_mia1acc.tex"), "w") as f:
        f.write(make_selected_table(
            sel, [("MIA-1 Acc. forget", "forget_mia1_bacc"), ("MIA-1 Acc. retain", "retain_mia1_bacc")],
            "Balanced accuracy of the shadow-model attack (MIA-1, RandomForest decision) on forget vs. "
            "non-members and retain vs. non-members (chance = 0.50). " + note, "tab:mia1_acc_heldout"))
    with open(os.path.join(OUT_DIR, "table_utility_mia_all.tex"), "w") as f:
        f.write(make_paper_table(sel))
    figures = []
    if learning_rates is None:  # the grid table and sensitivity figures need every learning rate
        with open(os.path.join(OUT_DIR, "table_lr_grid.tex"), "w") as f:
            f.write(make_grid_table(per_lr, sel))
        figures = make_figures(per_lr, methods)

    view = sel[["exp_id", "method", "selected_lr", "selected_epoch", "sel_distance", "eval_distance",
                "retr_eval_forget_mia1_auc", "sel_mia1_forget_auc", "sel_forget_mia1_auc",
                "sel_forget_mia1_bacc", "sel_retain_mia1_bacc", "sel_test_acc"]].rename(columns={
                    "retr_eval_forget_mia1_auc": "retr_auc", "sel_mia1_forget_auc": "auc_selhalf",
                    "sel_forget_mia1_auc": "auc_evalhalf", "sel_forget_mia1_bacc": "acc_F",
                    "sel_retain_mia1_bacc": "acc_R", "sel_test_acc": "test_acc"}) \
        if "selected_lr" in sel else sel
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print("\nSelected configuration per (experiment, method); metrics of the selected model on the "
              "evaluation half (auc_selhalf = same model on the selection half):")
        print(view.to_string(index=False))
    print(f"\nSaved {OUT_DIR}/: selection.csv, grid_per_lr.csv, grid_epochs.csv, table_selected.tex, "
          f"table_selected_mia1acc.tex, table_utility_mia_all.tex" + (f", table_lr_grid.tex, {len(figures)} figures" if figures else ""))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        job = json.loads(sys.argv[2])
        worker(job["exp_id"], job["methods"], job.get("learning_rates"))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=er.ALL_EXPERIMENTS, choices=er.ALL_EXPERIMENTS)
    parser.add_argument("--methods", nargs="+", default=er.METHODS, choices=er.METHODS)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not args.report_only:
        run(args.experiments, args.methods)
    report(args.experiments, args.methods)


if __name__ == "__main__":
    main()
