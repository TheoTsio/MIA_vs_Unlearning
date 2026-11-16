"""
Learning-rate variation: the same logic as run_seed_repeats.py, but the learning rate changes instead of
the seed. Every target model x unlearning method (all 36 by default) is run with several learning rates,
and the script reports how much the results vary. Main quantity: the MIA-1 (shadow-model attack)
ACCURACY on the forget set and on the retain set (balanced member/non-member samples, chance = 0.50).
Also reported: test accuracy and the MIA-1/2/3 AUC. Nothing is selected; every learning rate is reported.

Usage:
    python run_lr_grid.py                                  # all 12 experiments x 3 methods x LRS
    python run_lr_grid.py --key-only                       # only the 12 key configurations
    python run_lr_grid.py --only 2a:sftc 3b:sftc           # a subset
    python run_lr_grid.py --lrs 1e-5 1e-4 1e-3             # other learning rates
    python run_lr_grid.py --report-only                    # only rebuild tables / figures

Each experiment runs in its own subprocess (data and models loaded once for all its runs). Runs are cached
in results/lr_grid/lr_<lr>/<exp>/<method>.json (resumable). A run identical to one in results/<exp>/
(same lr, seed, epochs, ... and with the attack accuracy recorded) is reused.

Outputs (results/lr_grid/):
    lr_runs.csv                 one row per (configuration, lr): Start / Retr. / Best / Final values
    lr_variation.csv            per configuration, metric and set (forget / retain): value at each lr,
                                mean, std, min, max and range over the learning rates
    table_lr_grid.tex           LaTeX table: MIA-1 attack accuracy forget / retain after the last epoch
    fig_lr_final_mia1acc.pdf    MIA-1 attack accuracy vs. learning rate (forget solid, retain dashed)
    fig_lr_<state>_<metric>.pdf the same for metric = acc (test), mia1, mia2, mia3 (AUC);
                                state: final = last epoch, best = epoch closest to the retrained model
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import experiment_runner as er
from run_seed_repeats import KEY_CONFIGS

LRS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
SEED, EPOCHS, BATCH_SIZE, MOMENTUM = 42, 30, 32, 0.9
OUT_DIR = os.path.join("results", "lr_grid")
SETS = ("forget", "retain")
ATTACK_NAMES = {"mia1acc": "MIA-1 accuracy", "mia1": "MIA-1 AUC", "mia2": "MIA-2 AUC", "mia3": "MIA-3 AUC"}
ATTACK_KEYS = list(ATTACK_NAMES)  # mia1acc = attack accuracy of the shadow-model RandomForest


def lr_dir(lr):
    return os.path.join(OUT_DIR, f"lr_{lr:g}")


def config_for(method, lr):
    return er.run_config(method, lr, EPOCHS, BATCH_SIZE, MOMENTUM, SEED)


def has_attack_acc(rec):
    return rec is not None and "forget_mia1_attack_acc" in (rec.get("start") or {})


def cached(exp_id, method, lr):
    rec = er.load_cached(lr_dir(lr), exp_id, method, config_for(method, lr))
    return rec if has_attack_acc(rec) else None


def reuse_main_results(exp_id, method, lr):
    """Copy an identical run from results/<exp>/ (the main table) instead of recomputing it."""
    if cached(exp_id, method, lr) is not None:
        return True
    if not has_attack_acc(er.load_cached("results", exp_id, method, config_for(method, lr))):
        return False
    os.makedirs(os.path.join(lr_dir(lr), exp_id), exist_ok=True)
    for name in (f"{method}.json", f"{method}_epochs.csv"):
        src = os.path.join("results", exp_id, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(lr_dir(lr), exp_id, name))
    print(f"[{exp_id}] {er.METHOD_NAMES[method]} lr={lr:g}: reused results/{exp_id}/{method}.json", flush=True)
    return True


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def worker(exp_id, methods, lrs):
    """Runs in a subprocess: load one experiment once, then run all its missing (method, lr) points."""
    todo = [(m, lr) for lr in lrs for m in methods if not reuse_main_results(exp_id, m, lr)]
    if not todo:
        print(f"[{exp_id}] all runs cached", flush=True)
        return
    prep = er.prepare_experiment(exp_id, seed=SEED)
    failed = []
    for i, (method, lr) in enumerate(todo, 1):
        print(f"[{exp_id}] run {i}/{len(todo)}: {er.METHOD_NAMES[method]} lr={lr:g}", flush=True)
        try:
            er.run_method(prep, method, lr, epochs=EPOCHS, batch_size=BATCH_SIZE, momentum=MOMENTUM,
                          results_dir=lr_dir(lr))
        except Exception:  # one failing run must not stop the others
            import traceback
            traceback.print_exc()
            failed.append((method, lr))
            er.free_memory()
    if failed:
        print(f"[{exp_id}] FAILED runs: {failed}", flush=True)
        raise SystemExit(1)


def run(configs, lrs):
    by_exp = {}
    for exp_id, method in configs:
        by_exp.setdefault(exp_id, []).append(method)
    failed = []
    for exp_id in [e for e in er.ALL_EXPERIMENTS if e in by_exp]:
        job = dict(exp_id=exp_id, methods=by_exp[exp_id], lrs=lrs)
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
        print(f"\nFailed experiments: {failed}. Re-run the script; finished runs are cached.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def collect(configs, lrs):
    rows, missing = [], []
    for exp_id, method in configs:
        for lr in lrs:
            rec = cached(exp_id, method, lr)
            if rec is None:
                missing.append((exp_id, method, lr))
                continue
            row = dict(exp_id=exp_id, dataset=er.DATASET_NAMES[er.EXPERIMENTS[exp_id]["dataset"]],
                       method=method, lr=lr, best_epoch=rec["best_epoch"],
                       diverged_at_epoch=rec.get("diverged_at_epoch"))
            for state in ("start", "retrain", "best", "final"):
                m = rec.get(state) or {}
                row[f"{state}_acc"] = m.get("test_acc", np.nan)
                for s in SETS:
                    row[f"{state}_{s}_mia1acc"] = m.get(f"{s}_mia1_attack_acc", np.nan)
                    for a in er.ATTACKS:
                        row[f"{state}_{s}_{a}"] = m.get(f"{s}_{a}_auc", np.nan)
            rows.append(row)
    if missing:
        print(f"Missing runs ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return pd.DataFrame(rows)


def variation(runs, state="final"):
    """Spread over the learning rates (diverged runs excluded), per configuration, attack and set."""
    out = []
    for (exp_id, method), g in runs.groupby(["exp_id", "method"], sort=False):
        g = g.sort_values("lr")
        for s in SETS:
            for a in ATTACK_KEYS:
                col = f"{state}_{s}_{a}"
                v = g[col].dropna()
                row = dict(exp_id=exp_id, method=method, state=state, set=s, attack=ATTACK_NAMES[a],
                           start=g[f"start_{s}_{a}"].iloc[0], retrain=g[f"retrain_{s}_{a}"].iloc[0],
                           n_lrs=len(g), n_diverged=int(g["diverged_at_epoch"].notna().sum()),
                           mean=v.mean(), std=v.std(ddof=1), min=v.min(), max=v.max(),
                           range=v.max() - v.min() if len(v) else np.nan)
                row.update({f"lr_{lr:g}": val for lr, val in zip(g["lr"], g[col])})
                out.append(row)
    return pd.DataFrame(out)


def make_figures(runs, configs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [c for c in configs if ((runs["exp_id"] == c[0]) & (runs["method"] == c[1])).any()]
    cols = 4
    rows = int(np.ceil(len(present) / cols))
    paths = []
    for state in ("final", "best"):
        for key in ["acc"] + ATTACK_KEYS:
            fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
            for ax, (exp_id, method) in zip(axes.flat, present):
                g = runs[(runs["exp_id"] == exp_id) & (runs["method"] == method)].sort_values("lr")
                if key == "acc":
                    lines = [("test_acc", f"{state}_acc", "start_acc", "retrain_acc", "-")]
                else:
                    lines = [(s, f"{state}_{s}_{key}", f"start_{s}_{key}", f"retrain_{s}_{key}",
                              "-" if s == "forget" else "--") for s in SETS]
                for c, (name, col, start_col, retr_col, ls) in enumerate(lines):
                    ax.plot(g["lr"], g[col], marker="o", ls=ls, color=f"C{c}", ms=4, lw=1.3, label=name)
                    div = g[g[col].isna()]
                    if len(div):  # diverged before this state could be measured
                        ax.plot(div["lr"], np.full(len(div), 0.0 if key == "acc" else 1.0), "x",
                                color=f"C{c}", ms=7, mew=1.5)
                    ax.axhline(g[retr_col].iloc[0], color=f"C{c}", ls=":", lw=1)
                ax.axhline(g[lines[0][2]].iloc[0], color="grey", ls="-.", lw=0.8)
                ax.set_xscale("log")
                ax.set_title(f"{exp_id} {er.METHOD_NAMES[method]}", fontsize=9)
                ax.set_ylim((0.0, 1.02) if key == "acc" else (0.4, 1.02))
                ax.grid(alpha=0.3)
            for ax in list(axes.flat)[len(present):]:
                ax.axis("off")
            for ax in axes[-1]:
                ax.set_xlabel("Learning rate")
            label = "Test accuracy" if key == "acc" else ATTACK_NAMES[key]
            for ax in axes[:, 0]:
                ax.set_ylabel(label)
            axes.flat[0].legend(fontsize=7, loc="best")
            state_txt = "last epoch" if state == "final" else "epoch closest to the retrained model"
            fig.suptitle(f"{label} vs. learning rate ({state_txt}). Dotted: retrained model, "
                         f"dash-dot: original model, x: diverged", fontsize=9)
            fig.tight_layout()
            path = os.path.join(OUT_DIR, f"fig_lr_{state}_{key}.pdf")
            fig.savefig(path)
            plt.close(fig)
            paths.append(path)
    return paths


def make_table(runs, configs, lrs):
    """MIA-1 attack accuracy forget / retain after the last epoch for every learning rate."""
    idx = runs.set_index(["exp_id", "method", "lr"])
    lr_head = " & ".join(f"$10^{{{int(round(np.log10(lr)))}}}$" if np.isclose(np.log10(lr) % 1, 0)
                         or np.isclose(np.log10(lr) % 1, 1) else f"{lr:g}" for lr in lrs)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Effect of the unlearning learning rate. Each cell: MIA-1 (shadow-model attack) accuracy on the "
        rf"forget set / on the retain set (balanced member/non-member samples, chance = 0.50) after the last epoch ({EPOCHS} epochs, SGD with momentum {MOMENTUM}, batch size "
        rf"{BATCH_SIZE}, seed {SEED}). Start and Retr.: original and retrained model. "
        r"div.: weights became inf/NaN.}",
        r"\label{tab:lr_grid}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c l c c " + "c" * len(lrs) + "}",
        r"\toprule",
        r"Dataset & Model & Method & Start & Retr. & " + lr_head + r" \\",
        r"\midrule",
    ]
    previous = None
    for exp_id, method in configs:
        sub = runs[(runs["exp_id"] == exp_id) & (runs["method"] == method)]
        if sub.empty:
            continue
        first = sub.iloc[0]
        dataset = first["dataset"]
        if previous is not None and dataset != previous:
            lines.append(r"\midrule")
        cells = [dataset if dataset != previous else "", exp_id, er.METHOD_NAMES[method],
                 f"{first['start_forget_mia1acc']:.2f} / {first['start_retain_mia1acc']:.2f}",
                 f"{first['retrain_forget_mia1acc']:.2f} / {first['retrain_retain_mia1acc']:.2f}"]
        for lr in lrs:
            if (exp_id, method, lr) not in idx.index:
                cells.append("--")
                continue
            r = idx.loc[(exp_id, method, lr)]
            cells.append("div." if pd.notna(r["diverged_at_epoch"])
                         else f"{r['final_forget_mia1acc']:.2f} / {r['final_retain_mia1acc']:.2f}")
        lines.append(" & ".join(cells) + r" \\")
        previous = dataset
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def report(configs, lrs):
    runs = collect(configs, lrs)
    if runs.empty:
        print("No finished runs yet.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(os.path.join(OUT_DIR, "lr_runs.csv"), index=False)
    var = pd.concat([variation(runs, "final"), variation(runs, "best")])
    var.to_csv(os.path.join(OUT_DIR, "lr_variation.csv"), index=False)
    with open(os.path.join(OUT_DIR, "table_lr_grid.tex"), "w") as f:
        f.write(make_table(runs, configs, lrs))
    figures = make_figures(runs, configs)

    view = var[(var["state"] == "final") & (var["attack"] == "MIA-1 accuracy")]
    cols = ["exp_id", "method", "set", "start", "retrain", "n_diverged"] + \
           [c for c in view.columns if c.startswith("lr_")] + ["std", "range"]
    with pd.option_context("display.width", 220, "display.max_rows", 200,
                           "display.float_format", "{:.2f}".format):
        print("\nMIA-1 attack accuracy after the last epoch, per learning rate (NaN = diverged):")
        print(view[cols].to_string(index=False))
    print(f"\nSaved {OUT_DIR}/lr_runs.csv, lr_variation.csv, table_lr_grid.tex and "
          f"{len(figures)} figures (fig_lr_*.pdf)")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        job = json.loads(sys.argv[2])
        worker(job["exp_id"], job["methods"], job["lrs"])
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lrs", type=float, nargs="+", default=LRS)
    parser.add_argument("--only", nargs="+", metavar="EXP:METHOD", help="e.g. 2a:sftc 3b:sftc")
    parser.add_argument("--key-only", action="store_true",
                        help="only the 12 key configurations of run_seed_repeats.py (default: all 36)")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    lrs = sorted(args.lrs)

    if args.key_only:
        configs = [(e, m) for e, m, _ in KEY_CONFIGS]
    else:
        configs = [(e, m) for e in er.ALL_EXPERIMENTS for m in er.METHODS]
    if args.only:
        wanted = [tuple(s.split(":")) for s in args.only]
        bad = [w for w in wanted if w[0] not in er.EXPERIMENTS or w[1] not in er.METHODS]
        if bad:
            parser.error(f"unknown EXP:METHOD {bad}")
        configs = wanted

    if not args.report_only:
        run(configs, lrs)
    report(configs, lrs)


if __name__ == "__main__":
    main()
