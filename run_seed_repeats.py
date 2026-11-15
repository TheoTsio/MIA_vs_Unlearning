"""
Repeated runs with different random seeds for the key configurations (reviewer: variability / significance).

For every key configuration (target model x unlearning method, fixed learning rate) and every seed, this
script runs the full unlearning procedure with per-epoch MIA-1/2/3 evaluation (experiment_runner.run_all),
then reports mean +- std over seeds, a 95% confidence interval of the AUC reduction and a one-sample t-test.

What the seed controls (the trained models and the data splits stay fixed, data seed 42):
  * the unlearning procedure: data order, dropout, random targets of SFTC;
  * the evaluation sample: which forget / retain / non-member samples enter the AUC.
So the reported std covers both the stochasticity of unlearning and of the attack evaluation.

Usage:
    python run_seed_repeats.py                        # run everything that is missing, then report
    python run_seed_repeats.py --report-only          # only rebuild tables / figures from saved runs
    python run_seed_repeats.py --seeds 42 0 1         # other seeds
    python run_seed_repeats.py --only 2a:sftc 3b:sftc # a subset of the key configurations

Runs are cached per seed in results/seeds/seed_<s>/<exp>/<method>.json (resumable; an interrupted run
continues where it stopped). Seed 42 reuses results/<exp>/<method>.json if it has exactly the same config.

Outputs (results/seeds/):
    seed_runs.csv          one row per (configuration, seed)
    seed_summary.csv       mean / std / 95% CI / p-value per configuration and attack
    table_seeds.tex        LaTeX table (label tab:seeds)
    fig_seeds_mia{1,2,3}.pdf   forget-set AUC per epoch, mean +- std band, retrained-model reference
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from scipy import stats

import experiment_runner as er

# Key configurations: (experiment, method, learning rate). The learning rates are the ones of the
# seed-42 runs in the main table. Edit here if you change them in run_all_experiments.ipynb.
KEY_CONFIGS = [
    ("1a", "sftc", 1e-4),
    ("2a", "scrub", 1e-5),
    ("2a", "sftc", 9e-4),
    ("3a", "sftc", 1e-2),
    ("1b", "sftc", 1e-3),
    ("2b", "scrub", 7.5e-5),
    ("3b", "sftc", 1e-3),
    ("1c", "sftc", 3e-3),
    ("2c", "scrub", 5e-4),
    ("2c", "sftc", 1e-2),
    ("1d", "sftc", 1e-2),
    ("2d", "sftc", 1e-2),
]
SEEDS = [42, 0, 1, 2, 3]
EPOCHS, BATCH_SIZE, MOMENTUM = 30, 32, 0.9
OUT_DIR = os.path.join("results", "seeds")
ATTACK_NAMES = {"mia1": "MIA-1", "mia2": "MIA-2", "mia3": "MIA-3"}


def seed_dir(seed):
    return os.path.join(OUT_DIR, f"seed_{seed}")


def reuse_main_results(configs, seed):
    """Copy a matching run from results/<exp>/ (the main table) instead of recomputing it."""
    for exp_id, method, lr in configs:
        config = er.run_config(method, lr, EPOCHS, BATCH_SIZE, MOMENTUM, seed)
        if er.load_cached(seed_dir(seed), exp_id, method, config) is not None:
            continue
        if er.load_cached("results", exp_id, method, config) is not None:
            os.makedirs(os.path.join(seed_dir(seed), exp_id), exist_ok=True)
            for name in (f"{method}.json", f"{method}_epochs.csv"):
                src = os.path.join("results", exp_id, name)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(seed_dir(seed), exp_id, name))
            print(f"[{exp_id}] {er.METHOD_NAMES[method]} seed {seed}: reused results/{exp_id}/{method}.json")


def run(configs, seeds):
    """Seeds in the outer loop: after k seeds, every configuration already has k repetitions."""
    learning_rates = {}
    for exp_id, method, lr in configs:
        learning_rates.setdefault(exp_id, {})[method] = lr
    experiments = [e for e in er.ALL_EXPERIMENTS if e in learning_rates]
    methods = sorted({m for _, m, _ in configs}, key=er.METHODS.index)

    failed = []
    for seed in seeds:
        print(f"\n================ seed {seed} ================", flush=True)
        reuse_main_results(configs, seed)
        failed += [(seed, e) for e in er.run_all(
            experiments, methods, learning_rates, epochs=EPOCHS, batch_size=BATCH_SIZE,
            momentum=MOMENTUM, seed=seed, results_dir=seed_dir(seed), isolate=True) or []]
    return failed


# ---------------------------------------------------------------------------
# Collect and summarise
# ---------------------------------------------------------------------------

def collect(configs, seeds):
    rows, missing = [], []
    for exp_id, method, lr in configs:
        for seed in seeds:
            config = er.run_config(method, lr, EPOCHS, BATCH_SIZE, MOMENTUM, seed)
            rec = er.load_cached(seed_dir(seed), exp_id, method, config)
            if rec is None:
                missing.append((exp_id, method, seed))
                continue
            row = dict(exp_id=exp_id, method=method, lr=lr, seed=seed,
                       best_epoch=rec["best_epoch"], diverged_at_epoch=rec.get("diverged_at_epoch"))
            for state in ("start", "retrain", "best", "final"):
                m = rec.get(state) or {}
                row[f"{state}_test_acc"] = m.get("test_acc", np.nan)
                for a in er.ATTACKS:
                    row[f"{state}_{a}"] = m.get(f"forget_{a}_auc", np.nan)
                    row[f"{state}_{a}_tpr1"] = m.get(f"forget_{a}_tpr1", np.nan)
            for a in er.ATTACKS:
                row[f"delta_{a}"] = row[f"start_{a}"] - row[f"best_{a}"]          # AUC reduction
                row[f"gap_{a}"] = row[f"best_{a}"] - row[f"retrain_{a}"]          # remaining gap to retrain
            rows.append(row)
    if missing:
        print(f"\nMissing runs (not included): {missing}")
    return pd.DataFrame(rows)


def ci95(x):
    """Mean, std, t-based 95% CI and one-sample t-test p-value (H0: mean = 0) over seeds."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    mean = x.mean()
    if n < 2:
        return dict(n=n, mean=mean, std=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
    std = x.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)
    p = stats.ttest_1samp(x, 0.0).pvalue if std > 0 else (0.0 if mean != 0 else 1.0)
    return dict(n=n, mean=mean, std=std, ci_low=mean - half, ci_high=mean + half, p=p)


def summarise(runs):
    out = []
    for (exp_id, method), g in runs.groupby(["exp_id", "method"], sort=False):
        base = dict(exp_id=exp_id, method=method, lr=g["lr"].iloc[0], n_seeds=len(g),
                    n_diverged=int(g["diverged_at_epoch"].notna().sum()),
                    best_epoch_mean=g["best_epoch"].mean(), best_epoch_std=g["best_epoch"].std(ddof=1))
        for col in ["start_test_acc", "retrain_test_acc", "best_test_acc"]:
            base[f"{col}_mean"], base[f"{col}_std"] = g[col].mean(), g[col].std(ddof=1)
        for a in er.ATTACKS:
            row = dict(base, attack=ATTACK_NAMES[a])
            for state in ("start", "retrain", "best", "final"):
                row[f"{state}_auc_mean"] = g[f"{state}_{a}"].mean()
                row[f"{state}_auc_std"] = g[f"{state}_{a}"].std(ddof=1)
            for name in ("delta", "gap"):
                for k, v in ci95(g[f"{name}_{a}"]).items():
                    row[f"{name}_{k}"] = v
            row["significant"] = bool(row["delta_ci_low"] > 0) if np.isfinite(row["delta_ci_low"]) else False
            out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# LaTeX table and figures
# ---------------------------------------------------------------------------

def _pm(mean, std, d=2):
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{d}f}" if not np.isfinite(std) else f"{mean:.{d}f} $\\pm$ {std:.{d}f}"


def make_seed_table(summary, seeds):
    idx = summary.set_index(["exp_id", "method", "attack"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Variability of the key configurations over {len(seeds)} random seeds (mean $\pm$ std). "
        r"Each seed changes the unlearning run (data order, stochastic layers) and the member/non-member "
        r"evaluation sample; the trained models and data splits are fixed. Test Acc.\ and AUC are given at the "
        r"selected (Best) epoch. $\Delta$ = forget-set AUC reduction from the original model (Start) to Best, "
        r"with a $t$-based 95\% confidence interval for MIA-1. "
        r"$^{*}$: the 95\% CI of $\Delta$ excludes 0 (one-sample $t$-test, $p<0.05$).}",
        r"\label{tab:seeds}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c l c cc c cc cc}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{MIA-1} & \multicolumn{2}{c}{MIA-2} & \multicolumn{2}{c}{MIA-3} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"Dataset & Model & Method & Test Acc. & Start & Best & $\Delta$ [95\% CI] "
        r"& Best & $\Delta$ & Best & $\Delta$ \\",
        r"\midrule",
    ]
    previous_dataset = None
    for exp_id, method, _ in [c for c in KEY_CONFIGS if (c[0], c[1], "MIA-1") in idx.index]:
        r1 = idx.loc[(exp_id, method, "MIA-1")]
        dataset = er.DATASET_NAMES[er.EXPERIMENTS[exp_id]["dataset"]]
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\midrule")
        name = er.METHOD_NAMES[method] + (f" ({int(r1['n_diverged'])} div.)" if r1["n_diverged"] else "")
        star = lambda r: "$^{*}$" if r["significant"] else ""
        cells = [dataset if dataset != previous_dataset else "", exp_id, name,
                 _pm(r1["best_test_acc_mean"], r1["best_test_acc_std"]),
                 _pm(r1["start_auc_mean"], r1["start_auc_std"]),
                 _pm(r1["best_auc_mean"], r1["best_auc_std"]),
                 f"{r1['delta_mean']:.2f} [{r1['delta_ci_low']:.2f}, {r1['delta_ci_high']:.2f}]{star(r1)}"
                 if np.isfinite(r1["delta_ci_low"]) else _pm(r1["delta_mean"], np.nan)]
        for a in ("MIA-2", "MIA-3"):
            r = idx.loc[(exp_id, method, a)]
            cells += [_pm(r["best_auc_mean"], r["best_auc_std"]), f"{r['delta_mean']:.2f}{star(r)}"]
        lines.append(" & ".join(cells) + r" \\")
        previous_dataset = dataset
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def make_figures(configs, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = {}
    for exp_id, method, lr in configs:
        frames = []
        for seed in seeds:
            path = os.path.join(seed_dir(seed), exp_id, f"{method}_epochs.csv")
            config = er.run_config(method, lr, EPOCHS, BATCH_SIZE, MOMENTUM, seed)
            rec = er.load_cached(seed_dir(seed), exp_id, method, config)
            if rec is not None and os.path.exists(path):
                df = pd.read_csv(path)
                for a in er.ATTACKS:
                    df[f"retrain_{a}"] = rec["retrain"][f"forget_{a}_auc"]
                frames.append(df.assign(seed=seed))
        if frames:
            curves[(exp_id, method)] = pd.concat(frames)

    if not curves:
        return []
    paths = []
    n = len(curves)
    cols = 4
    rows = int(np.ceil(n / cols))
    for a in er.ATTACKS:
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.5 * rows), sharex=True, squeeze=False)
        for ax, ((exp_id, method), df) in zip(axes.flat, curves.items()):
            g = df.groupby("epoch")[f"forget_{a}_auc"]
            mean, std = g.mean(), g.std(ddof=1).fillna(0)
            ax.plot(mean.index, mean.values, color="C0", lw=1.5, label="unlearned (mean)")
            ax.fill_between(mean.index, mean - std, mean + std, color="C0", alpha=0.25, label="$\\pm$ 1 std")
            ax.axhline(df[f"retrain_{a}"].mean(), color="C3", ls="--", lw=1, label="retrained")
            ax.axhline(0.5, color="grey", ls=":", lw=0.8)
            ax.set_title(f"{exp_id} {er.METHOD_NAMES[method]} (n={df['seed'].nunique()})", fontsize=9)
            ax.set_ylim(0.45, 1.0)
            ax.grid(alpha=0.3)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Unlearning epoch")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{ATTACK_NAMES[a]} forget AUC")
        axes.flat[0].legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        path = os.path.join(OUT_DIR, f"fig_seeds_{a}.pdf")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def report(configs, seeds):
    runs = collect(configs, seeds)
    if runs.empty:
        print("No finished runs yet.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    runs.to_csv(os.path.join(OUT_DIR, "seed_runs.csv"), index=False)
    summary = summarise(runs)
    summary.to_csv(os.path.join(OUT_DIR, "seed_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "table_seeds.tex"), "w") as f:
        f.write(make_seed_table(summary, seeds))
    figures = make_figures(configs, seeds)

    view = summary[summary["attack"] == "MIA-1"][[
        "exp_id", "method", "n_seeds", "n_diverged", "start_auc_mean", "best_auc_mean", "best_auc_std",
        "delta_mean", "delta_ci_low", "delta_ci_high", "delta_p", "significant"]]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\nMIA-1 forget AUC over seeds:")
        print(view.to_string(index=False))
    sig = summary.groupby("attack")["significant"].agg(["sum", "count"])
    print("\nSignificant AUC reductions (95% CI excludes 0):")
    for attack, r in sig.iterrows():
        print(f"  {attack}: {int(r['sum'])}/{int(r['count'])} configurations")
    print(f"\nSaved {OUT_DIR}/seed_runs.csv, seed_summary.csv, table_seeds.tex"
          + (", " + ", ".join(os.path.basename(p) for p in figures) if figures else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--only", nargs="+", metavar="EXP:METHOD",
                        help="subset of KEY_CONFIGS, e.g. 2a:sftc 3b:sftc")
    parser.add_argument("--report-only", action="store_true", help="do not run, only summarise saved runs")
    args = parser.parse_args()

    configs = KEY_CONFIGS
    if args.only:
        wanted = {tuple(s.split(":")) for s in args.only}
        configs = [c for c in KEY_CONFIGS if (c[0], c[1]) in wanted]
        unknown = wanted - {(c[0], c[1]) for c in configs}
        if unknown:
            parser.error(f"not in KEY_CONFIGS: {sorted(unknown)}")

    for seed in args.seeds:
        reuse_main_results(configs, seed)
    if not args.report_only:
        failed = run(configs, args.seeds)
        if failed:
            print(f"\nFailed (seed, experiment): {failed}. Re-run the script; finished runs are cached.")
    report(configs, args.seeds)


if __name__ == "__main__":
    main()
