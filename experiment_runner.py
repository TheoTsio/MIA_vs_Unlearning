"""
Runs the full "unlearning vs. membership inference" experiment grid and builds the paper table.

For every target model (1a ... 3d, see Table 1) and every unlearning method (NegGrad, SCRUB, SFTC):
    1. Rebuild the exact data splits used to train the saved models (seed 42).
    2. Load the saved target model, retrain-from-scratch baseline and shadow-model attack (MIA-1).
    3. Unlearn for `epochs` epochs and, after every epoch, measure test accuracy and the AUC of
         MIA-1  shadow-model + RandomForest attack (Shokri et al., 2017)   [saved .jolib model]
         MIA-2  loss-threshold attack (Yeom et al., 2018)
         MIA-3  confidence-based attack: max softmax probability (Salem et al., 2019)
       on forget-vs-non-members and retain-vs-non-members.
    4. Pick the "Best" epoch: the one closest to the retrained model, measured as
         |MIA-1 forget AUC - Retr. MIA-1 forget AUC| + |Test Acc - Retr. Test Acc|
       and the "Final" epoch (the last one).
    5. Save every run to results/<exp>/<method>.json (+ per-epoch .csv). Runs whose config did not
       change are loaded from disk instead of re-run, so an interrupted notebook resumes.

MIA-2 and MIA-3 AUCs are threshold-free, so no extra shadow models are trained for them.
"""
import os

# Must be set before CUDA is initialised for deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import contextlib
import copy
import gc
import io
import json
import platform
import random
import subprocess
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

import utils.unlearning_alg.neg_grad as _neg_grad_module
import utils.unlearning_alg.scrub as _scrub_module
import utils.unlearning_alg.sftc_unlearn as _sftc_module
from preprocess_data import (create_merged_loader, get_cifar10_dataset,
                             get_purchase_dataset, get_texas_100_dataset)
from utils.data_split import get_retain_forget_split, get_target_shadow_split, get_train_test_split
from utils.unlearning_utils import RandomDistributionGenerator
# Model classes must be importable for torch.load of the saved (pickled) models.
from Target_Models.target_model_1a import TargetModel_1a  # noqa: F401
from Target_Models.target_model_1c import TargetModel_1c  # noqa: F401
from Target_Models.target_model_2a import TargetModel_2a  # noqa: F401
from Target_Models.target_model_2b import TargetModel_2b  # noqa: F401
from Target_Models.target_model_2c import TargetModel_2c  # noqa: F401
from Target_Models.target_model_3a import TargetModel_3a  # noqa: F401
from Target_Models.target_model_3b import TargetModel_3b  # noqa: F401
from Target_Models.target_model_3c import TargetModel_3c  # noqa: F401
from Target_Models.target_model_3d import TargetModel_3d  # noqa: F401

RUNNER_VERSION = 1  # bump when the evaluation logic changes, to invalidate cached results
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Experiment registry (Table 1). input_shape: reshape for CNNs that need (C, H, W) input;
# the FC models and the ResNets (3a/3b reshape internally) take flat vectors.
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "1a": dict(dataset="cifar10", input_shape=None),
    "1b": dict(dataset="mufac", input_shape=None),
    "1c": dict(dataset="purchase", input_shape=None),
    "1d": dict(dataset="texas", input_shape=None),
    "2a": dict(dataset="cifar10", input_shape=(3, 32, 32)),
    "2b": dict(dataset="mufac", input_shape=(3, 128, 128)),
    "2c": dict(dataset="purchase", input_shape=None),
    "2d": dict(dataset="texas", input_shape=None),
    "3a": dict(dataset="cifar10", input_shape=None),
    "3b": dict(dataset="mufac", input_shape=None),
    "3c": dict(dataset="purchase", input_shape=None),
    "3d": dict(dataset="texas", input_shape=None),
}
ALL_EXPERIMENTS = list(EXPERIMENTS)
METHODS = ["neg_grad", "scrub", "sftc"]
METHOD_NAMES = {"neg_grad": "NegGrad", "scrub": "SCRUB", "sftc": "SFTC"}
DATASET_NAMES = {"cifar10": "CIFAR-10", "mufac": "MuFac", "purchase": "Purchase-100", "texas": "Texas-100"}
TABLE_LAYOUT = [  # (dataset, experiments) in the order of the paper table
    ("cifar10", ["1a", "2a", "3a"]),
    ("mufac", ["1b", "2b", "3b"]),
    ("purchase", ["1c", "2c", "3c"]),
    ("texas", ["1d", "2d", "3d"]),
]
ATTACKS = ["mia1", "mia2", "mia3"]
EVAL_BATCH_SIZE = 128  # small enough for ResNet-18 on 128x128 images on a 4 GB GPU


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only: a few CUDA backward kernels have no deterministic version; they warn instead of failing.
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "*.py"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": commit,
        "uncommitted_python_changes": dirty,
        "runner_version": RUNNER_VERSION,
    }


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------

def _load_mufac_low_memory(meta_path, image_dir, batch_size=64, percentage_of_rows_to_drop=0.4):
    """Same output as preprocess_data.get_MUFAC_dataset (same transforms, shuffle order and split, so
    the same random augmentations), but writes batches into one preallocated array instead of
    concatenating a list: peak memory ~2 GB instead of ~4 GB."""
    from sklearn.model_selection import train_test_split
    from torchvision import transforms
    from preprocess_data import MUFACCustomDataset

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    dataset = MUFACCustomDataset(pd.read_csv(meta_path), image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    images, labels, pos = None, np.empty(len(dataset), dtype=np.int64), 0
    for imgs, labs in loader:
        flat = imgs.view(imgs.size(0), -1).numpy()
        if images is None:
            images = np.empty((len(dataset), flat.shape[1]), dtype=flat.dtype)
        images[pos:pos + len(flat)] = flat
        labels[pos:pos + len(flat)] = labs.numpy()
        pos += len(flat)

    X = pd.DataFrame(images)
    y = pd.DataFrame(labels, columns=["label"])
    X, _, y, _ = train_test_split(X, y, test_size=percentage_of_rows_to_drop, random_state=42, stratify=y)
    return X, y, X.shape[1], y["label"].nunique()


def load_dataset(name: str, data_seed: int = 42):
    """Load a dataset exactly as train_target_and_perform_MIA_attack.py did.

    The seed is set right before loading, as in the training script. This matters for MuFac,
    whose loader applies random augmentations and shuffles while building the DataFrame.
    """
    set_random_seed(data_seed)
    if name == "cifar10":
        return get_cifar10_dataset()
    if name == "mufac":
        return _load_mufac_low_memory(
            "data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv",
            "data/custom_korean_family_dataset_resolution_128/train_images",
            percentage_of_rows_to_drop=0.4)
    if name == "purchase":
        return get_purchase_dataset(dataset_path="data/dataset_purchase.csv", keep_rows=40_000)
    if name == "texas":
        return get_texas_100_dataset(path="data/texas100.npz", limit_rows=40_000)
    raise ValueError(f"Unknown dataset: {name}")


def _labels(y):
    return np.asarray(y).ravel().astype(np.int64)


def _features(X, input_shape):
    t = torch.tensor(np.asarray(X, dtype=np.float32))
    return t.reshape(-1, *input_shape) if input_shape is not None else t


def prepare_experiment(exp_id: str, seed: int = 42, data_seed: int = 42, verbose: bool = True):
    """Rebuild data splits and load the saved models for one target model (shared by all methods)."""
    cfg = EXPERIMENTS[exp_id]
    shape = cfg["input_shape"]
    t0 = time.time()

    X, y, num_features, num_classes = load_dataset(cfg["dataset"], data_seed)
    X_target, _, y_target, _, X_mia, y_mia = get_target_shadow_split(X, y, seed=data_seed)
    del X, y
    X_train, X_test, y_train, y_test = get_train_test_split(X_target, y_target, seed=data_seed)
    del X_target, y_target
    X_retain, X_forget, y_retain, y_forget = get_retain_forget_split(
        X_train, y_train, seed=data_seed, forget_class=None)
    del X_train, y_train

    # Equal-sized member / non-member samples for the attacks (as in main.py, but seeded).
    n = min(len(X_mia), len(X_forget), len(X_retain))
    forget_m = X_forget.sample(n=n, replace=False, random_state=seed)
    retain_m = X_retain.sample(n=n, replace=False, random_state=seed)
    non_m = X_mia.sample(n=n, replace=False, random_state=seed)

    prep = SimpleNamespace(
        exp_id=exp_id, dataset=cfg["dataset"], input_shape=shape,
        num_features=num_features, num_classes=num_classes, seed=seed,
        # full sets for unlearning and utility
        X_retain=_features(X_retain, shape), y_retain=torch.tensor(_labels(y_retain)),
        X_forget=_features(X_forget, shape), y_forget=torch.tensor(_labels(y_forget)),
        X_test=_features(X_test, shape), y_test=torch.tensor(_labels(y_test)),
        # attack evaluation samples
        mia_sets={
            "forget": (_features(forget_m, shape), _labels(y_forget.loc[forget_m.index])),
            "retain": (_features(retain_m, shape), _labels(y_retain.loc[retain_m.index])),
        },
        non_member=(_features(non_m, shape), _labels(y_mia.loc[non_m.index])),
        sizes=dict(retain=len(X_retain), forget=len(X_forget), test=len(X_test), mia_sample=n),
    )
    del X_retain, X_forget, X_test, X_mia, forget_m, retain_m, non_m

    load = lambda kind: torch.load(f"models/{exp_id}_dataset_{kind}_model.pth",
                                   map_location=device, weights_only=False).to(device).eval()
    prep.target_model = load("target")
    prep.retrain_model = load("retrain_baseline")
    prep.attack_model = joblib.load(f"models/{exp_id}_dataset_attack_model.jolib")
    if hasattr(prep.attack_model, "n_jobs"):
        prep.attack_model.n_jobs = 1  # deterministic and avoids oversubscription
    if hasattr(prep.attack_model, "verbose"):
        prep.attack_model.verbose = 0

    prep.start = evaluate_model(prep, prep.target_model)
    prep.retrain = evaluate_model(prep, prep.retrain_model)

    if verbose:
        s = prep.start
        print(f"[{exp_id}] {DATASET_NAMES[prep.dataset]} | {type(prep.target_model).__name__} | "
              f"retain {prep.sizes['retain']}, forget {prep.sizes['forget']}, test {prep.sizes['test']}, "
              f"MIA sample {n} | loaded in {time.time() - t0:.0f}s")
        print(f"  Original : test acc {s['test_acc']:.3f}, forget acc {s['forget_acc']:.3f}, "
              f"retain acc {s['retain_acc']:.3f}, MIA-1 forget AUC {s['forget_mia1_auc']:.3f}")
        r = prep.retrain
        print(f"  Retrained: test acc {r['test_acc']:.3f}, forget acc {r['forget_acc']:.3f}, "
              f"retain acc {r['retain_acc']:.3f}, MIA-1 forget AUC {r['forget_mia1_auc']:.3f}")
        if s["forget_acc"] - s["test_acc"] < 0.02 and s["retain_acc"] - s["test_acc"] < 0.02:
            print("  WARNING: the original model is not more accurate on its members than on the test set. "
                  "The rebuilt member sets may not match the data the model was trained on.")
    return prep


# ---------------------------------------------------------------------------
# Evaluation: utility + MIA-1/2/3 on forget and retain sets
# ---------------------------------------------------------------------------

@torch.no_grad()
def _probabilities(model, X_t):
    model.eval()
    out = [torch.softmax(model(X_t[i:i + EVAL_BATCH_SIZE].to(device)), dim=1).double().cpu()
           for i in range(0, len(X_t), EVAL_BATCH_SIZE)]
    return torch.cat(out).numpy()


def _tpr_at_fpr(labels, scores, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def _attack_scores(attack_model, probs, y):
    """Membership scores per attack; higher = more likely member."""
    idx = np.arange(len(y))
    p_y = np.clip(probs[idx, y], 1e-12, 1.0)
    member_col = list(attack_model.classes_).index(True)
    return {
        "mia1": attack_model.predict_proba(pd.DataFrame(probs))[:, member_col],  # same features as training
        "mia2": np.log(p_y),          # -loss: low loss -> member
        "mia3": probs.max(axis=1),    # max confidence -> member (label-free)
    }


def evaluate_model(prep, model):
    """Test accuracy and, for forget/retain vs. non-members, AUC / AUC_sym / TPR@1%FPR of every attack."""
    metrics = {}
    p_test = _probabilities(model, prep.X_test)
    metrics["test_acc"] = float((p_test.argmax(1) == prep.y_test.numpy()).mean())

    nm_X, nm_y = prep.non_member
    p_out = _probabilities(model, nm_X)
    s_out = _attack_scores(prep.attack_model, p_out, nm_y)
    for split, (X_in, y_in) in prep.mia_sets.items():
        p_in = _probabilities(model, X_in)
        metrics[f"{split}_acc"] = float((p_in.argmax(1) == y_in).mean())
        s_in = _attack_scores(prep.attack_model, p_in, y_in)
        labels = np.r_[np.ones(len(p_in)), np.zeros(len(p_out))]
        for a in ATTACKS:
            scores = np.r_[s_in[a], s_out[a]]
            auc = float(roc_auc_score(labels, scores))
            metrics[f"{split}_{a}_auc_raw"] = auc
            metrics[f"{split}_{a}_auc"] = max(auc, 1 - auc)  # reported: max(AUC, 1 - AUC)
            metrics[f"{split}_{a}_tpr1"] = _tpr_at_fpr(labels, scores)
    return metrics


def distance_to_retrain(m, retrain):
    """Best-epoch criterion: MIA distance + utility distance to the retrained model."""
    return abs(m["forget_mia1_auc"] - retrain["forget_mia1_auc"]) + abs(m["test_acc"] - retrain["test_acc"])


# ---------------------------------------------------------------------------
# Unlearning runs
# ---------------------------------------------------------------------------

def _no_tqdm(iterable, *args, **kwargs):
    return iterable


def run_config(method, lr, epochs, batch_size, momentum, seed):
    return dict(method=method, lr=lr, epochs=epochs, batch_size=batch_size, momentum=momentum,
                seed=seed, best_criterion="|MIA-1 forget AUC - retr| + |test acc - retr|",
                runner_version=RUNNER_VERSION)


def result_path(results_dir, exp_id, method):
    return os.path.join(results_dir, exp_id, f"{method}.json")


def load_cached(results_dir, exp_id, method, config):
    path = result_path(results_dir, exp_id, method)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        record = json.load(f)
    return record if record.get("config") == config else None


def run_method(prep, method, lr, epochs=30, batch_size=32, momentum=0.9, results_dir="results",
               verbose_algorithm=False):
    """Unlearn prep.target_model with `method`, evaluating every epoch. Saves and returns the record."""
    config = run_config(method, lr, epochs, batch_size, momentum, prep.seed)
    set_random_seed(prep.seed)  # every run starts from the same RNG state, independent of run order

    model = copy.deepcopy(prep.target_model).to(device)
    teacher = copy.deepcopy(prep.target_model).to(device).eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    retain_loader = DataLoader(TensorDataset(prep.X_retain, prep.y_retain), batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(TensorDataset(prep.X_forget, prep.y_forget), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(prep.X_test, prep.y_test), batch_size=batch_size, shuffle=False)

    kwargs = {}
    if method == "neg_grad":
        algorithm = _neg_grad_module.neg_grad
    elif method == "scrub":
        algorithm = _scrub_module.scrub
        kwargs = dict(teacher_model=teacher)
    elif method == "sftc":
        algorithm = _sftc_module.sftc_unlearn
        kwargs = dict(teacher_model=teacher,
                      dummy_model=RandomDistributionGenerator(dist="normal", dimensions=prep.num_classes),
                      merged_loader=create_merged_loader(prep.X_retain, prep.y_retain, prep.X_forget,
                                                         prep.y_forget, batch_size=batch_size, shuffle=True))
    else:
        raise ValueError(f"Unknown method: {method}")

    modules = [_neg_grad_module, _scrub_module, _sftc_module]
    saved_tqdm = [m.tqdm for m in modules]
    if not verbose_algorithm:
        for m in modules:
            m.tqdm = _no_tqdm

    history = [dict(epoch=0, **prep.start)]
    diverged_at = None
    t0 = time.time()
    try:
        for epoch in range(1, epochs + 1):
            # One epoch of the original algorithm; the optimizer (and its momentum) persists across calls,
            # so this equals one call with epochs=epochs. Same val/test loader as main.py (test set).
            quiet = contextlib.nullcontext() if verbose_algorithm else contextlib.redirect_stdout(io.StringIO())
            with quiet:
                algorithm(model, retain_loader, test_loader, test_loader, forget_loader, optimizer, None,
                          criterion, epochs=1, device=device, **kwargs)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                # e.g. SCRUB's forget step maximises an unbounded KL loss and can blow up at a high lr
                diverged_at = epoch
                print(f"  [{prep.exp_id} {METHOD_NAMES[method]} lr={lr:g}] epoch {epoch:2d}/{epochs} | "
                      f"DIVERGED (weights are inf/NaN). Stopping this run; try a smaller learning rate.",
                      flush=True)
                break
            m = evaluate_model(prep, model)
            history.append(dict(epoch=epoch, **m))
            print(f"  [{prep.exp_id} {METHOD_NAMES[method]} lr={lr:g}] epoch {epoch:2d}/{epochs} | "
                  f"test {m['test_acc']:.3f} | forget AUC MIA-1 {m['forget_mia1_auc']:.3f} "
                  f"MIA-2 {m['forget_mia2_auc']:.3f} MIA-3 {m['forget_mia3_auc']:.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        for mod, t in zip(modules, saved_tqdm):
            mod.tqdm = t

    unlearn_epochs = history[1:]  # only epochs with finite weights
    best = (min(unlearn_epochs, key=lambda h: distance_to_retrain(h, prep.retrain))  # ties -> earliest
            if unlearn_epochs else None)
    record = dict(
        exp_id=prep.exp_id, dataset=prep.dataset, method=method, config=config,
        best_epoch=best["epoch"] if best else None, start=prep.start, retrain=prep.retrain,
        best=best, final=None if diverged_at else unlearn_epochs[-1], diverged_at_epoch=diverged_at,
        sizes=prep.sizes, runtime_seconds=round(time.time() - t0, 1), environment=environment_info(),
    )

    os.makedirs(os.path.join(results_dir, prep.exp_id), exist_ok=True)
    pd.DataFrame(history).to_csv(os.path.join(results_dir, prep.exp_id, f"{method}_epochs.csv"), index=False)
    with open(result_path(results_dir, prep.exp_id, method), "w") as f:
        json.dump(record, f, indent=2)

    del model, teacher, optimizer
    free_memory()
    best_txt = (f"best epoch {record['best_epoch']} (distance {distance_to_retrain(best, prep.retrain):.3f})"
                if best else "no finite epoch")
    div_txt = f", DIVERGED at epoch {diverged_at}" if diverged_at else ""
    print(f"  -> {best_txt}{div_txt}, saved {result_path(results_dir, prep.exp_id, method)}")
    return record


def run_all(experiments, methods, learning_rates, epochs=30, batch_size=32, momentum=0.9, seed=42,
            results_dir="results", isolate=True):
    """Run every (experiment, method) with a learning rate set; reuse cached runs with the same config.

    isolate=True runs each target model in its own Python subprocess, so all of its memory (datasets of
    up to ~3 GB) is returned to the OS before the next one starts, and every experiment starts from a
    clean process. Progress is streamed live. A failed experiment is reported and the rest continue.
    """
    if isolate:
        import subprocess as sp
        import sys
        failed = []
        for exp_id in experiments:
            job = dict(experiments=[exp_id], methods=methods, learning_rates=learning_rates, epochs=epochs,
                       batch_size=batch_size, momentum=momentum, seed=seed, results_dir=results_dir)
            cmd = [sys.executable, "-u", "-W", "ignore", os.path.abspath(__file__), "--worker", json.dumps(job)]
            proc = sp.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=sp.PIPE,
                            stderr=sp.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
            code = proc.wait()
            if code != 0:
                reason = "killed: out of memory?" if code in (-9, 137) else f"exit code {code}"
                print(f"[{exp_id}] FAILED ({reason}); continuing with the next experiment", flush=True)
                failed.append(exp_id)
        if failed:
            print(f"\nFailed experiments: {failed}. Re-run them; finished runs are cached and not repeated.")
        return failed

    for exp_id in experiments:
        todo = []
        for method in methods:
            lr = learning_rates.get(exp_id, {}).get(method)
            if lr is None:
                print(f"[{exp_id}] {METHOD_NAMES[method]}: no learning rate set, skipped")
                continue
            config = run_config(method, lr, epochs, batch_size, momentum, seed)
            if load_cached(results_dir, exp_id, method, config) is not None:
                print(f"[{exp_id}] {METHOD_NAMES[method]}: cached result found (lr={lr:g}), skipped")
            else:
                todo.append((method, lr))
        if not todo:
            continue

        prep = prepare_experiment(exp_id, seed=seed)
        failed = []
        for method, lr in todo:
            try:
                run_method(prep, method, lr, epochs=epochs, batch_size=batch_size, momentum=momentum,
                           results_dir=results_dir)
            except Exception:  # one failing method must not stop the other methods of this experiment
                import traceback
                traceback.print_exc()
                print(f"[{exp_id}] {METHOD_NAMES[method]} FAILED; continuing with the next method", flush=True)
                failed.append(method)
                free_memory()
        del prep
        free_memory()
        if failed:
            raise SystemExit(1)  # makes the parent report this experiment as failed


# ---------------------------------------------------------------------------
# Results and the LaTeX table
# ---------------------------------------------------------------------------

def load_results(results_dir="results"):
    """{exp_id: {method: record}} for every saved run."""
    results = {}
    for exp_id in ALL_EXPERIMENTS:
        for method in METHODS:
            path = result_path(results_dir, exp_id, method)
            if os.path.exists(path):
                with open(path) as f:
                    results.setdefault(exp_id, {})[method] = json.load(f)
    return results


def summary_dataframe(results):
    rows = []
    for exp_id, per_method in results.items():
        for method, r in per_method.items():
            row = {"exp": exp_id, "dataset": DATASET_NAMES[r["dataset"]], "method": METHOD_NAMES[method],
                   "lr": r["config"]["lr"], "best_epoch": r["best_epoch"],
                   "diverged_at_epoch": r.get("diverged_at_epoch")}
            for state in ["start", "retrain", "best", "final"]:
                s = r[state] or {}  # best/final are None for diverged runs
                row[f"test_acc_{state}"] = s.get("test_acc")
                for a in ATTACKS:
                    row[f"forget_{a}_auc_{state}"] = s.get(f"forget_{a}_auc")
            rows.append(row)
    return pd.DataFrame(rows)


def make_latex_table(results, collapse_ratio=0.25):
    """Fill the paper's sidewaystable. Test accuracies below collapse_ratio x Retr. test accuracy are
    bolded (model collapse). Missing runs are shown as '--'."""
    f = lambda v: "--" if v is None else v if isinstance(v, str) else f"{v:.2f}"
    DIVERGED = "div."  # shown in the Final columns of runs whose weights became inf/NaN
    start = lambda v: f"\\startcell{{{v}}}"

    def get(exp_id, method, state, key):
        r = results.get(exp_id, {}).get(method)
        if r is None:
            return None
        if r[state] is None:  # diverged run: no final model (and no best if it diverged at epoch 1)
            return DIVERGED if state == "final" else None
        return r[state][key]

    def ref(exp_id, state, key):  # Start / Retr. are shared by the methods of one target model
        for method in METHODS:
            v = get(exp_id, method, state, key)
            if v is not None:
                return v
        return None

    def acc_cell(exp_id, method, state):
        v = get(exp_id, method, state, "test_acc")
        retr = ref(exp_id, "retrain", "test_acc")
        if isinstance(v, float) and retr and v < collapse_ratio * retr:
            return f"\\textbf{{{f(v)}}}"
        return f(v)

    metric_keys = ["test_acc"] + [f"forget_{a}_auc" for a in ATTACKS]
    any_diverged = any(r.get("diverged_at_epoch") for per in results.values() for r in per.values())
    lines = [
        r"% Requires: \usepackage{booktabs,multirow,graphicx,rotating}",
        r"% Generated by experiment_runner.make_latex_table. Rows: Dataset -> Target model ID (Table 1) -> method.",
        r"% Start = original model, Retr. = retrained from scratch on the retain set (merged cells).",
        r"% Best epoch = argmin |MIA-1 forget AUC - Retr.| + |Test Acc - Retr.|. '--' = not run yet.",
        "",
        r"\providecommand{\startcell}[1]{\multirow{3}{*}{#1}}",
        "",
        r"\begin{sidewaystable}",
        r"\centering",
        r"\caption{Test accuracy and forget-set MIA AUC for all 12 target models (Table~1) and all three",
        r"unlearning methods: the original model before unlearning (Start), the model retrained from scratch",
        r"on the retain set (Retr.), the unlearned model at the epoch closest to the retrained model (Best;",
        r"epoch in parentheses), and after the last unlearning epoch (Final). Start and Retr. do not depend on",
        r"the unlearning method and are therefore given once per target model. MIA-2 and MIA-3 are evaluated at",
        r"the same epochs as MIA-1. AUC is reported as $\max(\mathrm{AUC}, 1-\mathrm{AUC})$.",
        r"MIA-1: shadow-model attack (Section~5.2); MIA-2: loss-threshold attack; MIA-3: confidence-based attack."
        + (r" div.: unlearning diverged (inf/NaN weights) before the last epoch." if any_diverged else "") + "}",
        r"\label{tab:utility_mia_all}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l c l cccc cccc cccc cccc}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{Test Acc.} & \multicolumn{4}{c}{MIA-1 AUC}",
        r"     & \multicolumn{4}{c}{MIA-2 AUC} & \multicolumn{4}{c}{MIA-3 AUC} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-15}\cmidrule(lr){16-19}",
        r"Dataset & Model & Method & Start & Retr. & Best & Final & Start & Retr. & Best & Final",
        r"        & Start & Retr. & Best & Final & Start & Retr. & Best & Final \\",
        r"\midrule",
    ]
    for d_i, (dataset, exp_ids) in enumerate(TABLE_LAYOUT):
        name = DATASET_NAMES[dataset]
        lines.append(f"% {'=' * 21} {name} {'=' * 21}")
        for e_i, exp_id in enumerate(exp_ids):
            for m_i, method in enumerate(METHODS):
                r = results.get(exp_id, {}).get(method)
                epoch = r["best_epoch"] if r and r["best_epoch"] is not None else "--"
                cells = [
                    f"\\multirow{{9}}{{*}}{{{name}}}" if e_i == 0 and m_i == 0 else "",
                    start(exp_id) if m_i == 0 else "",
                    f"{METHOD_NAMES[method]} ({epoch})",
                ]
                for key in metric_keys:
                    first = m_i == 0
                    cells.append(start(f(ref(exp_id, "start", key))) if first else "")
                    cells.append(start(f(ref(exp_id, "retrain", key))) if first else "")
                    if key == "test_acc":
                        cells += [acc_cell(exp_id, method, "best"), acc_cell(exp_id, method, "final")]
                    else:
                        cells += [f(get(exp_id, method, "best", key)), f(get(exp_id, method, "final", key))]
                lines.append(" & ".join(cells) + r" \\")
            if e_i < len(exp_ids) - 1:
                lines.append(r"\cmidrule(lr){2-19}")
        lines.append(r"\midrule" if d_i < len(TABLE_LAYOUT) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{sidewaystable}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        job = json.loads(sys.argv[2])
        run_all(isolate=False, **job)
    else:
        print("Usage: run from run_all_experiments.ipynb, or: python experiment_runner.py --worker '<json job>'")
