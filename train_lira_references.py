"""
Train the LiRA reference ("OUT") models and store their scores on the evaluation pools (see lira.py).

Usage:
    python train_lira_references.py                         # all 12 experiments, N_REFERENCE models each
    python train_lira_references.py --experiments 2c 2d --n-refs 16
    python train_lira_references.py --check                 # only print the status

Each experiment runs in its own subprocess; each finished reference model is saved immediately
(models/lira/<exp>/ref_<k>.npz), so an interrupted run resumes. Reference model k uses seed 1000 + k for its
training subset and initialisation.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import experiment_runner as er
import lira


def reference_path(exp_id, k):
    return os.path.join(lira.LIRA_DIR, exp_id, f"ref_{k}.npz")


def train_one(exp_id, k, data, model_cls, cfg):
    seed = 1000 + k
    er.set_random_seed(seed)
    X_sh, y_sh = data["shadow"]
    rng = np.random.default_rng(seed)
    n = min(data["n_train"], len(X_sh))
    perm = rng.permutation(len(X_sh))
    tr, te = perm[:n], perm[n:]

    model = model_cls(input_size=data["num_features"], output_size=data["num_classes"]).to(er.device)
    optimizer = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"])
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_sh[tr], torch.as_tensor(y_sh[tr])), batch_size=cfg["batch_size"],
                        shuffle=True)
    t0 = time.time()
    epochs_run, train_acc = 0, 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        correct = total = 0
        for xb, yb in loader:
            xb, yb = xb.to(er.device), yb.to(er.device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        epochs_run, train_acc = epoch + 1, correct / total
        if not np.isfinite(loss.item()):
            raise RuntimeError(f"reference model {k} diverged (loss {loss.item()})")
        if cfg.get("stop_train_acc") and train_acc >= cfg["stop_train_acc"]:
            break

    with torch.no_grad():
        model.eval()
        preds = torch.cat([model(X_sh[te][i:i + 256].to(er.device)).argmax(1).cpu()
                           for i in range(0, len(te), 256)]) if len(te) else torch.empty(0)
    held_out_acc = float((preds.numpy() == y_sh[te]).mean()) if len(te) else float("nan")

    pools = data["pools"]
    phi = np.concatenate([lira.phi_scores(model, *pools[p]) for p in ("forget", "retain", "non_member")])
    info = dict(exp_id=exp_id, k=k, seed=seed, n_train=int(n), epochs_run=epochs_run, train_acc=train_acc,
                shadow_held_out_acc=held_out_acc, config=cfg, seconds=round(time.time() - t0, 1))
    np.savez(reference_path(exp_id, k), phi=phi.astype(np.float64), info=json.dumps(info))
    del model, optimizer
    er.free_memory()
    return info


def worker(exp_id, n_ref):
    todo = [k for k in range(n_ref) if not os.path.exists(reference_path(exp_id, k))]
    if not todo:
        print(f"[{exp_id}] {n_ref} reference models already trained", flush=True)
        return
    t0 = time.time()
    data = lira.rebuild_pools(exp_id)
    target = torch.load(f"models/{exp_id}_dataset_target_model.pth", map_location=er.device, weights_only=False)
    model_cls = type(target)
    del target
    os.makedirs(os.path.join(lira.LIRA_DIR, exp_id), exist_ok=True)
    keys = np.concatenate([lira.row_keys(*data["pools"][p]) for p in ("forget", "retain", "non_member")])
    keys_path = os.path.join(lira.LIRA_DIR, exp_id, "pool_keys.npy")
    if os.path.exists(keys_path) and not np.array_equal(np.load(keys_path), keys):
        raise RuntimeError(f"{keys_path} does not match the rebuilt pools; delete models/lira/{exp_id}/")
    np.save(keys_path, keys)
    cfg = lira.TRAIN_CONFIG[exp_id]
    print(f"[{exp_id}] {model_cls.__name__} | shadow {len(data['shadow'][1])}, reference train size "
          f"{data['n_train']} | pools: " + ", ".join(f"{p} {len(v[1])}" for p, v in data["pools"].items())
          + f" | config {cfg} | loaded in {time.time() - t0:.0f}s", flush=True)
    for k in todo:
        info = train_one(exp_id, k, data, model_cls, cfg)
        print(f"  [{exp_id}] reference {k + 1}/{n_ref}: {info['epochs_run']} epochs, train acc "
              f"{info['train_acc']:.3f}, held-out shadow acc {info['shadow_held_out_acc']:.3f}, "
              f"{info['seconds']:.0f}s", flush=True)


def status(experiments, n_ref):
    for exp_id in experiments:
        d = os.path.join(lira.LIRA_DIR, exp_id)
        files = sorted(f for f in os.listdir(d) if f.startswith("ref_")) if os.path.isdir(d) else []
        infos = [json.loads(str(np.load(os.path.join(d, f))["info"])) for f in files]
        acc = np.mean([i["shadow_held_out_acc"] for i in infos]) if infos else float("nan")
        print(f"{exp_id}: {len(files)}/{n_ref} reference models; mean held-out acc {acc:.3f}")


def run(experiments, n_refs=lira.N_REFERENCE):
    """Train the missing reference models, one subprocess per experiment (output streamed, also in Jupyter)."""
    failed = []
    for exp_id in experiments:
        cmd = [sys.executable, "-u", "-W", "ignore", os.path.abspath(__file__), "--worker",
               json.dumps(dict(exp_id=exp_id, n=n_refs))]
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end="", flush=True)
        code = proc.wait()
        if code != 0:
            reason = "killed: out of memory?" if code in (-9, 137) else f"exit code {code}"
            print(f"[{exp_id}] FAILED ({reason}); continuing with the next experiment", flush=True)
            failed.append(exp_id)
    status(experiments, n_refs)
    return failed


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        job = json.loads(sys.argv[2])
        worker(job["exp_id"], job["n"])
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=er.ALL_EXPERIMENTS, choices=er.ALL_EXPERIMENTS)
    parser.add_argument("--n-refs", type=int, default=lira.N_REFERENCE, help="reference models per experiment")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        status(args.experiments, args.n_refs)
    else:
        run(args.experiments, args.n_refs)


if __name__ == "__main__":
    main()
