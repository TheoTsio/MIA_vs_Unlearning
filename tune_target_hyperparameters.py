"""
Hyperparameter tuning for the basic target models with Optuna.

Data split (consistent with train_target_and_perform_MIA_attack.py / main.py):

    X  ──► get_target_shadow_split ──► X_target | X_shadow | X_test_MIA
    X_target ──► get_train_test_split ──► X_train_full | X_test     (same test set as before)
    X_train_full ──► stratified split  ──► X_train | X_val          (new: validation set)

Optuna trains on X_train, selects on X_val, and only the best configuration is
evaluated once on X_test. The test set never influences the search.

Usage examples:
    python tune_target_hyperparameters.py --dataset texas --model 3d --n-trials 50
    python tune_target_hyperparameters.py --dataset purchase --model 1c --n-trials 30 --max-epochs 60
    python tune_target_hyperparameters.py --dataset cifar10 --model 2a --n-trials 20

Results are written to hyperparameters/<dataset>_<model>_best_params.json
and hyperparameters/<dataset>_<model>_trials.csv.
"""
import argparse
import json
import os
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from preprocess_data import (get_cifar10_dataset, get_MUFAC_dataset,
                             get_purchase_dataset, get_texas_100_dataset)
from utils.data_split import get_target_shadow_split, get_train_test_split
from Target_Models.target_model_1a import TargetModel_1a
from Target_Models.target_model_1c import TargetModel_1c
from Target_Models.target_model_2a import TargetModel_2a
from Target_Models.target_model_2b import TargetModel_2b
from Target_Models.target_model_2c import TargetModel_2c
from Target_Models.target_model_3a import TargetModel_3a
from Target_Models.target_model_3b import TargetModel_3b
from Target_Models.target_model_3c import TargetModel_3c
from Target_Models.target_model_3d import TargetModel_3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model name -> (class, input reshape). CNNs 2a/2b need (C, H, W) input;
# all others accept flat vectors (3a/3b reshape internally).
MODELS = {
    "1a": (TargetModel_1a, None),
    "1c": (TargetModel_1c, None),
    "2a": (TargetModel_2a, (3, 32, 32)),
    "2b": (TargetModel_2b, (3, 128, 128)),
    "2c": (TargetModel_2c, None),
    "3a": (TargetModel_3a, None),
    "3b": (TargetModel_3b, None),
    "3c": (TargetModel_3c, None),
    "3d": (TargetModel_3d, None),
}


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(name: str):
    if name == "texas":
        return get_texas_100_dataset(path="data/texas100.npz", limit_rows=40_000)
    if name == "purchase":
        return get_purchase_dataset(dataset_path="data/dataset_purchase.csv", keep_rows=40_000)
    if name == "cifar10":
        return get_cifar10_dataset()
    if name == "mufac":
        return get_MUFAC_dataset(
            "data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv",
            "data/custom_korean_family_dataset_resolution_128/train_images",
            percentage_of_rows_to_drop=0.4)
    raise ValueError(f"Unknown dataset: {name}")


def get_train_val_test_split(X, y, val_size=0.2, seed=42):
    """Split the full dataset into target train / validation / test sets.

    The test set is identical to the one produced by get_train_test_split, so the
    tuned hyperparameters can be dropped straight into the existing pipeline.
    The validation set is a stratified slice of the target training data.
    """
    X_target, _, y_target, _, _, _ = get_target_shadow_split(X, y, seed=seed)
    X_train_full, X_test, y_train_full, y_test = get_train_test_split(X_target, y_target, seed=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed, stratify=y_train_full)
    return X_train, X_val, X_test, y_train, y_val, y_test


def to_tensors(X, y, input_shape=None):
    X_t = torch.tensor(X.values, dtype=torch.float32)
    if input_shape is not None:
        X_t = X_t.reshape(-1, *input_shape)
    y_t = torch.tensor(np.asarray(y).ravel(), dtype=torch.long)
    return X_t, y_t


def evaluate(model, X_t, y_t, criterion, batch_size=512):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            xb, yb = X_t[i:i + batch_size].to(device), y_t[i:i + batch_size].to(device)
            out = model(xb)
            total_loss += criterion(out, yb).item() * len(xb)
            correct += (out.argmax(dim=1) == yb).sum().item()
    return total_loss / len(X_t), correct / len(X_t)


def build_optimizer(params, model):
    name = params["optimizer"]
    if name == "SGD":
        return optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"],
                         weight_decay=params["weight_decay"])
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    return optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])


def train_and_validate(params, model_cls, num_features, num_classes, train_ds, X_val_t, y_val_t,
                       max_epochs, patience, trial=None):
    """Train one model with `params`; return best validation accuracy and the epoch it occurred.

    Uses early stopping on validation loss. If `trial` is given, reports validation
    accuracy every epoch so Optuna can prune unpromising trials.
    """
    model = model_cls(input_size=num_features, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(params, model)
    loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)

    best_val_loss, best_val_acc, best_epoch, bad_epochs = float("inf"), 0.0, 0, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_loss, val_acc = evaluate(model, X_val_t, y_val_t, criterion)
        if not np.isfinite(val_loss):
            break  # diverged

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss, bad_epochs = val_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    return best_val_acc, best_epoch, model


def suggest_params(trial):
    params = {
        "optimizer": trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }
    if params["optimizer"] == "SGD":
        params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
    return params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="texas", choices=["texas", "purchase", "cifar10", "mufac"])
    parser.add_argument("--model", default="3d", choices=sorted(MODELS))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5, help="early-stopping patience on val loss")
    parser.add_argument("--val-size", type=float, default=0.2, help="fraction of target train data used for validation")
    parser.add_argument("--timeout", type=int, default=None, help="stop the study after this many seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="hyperparameters")
    args = parser.parse_args()

    set_random_seed(args.seed)
    print(f"Using device: {device}")

    X, y, num_features, num_classes = load_dataset(args.dataset)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_split(X, y, args.val_size, args.seed)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    model_cls, input_shape = MODELS[args.model]
    X_train_t, y_train_t = to_tensors(X_train, y_train, input_shape)
    X_val_t, y_val_t = to_tensors(X_val, y_val, input_shape)
    X_test_t, y_test_t = to_tensors(X_test, y_test, input_shape)
    train_ds = TensorDataset(X_train_t, y_train_t)

    def objective(trial):
        set_random_seed(args.seed + trial.number)
        params = suggest_params(trial)
        best_val_acc, best_epoch, _ = train_and_validate(
            params, model_cls, num_features, num_classes, train_ds, X_val_t, y_val_t,
            args.max_epochs, args.patience, trial)
        trial.set_user_attr("best_epoch", best_epoch)
        return best_val_acc

    study = optuna.create_study(
        study_name=f"{args.dataset}_{args.model}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)

    best = study.best_trial
    best_params = dict(best.params)
    best_params["num_epochs"] = best.user_attrs["best_epoch"]
    print(f"\nBest val accuracy: {best.value:.4f}")
    print("Best params:", best_params)

    # Retrain once with the best params and report test performance (test set used only here).
    set_random_seed(args.seed)
    _, _, final_model = train_and_validate(
        best_params, model_cls, num_features, num_classes, train_ds, X_val_t, y_val_t,
        args.max_epochs, args.patience)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_acc = evaluate(final_model, X_train_t, y_train_t, criterion)
    val_loss, val_acc = evaluate(final_model, X_val_t, y_val_t, criterion)
    test_loss, test_acc = evaluate(final_model, X_test_t, y_test_t, criterion)
    print(f"Final model | train acc {train_acc:.4f} | val acc {val_acc:.4f} | test acc {test_acc:.4f}")
    print(f"Generalization gap (train - test acc): {train_acc - test_acc:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = os.path.join(args.out_dir, f"{args.dataset}_{args.model}")
    with open(f"{prefix}_best_params.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "model": args.model,
            "best_params": best_params,
            "best_val_acc": best.value,
            "final_train_acc": train_acc,
            "final_val_acc": val_acc,
            "final_test_acc": test_acc,
            "n_trials": len(study.trials),
        }, f, indent=2)
    study.trials_dataframe().to_csv(f"{prefix}_trials.csv", index=False)
    print(f"Saved results to {prefix}_best_params.json and {prefix}_trials.csv")


if __name__ == "__main__":
    main()
