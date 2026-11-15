# Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks

This repository contains the code for our study of whether **machine unlearning** can be used as a
**defense against membership inference attacks (MIAs)**.

A membership inference attack tries to tell whether a given sample was in a model's training set.
Machine unlearning tries to remove the influence of a chosen set of training samples (the *forget
set*) from a trained model without retraining it from scratch. We ask:

> After a model "unlearns" the forget set, can an attacker still detect that those samples were
> once training members? How close does the unlearned model get to a model that never saw them?

We train target models on four datasets with three families of architectures, unlearn a forget
set with three unlearning algorithms, and **after every unlearning epoch** we run three different
MIAs. The **gold standard** is a model retrained from scratch on the retain set only. For that
model the forget samples really are non-members, so its MIA AUC is about 0.5.

---

## Pipeline overview

```
            dataset
               │
   get_target_shadow_split (seed 42)
     ┌─────────┼──────────────┐
  X_target  X_shadow     X_test_MIA  (non-members for the attack evaluation)
     │          │
 train / test   └─► 5 shadow models ─► RandomForest attack model (MIA-1)
     │
 train ─► retain (80%) / forget (20%)
     │
     ├─► target model          (trained on retain + forget)
     ├─► retrain baseline      (trained on retain only = ideal unlearning)
     │
     └─► unlearning of the target model (NegGrad / SCRUB / SFTC), 30 epochs
             after each epoch: test accuracy + MIA-1/2/3 AUC on forget and retain
```

### Datasets

| Dataset      | Type                         | Classes | Notes                                    |
|--------------|------------------------------|---------|------------------------------------------|
| CIFAR-10     | 32×32 RGB images             | 10      | downloaded automatically by torchvision  |
| MuFac        | 128×128 face images (age)    | 8       | Korean family dataset, 40% of rows dropped |
| Purchase-100 | 600 binary purchase features | 100     | first 40,000 rows                        |
| Texas-100    | 6,169 binary hospital features | 100   | first 40,000 rows                        |

### Target models (experiments `1a` … `3d`)

The digit is the model family and the letter is the dataset (`a` CIFAR-10, `b` MuFac, `c` Purchase-100, `d` Texas-100).

| Exp. | Dataset      | Architecture class | Exp. | Dataset | Architecture class |
|------|--------------|--------------------|------|---------|--------------------|
| 1a   | CIFAR-10     | `TargetModel_1a` (MLP) | 2c | Purchase-100 | `TargetModel_2c` |
| 1b   | MuFac        | `TargetModel_1a` (MLP) | 2d | Texas-100 | `TargetModel_2c` |
| 1c   | Purchase-100 | `TargetModel_1c` (MLP) | 3a | CIFAR-10 | `TargetModel_3a` (ResNet18) |
| 1d   | Texas-100    | `TargetModel_1c` (MLP) | 3b | MuFac | `TargetModel_3b` (ResNet18) |
| 2a   | CIFAR-10     | `TargetModel_2a` (CNN) | 3c | Purchase-100 | `TargetModel_3c` (deep MLP) |
| 2b   | MuFac        | `TargetModel_2b` (CNN) | 3d | Texas-100 | `TargetModel_3d` (deep MLP) |

### Unlearning methods

| Method  | Idea |
|---------|------|
| **NegGrad** | gradient ascent on the forget set (optionally combined with descent on the retain set) |
| **SCRUB**   | teacher–student: the student moves away from the original (teacher) model on the forget set and stays close to it on the retain set |
| **SFTC**    | teacher–student on a merged retain/forget loader: imitate the teacher on retain samples and a random-distribution "dummy" model on forget samples |

The implementations are in [`utils/unlearning_alg/`](utils/unlearning_alg). The versions in
[`Unlearning_Functions_Tracking_MIA/`](Unlearning_Functions_Tracking_MIA) call the MIAs after every epoch.

### Membership inference attacks

All attacks are evaluated as **forget set vs. unseen samples** and **retain set vs. unseen samples**, using balanced sets.
AUC is reported as `max(AUC, 1 − AUC)`. TPR at 1% FPR is also saved.

| Attack | Paper | Score | Attacker needs |
|--------|-------|-------|----------------|
| **MIA-1**: shadow-model attack | Shokri et al., 2017 | P(member) from a RandomForest trained on the softmax outputs of 5 shadow models | shadow data, full probability vector |
| **MIA-2**: loss-threshold attack | Yeom et al., 2018 | `log p(y_true \| x)` (= −cross-entropy loss) | true label, probabilities |
| **MIA-3**: confidence attack | Salem et al., 2019 (ML-Leaks) | `max_c p(c \| x)` | probabilities only |

[`MIA_Threshold_Attacks.py`](MIA_Threshold_Attacks.py) also has a modified-entropy attack (Song &
Mittal, 2021) and a label-only gap attack. [`main.py`](main.py) uses them for the single-experiment analysis.

### How results are reported

For each experiment and unlearning method we report test accuracy and MIA-1/2/3 forget-set AUC at four points:

- **Start**: the original target model before unlearning.
- **Retr.**: the model retrained from scratch on the retain set. This is the target.
- **Best**: the unlearning epoch closest to the retrained model, chosen by
  `|MIA-1 AUC − Retr.| + |Test Acc − Retr. Test Acc|`. Accuracy is part of the criterion so a collapsed model is not picked.
- **Final**: the model after the last epoch, or `div.` if the weights became inf/NaN.

---

## Repository structure

```
├── train_target_and_perform_MIA_attack.py  # train target, retrain baseline, shadow models + attack model
├── tune_target_hyperparameters.py          # Optuna tuning of the target models (train/val/test split)
├── main.py                                 # one experiment: unlearning + all MIAs, plots in curves/
├── experiment_runner.py                    # all 12 experiments × 3 methods, reproducible, cached
├── run_all_experiments.ipynb               # notebook front-end: config, run, summary, LaTeX table
├── MIA_Functions.py                        # shadow-model MIA (MIA-1) helpers
├── MIA_Threshold_Attacks.py                # loss / confidence / entropy / label-only attacks
├── preprocess_data.py                      # dataset loaders (CIFAR-10, MuFac, Purchase, Texas)
├── Target_Models/                          # architectures TargetModel_1a … TargetModel_3d
├── Unlearning_Functions_Tracking_MIA/      # NegGrad / SCRUB / SFTC with per-epoch MIA tracking
├── utils/                                  # data splits, training loops, losses, unlearning algorithms
├── data/                                   # datasets (not in git)
├── models/                                 # trained models <exp>_dataset_*.pth / .jolib (not in git)
├── results/                                # per-experiment results + LaTeX table
└── curves/                                 # figures from main.py
```

---

## Installation

```bash
conda create -n unlearn python=3.11
conda activate unlearn
pip install -r requirements.txt
pip install jupyter            # only needed for the notebook
```

A CUDA GPU is recommended, but the code also runs on CPU. The experiments were run on a 4 GB GPU with 14 GB RAM.

### Data

Put the datasets under `data/`. The paths are the ones used in the code:

| Dataset      | Expected path |
|--------------|---------------|
| CIFAR-10     | `data/` (downloaded automatically) |
| MuFac        | `data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv` and `.../train_images/` |
| Purchase-100 | `data/dataset_purchase.csv` |
| Texas-100    | `data/texas100.npz` |

---

## How to run

### 1. Train the target, retrain and attack models

Choose the dataset and architecture at the top of
[`train_target_and_perform_MIA_attack.py`](train_target_and_perform_MIA_attack.py) (dataset loader, model class, learning rate, epochs), then run:

```bash
python train_target_and_perform_MIA_attack.py
```

It saves `models/dataset_target_model.pth`, `models/dataset_retrain_baseline_model.pth` and
`models/dataset_attack_model.jolib`. **Add the experiment id as a prefix** (for example
`models/3d_dataset_target_model.pth`) so the other scripts can find them. Repeat for each experiment.

*Optional:* tune the target model hyperparameters first:

```bash
python tune_target_hyperparameters.py --dataset texas --model 3d --n-trials 50
```

The results go to `hyperparameters/<dataset>_<model>_best_params.json`.

### 2a. Run all experiments (recommended)

Open [`run_all_experiments.ipynb`](run_all_experiments.ipynb), set the per-experiment / per-method
learning rates in `LEARNING_RATES` (along with `EPOCHS`, `BATCH_SIZE` and `SEED` if needed), and run all cells.
To run it without opening Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 run_all_experiments.ipynb
```

The notebook:

- runs every experiment in its own subprocess to keep memory low;
- **caches** each `(experiment, method)` result in `results/<exp>/<method>.json` together with its config.
  An interrupted run continues where it stopped. A run is redone only if its config changes;
- writes the per-epoch metrics to `results/<exp>/<method>_epochs.csv`;
- writes `results/summary.csv` and the LaTeX table `results/table_utility_mia_all.tex`.

### 2b. Run a single experiment in detail

Set the experiment at the top of the `__main__` block in [`main.py`](main.py):

```python
experiment_id = '1d'
target_architecture = TargetModel_1c   # must match the experiment (see table above)
input_shape = None                     # (3, H, W) for CNNs 2a / 2b
```

then run:

```bash
python main.py
```

It runs the three unlearning methods, tracks all the attacks every epoch and writes the plots and CSVs to `curves/`
(`accuracy_curves.pdf`, `mia_attacks_auc.pdf`, `mia_attacks_tpr_at_1fpr.pdf`, `mia_attacks_summary.csv`, …).

---

## Reproducibility

- All splits use `seed=42`. Every run is reseeded before it starts.
- `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are set in
  `experiment_runner.py`. Repeated runs give identical results.
- Each saved result stores its full config (method, learning rate, epochs, batch size, momentum, seed),
  so you can check exactly which settings produced it.

## License

See [LICENSE](LICENSE).
