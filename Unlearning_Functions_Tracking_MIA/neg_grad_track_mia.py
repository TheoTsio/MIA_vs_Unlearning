import joblib
import os
import random
import numpy as np
import pandas as pd
import sys


from utils.unlearning_alg.scrub import scrub
from utils.unlearning_alg.sftc_unlearn import sftc_unlearn
from utils.unlearning_alg.neg_grad import neg_grad
from utils.unlearning_utils import RandomDistributionGenerator, CustomPseudoLabelDataset
from utils.preprocessing import to_torch_loader
from utils.train_utils import predict_epoch
from utils.loss_utils import kl_loss, custom_kl_loss, SelectiveCrossEntropyLoss
import copy
import torchvision.models as models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image
from MIA_Functions import *

def neg_grad_tracking_MIA(
            retain_member_df: pd.DataFrame,
            forget_member_df: pd.DataFrame,
            non_member_df: pd.DataFrame,
            attack_model_trained: RandomForestClassifier,
            model: torch.nn.Module,
            retain_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            forget_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            criterion: torch.nn.Module,
            epochs: int = 5,
            return_history: bool = False,
            device: str = 'cuda',
            X_shadow: np.ndarray = None,
            y_shadow: np.ndarray = None,
            num_shadow_models: int = 5,
            num_features: int = 0, 
            num_classes: int = 0,
            batch_size: int =32,
            learning_rate: float =0.001,
            num_epochs_MIA: int =5,
            shadow_model_architecture: Callable = None,
            **kwargs
            ):
    advanced_neg_grad = kwargs.get("advanced_neg_grad", False)

    MIA_acc_retain, MIA_f1_retain, MIA_prec_retain, MIA_recall_retain = [], [], [], []
    MIA_acc_forget, MIA_f1_forget, MIA_prec_forget, MIA_recall_forget = [], [], [], []

    train_losses, val_losses, test_losses, forget_losses = [], [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []

    retain_iterator = iter(retain_loader)
    for epoch in range(epochs):
        model.train()

        running_loss = []
        for x_forget, y_forget in tqdm(forget_loader, desc=f"Epoch {epoch + 1} - Training"):
            x_forget, y_forget = x_forget.to(device), y_forget.to(device)

            try:
                x, y = next(retain_iterator)
            except StopIteration:
                retain_iterator = iter(retain_loader)
                x, y = next(retain_iterator)

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # make a prediction on the forget set
            outputs_forget = model(x_forget)
            total_loss = -criterion(outputs_forget, y_forget)

            if advanced_neg_grad:
                # make a prediction on the retain set
                outputs = model(x)
                total_loss += criterion(outputs, y)

            total_loss.backward()
            optimizer.step()
            running_loss.append(total_loss.item())
        if scheduler is not None:
            scheduler.step()
        epoch_loss = sum(running_loss) / len(running_loss)
        train_losses.append(epoch_loss)
        _, train_acc = predict_epoch(model, retain_loader, criterion, device)
        val_loss, val_acc = predict_epoch(model, val_loader, criterion, device)
        test_loss, test_acc = predict_epoch(model, test_loader, criterion, device)

        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        if forget_loader is not None:
            forget_loss, forget_acc = predict_epoch(model, forget_loader, criterion, device)
            forget_losses.append(forget_loss)
            forget_accs.append(forget_acc)

        print(f"[Epoch {epoch + 1}]\n\t[Train]\tLoss={epoch_loss:.4f}, Acc={train_acc:.4f}\n\t"
              f"[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")

        
        # attack_model_trained = train_shadow_models_and_attack_model(shadow_model_architecture, X_shadow, y_shadow, num_shadow_models, num_features, num_classes, batch_size, learning_rate, num_epochs_MIA)

        dataset_from_model_outputs = create_membership_dataframe(model, retain_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_retain.append(acc)
        MIA_f1_retain.append(f1)
        MIA_prec_retain.append(prec)
        MIA_recall_retain.append(recall)

        dataset_from_model_outputs = create_membership_dataframe(model, forget_member_df, non_member_df)
        acc, f1, prec, recall = evaluate_attack_model_get_stats(dataset_from_model_outputs, attack_model_trained)
        MIA_acc_forget.append(acc)
        MIA_f1_forget.append(f1)
        MIA_prec_forget.append(prec)
        MIA_recall_forget.append(recall)

        losses = {"train": train_losses,
                  "val": val_losses,
                  "test": test_losses}
        accs = {"train": train_accs,
                "val": val_accs,
                "test": test_accs}
        if forget_loader is not None:
            losses["forget"] = forget_losses
            accs["forget"] = forget_accs

        mia_stats = {"retain": {"acc": MIA_acc_retain,
                                "f1": MIA_f1_retain,
                                "precision": MIA_prec_retain,
                                "recall": MIA_recall_retain},
                     "forget": {"acc": MIA_acc_forget,
                                "f1": MIA_f1_forget,
                                "precision": MIA_prec_forget,
                                "recall": MIA_recall_forget}}            
    return mia_stats, accs, losses
