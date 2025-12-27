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

def scrub_tracking_MIA(
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
        epochs: int = 3,
        return_history: bool = False,
        device: str = 'cuda',
        **kwargs
):
    """"""
    try:
        teacher_model = kwargs['teacher_model']
    except KeyError:
        raise ValueError("SCRUB requires the teacher_model to be given as input!")

    temperature = kwargs.get("temperature", 4.)

    teacher_model.eval()
    model.train()

    MIA_acc_retain, MIA_f1_retain, MIA_prec_retain, MIA_recall_retain = [], [], [], []
    MIA_acc_forget, MIA_f1_forget, MIA_prec_forget, MIA_recall_forget = [], [], [], []

    train_losses_retain, train_losses_forget = [], []
    val_losses, test_losses, forget_losses = [], [], []
    train_accs, val_accs, test_accs, forget_accs = [], [], [], []
    # Training with retain data
    for epoch in range(epochs):
        running_classification_loss_retain, running_kl_retain = [], []
        total_retain_loss, running_kl_forget = [], []

        model.train()
        for x_forget, y_forget in tqdm(forget_loader, desc=f"Epoch {epoch + 1} - Training on Forget"):
            x_forget, y_forget = x_forget.to(device), y_forget.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                # local_x = x_forget.view(-1, 3, 32, 32)
                teacher_outputs = teacher_model(x_forget)


            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x_forget)
            # print('my model ', outputs.shape)

            # maximize the kl div loss
            loss = -kl_loss(model_logits=outputs,
                            teacher_logits=teacher_outputs,
                            temperature=temperature,
                            distill=True)
            loss.backward()
            optimizer.step()
            running_kl_forget.append(loss.item())

        model.train()

        for x, y in tqdm(retain_loader, desc=f"Epoch {epoch + 1} - Training"):
            x, y = x.to(device), y.to(device)

            # Make a prediction using the teacher
            with torch.no_grad():
                # temp_x = x.view(-1, 3, 32, 32)
                teacher_outputs = teacher_model(x)

            optimizer.zero_grad()
            # Make a prediction using the student
            outputs = model(x)

            loss = criterion(outputs, y)
            running_classification_loss_retain.append(loss.item())
            kl = kl_loss(model_logits=outputs,
                         teacher_logits=teacher_outputs,
                         temperature=temperature,
                         distill=True)
            running_kl_retain.append(kl.item())
            total_loss = loss + kl
            total_loss.backward()
            optimizer.step()
            total_retain_loss.append(total_loss.item())

        if scheduler is not None:
            scheduler.step()

        epoch_classification_loss = sum(running_classification_loss_retain) / len(running_classification_loss_retain)
        epoch_kl_retain_loss = sum(running_kl_retain) / len(running_kl_retain)
        epoch_kl_forget_loss = sum(running_kl_forget) / len(running_kl_forget)

        train_losses_retain.append(epoch_classification_loss)
        train_losses_forget.append(epoch_kl_forget_loss)

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

        print(f"[Epoch {epoch + 1}]\n\t[Train]\tRetain Classification Loss={epoch_classification_loss:.4f}, "
              f"Retain KL Loss={epoch_kl_retain_loss:.4f}, Forget KL Loss={epoch_kl_forget_loss}, Acc={train_acc:.4f}\n\t"
              f"[Val]\tLoss={val_loss:.4f}, Acc={val_acc:.4f}\n\t"
              f"[Test]\tLoss={test_loss:.4f}, Acc={test_acc:.4f}")
        if forget_loader is not None:
            print(f"\t[Forget] Loss={forget_loss:.4f}, Acc={forget_acc:.4f}")

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

        losses = {"train": train_losses_retain,
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
