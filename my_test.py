import torchvision
from torchvision import transforms
import torch
from argparse import Namespace 
import numpy as np
from typing import Tuple, Union, List, Dict, Callable
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import pandas as pd
import torch.nn as nn
from helpers import store_training_history, store_trained_model, get_model
import os
import random
from utils.train_utils import base_fit
from Target_Models.target_model_1a import TargetModel_1a

import time
from sklearn.utils import compute_class_weight
rng = torch.Generator().manual_seed(42)

def pre_process_cifar10(
        rng,
        training=False,
        imbalanced=False,
        note=False,
):
    if rng is None:
        rng = torch.Generator().manual_seed(42)

    if note:
        pth = "../data"
    else:
        pth = "./data"

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    if training:
        train_normalize = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        train_normalize = normalize

    train_set = torchvision.datasets.CIFAR10(
        root=pth, train=True, download=True, transform=train_normalize
    )

    held_out = torchvision.datasets.CIFAR10(
        root=pth, train=False, download=True, transform=normalize
    )

    
    # split held out data into validation and test set
    test_set, val_set = torch.utils.data.random_split(
            held_out, [0.5, 0.5], generator=rng
        )
    # download the forget and retain index split
    try:
            local_path = "./data/forget_idx_cifar.npy"
            forget_idx = np.load(local_path)
    except:
            local_path = "../data/forget_idx_cifar.npy"
            forget_idx = np.load(local_path)

    # construct indices of retain set from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set
    forget_set = torch.utils.data.Subset(train_set, forget_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)

    print(f"#Training: {len(train_set)}")
    print(f"#Validation: {len(val_set)}")
    print(f"#Test: {len(test_set)}")
    print(f"#Retain: {len(retain_set)}")
    print(f"#Forget: {len(forget_set)}")

    print(train_normalize)

    return train_set, val_set, test_set, retain_set, forget_set



def to_torch_loader(
        data,
        batch_size: int = 128,
        seed: Union[int, None] = None,
        shuffle: bool = False
) -> DataLoader:
    """
    Create a PyTorch DataLoader with customizable batch size, shuffling, and optional seed for reproducibility.

    Args:
        data: Dataset to be loaded.
        batch_size (int): Batch size for the DataLoader.
        seed (Union[int, None]): Random seed for shuffling. If None, a random seed is generated.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    if shuffle:
        if seed is None:
            seed = np.random.randint(0, 2 ** 32 - 1)
            print(f"Using seed={seed}")
        data_generator = torch.Generator().manual_seed(seed)
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=data_generator,
            num_workers=2,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    return data_loader

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = self.forget_data[index][1]
            pseudo_label = 1
            return x, y, pseudo_label
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = self.retain_data[index - self.forget_len][1]
            pseudo_label = 0
            return x, y, pseudo_label

class CustomPseudoLabelDataset(Dataset):
    """
    A custom dataset that includes pseudo labels indicating whether a sample belongs to the retain set or not.
    """

    def __init__(self, dataset, pseudo_label):
        self.dataset = dataset
        self.pseudo_label = pseudo_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, self.pseudo_label

def get_data_training(
        args, return_sets: bool = False
) -> Union[Tuple[DataLoader, DataLoader, DataLoader, DataLoader], Tuple[
    DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]
]:
    try:
        note = args.note
    except KeyError:
        note = False
    except AttributeError:
        note = False
    try:
        unlearn = args.unlearn
    except KeyError:
        unlearn = False
    except AttributeError:
        unlearn = False
    training = False if unlearn else True
    dataset = args.dataset.lower()
    if dataset == "cifar":
        train_set, val_set, test_set, retain_set, forget_set = pre_process_cifar10(
            rng=None,
            training=training,
            note=note
        )
    else:
        raise ValueError

    if return_sets:
        return retain_set, val_set, test_set, forget_set

    try:
        unlearn_alg = args.algorithm
    except KeyError:
        unlearn_alg = None
    except AttributeError:
        unlearn_alg = None
    if unlearn_alg is not None and unlearn_alg == "bad_teaching":
        retain_set = UnLearningData(forget_data=forget_set, retain_data=retain_set)

    test_loader = to_torch_loader(test_set, batch_size=args.batch_size, shuffle=False)
    val_loader = to_torch_loader(val_set, batch_size=args.batch_size, shuffle=False)
    retain_loader = to_torch_loader(retain_set, batch_size=args.batch_size, shuffle=True)
    train_loader = to_torch_loader(train_set, batch_size=args.batch_size, shuffle=True)
    forget_loader = to_torch_loader(forget_set, batch_size=args.batch_size, shuffle=True)

    if unlearn_alg is not None and unlearn_alg == "sftc":
        retain_set_pseudo = CustomPseudoLabelDataset(retain_set, 0)
        forget_set_pseudo = CustomPseudoLabelDataset(forget_set, 1)
        merged_set = ConcatDataset([retain_set_pseudo, forget_set_pseudo])
        merged_loader = to_torch_loader(merged_set, batch_size=args.batch_size, shuffle=True)
        return retain_loader, val_loader, test_loader, forget_loader, merged_loader

    if args.mode.lower() == "full":
        return train_loader, val_loader, test_loader, forget_loader
    else:
        return retain_loader, val_loader, test_loader, forget_loader
    
    

# class CifarCNN(nn.Module):
#     """
#     Light CNN for 32×32 CIFAR-10/100.
#     """
#     def __init__(self, num_classes: int = 10):
#         super().__init__()

#         # (N,3,32,32)  →  (N,64,16,16)
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64, 3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )

#         # (N,64,16,16) →  (N,128,8,8)
#         self.block2 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, 3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )

#         # (N,128,8,8)  →  (N,256,4,4)
#         self.block3 = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )

#         # classifier
#         self.avgpool = nn.AdaptiveAvgPool2d(1)   # (N,256,1,1)
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(128, num_classes)
#         )

#         # He init
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.avgpool(x)
#         return self.fc(x)
# # params
full_train_args = Namespace(
    dataset='cifar', # CIFAR dataset
    mode='full', # do not get the full training set
    model='TargetModel_1a', # the model to use
    lr=1e-3, # learning rate
    optimizer='adam', # optimizer
    decay=0, # weight decay
    epochs=30, # number of epochs
    batch_size=64, # batch size
    scheduler=True, # 
    class_weights=False,
    store_model=True,
    store_history=True,
    num_runs=1
)   

def get_optimizer(args, model) -> torch.optim.Optimizer:
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.decay
        )
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.decay, momentum=args.momentum
        )
    else:
        raise ValueError
    return optimizer



def get_scheduler(args, optimizer) -> Union[torch.optim.lr_scheduler.LRScheduler, None]:
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.epochs, verbose=True
        )
        return scheduler
    return None



def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set as {seed}")
    
def calculate_class_weights(data_loader, log_count=True):
    """Calculates class weights based on the frequency of each class."""
    # get the labels from the training set
    all_labels = []
    for _, labels in data_loader:
        all_labels.extend(labels.numpy())
    if log_count:
        print(f"Counts per class: {np.bincount(all_labels)}")

    # calculate class weights
    classes = np.unique(all_labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=all_labels)

    return torch.tensor(class_weights, dtype=torch.float)


def get_criterion(args, data_loader, device) -> torch.nn.Module:
    if args.class_weights:
        class_weights = calculate_class_weights(data_loader)
        class_weights = class_weights.to(device)
        print(f"Class Weights: {class_weights}")
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    return criterion

def train(args):
    train_loader, val_loader, test_loader, forget_loader = get_data_training(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(args.num_runs):
        model = get_model(args)
        model.to(device)

        optimizer = get_optimizer(args, model)
        scheduler = get_scheduler(args, optimizer)

        criterion = get_criterion(args, train_loader, device)
        model, history = base_fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            forget_loader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epochs=args.epochs,
            return_history=True,
            device=device,
        )
        timestamp = str(time.time())
        name = str(args.model) + "_" + str(args.mode) + "_" + str(args.dataset) + "_" + timestamp
        if args.store_history:
            fig_name = name + "_history"
            store_training_history(args, history, fig_name)
        if args.store_model:
            store_trained_model(args, model, name)
        return model



set_seed(42)
trained_model = train(full_train_args)