import joblib
import os
import random
import numpy as np
import pandas as pd
import sys
# Add the SFTC-Unlearn directory to Python path
sftc_unlearn_path = '/home/theo/Desktop/Evaluate_Mia_Through_Unlearning'
if sftc_unlearn_path not in sys.path:
    sys.path.insert(0, sftc_unlearn_path)

# Verify the path was added
print("Python path:", sys.path)
print("Looking for utils at:", os.path.join(sftc_unlearn_path, 'utils'))
print("Utils exists:", os.path.exists(os.path.join(sftc_unlearn_path, 'utils')))
print("unlearning_alg exists:", os.path.exists(os.path.join(sftc_unlearn_path, 'utils', 'unlearning_alg')))

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
'''
Datasets
'''


def create_merged_loader(X_retain, y_retain, X_forget, y_forget, batch_size=32, shuffle=True):
    """
    Create a merged DataLoader from retain and forget datasets where the inputs are PyTorch tensors.

    Args:
        X_retain (torch.Tensor): Features for the retain dataset (tensor).
        y_retain (torch.Tensor): Labels for the retain dataset (tensor).
        X_forget (torch.Tensor): Features for the forget dataset (tensor).
        y_forget (torch.Tensor): Labels for the forget dataset (tensor).
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
        DataLoader: Merged DataLoader with features, labels, and pseudo-labels.
    """
    # Create pseudo-labels: 0 for retain, 1 for forget
    pseudo_labels_retain = torch.zeros(len(X_retain), dtype=torch.long)  # Pseudo-label 0 for retain
    pseudo_labels_forget = torch.ones(len(X_forget), dtype=torch.long)   # Pseudo-label 1 for forget

    # Combine retain and forget datasets
    X_merged = torch.cat([X_retain, X_forget], dim=0)
    y_merged = torch.cat([y_retain, y_forget], dim=0)
    pseudo_labels_merged = torch.cat([pseudo_labels_retain, pseudo_labels_forget], dim=0)

    # Create a TensorDataset with X, y, and pseudo-labels
    merged_dataset = TensorDataset(X_merged, y_merged, pseudo_labels_merged)

    # Create the DataLoader for the merged dataset
    merged_loader = DataLoader(merged_dataset, batch_size=batch_size, shuffle=shuffle)

    return merged_loader



def parsing(meta_data):
    image_age_list = []
    for idx, row in meta_data.iterrows():
        image_path = row['image_path']
        age_class = row['age_class']
        image_age_list.append([image_path, age_class])
    return image_age_list

# Custom Dataset class to handle image loading and transformations
class MUFACCustomDataset(Dataset):
    def __init__(self, meta_data, image_directory, transform=None, forget=False, retain=False):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata
        image_age_list = parsing(meta_data)

        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7
        }

        # Splitting dataset for machine unlearning (forget and retain datasets)
        if forget:
            self.image_age_list = self.image_age_list[:1500]
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_texas_100_dataset(path='data/texas100.npz', limit_rows=None):
    """
    Processes the Texas 100 dataset.

    Returns:
    X (DataFrame): DataFrame of features.
    y (DataFrame): DataFrame with a single 'label' column representing the class index.
    num_features (int): Number of features.
    num_classes (int): Number of unique classes in the dataset.
    """

    data = np.load(path)
    features = data['features']
    labels = data['labels']

    if limit_rows:
        features = features[:limit_rows]
        labels = labels[:limit_rows]

    # Convert features and labels to DataFrames if they are not already
    X = pd.DataFrame(features)

    # Convert one-hot encoded labels to class indices
    y = pd.DataFrame()
    y['label'] = pd.DataFrame(labels).idxmax(axis=1).astype(int)

    # Get the number of features and classes
    num_features = X.shape[1]
    num_classes = len(np.unique(y['label']))

    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")

    return X, y, num_features, num_classes


def get_MUFAC_dataset(train_meta_data_path, train_image_directory, batch_size=64, percentage_of_rows_to_drop=0.2):
    # Define transformations for training data
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])

    # Load metadata, limiting to 2000 rows
    train_meta_data = pd.read_csv(train_meta_data_path)

    # Initialize the Dataset and DataLoader
    train_dataset = MUFACCustomDataset(train_meta_data, train_image_directory, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Convert DataLoader to Pandas DataFrames (dataloader_to_dataframe logic)
    all_images = []
    all_labels = []

    for images, labels in train_dataloader:
        # Flatten the image tensors (e.g., 64, 3, 128, 128 -> 64, 3*128*128)
        flattened_images = images.view(images.size(0), -1).numpy()

        # Convert labels to numpy array
        labels = labels.numpy()

        # Accumulate images and labels
        all_images.append(flattened_images)
        all_labels.append(labels)

    # Concatenate all batches into a single array
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Create DataFrames
    X_train = pd.DataFrame(all_images)
    y_train = pd.DataFrame(all_labels, columns=['label'])

    # limit dataset size
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=percentage_of_rows_to_drop, random_state=42, stratify=y_train)

    # Calculate the number of features and classes
    num_features = X_train.shape[1]
    num_classes = y_train['label'].nunique()

    return X_train, y_train, num_features, num_classes


def get_cifar10_dataset():
    datasets.CIFAR10.url="https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"

    # Define transformations (convert to tensor and normalize the images)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images for RGB channels
    ])


    # Load the CIFAR-10 dataset
    cifar10_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Convert to features (X) and targets (y)
    X = np.array([np.transpose(img.numpy(), (1, 2, 0)).flatten() for img, _ in cifar10_dataset])  # Flatten RGB images
    y = np.array([label for _, label in cifar10_dataset])

    # Convert to Pandas DataFrame
    X_df = pd.DataFrame(X, columns=[f'pixel_{i}' for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['label'])

    num_features = X_df.shape[1]  # Number of features (pixels)
    num_classes = y_df['label'].nunique()  # Number of unique classes (0-9)

    return X_df, y_df, num_features, num_classes


def get_purchase_dataset(dataset_path='/content/dataset_purchase.csv', keep_rows=50_000):
    # Load the dataset, restricting to first 50,000 rows
    purchase_data = pd.read_csv(dataset_path).head(keep_rows)

    # Extract features (X) and adjust target labels (y)
    X = purchase_data.drop(columns=purchase_data.columns[0], axis=1)
    y = purchase_data.iloc[:, 0] - 1  # Adjust class labels from 1-100 to 0-99

    num_features = X.shape[1]  # Number of features (columns in X)
    num_classes = y.nunique()  # Number of unique classes in y

    return X, y, num_features, num_classes
