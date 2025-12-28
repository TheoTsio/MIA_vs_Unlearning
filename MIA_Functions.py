import os
from pyexpat import model
import sys
from pathlib import Path

import joblib
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets, transforms
from PIL import Image
from Target_Models.target_model_1a import TargetModel_1a

from preprocess_data import *
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Tuple, Union, List, Dict, Callable
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import resnet18, efficientnet_b0
from Target_Models.target_model_1a import TargetModel_1a


from utils.preprocessing import pre_process_cifar10, to_torch_loader, pre_process_korean_family, pre_process_fer

from utils.unlearning_utils import UnLearningData, RandomDistributionGenerator, CustomPseudoLabelDataset

from utils.unlearning_alg.simple_finetuning import fine_tune
from utils.unlearning_alg.cf_eu_k import catastrophic_forgetting_k
from utils.unlearning_alg.neg_grad import neg_grad
from utils.unlearning_alg.scrub import scrub
from utils.unlearning_alg.bad_teaching import bad_teaching
from utils.unlearning_alg.sftc_unlearn import sftc_unlearn

def create_membership_dataframe(
    model: nn.Module,
    member_data: pd.DataFrame,
    non_member_data: pd.DataFrame
) -> pd.DataFrame:
    """Create a DataFrame with model outputs and membership status. Also apply softmax on the outputs"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.eval()
    model = model.to(device)
    member_tensor = torch.tensor(member_data.values, dtype=torch.float32).to(device)
    non_member_tensor = torch.tensor(non_member_data.values, dtype=torch.float32).to(device)

    with torch.no_grad():
        member_outputs = F.softmax(model(member_tensor), dim=1)
        non_member_outputs = F.softmax(model(non_member_tensor), dim=1)

    member_df = pd.DataFrame(member_outputs.cpu().detach().numpy())
    member_df['membership'] = True

    non_member_df = pd.DataFrame(non_member_outputs.cpu().detach().numpy())
    non_member_df['membership'] = False

    membership_df = pd.concat([member_df, non_member_df], ignore_index=True)

    return membership_df

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    patience: int = 5,
    early_stopping: bool = True
) -> None:
    best_validation_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        total_training_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

        average_training_loss = total_training_loss / len(train_loader)

        if early_stopping:
            model.eval()
            total_validation_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_validation_loss += loss.item()
            average_validation_loss = total_validation_loss / \
                len(val_loader)

            if average_validation_loss < best_validation_loss:
                best_validation_loss = average_validation_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\nEarly stopping triggered.")
                    break
            print(f'\rEpoch {epoch + 1}/{num_epochs} | Train Loss: {average_training_loss:.4f}, '
                  f'Validation Loss: {average_validation_loss:.4f}', end='', flush=True)
        else:
            print(f'\rEpoch {epoch + 1}/{num_epochs} | Train Loss: {average_training_loss:.4f}',
                  end='', flush=True)

def evaluate_attack_model_get_stats(dataset, model):
    """
    Evaluates and visualizes the performance of the attack model.

    Parameters:
        dataset (pd.DataFrame): DataFrame containing features and 'membership' column.
        attack_model (sklearn.base.ClassifierMixin): Trained attack model.
        plot_choice (str): Type of plot to display. 'roc' for ROC curve, 'confusion' for confusion matrix.

    Returns:
        dict: Dictionary containing precision, recall, accuracy, F1 score, and confusion matrix.
    """
    # Extract features and labels from the dataset
    features = dataset.drop(columns=['membership'])
    labels = dataset['membership']

    # Predict probabilities using the trained attack model
    predictions = model.predict(features)
    # Probabilities for the positive class
    probabilities = model.predict_proba(features)[:, 1]

    # Compute metrics
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    accuracy = model.score(features, labels)
    f1 = f1_score(labels, predictions)

    return accuracy, f1, precision, recall

def train_attack_model_on_output_data(
    shadow_model_outputs_df: pd.DataFrame,
    random_seed: int = 42,
    test_model_capabilities: bool = False,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Train an attack model on the output data of a shadow model.

    Args:
        shadow_model_outputs_df (pd.DataFrame): DataFrame containing the shadow model outputs.
        random_seed (int): Random seed for reproducibility.
        test_model_capabilities (bool): Flag to indicate if the model should be evaluated.

    Returns:
        RandomForestClassifier: The trained attack model.
    """
    # Shuffle the DataFrame and reset index
    training_df = shuffle(shadow_model_outputs_df, random_state=random_seed).reset_index(drop=True)

    # Prepare features and labels
    features = training_df.drop(columns=['membership'])
    labels = training_df['membership']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_seed)

    verbose_val = 2 if verbose else 0

    # Initialize and train the attack model
    attack_model = RandomForestClassifier(n_estimators=100, verbose=verbose_val, n_jobs=-1, random_state=random_seed)
    attack_model.fit(X_train, y_train)

    if test_model_capabilities:
        # Predict and evaluate the model
        y_pred = attack_model.predict(X_test)
        test_precision = precision_score(y_test, y_pred)
        test_accuracy = attack_model.score(X_test, y_test)
        print('Attack Model Evaluation:')
        print(f"Test Precision: {test_precision}")
        print(f"Test Accuracy: {test_accuracy}")

    return attack_model


def train_shadow_models_and_attack_model(shadow_model_architecture, X_shadow, y_shadow, num_shadow_models, num_features, num_classes, batch_size, learning_rate, num_epochs):
    # DataFrame to store outputs from shadow models
    shadow_model_outputs_df = pd.DataFrame()

    # Prepare features and labels
    features = X_shadow
    labels = y_shadow

    for model_index in tqdm(range(num_shadow_models), desc='Training Shadow Models'):
        print(f'\nTraining shadow model ({model_index + 1})\n')

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

        # Create DataLoader instances for training and testing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        input_size = num_features
        output_size = num_classes
        print(f'Input size: {input_size}, Output size: {output_size}')
        shadow_model = shadow_model_architecture(input_size, output_size)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=learning_rate, momentum=0.9)

        # Train the shadow model
        train_model(shadow_model, train_loader, test_loader, criterion, optimizer, num_epochs, patience=5, early_stopping=True)

        print(f'\nGenerating data from shadow model ({model_index + 1}) outputs\n')

        # Sample equal number of members and non-members
        sample_size = min(len(X_train), len(X_test))
        X_member_sample = X_train.sample(n=sample_size, replace=False)
        X_non_member_sample = X_test.sample(n=sample_size, replace=False)

        # Generate dataset with membership status from current shadow model
        current_shadow_model_outputs_df = create_membership_dataframe(shadow_model, X_member_sample, X_non_member_sample)

        # Append to the results DataFrame
        shadow_model_outputs_df = pd.concat([shadow_model_outputs_df, current_shadow_model_outputs_df], ignore_index=True)

    print('\nResulting shadow model outputs dataset shape:', shadow_model_outputs_df.shape)

    """
    Train the attack model using the shadow models outputs
    """

    attack_model = train_attack_model_on_output_data(shadow_model_outputs_df, 42, test_model_capabilities=True)
    return attack_model
