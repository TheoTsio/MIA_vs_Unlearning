import joblib
import os
import random
import numpy as np
import pandas as pd
import sys
import os

import sys

image_size = 128
image_size = 32

current_dir = os.getcwd()
sftc_unlearn_path = current_dir
print(sftc_unlearn_path)
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

from Unlearning_Functions_Tracking_MIA.scrub_track_mia import scrub_tracking_MIA
from Unlearning_Functions_Tracking_MIA.sftc_track_mia import sftc_unlearn_tracking_MIA
from Unlearning_Functions_Tracking_MIA.neg_grad_track_mia import neg_grad_tracking_MIA

from preprocess_data import *
from Target_Models.target_model_1a import *
from Target_Models.target_model_2a import *
from Target_Models.target_model_2b import *
from Target_Models.target_model_2c import *
from Target_Models.target_model_1c import *
from Target_Models.target_model_3a import *
from Target_Models.target_model_3b import *
from Target_Models.target_model_3c import *
from Target_Models.target_model_3d import *
from utils.data_split import *
from MIA_Threshold_Attacks import AttackSuite, load_or_calibrate_thresholds


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

'''
Default Functions
'''

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

def create_membership_dataframe(
    model: nn.Module,
    member_data: pd.DataFrame,
    non_member_data: pd.DataFrame
) -> pd.DataFrame:
    """Create a DataFrame with model outputs and membership status. Also apply softmax on the outputs"""

    model.eval()
    model.to(device)
    '''For traditional ML models Only Fc'''
    member_tensor = torch.tensor(member_data.values, dtype=torch.float32).to(device)
    non_member_tensor = torch.tensor(non_member_data.values, dtype=torch.float32).to(device)
    
    '''For CNNs'''
    # member_tensor = torch.tensor(member_data.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size).to(device)
    # non_member_tensor = torch.tensor(non_member_data.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size).to(device)

    with torch.no_grad():
        member_outputs = F.softmax(model(member_tensor), dim=1)
        non_member_outputs = F.softmax(model(non_member_tensor), dim=1)

    member_df = pd.DataFrame(member_outputs.cpu().detach().numpy())
    member_df['membership'] = True

    non_member_df = pd.DataFrame(non_member_outputs.cpu().detach().numpy())
    non_member_df['membership'] = False

    membership_df = pd.concat([member_df, non_member_df], ignore_index=True)

    return membership_df


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

if __name__ == "__main__":
    # load pre-trained models

    # Experiment configuration: change these together when switching experiments
    experiment_id = '1d'
    target_architecture = TargetModel_3d  # architecture of the target/shadow models for this experiment
    input_shape = None                    # (3, image_size, image_size) for CNNs, None for fully connected models

    target_model = torch.load(f'models/{experiment_id}_dataset_target_model.pth')
    attack_model = joblib.load(f"models/{experiment_id}_dataset_attack_model.jolib")


    """
    Data preprocessing
    """
    # load the dataset

    # X, y, num_features, num_classes = get_cifar10_dataset()
    # X, y, num_features, num_classes = get_purchase_dataset(dataset_path='data/dataset_purchase.csv', keep_rows=40_000)
    # X, y, num_features, num_classes = get_MUFAC_dataset("data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv", "data/custom_korean_family_dataset_resolution_128/train_images", percentage_of_rows_to_drop = 0.4)
    X, y, num_features, num_classes = get_texas_100_dataset(path='data/texas100.npz', limit_rows=40_000)
    print(X.shape)

    input_size = num_features 
    output_size = num_classes

    """
    Split data into Target training data, Shadow training data and data as Non Member to test the Attack model
    """

    X_target, X_shadow, y_target, y_shadow, X_test_MIA, y_test_MIA = get_target_shadow_split(X, y)
    print(X.shape)
    print(X_target.shape)

    X_train, X_test, y_train, y_test = get_train_test_split(X_target, y_target)

    # Keep this to create a test dataset for the attack model
    X_target_train_set, y_target_train_set = X_train, y_train

    """ 
    Run MIA during each unlearning epoch
    """

    unlearned_model = copy.deepcopy(target_model)
    teacher_model = copy.deepcopy(target_model)
    dummy_model = RandomDistributionGenerator(dist='normal', dimensions=num_classes)

    X_retain, X_forget, y_retain, y_forget = get_retain_forget_split(
    X_target_train_set, y_target_train_set, forget_class=None)  # Seven in forget class for CIFAR-10, or None for full 20% forget split for Purchase/Texas/MuFac


    '''Classic For Fully Connected Models'''
    X_forget_tensor = torch.tensor(X_forget.values, dtype=torch.float32)
    y_forget_tensor = torch.tensor(y_forget.values, dtype=torch.long).squeeze()

    X_retain_tensor = torch.tensor(X_retain.values, dtype=torch.float32)
    y_retain_tensor = torch.tensor(y_retain.values, dtype=torch.long).squeeze()

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()
    
    # Reshape for CNN (CIFAR-10: 3 channels, 32x32 pixels)
    # X_forget_tensor = torch.tensor(X_forget.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
    # y_forget_tensor = torch.tensor(y_forget.values, dtype=torch.long).squeeze()

    # X_retain_tensor = torch.tensor(X_retain.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
    # y_retain_tensor = torch.tensor(y_retain.values, dtype=torch.long).squeeze()

    # X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
    # y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    # Merge Loader only for sftc
    merged_loader = create_merged_loader(X_retain_tensor, y_retain_tensor, X_forget_tensor, y_forget_tensor, batch_size=32, shuffle=True)

    # Hyperparameters for Unlearning
    learning_rate = 0.0001
    epochs = 30
    batch_size = 32

    # Create DataLoader for forget set and test set
    forget_data = TensorDataset(X_forget_tensor, y_forget_tensor)
    forget_loader = DataLoader(forget_data, batch_size=batch_size, shuffle=True)

    retain_data = TensorDataset(X_retain_tensor, y_retain_tensor)
    retain_loader = DataLoader(retain_data, batch_size=batch_size, shuffle=True)

    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=learning_rate, momentum=0.9)

    sample_size = min(len(X_test_MIA), len(X_forget), len(X_retain))

    # sample_size = min(len(X_test_MIA), len(X_retain))
    retain_member_df = X_retain.sample(n=sample_size, replace=False)

    # sample_size = min(len(X_test_MIA), len(X_forget))
    forget_member_df = X_forget.sample(n=sample_size, replace=False)

    train_non_member_data = X_test_MIA.sample(n=sample_size, replace=False)

    num_epochs_MIA = 50
    num_shadow_models = 5

    """
    Additional MIAs: loss / confidence / modified-entropy thresholds + label-only gap attack.
    Thresholds are calibrated on shadow models trained on X_shadow with the same recipe as the
    target model (SGD, lr=0.01, 100 epochs, batch 32) and cached per experiment.
    """
    mia_thresholds = load_or_calibrate_thresholds(
        f'models/{experiment_id}_mia_thresholds.json',
        target_architecture, X_shadow, y_shadow, num_features, num_classes,
        num_shadow_models=1, learning_rate=0.01, num_epochs=100, batch_size=32,
        input_shape=input_shape, device=device)

    attack_suite = AttackSuite(
        retain_X=retain_member_df, retain_y=y_retain.loc[retain_member_df.index],
        forget_X=forget_member_df, forget_y=y_forget.loc[forget_member_df.index],
        non_member_X=train_non_member_data, non_member_y=y_test_MIA.loc[train_non_member_data.index],
        thresholds=mia_thresholds, attack_model=attack_model, input_shape=input_shape, device=device)

    # Reference point before unlearning
    original_attack_stats = attack_suite.evaluate(target_model)

    # Unlearn the target neural network
    # mia_stats, accs, losses = scrub_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model,  unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=device, teacher_model=teacher_model, X_shadow=X_shadow, y_shadow=y_shadow, num_shadow_models=num_shadow_models, num_features=num_features, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, num_epochs_MIA=num_epochs_MIA, shadow_model_architecture=target_architecture, attack_suite=attack_suite)
    mia_stats, accs, losses = neg_grad_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model, unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=device, X_shadow=X_shadow, y_shadow=y_shadow, num_shadow_models=num_shadow_models, num_features=num_features, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, num_epochs_MIA=num_epochs_MIA, shadow_model_architecture=target_architecture, attack_suite=attack_suite)
    # mia_stats, accs, losses = sftc_unlearn_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model, unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=device, merged_loader=merged_loader, teacher_model=teacher_model, dummy_model=dummy_model,  X_shadow=X_shadow, y_shadow=y_shadow, num_shadow_models=num_shadow_models, num_features=num_features, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, num_epochs_MIA=num_epochs_MIA, shadow_model_architecture=target_architecture, attack_suite=attack_suite)


    """
    =========================================
    Retrain-from-scratch baseline MIA (gold standard reference line)
    =========================================
    """
    retrain_baseline_model = torch.load(f'models/{experiment_id}_dataset_retrain_baseline_model.pth')

    # Forget set vs unseen
    df_forget_baseline = create_membership_dataframe(retrain_baseline_model, forget_member_df, train_non_member_data)
    baseline_forget_acc, baseline_forget_f1, baseline_forget_prec, baseline_forget_rec = \
        evaluate_attack_model_get_stats(df_forget_baseline, attack_model)

    # Retain set vs unseen
    df_retain_baseline = create_membership_dataframe(retrain_baseline_model, retain_member_df, train_non_member_data)
    baseline_retain_acc, baseline_retain_f1, baseline_retain_prec, baseline_retain_rec = \
        evaluate_attack_model_get_stats(df_retain_baseline, attack_model)

    print(f"\n[Retrain Baseline] MIA Forget Acc: {baseline_forget_acc:.4f}, MIA Retain Acc: {baseline_retain_acc:.4f}")
    print(f"[Retrain Baseline] MIA Forget F1: {baseline_forget_f1:.4f}, MIA Retain F1: {baseline_retain_f1:.4f}")
    
    '''
    Plots
    '''

    # Prepare the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each accuracy curve
    ax.plot(accs['train'], label='Train Accuracies', linestyle='-.', color='blue')
    ax.plot(accs['val'], label='Val Accuracies', linestyle='--', color='green')
    ax.plot(accs['test'], label='Test Accuracies', linestyle='-.', color='orange')
    ax.plot(accs['forget'], label='Forget Accuracies', linestyle='--', color='red')

    # Add grid and minor ticks
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)

    # Set labels, title, and legend
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')

    ax.legend(loc='lower left')

    # Save the plot
    plt.savefig('curves/accuracy_curves.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mia_stats['forget']['acc'], label='MIA Forget Accuracies', linestyle='-')
    ax.plot(mia_stats['retain']['acc'], label='MIA Retain Accuracies', linestyle='-')
    
    # Retrain-from-scratch baseline reference lines
    ax.axhline(y=baseline_forget_acc, color='C0', linestyle='--', linewidth=2,
            label='Retrain Baseline: Forget')
    ax.axhline(y=baseline_retain_acc, color='C1', linestyle='--', linewidth=2,
            label='Retrain Baseline: Retain')

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('MIA Accuracy as the Target Model Unlearns')

    ax.legend(loc='lower left')

    # Save the plot
    plt.savefig('curves/mia_accuracy.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(mia_stats['forget']['f1'], label='Forget F1 Scores', linestyle='-')
    ax.plot(mia_stats['retain']['f1'], label='Retain F1 Scores', linestyle='-')
    
    
    ax.axhline(y=baseline_forget_f1, color='C0', linestyle='--', linewidth=2, label='Retrain Baseline: Forget F1')
    ax.axhline(y=baseline_retain_f1, color='C1', linestyle='--', linewidth=2, label='Retrain Baseline: Retain F1')

    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('F1 Score')
    ax.set_title('MIA F1 Score as the Target Model Unlearns')
    ax.legend(loc='lower left')

    # Save the plot
    plt.savefig('curves/mia_f1_score.pdf')
    plt.close(fig)


    # Plot for Precision
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mia_stats['forget']['precision'], label='Forget Precision Scores', linestyle='-')
    ax.plot(mia_stats['retain']['precision'], label='Retain Precision Scores', linestyle='--')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Precision')
    ax.set_title('MIA Precision as the Target Model Unlearns')
    ax.legend(loc='lower left')

    # Save the Precision plot
    plt.savefig('curves/mia_precision.pdf')
    plt.close(fig)

    # Plot for Recall
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(mia_stats['forget']['recall'], label='Forget Recall Scores', linestyle='-')
    ax.plot(mia_stats['retain']['recall'], label='Retain Recall Scores', linestyle='--')
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', linewidth=0.25)
    ax.set_xlabel('Number of Epochs')
    ax.set_ylabel('Recall')
    ax.set_title('MIA Recall as the Target Model Unlearns')
    ax.legend(loc='lower left')

    # Save the Recall plot
    plt.savefig('curves/mia_recall.pdf')
    plt.close(fig)

    epochs = list(range(0, len(mia_stats['forget']['acc'])))
    forget_acc = mia_stats['forget']['acc']
    retain_acc = mia_stats['retain']['acc']
    forget_f1 = mia_stats['forget']['f1']
    retain_f1 = mia_stats['retain']['f1']
    forget_precision = mia_stats['forget']['precision']
    retain_precision = mia_stats['retain']['precision']
    forget_recall = mia_stats['forget']['recall']
    retain_recall = mia_stats['retain']['recall']

    # Create a DataFrame
    df = pd.DataFrame({
        'Epoch': epochs,
        'MIA Forget Acc': forget_acc,
        'MIA Retain Acc': retain_acc,
        'MIA Forget F1': forget_f1,
        'MIA Retain F1': retain_f1,
        'MIA Forget Precision': forget_precision,
        'MIA Retain Precision': retain_precision,
        'MIA Forget Recall': forget_recall,
        'MIA Retain Recall': retain_recall,
        'Train Acc': accs['train'],
        'Val Acc': accs['val'],
        'Test Acc': accs['test'],
        'Forget Acc': accs['forget'],
        # Constant reference columns (same value every row) — easy to pull for table/plot later
        'Retrain Baseline MIA Forget Acc': baseline_forget_acc,
        'Retrain Baseline MIA Retain Acc': baseline_retain_acc,
        'Retrain Baseline MIA Forget F1': baseline_forget_f1,
        'Retrain Baseline MIA Retain F1': baseline_retain_f1,
    })

    # Define the CSV file name
    csv_file = 'curves/unlearn_algo_mia_stats.csv'

    # Write to CSV
    df.to_csv(csv_file, index=False)

    """
    =========================================
    Comparison across MIA strategies (shadow RF, loss, confidence, modified entropy, label-only)
    =========================================
    """
    retrain_attack_stats = attack_suite.evaluate(retrain_baseline_model)
    attack_references = {'Retrain Baseline': retrain_attack_stats, 'Original': original_attack_stats}

    attack_suite.history_dataframe(attack_references).to_csv('curves/mia_attacks_comparison.csv', index=False)

    attack_summary = attack_suite.summary_table(attack_references)
    attack_summary.to_csv('curves/mia_attacks_summary.csv', index=False)
    print("\nMIA strategies — final epoch vs references:")
    print(attack_summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    attack_suite.plot('acc', 'curves/mia_attacks_accuracy.pdf', attack_references, ylabel='MIA Accuracy')
    attack_suite.plot('auc', 'curves/mia_attacks_auc.pdf', attack_references, ylabel='MIA AUC')
    attack_suite.plot('tpr_at_1fpr', 'curves/mia_attacks_tpr_at_1fpr.pdf', attack_references, ylabel='TPR @ 1% FPR')

    """
    =========================================
    Shadow-model ablation: K in {1, 3, 5, 10} shadow models, 3 seeds each (slow: 57 shadow models)
    =========================================
    """
    RUN_SHADOW_ABLATION = False
    if RUN_SHADOW_ABLATION:
        from MIA_Functions import train_shadow_models_and_attack_model
        results = []
        for K in [1, 3, 5, 10]:
            for seed in [0, 1, 2]:
                np.random.seed(seed); torch.manual_seed(seed)
                attack = train_shadow_models_and_attack_model(
                    target_architecture, X_shadow, y_shadow, K, num_features, num_classes,
                    batch_size=32, learning_rate=0.01, num_epochs=100)
                row = {'K': K, 'seed': seed}
                for name, model in [('target', target_model), ('retrain', retrain_baseline_model)]:
                    for split, members in [('forget', forget_member_df), ('retain', retain_member_df)]:
                        df_ablation = create_membership_dataframe(model, members, train_non_member_data)
                        row[f'{name}_{split}'] = evaluate_attack_model_get_stats(df_ablation, attack)[0]
                results.append(row)
        ablation_df = pd.DataFrame(results)
        ablation_df.to_csv(f'curves/{experiment_id}_shadow_ablation.csv', index=False)
        print(ablation_df.groupby('K').agg(['mean', 'std']))
