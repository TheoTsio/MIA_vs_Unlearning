import joblib
import os
import random
import numpy as np
import pandas as pd
import sys
import os

import sys

image_size = 128

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
from Target_Models.target_model_1c import *
from Target_Models.target_model_2a import *


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


def create_membership_dataframe(
    model: nn.Module,
    member_data: pd.DataFrame,
    non_member_data: pd.DataFrame
) -> pd.DataFrame:
    """Create a DataFrame with model outputs and membership status. Also apply softmax on the outputs"""

    model.eval()
    model.to(device)
    '''For traditional ML models Only Fc'''
    # member_tensor = torch.tensor(member_data.values, dtype=torch.float32).to(device)
    # non_member_tensor = torch.tensor(non_member_data.values, dtype=torch.float32).to(device)
    
    '''For CNNs'''
    member_tensor = torch.tensor(member_data.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size).to(device)
    non_member_tensor = torch.tensor(non_member_data.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size).to(device)

    with torch.no_grad():
        member_outputs = F.softmax(model(member_tensor), dim=1)
        non_member_outputs = F.softmax(model(non_member_tensor), dim=1)

    member_df = pd.DataFrame(member_outputs.cpu().detach().numpy())
    member_df['membership'] = True

    non_member_df = pd.DataFrame(non_member_outputs.cpu().detach().numpy())
    non_member_df['membership'] = False

    membership_df = pd.concat([member_df, non_member_df], ignore_index=True)

    return membership_df



if __name__ == "__main__":
    # load pre-trained models

    target_model = torch.load('models/2b_MuFac_target_model.pth')
    attack_model = joblib.load("models/2b_MuFac_attack_model.jolib")
    
    
    """
    Data preprocessing
    """
    # load the dataset
    set_random_seed(42)

    # X, y, num_features, num_classes = get_cifar10_dataset()
    # X, y, num_features, num_classes = get_purchase_dataset(dataset_path='data/dataset_purchase.csv', keep_rows=40_000)
    X, y, num_features, num_classes = get_MUFAC_dataset("data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv", "data/custom_korean_family_dataset_resolution_128/train_images", percentage_of_rows_to_drop = 0.4)
    # X, y, num_features, num_classes = get_texas_100_dataset(path='data/texas100.npz', limit_rows=40_000)
    print(X.shape)

    input_size = num_features 
    output_size = num_classes

    """
    Split data into Target training data, Shadow training data and data as Non Member to test the Attack model
    """

    # Split the data into 80% temp and 20% test
    X_temp, X_test_MIA, y_temp, y_test_MIA = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Now, split the remaining 80% data into 40% target train and 40% shadow train
    X_target, X_shadow, y_target, y_shadow = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(X.shape)
    print(X_target.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_target, y_target, test_size=0.2, random_state=42)

    # Keep this to create a test dataset for the attack model
    X_target_train_set = X_train
    y_target_train_set = y_train

    """ 
    Run MIA during each unlearning epoch
    """

    unlearned_model = copy.deepcopy(target_model)
    teacher_model = copy.deepcopy(target_model)
    dummy_model = RandomDistributionGenerator(dist='normal', dimensions=num_classes)

    # Define the forget set to forget class 7
    X_retain, X_forget, y_retain, y_forget = train_test_split(X_target_train_set, y_target_train_set, test_size=0.2, random_state=42, stratify=y_target_train_set)

    # # on CIFAR10 data specifically unlearn only class 7 
    # y_forget = y_forget[y_forget['label'] == 7]
    # X_forget = X_forget.loc[y_forget.index]


    '''Classic For Fully Connected Models'''
    # X_forget_tensor = torch.tensor(X_forget.values, dtype=torch.float32)
    # y_forget_tensor = torch.tensor(y_forget.values, dtype=torch.long).squeeze()

    # X_retain_tensor = torch.tensor(X_retain.values, dtype=torch.float32)
    # y_retain_tensor = torch.tensor(y_retain.values, dtype=torch.long).squeeze()

    # X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    # y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()
    
    # Reshape for CNN (CIFAR-10: 3 channels, 32x32 pixels)
    X_forget_tensor = torch.tensor(X_forget.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
    y_forget_tensor = torch.tensor(y_forget.values, dtype=torch.long).squeeze()

    X_retain_tensor = torch.tensor(X_retain.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
    y_retain_tensor = torch.tensor(y_retain.values, dtype=torch.long).squeeze()

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    # Merge Loader only for sftc
    merged_loader = create_merged_loader(X_retain_tensor, y_retain_tensor, X_forget_tensor, y_forget_tensor, batch_size=32, shuffle=True)

    # Hyperparameters for Unlearning
    learning_rate = 0.000065
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
    # Unlearn the target neural network
    mia_stats, accs, losses = scrub_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model,  unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=device, teacher_model=teacher_model)
    # mia_stats, accs, losses = neg_grad_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model, unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=device, X_shadow=X_shadow, y_shadow=y_shadow, num_shadow_models=num_shadow_models, num_features=num_features, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, num_epochs_MIA=num_epochs_MIA, shadow_model_architecture=TargetModel_1a)
    # mia_stats, accs, losses = sftc_unlearn_tracking_MIA(retain_member_df, forget_member_df, train_non_member_data, attack_model, unlearned_model, retain_loader, test_loader, test_loader, forget_loader, optimizer=optimizer, scheduler=None, criterion=criterion, epochs=epochs, device=device, merged_loader=merged_loader, teacher_model=teacher_model, dummy_model=dummy_model,  X_shadow=X_shadow, y_shadow=y_shadow, num_shadow_models=num_shadow_models, num_features=num_features, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, num_epochs_MIA=num_epochs_MIA, shadow_model_architecture=TargetModel_1a)

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
    ax.plot(mia_stats['retain']['acc'], label='MIA Retain Accuracies', linestyle='--')
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
    ax.plot(mia_stats['retain']['f1'], label='Retain F1 Scores', linestyle='--')
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
    })

    # Define the CSV file name
    csv_file = 'curves/unlearn_algo_mia_stats.csv'

    # Write to CSV
    df.to_csv(csv_file, index=False)
