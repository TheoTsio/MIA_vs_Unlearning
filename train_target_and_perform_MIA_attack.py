
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
from Target_Models.target_model_1a import *
from Target_Models.target_model_2a import *
from Target_Models.target_model_2b import *
from Target_Models.target_model_2c import *
from Target_Models.target_model_1c import *
import gc

from preprocess_data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
image_size = 128
def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model_architecture = TargetModel_2c


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    patience: int,
    early_stopping: bool = True
) -> None:
    best_validation_loss = float('inf')
    patience_counter = 0
    model.to(device)  # Force model to GPU

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        total_training_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
                    inputs, labels = inputs.to(device), labels.to(device)
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
def evaluate_model_performance(model, loss_function, X_tensor, y_tensor, eval_batch_size=256):
    """
    Evaluate model performance using batched processing to avoid OOM.
    Results are identical to processing all data at once.
    """
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    total_samples = 0
    
    with torch.no_grad():
        # Process in batches
        for i in range(0, len(X_tensor), eval_batch_size):
            '''For traditional ML models Only Fc'''
            batch_X = X_tensor[i:i+eval_batch_size].to(device)
            
            '''For CNNs'''
            # batch_X = X_tensor[i:i+eval_batch_size].reshape(-1, 3, image_size, image_size).to(device)
            
            batch_y = y_tensor[i:i+eval_batch_size].to(device)
            
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            
            # Accumulate loss weighted by batch size
            batch_size = batch_X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Store predictions and labels
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics exactly as before
    avg_loss = total_loss / total_samples
    accuracy = (all_predictions == all_labels).mean()
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall



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


def evaluate_attack_model(dataset, model, plot_title='Chart', plot_choice='roc'):
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

    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True) * 100

    # ROC Curve calculation
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot based on the chosen type
    if plot_choice == 'roc':
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(plot_title)
        plt.legend(loc='lower right')
        plt.show()

    elif plot_choice == 'confusion':
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (Percentages)')
        plt.show()

    # Print metrics
    print(f"Confusion Matrix (Percentages):\n{conf_matrix_percent}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")


"""
Data preprocessing
"""
# load the dataset
set_random_seed(42)

# X, y, num_features, num_classes = get_mnist_dataset()
# X, y, num_features, num_classes = get_cifar10_dataset()
# X, y, num_features, num_classes = get_adults_dataset()
# X, y, num_features, num_classes = get_purchase_dataset(dataset_path='data/dataset_purchase.csv', keep_rows=40_000)
# X, y, num_features, num_classes = get_MUFAC_dataset("data/custom_korean_family_dataset_resolution_128/custom_train_dataset.csv", "data/custom_korean_family_dataset_resolution_128/train_images", percentage_of_rows_to_drop = 0.4)
X, y, num_features, num_classes = get_texas_100_dataset(path='data/texas100.npz', limit_rows=40_000)


"""
Split data into Target training data, Shadow training data and data as Non Member to test the Attack model
"""

# Split the data into 80% temp and 20% test
X_temp, X_test_MIA, y_temp, y_test_MIA = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Now, split the remaining 80% data into 40% target train and 40% shadow train
X_target, X_shadow, y_target, y_shadow = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print(X.shape)
print(X_target.shape)

"""
Train Target Model
"""
set_random_seed(42)

features = X_target
labels = y_target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Keep this to create a test dataset for the attack model
X_target_train_set = X_train
y_target_train_set = y_train

'''For traditional ML models Only Fc'''
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

''''For CNNs'''
# X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
# X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
# y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()


# Hyperparameters for images
learning_rate = 0.001
num_epochs = 300
batch_size = 32
early_stopping_patience = 3
enable_early_stopping = False 

# for tabular data
# learning_rate = 0.01
# num_epochs = 100
# batch_size = 32
# early_stopping_patience = 3
# enable_early_stopping = False

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = num_features
output_size = num_classes
print("NN input size", input_size)
print("NN output size", output_size)

target_model = model_architecture(input_size=input_size, output_size=output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(target_model.parameters(), lr=learning_rate, momentum=0.9)

train_model(target_model, train_loader, test_loader, criterion, optimizer, num_epochs, early_stopping_patience, enable_early_stopping)

train_loss, train_accuracy, train_precision, train_recall = evaluate_model_performance( target_model, criterion, X_train_tensor, y_train_tensor)
test_loss, test_accuracy, test_precision, test_recall = evaluate_model_performance( target_model, criterion, X_test_tensor, y_test_tensor)

print(f'\nTraining Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
print(f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}')
# Save the models 
torch.save(target_model, 'models/dataset_target_model.pth')
del target_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
# target_model = torch.load('models/dataset_target_model.pth' )
"""
Train shadow models
"""
# Hyperparameters for training the shadow model
# learning_rate = 0.001
# num_epochs = 10
# batch_size = 32
# early_stopping_patience = 3
# enable_early_stopping = False
num_shadow_models = 5

def train_shadow_models_and_attack_model(target_model, X_shadow, y_shadow, num_shadow_models, num_features, num_classes, batch_size, learning_rate, num_epochs):
    # DataFrame to store outputs from shadow models
    shadow_model_outputs_df = pd.DataFrame()

    # Prepare features and labels
    features = X_shadow
    labels = y_shadow

    for model_index in tqdm(range(num_shadow_models), desc='Training Shadow Models'):

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5)

        '''For traditional ML models Only Fc'''
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

        ''''For CNNs'''
        # X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
        # y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze()
        # X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).reshape(-1, 3, image_size, image_size)
        # y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()


        # Create DataLoader instances for training and testing
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        input_size = num_features
        output_size = num_classes

        shadow_model = model_architecture(input_size=input_size, output_size=output_size)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(shadow_model.parameters(), lr=learning_rate, momentum=0.9)

        # Train the shadow model
        train_model(shadow_model, train_loader, test_loader, criterion, optimizer, num_epochs, early_stopping_patience, enable_early_stopping)

        print(f'\nGenerating data from shadow model ({model_index + 1}) outputs\n')

        # Sample equal number of members and non-members
        sample_size = min(len(X_train), len(X_test))
        X_member_sample = X_train.sample(n=sample_size, replace=False)
        X_non_member_sample = X_test.sample(n=sample_size, replace=False)
        # Generate dataset with membership status from current shadow model
        current_shadow_model_outputs_df = create_membership_dataframe(shadow_model, X_member_sample, X_non_member_sample)

        # Append to the results DataFrame
        shadow_model_outputs_df = pd.concat([shadow_model_outputs_df, current_shadow_model_outputs_df], ignore_index=True)
        
        print(f"Finished Model {model_index + 1}. Clearing memory...")
        del shadow_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print('\nResulting shadow model outputs dataset shape:', shadow_model_outputs_df.shape)

    """
    Train the attack model using the shadow models outputs
    """

    attack_model = train_attack_model_on_output_data(shadow_model_outputs_df, 42, test_model_capabilities=True)
    return attack_model

""" ##### MIA ##### """

"""
Based on the Target Model predictions, use the Attack Model to predict membership status for each data point
"""
attack_model = train_shadow_models_and_attack_model(model_architecture, X_shadow, y_shadow, num_shadow_models, num_features, num_classes, batch_size, learning_rate, num_epochs)
# Sample an equal number of non-member data
sample_size = min(len(X_test_MIA), len(X_target_train_set))
train_member_data = X_target_train_set.sample(n=sample_size, replace=False)
train_non_member_data = X_test_MIA.sample(n=sample_size, replace=False)
# Create a dataset by QUERYING the TARGET model on member and non-member data

target_model = torch.load('models/dataset_target_model.pth')

dataset_from_target_model_outputs = create_membership_dataframe(target_model, train_member_data, train_non_member_data)
# Use the ATTACK model to predict membership status
evaluate_attack_model(dataset_from_target_model_outputs, attack_model, 'ROC Curve - Test set', plot_choice='roc')

# Get a typical input tensor from the training data loader


joblib.dump(attack_model , 'models/dataset_attack_model.jolib')