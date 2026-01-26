import os
# Ensure cuBLAS determinism is enabled before any CUDA ops
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from sklearn.metrics import confusion_matrix, classification_report, f1_score, balanced_accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import json
from datetime import datetime
from LoadDataset import myDataset
from cnnmodels_fixed import create_model

# Add near your imports (optional helper)
SUPPORTED_MODELS = ['cnn1d', 'resnet1d', 'resnet50v2', 'mobilenet', 'efficientnetb0', 'densenet201']


# ----- Reproducibility helpers ----- #
def seed_everything(seed: int):
    """Seed Python, NumPy, PyTorch (CPU/GPU) and set deterministic flags."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _make_worker_init_fn(base_seed: int):
    def _seed_worker(worker_id: int):
        s = int(base_seed) + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _seed_worker


def make_dataloader(dataset, batch_size, shuffle, num_workers, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    worker_fn = _make_worker_init_fn(int(seed)) if num_workers and num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=worker_fn,
        persistent_workers=False,
    )


class SyntheticDataset(Dataset):
    """Dataset for synthetic data (no pH labels needed)"""
    def __init__(
        self,
        normal_folder,
        abnormal_folder,
        sequence_length=1000,
        max_samples_per_class=None,
        max_normal_samples=None,
        max_abnormal_samples=None,
    ):
        self.sequence_length = sequence_length
        self.min_bpm = 50
        self.max_bpm = 200

        # Load normal files
        normal_files = sorted(glob.glob(os.path.join(normal_folder, "*.csv")))
        normal_limit = max_normal_samples if max_normal_samples is not None else max_samples_per_class
        if normal_limit is not None:
            normal_files = normal_files[:normal_limit]

        # Load abnormal files
        abnormal_files = sorted(glob.glob(os.path.join(abnormal_folder, "*.csv")))
        abnormal_limit = max_abnormal_samples if max_abnormal_samples is not None else max_samples_per_class
        if abnormal_limit is not None:
            abnormal_files = abnormal_files[:abnormal_limit]

        self.file_list = normal_files + abnormal_files
        # Assign labels: abnormal=0, normal=1 to match evaluation conventions
        self.labels = [1] * len(normal_files) + [0] * len(abnormal_files)
        print(f"Synthetic Dataset: {len(normal_files)} normal, {len(abnormal_files)} abnormal")

    def _load_fhr_signal(self, file_path):
        """Load FHR signal from CSV file"""
        try:
            df = pd.read_csv(file_path, skiprows=2, header=None, usecols=[1], names=["FHR"])
            fhr_values = df["FHR"].values.astype(np.float32)
            return fhr_values
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return None

    def _normalize_signal(self, fhr_values):
        x = np.asarray(fhr_values, dtype=np.float32)
        # If values look like BPM, map to [-1, 1]; otherwise assume already normalized
        if np.nanmax(x) > 10:
            x = 2 * (x - self.min_bpm) / (self.max_bpm - self.min_bpm) - 1
        return np.clip(x, -1.0, 1.0)


    def _handle_sequence_length(self, fhr_values):
        """True cropping: Find valid samples, skip zeros entirely"""
        target_length = self.sequence_length

        if len(fhr_values) < target_length:
            padding_size = target_length - len(fhr_values)
            return np.pad(fhr_values, (0, padding_size), 'constant', constant_values=0)

        if len(fhr_values) == target_length:
            return fhr_values

        non_zero_indices = np.where(fhr_values != 0)[0]
        if len(non_zero_indices) == 0:
            return fhr_values[:target_length]

        start_idx = non_zero_indices[0]
        if start_idx + target_length <= len(fhr_values):
            selected_segment = fhr_values[start_idx:start_idx + target_length]
            return selected_segment
        else:
            return fhr_values[:target_length]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        fhr_values = self._load_fhr_signal(file_path)
        if fhr_values is None:
            data_tensor = torch.zeros((1, self.sequence_length), dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            return data_tensor, label_tensor

        fhr_values = self._handle_sequence_length(fhr_values)
        normalized_fhr = self._normalize_signal(fhr_values)
        data_tensor = torch.tensor(normalized_fhr, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return data_tensor, label_tensor


def create_stratified_kfold_splits(dataset, n_folds=4, test_per_class=25, seed=42):
    """
    Create stratified k-fold splits where each fold has a fixed test set size per class.

    Args:
        dataset: The dataset to split
        n_folds: Number of folds
        test_per_class: Number of samples per class in each test fold
        seed: Random seed

    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Separate indices by class
    normal_indices = []
    path_indices = []

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if int(label) == 1:  # Normal
            normal_indices.append(idx)
        else:  # Pathological (0)
            path_indices.append(idx)

    # Shuffle with seed
    random_gen = random.Random(seed)
    random_gen.shuffle(normal_indices)
    random_gen.shuffle(path_indices)

    print(f"\nTotal Normal samples: {len(normal_indices)}")
    print(f"Total Pathological samples: {len(path_indices)}")
    print(f"Creating {n_folds} folds with {test_per_class} samples per class in each test set")

    # Check if we have enough samples
    required_normal = n_folds * test_per_class
    required_path = n_folds * test_per_class

    if len(normal_indices) < required_normal:
        raise ValueError(f"Not enough normal samples. Need {required_normal}, have {len(normal_indices)}")
    if len(path_indices) < required_path:
        raise ValueError(f"Not enough pathological samples. Need {required_path}, have {len(path_indices)}")

    folds = []
    for fold_idx in range(n_folds):
        # Select test indices for this fold
        test_normal = normal_indices[fold_idx * test_per_class:(fold_idx + 1) * test_per_class]
        test_path = path_indices[fold_idx * test_per_class:(fold_idx + 1) * test_per_class]
        test_indices = test_normal + test_path

        # Train indices are everything except the test indices
        train_normal = [idx for i, idx in enumerate(normal_indices) 
                       if i < fold_idx * test_per_class or i >= (fold_idx + 1) * test_per_class]
        train_path = [idx for i, idx in enumerate(path_indices) 
                     if i < fold_idx * test_per_class or i >= (fold_idx + 1) * test_per_class]
        train_indices = train_normal + train_path

        # Shuffle train indices
        random_gen.shuffle(train_indices)
        random_gen.shuffle(test_indices)

        print(f"Fold {fold_idx + 1}: Train={len(train_indices)} ({len(train_normal)}N/{len(train_path)}P), "
              f"Test={len(test_indices)} ({len(test_normal)}N/{len(test_path)}P)")

        folds.append((train_indices, test_indices))

    return folds


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50, 
                validation_split=0.2, patience=100, verbose=False, seed=None):
    """
    Train the model with STRATIFIED validation split and return training history
    """
    from sklearn.model_selection import train_test_split
    
    dataset = train_loader.dataset
    
    # Check if dataset is ConcatDataset (real + synthetic)
    if isinstance(dataset, ConcatDataset):
        real_dataset = dataset.datasets[0]
        synthetic_dataset = dataset.datasets[1] if len(dataset.datasets) > 1 else None
        
        # ✅ Extract labels from real dataset for stratification
        real_labels = []
        for i in range(len(real_dataset)):
            _, label = real_dataset[i]
            real_labels.append(label)
        
        real_size = len(real_dataset)
        indices = list(range(real_size))
        
        # ✅ STRATIFIED split to maintain class balance
        rs = seed if seed is not None else 42
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_split,
            stratify=real_labels,  # ✅ Maintain class distribution
            random_state=rs
        )
        
        # Create train subset from real data only
        train_real_subset = Subset(real_dataset, train_indices)
        
        # Add synthetic data ONLY to training set
        if synthetic_dataset is not None:
            train_subset = ConcatDataset([train_real_subset, synthetic_dataset])
        else:
            train_subset = train_real_subset
        
        # Validation set is ONLY from real data
        val_subset = Subset(real_dataset, val_indices)
        
        if verbose:
            # ✅ Print class distribution to verify stratification
            train_real_labels = [real_labels[i] for i in train_indices]
            val_labels = [real_labels[i] for i in val_indices]
            
            print(f"  Real training samples: {len(train_real_subset)}")
            print(f"    - Normal: {train_real_labels.count(1)}, Pathological: {train_real_labels.count(0)}")
            
            if synthetic_dataset is not None:
                print(f"  Synthetic training samples: {len(synthetic_dataset)}")
                print(f"  Total training samples: {len(train_subset)}")
            
            print(f"  Validation samples (real only): {len(val_subset)}")
            print(f"    - Normal: {val_labels.count(1)}, Pathological: {val_labels.count(0)}")
    
    else:
        # No synthetic data, normal stratified split
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(label)
        
        indices = list(range(len(dataset)))
        
        # ✅ STRATIFIED split
        rs = seed if seed is not None else 42
        train_indices, val_indices = train_test_split(
            indices,
            test_size=validation_split,
            stratify=labels,  # ✅ Maintain class distribution
            random_state=rs
        )
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        if verbose:
            train_labels = [labels[i] for i in train_indices]
            val_labels = [labels[i] for i in val_indices]
            
            print(f"  Training samples: {len(train_subset)}")
            print(f"    - Normal: {train_labels.count(1)}, Pathological: {train_labels.count(0)}")
            print(f"  Validation samples: {len(val_subset)}")
            print(f"    - Normal: {val_labels.count(1)}, Pathological: {val_labels.count(0)}")
    
    # Create data loaders
    g_seed = seed if seed is not None else 0
    train_sub_loader = make_dataloader(
        train_subset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        seed=g_seed,
    )
    
    val_loader = make_dataloader(
        val_subset,
        batch_size=train_loader.batch_size,
        shuffle=False,
        num_workers=train_loader.num_workers,
        seed=g_seed,
    )
    
    # Rest of training code stays the same...
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = patience
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        for inputs, labels in train_sub_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        epoch_train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['train_f1'].append(epoch_train_f1)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, "
                      f"Val Loss: {epoch_val_loss:.4f} ✓ (Best)")
        else:
            patience_counter += 1
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, "
                      f"Val Loss: {epoch_val_loss:.4f}")
        
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if verbose:
            print(f"  Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return model, history





def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }


def compute_detailed_metrics(cm, positive_label=0):
    """Compute extended metrics from confusion matrix for binary classification"""
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2 for binary classification")

    neg_label = 1 - positive_label
    tp = int(cm[positive_label, positive_label])
    fn = int(cm[positive_label, neg_label])
    fp = int(cm[neg_label, positive_label])
    tn = int(cm[neg_label, neg_label])

    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0  # recall for positive class (abnormal)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # recall for negative class (normal)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = sensitivity
    f1_score_pos = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score_pos,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total_samples': total
    }


def plot_confusion_matrix_with_metrics(cm, metrics, class_names=None, title='', save_path=None):
    """Create confusion matrix with metrics table"""
    if class_names is None:
        class_names = ['Abnormal (0)', 'Normal (1)']

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Confusion matrix heatmap
    im = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - K-Fold CV\n{title}', fontsize=14)
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Add cell annotations
    max_val = cm.max() if cm.size else 0
    high_contrast_threshold = max_val * 0.75 if max_val else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            text_color = '#f5f5f5' if value >= high_contrast_threshold else '#0a0a0a'
            outline_color = '#0a0a0a' if text_color == '#f5f5f5' else '#f5f5f5'
            text = axes[0].text(j, i, str(value), ha='center', va='center', 
                               color=text_color, fontsize=14)
            text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground=outline_color),
                path_effects.Normal()
            ])

    # Metrics table
    axes[1].axis('off')
    axes[1].set_title(f'Metrics Analysis - K-Fold CV\n{title}', fontsize=14, pad=20)

    table_rows = [
        ['Evaluation Metrics', '', ''],
        ['Accuracy (Acc)', f"{metrics['accuracy']:.3f}", 'Overall classification accuracy'],
        ['Balanced Accuracy', f"{metrics['balanced_accuracy']:.3f}", 'Mean of per-class recall'],
        ['Sensitivity (Sen)', f"{metrics['sensitivity']:.3f}", 'Pathological detection rate'],
        ['Specificity (Sp)', f"{metrics['specificity']:.3f}", 'Normal detection rate'],
        ['F1-Score', f"{metrics['f1_score']:.3f}", 'Harmonic mean of prec & rec'],
        ['Precision', f"{metrics['precision']:.3f}", 'Positive predictive value'],
        ['Recall', f"{metrics['recall']:.3f}", 'True positive rate'],
        ['', '', ''],
        ['Confusion Matrix Values', '', ''],
        ['True Positive (TP)', str(metrics['tp']), 'Correctly detected pathological'],
        ['True Negative (TN)', str(metrics['tn']), 'Correctly detected normal'],
        ['False Positive (FP)', str(metrics['fp']), 'False alarms'],
        ['False Negative (FN)', str(metrics['fn']), 'Missed pathological cases'],
        ['Total Samples', str(metrics['total_samples']), 'Total evaluated records']
    ]

    def row_color(row_name):
        if row_name == 'Evaluation Metrics':
            return ['#2e7d32'] * 3
        if row_name == 'Confusion Matrix Values':
            return ['#fbc02d'] * 3
        if row_name == '':
            return ['#ffffff'] * 3
        if 'Accuracy (Acc)' in row_name:
            return ['#1565c0', '#1565c0', '#bbdefb']
        return ['#e8f5e9', '#f1f8e9', '#f1f8e9']

    cell_colours = [row_color(row[0]) for row in table_rows]

    table = axes[1].table(
        cellText=table_rows,
        cellColours=cell_colours,
        colLabels=['Metric', 'Value', 'Description'],
        colColours=['#1b5e20', '#1b5e20', '#1b5e20'],
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#263238')
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif row > 0 and table_rows[row - 1][0] in ('Evaluation Metrics', 'Confusion Matrix Values'):
            cell.get_text().set_color('#0a0a0a')
            cell.get_text().set_weight('bold')
        elif row > 0 and 'Accuracy (Acc)' in table_rows[row - 1][0] and col < 2:
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        else:
            cell.get_text().set_color('#0a0a0a')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def save_fold_info(fold_idx, train_indices, test_indices, train_dataset, synthetic_dataset, fold_output_dir, test_dataset=None, val_indices=None):
    """
    Save detailed information about which files are in train/test sets
    
    Args:
        fold_idx: Current fold number
        train_indices: Indices of training samples in train_dataset
        test_indices: Indices of test samples in train_dataset
        train_dataset: The base dataset (real data - myDataset)
        synthetic_dataset: Synthetic dataset (or None)
        fold_output_dir: Directory to save the info
        test_dataset: Optional separate test dataset (for Scenario 5)  # 🆕 ADD THIS
        val_indices: Optional list of validation indices (absolute w.r.t. train_dataset)
    """
    fold_info = {
        'fold_number': fold_idx + 1,
        'train_samples': {},
        'test_samples': {},
        'summary': {}
    }
    
    # Helper function to get label from dataset
    def get_label_from_dataset(dataset, idx):
        """Get label from either myDataset or SyntheticDataset"""
        if hasattr(dataset, 'labels'):
            return dataset.labels[idx]
        else:
            _, label = dataset[idx]
            return int(label)
    
    # Helper function to get file path from dataset
    def get_filepath_from_dataset(dataset, idx):
        """Get file path from dataset"""
        if hasattr(dataset, 'file_list'):
            return dataset.file_list[idx]
        elif hasattr(dataset, 'files'):
            return dataset.files[idx]
        else:
            return f"sample_{idx}"
    
    # 🆕 Determine which dataset to use for test samples
    actual_test_dataset = test_dataset if test_dataset is not None else train_dataset
    
    # Get test sample information
    test_files = []
    test_labels = []
    for idx in test_indices:
        # 🆕 Use actual_test_dataset instead of train_dataset
        if hasattr(actual_test_dataset, 'dataset'):
            actual_dataset = actual_test_dataset.dataset
            actual_idx = actual_test_dataset.indices[idx]
        else:
            actual_dataset = actual_test_dataset
            actual_idx = idx
        
        try:
            file_path = get_filepath_from_dataset(actual_dataset, actual_idx)
            label = get_label_from_dataset(actual_dataset, actual_idx)
            
            # 🆕 Determine data type
            data_type = 'synthetic' if test_dataset is not None else 'real'
            
            test_files.append({
                'index': int(actual_idx),
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'label': int(label),
                'label_name': 'normal' if label == 1 else 'abnormal',
                'data_type': data_type  # 🆕 Add data type
            })
            test_labels.append(label)
        except Exception as e:
            print(f"Warning: Could not get info for test index {actual_idx}: {e}")
    
    # ... rest of function remains the same until summary ...
    
    # Get train sample information (from real data)
    train_real_files = []
    train_real_labels = []
    for idx in train_indices:
        if hasattr(train_dataset, 'dataset'):
            actual_dataset = train_dataset.dataset
            actual_idx = train_dataset.indices[idx]
        else:
            actual_dataset = train_dataset
            actual_idx = idx
        
        try:
            file_path = get_filepath_from_dataset(actual_dataset, actual_idx)
            label = get_label_from_dataset(actual_dataset, actual_idx)
            
            train_real_files.append({
                'index': int(actual_idx),
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'label': int(label),
                'label_name': 'normal' if label == 1 else 'abnormal',
                'data_type': 'real'
            })
            train_real_labels.append(label)
        except Exception as e:
            print(f"Warning: Could not get info for train index {actual_idx}: {e}")
    
    # Get synthetic sample information (if exists)
    train_synthetic_files = []
    train_synthetic_labels = []
    if synthetic_dataset is not None:
        for idx in range(len(synthetic_dataset)):
            try:
                file_path = get_filepath_from_dataset(synthetic_dataset, idx)
                label = get_label_from_dataset(synthetic_dataset, idx)
                
                train_synthetic_files.append({
                    'index': int(idx),
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'label': int(label),
                    'label_name': 'normal' if label == 1 else 'abnormal',
                    'data_type': 'synthetic'
                })
                train_synthetic_labels.append(label)
            except Exception as e:
                print(f"Warning: Could not get info for synthetic index {idx}: {e}")
    
    # Combine train files
    train_all_files = train_real_files + train_synthetic_files
    train_all_labels = train_real_labels + train_synthetic_labels
    
    # Save to fold_info dictionary
    fold_info['test_samples'] = {
        'files': test_files,
        'count': len(test_files),
        'count_by_label': {
            'normal': int(sum(1 for l in test_labels if l == 1)),
            'abnormal': int(sum(1 for l in test_labels if l == 0))
        }
    }
    
    # 🆕 Validation sample information (from train_dataset)
    if val_indices is not None:
        val_files = []
        val_labels = []
        for idx in val_indices:
            if hasattr(train_dataset, 'dataset'):
                actual_dataset = train_dataset.dataset
                actual_idx = idx if not hasattr(train_dataset, 'indices') else idx
            else:
                actual_dataset = train_dataset
                actual_idx = idx
            try:
                file_path = get_filepath_from_dataset(actual_dataset, actual_idx)
                label = get_label_from_dataset(actual_dataset, actual_idx)
                val_files.append({
                    'index': int(actual_idx),
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'label': int(label),
                    'label_name': 'normal' if label == 1 else 'abnormal',
                    'data_type': 'real' if not hasattr(train_dataset, 'labels') else 'synthetic'
                })
                val_labels.append(label)
            except Exception as e:
                print(f"Warning: Could not get info for val index {actual_idx}: {e}")
        fold_info['val_samples'] = {
            'files': val_files,
            'count': len(val_files),
            'count_by_label': {
                'normal': int(sum(1 for l in val_labels if l == 1)),
                'abnormal': int(sum(1 for l in val_labels if l == 0))
            }
        }
    
    fold_info['train_samples'] = {
        'real_files': train_real_files,
        'synthetic_files': train_synthetic_files,
        'all_files': train_all_files,
        'count_real': len(train_real_files),
        'count_synthetic': len(train_synthetic_files),
        'count_total': len(train_all_files),
        'count_by_label': {
            'normal': int(sum(1 for l in train_all_labels if l == 1)),
            'abnormal': int(sum(1 for l in train_all_labels if l == 0))
        },
        'count_real_by_label': {
            'normal': int(sum(1 for l in train_real_labels if l == 1)),
            'abnormal': int(sum(1 for l in train_real_labels if l == 0))
        }
    }
    
    if synthetic_dataset is not None:
        fold_info['train_samples']['count_synthetic_by_label'] = {
            'normal': int(sum(1 for l in train_synthetic_labels if l == 1)),
            'abnormal': int(sum(1 for l in train_synthetic_labels if l == 0))
        }
    
    # 🆕 Update summary to include test data type
    fold_info['summary'] = {
        'test_size': len(test_files),
        'test_data_type': 'synthetic' if test_dataset is not None else 'real',  # 🆕
        'train_real_size': len(train_real_files),
        'train_synthetic_size': len(train_synthetic_files),
        'train_total_size': len(train_all_files),
        'has_synthetic': synthetic_dataset is not None,
        **({'val_size': fold_info['val_samples']['count']} if 'val_samples' in fold_info else {})
    }
    
    # Save to JSON
    info_path = os.path.join(fold_output_dir, 'fold_info.json')
    with open(info_path, 'w') as f:
        json.dump(fold_info, f, indent=4)
    
    # Save text file
    txt_path = os.path.join(fold_output_dir, 'fold_info.txt')
    with open(txt_path, 'w') as f:
        f.write(f"="*80 + "\n")
        f.write(f"FOLD {fold_idx + 1} INFORMATION\n")
        f.write(f"="*80 + "\n\n")
        
        f.write(f"SUMMARY:\n")
        f.write(f"  Train (Real):      {len(train_real_files):4d} samples\n")
        f.write(f"  Train (Synthetic): {len(train_synthetic_files):4d} samples\n")
        f.write(f"  Train (Total):     {len(train_all_files):4d} samples\n")
        # 🆕 Show test data type
        test_type = "Synthetic" if test_dataset is not None else "Real"
        f.write(f"  Test ({test_type}):  {len(test_files):4d} samples\n\n")
        if 'val_samples' in fold_info:
            f.write(f"  Val (Real):        {fold_info['val_samples']['count']:4d} samples\n\n")
        
        f.write(f"TEST SET FILES ({len(test_files)} samples - {test_type}):\n")
        f.write(f"-"*80 + "\n")
        for item in sorted(test_files, key=lambda x: x['file_name']):
            f.write(f"  {item['file_name']:20s} - {item['label_name']:10s} (label={item['label']})\n")
        
        f.write(f"\n" + "="*80 + "\n")
        f.write(f"TRAIN SET FILES ({len(train_all_files)} samples)\n")
        f.write(f"="*80 + "\n\n")
        
        f.write(f"Real Training Files ({len(train_real_files)} samples):\n")
        f.write(f"-"*80 + "\n")
        for item in sorted(train_real_files, key=lambda x: x['file_name']):
            f.write(f"  {item['file_name']:20s} - {item['label_name']:10s} (label={item['label']})\n")
        
        if train_synthetic_files:
            f.write(f"\nSynthetic Training Files ({len(train_synthetic_files)} samples):\n")
            f.write(f"-"*80 + "\n")
            for item in sorted(train_synthetic_files, key=lambda x: x['file_name']):
                f.write(f"  {item['file_name']:20s} - {item['label_name']:10s} (label={item['label']})\n")
        
        # 🆕 Write validation files if available
        if 'val_samples' in fold_info:
            val_files = fold_info['val_samples']['files']
            f.write(f"\n" + "="*80 + "\n")
            f.write(f"VALIDATION SET FILES ({len(val_files)} samples)\n")
            f.write(f"="*80 + "\n")
            for item in sorted(val_files, key=lambda x: x['file_name']):
                f.write(f"  {item['file_name']:20s} - {item['label_name']:10s} (label={item['label']})\n")
    
    print(f"✓ Saved fold info to: {fold_output_dir}")
    return fold_info




def run_scenario(
    config,
    scenario_name,
    train_dataset,
    test_folds,
    synthetic_dataset=None,
    test_dataset=None,
    use_all_train=False,
    alt_test_folds=None,
):
    """
    Run k-fold cross-validation for a given scenario
    """
    print(f"\n{'='*80}")
    print(f"Running {scenario_name}")
    print(f"{'='*80}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store results for all folds
    all_fold_results = []
    cumulative_cm = None
    all_predictions = []
    all_labels = []
    
    # ✅ ADD: Store metrics per fold for mean±std calculation
    fold_accuracies = []
    fold_f1_scores = []
    fold_balanced_accs = []
    fold_precisions = []
    fold_recalls = []
    fold_sensitivities = []
    fold_specificities = []

    for fold_idx in range(len(test_folds)):
        print(f"\n--- Fold {fold_idx + 1}/{len(test_folds)} ---")

        # Resolve indices for this fold
        train_indices_real, test_indices_real = test_folds[fold_idx]
        # If alt_test_folds provided, use those test indices (e.g., synthetic test)
        if alt_test_folds is not None:
            _, test_indices = alt_test_folds[fold_idx]
        else:
            test_indices = test_indices_real

        # Create train subset (optionally use all training data)
        if use_all_train:
            train_subset = train_dataset
            train_indices_for_logging = list(range(len(train_dataset)))
        else:
            train_subset = Subset(train_dataset, train_indices_real)
            train_indices_for_logging = train_indices_real

        # Add synthetic data if provided (only to training set!)
        if synthetic_dataset is not None:
            print(f"Adding {len(synthetic_dataset)} synthetic samples to training set")
            train_combined = ConcatDataset([train_subset, synthetic_dataset])
        else:
            train_combined = train_subset

        # Create test subset
        if test_dataset is not None:
            test_subset = Subset(test_dataset, test_indices)
        else:
            test_subset = Subset(train_dataset, test_indices)

        # Compute validation indices for logging (deterministic, same as train_model)
        from sklearn.model_selection import train_test_split as _tts
        subset_space = list(range(len(train_indices_for_logging)))
        labels_for_split = []
        for i in subset_space:
            try:
                _, lab = train_dataset[train_indices_for_logging[i]]
                labels_for_split.append(int(lab))
            except Exception:
                labels_for_split.append(0)
        rs = int(config['random_seed'])
        _, val_subset_idx = _tts(
            subset_space,
            test_size=0.2,
            stratify=labels_for_split if len(set(labels_for_split)) > 1 else None,
            random_state=rs,
        )
        val_indices_abs = [train_indices_for_logging[i] for i in val_subset_idx]

        # Create deterministic data loaders (single global seed)
        global_seed = int(config['random_seed'])
        train_loader = make_dataloader(
            train_combined,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            seed=global_seed,
        )

        test_loader = make_dataloader(
            test_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            seed=global_seed,
        )

        print(f"Train set: {len(train_combined)} samples")
        print(f"Test set: {len(test_subset)} samples")

        # Create fold output directory
        fold_output_dir = os.path.join(config['output_dir'], scenario_name, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # Save fold info (with validation details)
        save_fold_info(
            fold_idx=fold_idx,
            train_indices=train_indices_for_logging,
            test_indices=test_indices,
            train_dataset=train_dataset,
            synthetic_dataset=synthetic_dataset,
            fold_output_dir=fold_output_dir,
            test_dataset=test_dataset,
            val_indices=val_indices_abs,
        )
        
        # Initialize RNGs and model for this fold (single global seed)
        seed_everything(global_seed)
        model = create_model(config['model_name'], num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        # Train model for this fold
        print(f"\nTraining model for fold {fold_idx + 1}...")
        model, history = train_model(
            model, 
            train_loader,
            criterion,
            optimizer,
            device,
            config['num_epochs'],
            validation_split = 0.2,
            verbose=True,
            seed=global_seed
        )
        
        # Evaluate model
        print(f"Evaluating fold {fold_idx + 1}...")
        fold_results = evaluate_model(model, test_loader, device)
        
        # ✅ Compute detailed metrics per fold
        fold_detailed_metrics = compute_detailed_metrics(fold_results['confusion_matrix'], positive_label=0)
        
        # ✅ Store metrics for this fold
        fold_accuracies.append(fold_results['accuracy'])
        fold_f1_scores.append(fold_results['f1_score'])
        fold_balanced_accs.append(fold_results['balanced_accuracy'])
        fold_precisions.append(fold_results['precision'])
        fold_recalls.append(fold_results['recall'])
        fold_sensitivities.append(fold_detailed_metrics['sensitivity'])
        fold_specificities.append(fold_detailed_metrics['specificity'])
        
        # Accumulate results
        all_fold_results.append(fold_results)
        all_predictions.extend(fold_results['predictions'])
        all_labels.extend(fold_results['labels'])

        if cumulative_cm is None:
            cumulative_cm = fold_results['confusion_matrix']
        else:
            cumulative_cm += fold_results['confusion_matrix']

        print(f"Fold {fold_idx + 1} Results:")
        print(f"  Accuracy: {fold_results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {fold_results['balanced_accuracy']:.4f}")
        print(f"  F1-Score: {fold_results['f1_score']:.4f}")
        print(f"  Confusion Matrix:\n{fold_results['confusion_matrix']}")

        # Save fold model
        torch.save(model.state_dict(), os.path.join(fold_output_dir, 'model.pth'))

        # Save fold results
        fold_results_serializable = {
            'accuracy': float(fold_results['accuracy']),
            'balanced_accuracy': float(fold_results['balanced_accuracy']),
            'f1_score': float(fold_results['f1_score']),
            'precision': float(fold_results['precision']),
            'recall': float(fold_results['recall']),
            'confusion_matrix': fold_results['confusion_matrix'].tolist(),
            'predictions': [int(x) for x in fold_results['predictions']],
            'labels': [int(x) for x in fold_results['labels']],
            'detailed_metrics': {
                'sensitivity': float(fold_detailed_metrics['sensitivity']),
                'specificity': float(fold_detailed_metrics['specificity'])
            }
        }

        with open(os.path.join(fold_output_dir, 'results.json'), 'w') as f:
            json.dump(fold_results_serializable, f, indent=4)

    # ✅ Compute mean ± std across folds
    print(f"\n{'='*80}")
    print(f"Overall Results for {scenario_name} (K-Fold Statistics)")
    print(f"{'='*80}")
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies, ddof=1)  # Use sample std
    
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores, ddof=1)
    
    mean_balanced_acc = np.mean(fold_balanced_accs)
    std_balanced_acc = np.std(fold_balanced_accs, ddof=1)
    
    mean_precision = np.mean(fold_precisions)
    std_precision = np.std(fold_precisions, ddof=1)
    
    mean_recall = np.mean(fold_recalls)
    std_recall = np.std(fold_recalls, ddof=1)
    
    mean_sensitivity = np.mean(fold_sensitivities)
    std_sensitivity = np.std(fold_sensitivities, ddof=1)
    
    mean_specificity = np.mean(fold_specificities)
    std_specificity = np.std(fold_specificities, ddof=1)
    
    # ✅ Print results in mean ± std format
    print(f"\n✅ K-Fold Cross-Validation Results (Mean ± Std):")
    print(f"  Accuracy:         {mean_accuracy:.4f} ± {std_accuracy:.4f} ({mean_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%)")
    print(f"  Balanced Acc:     {mean_balanced_acc:.4f} ± {std_balanced_acc:.4f} ({mean_balanced_acc*100:.2f}% ± {std_balanced_acc*100:.2f}%)")
    print(f"  F1-Score:         {mean_f1:.4f} ± {std_f1:.4f} ({mean_f1*100:.2f}% ± {std_f1*100:.2f}%)")
    print(f"  Precision:        {mean_precision:.4f} ± {std_precision:.4f} ({mean_precision*100:.2f}% ± {std_precision*100:.2f}%)")
    print(f"  Recall:           {mean_recall:.4f} ± {std_recall:.4f} ({mean_recall*100:.2f}% ± {std_recall*100:.2f}%)")
    print(f"  Sensitivity:      {mean_sensitivity:.4f} ± {std_sensitivity:.4f} ({mean_sensitivity*100:.2f}% ± {std_sensitivity*100:.2f}%)")
    print(f"  Specificity:      {mean_specificity:.4f} ± {std_specificity:.4f} ({mean_specificity*100:.2f}% ± {std_specificity*100:.2f}%)")
    
    print(f"\nCumulative Confusion Matrix:")
    print(cumulative_cm)

    # Calculate metrics from cumulative CM
    detailed_metrics = compute_detailed_metrics(cumulative_cm, positive_label=0)
    detailed_metrics['balanced_accuracy'] = mean_balanced_acc

    # Save overall results
    scenario_output_dir = os.path.join(config['output_dir'], scenario_name)
    os.makedirs(scenario_output_dir, exist_ok=True)

    # Plot cumulative confusion matrix
    cm_plot_path = os.path.join(scenario_output_dir, 'cumulative_confusion_matrix.png')
    plot_confusion_matrix_with_metrics(
        cumulative_cm,
        detailed_metrics,
        class_names=['Abnormal (0)', 'Normal (1)'],
        title=f"{scenario_name} - {config['model_name']}",
        save_path=cm_plot_path
    )

    # ✅ Save overall results with mean±std to JSON
    overall_results = {
        'scenario': scenario_name,
        'n_folds': len(test_folds),
        'cumulative_confusion_matrix': cumulative_cm.tolist(),
        # ✅ Add mean±std statistics
        'kfold_statistics': {
            'accuracy': {
                'mean': float(mean_accuracy), 
                'std': float(std_accuracy),
                'folds': [float(x) for x in fold_accuracies]  # ← ADD THIS
            },
            'balanced_accuracy': {
                'mean': float(mean_balanced_acc), 
                'std': float(std_balanced_acc),
                'folds': [float(x) for x in fold_balanced_accs]  # ← ADD THIS
            },
            'f1_score': {
                'mean': float(mean_f1), 
                'std': float(std_f1),
                'folds': [float(x) for x in fold_f1_scores]  # ← ADD THIS
            },
            'precision': {
                'mean': float(mean_precision), 
                'std': float(std_precision),
                'folds': [float(x) for x in fold_precisions]  # ← ADD THIS
            },
            'recall': {
                'mean': float(mean_recall), 
                'std': float(std_recall),
                'folds': [float(x) for x in fold_recalls]  # ← ADD THIS
            },
            'sensitivity': {
                'mean': float(mean_sensitivity), 
                'std': float(std_sensitivity),
                'folds': [float(x) for x in fold_sensitivities]  # ← ADD THIS
            },
            'specificity': {
                'mean': float(mean_specificity), 
                'std': float(std_specificity),
                'folds': [float(x) for x in fold_specificities]  # ← ADD THIS
            }
        },
        'overall_accuracy': float(mean_accuracy),
        'overall_f1_score': float(mean_f1),
        'overall_balanced_accuracy': float(mean_balanced_acc),
        'detailed_metrics': {
            'accuracy': float(detailed_metrics['accuracy']),
            'sensitivity': float(detailed_metrics['sensitivity']),
            'specificity': float(detailed_metrics['specificity']),
            'precision': float(detailed_metrics['precision']),
            'recall': float(detailed_metrics['recall']),
            'f1_score': float(detailed_metrics['f1_score']),
            'tp': int(detailed_metrics['tp']),
            'tn': int(detailed_metrics['tn']),
            'fp': int(detailed_metrics['fp']),
            'fn': int(detailed_metrics['fn']),
            'total_samples': int(detailed_metrics['total_samples'])
        },
        'fold_results': []
    }

    # Add individual fold results
    for fold_idx, fold_res in enumerate(all_fold_results):
        overall_results['fold_results'].append({
            'fold': fold_idx + 1,
            'accuracy': float(fold_res['accuracy']),
            'balanced_accuracy': float(fold_res['balanced_accuracy']),
            'f1_score': float(fold_res['f1_score']),
            'precision': float(fold_res['precision']),
            'recall': float(fold_res['recall']),
            'confusion_matrix': fold_res['confusion_matrix'].tolist()
        })

    with open(os.path.join(scenario_output_dir, 'overall_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)
    # Save human-readable metrics summary as CSV
    metrics_summary = pd.DataFrame([{
        'Scenario': scenario_name,
        'Model': config['model_name'],
        'N_Folds': len(test_folds),
        'Accuracy_Mean': f'{mean_accuracy:.4f}',
        'Accuracy_Std': f'{std_accuracy:.4f}',
        'Accuracy_Pct': f'{mean_accuracy*100:.2f}±{std_accuracy*100:.2f}%',
        'BalancedAcc_Mean': f'{mean_balanced_acc:.4f}',
        'BalancedAcc_Std': f'{std_balanced_acc:.4f}',
        'F1Score_Mean': f'{mean_f1:.4f}',
        'F1Score_Std': f'{std_f1:.4f}',
        'Precision_Mean': f'{mean_precision:.4f}',
        'Precision_Std': f'{std_precision:.4f}',
        'Recall_Mean': f'{mean_recall:.4f}',
        'Recall_Std': f'{std_recall:.4f}',
        'Sensitivity_Mean': f'{mean_sensitivity:.4f}',
        'Sensitivity_Std': f'{std_sensitivity:.4f}',
        'Sensitivity_Pct': f'{mean_sensitivity*100:.2f}±{std_sensitivity*100:.2f}%',
        'Specificity_Mean': f'{mean_specificity:.4f}',
        'Specificity_Std': f'{std_specificity:.4f}',
        'Specificity_Pct': f'{mean_specificity*100:.2f}±{std_specificity*100:.2f}%'
    }])

    csv_path = os.path.join(scenario_output_dir, 'kfold_metrics_summary.csv')
    metrics_summary.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics summary CSV to {csv_path}")

    print(f"\nResults saved to {scenario_output_dir}")

    return overall_results



def main():
    """Main function to run all scenarios with k-fold cross-validation"""

    config = {
        # Model and training parameters
        'model_name': 'cnn1d',              # default/legacy single model
        'model_names': ['cnn1d', 'resnet50v1', 'mobilenet', 'efficientnetb0'],  # <— run multiple models or set to 'all'
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'random_seed': 81,
        'sequence_length': 1000,
        'num_workers': 0,

        # K-fold parameters
        'n_folds': 4,
        'test_samples_per_class': 25,  # 25 normal + 25 pathological per test fold

        # Real dataset configuration
        'real_max_normal': 447,
        'real_max_pathological': 105,

        # How many real samples to use for k-fold
        'use_normal_samples': 100,
        'use_pathological_samples': 100,

        # Paths
        'real_data_folder': '/path/to/.csv',
        'real_ph_file': '/path/to/ph_labels.csv',
        'synthetic_normal_folder': '/path/to/.csv/synthetic_normal_FHRGAN',
        'synthetic_abnormal_folder': '/path/to/.csv/synthetic_pathological_FHRGAN',

        # Synthetic data limits for scenario 2 and 4
        'synthetic_normal_limit': 500,
        'synthetic_abnormal_limit': 500,

        # For scenario 3: synthetic train, real test (use all synthetic)
        'scenario3_synthetic_normal': 400,
        'scenario3_synthetic_abnormal': 400,

        # Output directory (per-model subfolders will be created inside this)
        'output_dir': './results_kfold_cv_all_scenarios/FHRGAN715100100',

        # Which scenarios to run
        'run_scenarios': [2]  # add 3 if you want it too
    }

    # Set random seeds (deterministic settings)
    seed_everything(config['random_seed'])

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Save configuration
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n{'='*80}")
    print("K-Fold Cross-Validation CNN Training - ALL 5 SCENARIOS")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Folds: {config['n_folds']}")
    print(f"  Test samples per class per fold: {config['test_samples_per_class']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Random seed: {config['random_seed']}")

    # Load full real dataset for creating folds
    print(f"\nLoading real dataset...")
    real_full_dataset = myDataset(
        config['real_data_folder'],
        config['real_ph_file'],
        mode='train',  # Load all available data
        sequence_length=config['sequence_length'],
        max_normal=config['use_normal_samples'],
        max_pathological=config['use_pathological_samples'],
        train_normal=config['use_normal_samples'],  # Use all for folding
        train_pathological=config['use_pathological_samples'],
        eval_normal=0,  # Not used in this context
        eval_pathological=0,
        random_seed=config['random_seed']
    )

    print(f"Loaded {len(real_full_dataset)} real samples")

    # Create or load cached stratified k-fold splits (persist for reuse)
    folds_cache_path = os.path.join(
        config['output_dir'],
        f"folds_real_n{config['n_folds']}_t{config['test_samples_per_class']}_seed{config['random_seed']}.json"
    )

    if os.path.exists(folds_cache_path):
        with open(folds_cache_path, 'r') as f:
            cached = json.load(f)
        test_folds = [(fold['train_indices'], fold['test_indices']) for fold in cached.get('folds', [])]
        print(f"Loaded cached folds from {folds_cache_path}")
    else:
        test_folds = create_stratified_kfold_splits(
            real_full_dataset,
            n_folds=config['n_folds'],
            test_per_class=config['test_samples_per_class'],
            seed=config['random_seed']
        )
        cache_payload = {
            'n_folds': int(config['n_folds']),
            'test_per_class': int(config['test_samples_per_class']),
            'seed': int(config['random_seed']),
            'timestamp': datetime.now().isoformat(),
            'folds': [
                {
                    'train_indices': list(map(int, tr)),
                    'test_indices': list(map(int, te))
                }
                for (tr, te) in test_folds
            ]
        }
        with open(folds_cache_path, 'w') as f:
            json.dump(cache_payload, f, indent=4)
        print(f"Saved folds to {folds_cache_path}")
    supported = ['cnn1d', 'resnet1d', 'resnet50v2', 'mobilenet', 'efficientnetb0', 'densenet201']
    models_to_run = supported if config.get('model_names') == 'all' else config.get('model_names', [config['model_name']])

    base_output_dir = config['output_dir']

    for model_name in models_to_run:
        print("\n" + "="*80)
        print(f"Running all selected scenarios for model: {model_name}")
        print("="*80)

        # Make a per-model copy of config
        cfg = dict(config)
        cfg['model_name'] = model_name
        cfg['output_dir'] = os.path.join(base_output_dir, model_name)
        os.makedirs(cfg['output_dir'], exist_ok=True)

        # Collect results for this model
        all_scenario_results = {}

        # # Dictionary to store all scenario results
        # all_scenario_results = {}

        # ------------------------------------------------------------------------
        # Scenario 1: Real train, Real test
        # ------------------------------------------------------------------------
        if 1 in config['run_scenarios']:
            scenario_results = run_scenario(
                config=cfg,
                scenario_name='scenario1_real_only',
                train_dataset=real_full_dataset,
                test_folds=test_folds,
                synthetic_dataset=None
            )
            all_scenario_results['scenario1'] = scenario_results

        # ------------------------------------------------------------------------
        # Scenario 2: Real + Synthetic train, Real test
        # ------------------------------------------------------------------------
        if 2 in config['run_scenarios']:
            print(f"\nLoading synthetic dataset for Scenario 2...")
            synthetic_dataset_s2 = SyntheticDataset(
                config['synthetic_normal_folder'],
                config['synthetic_abnormal_folder'],
                sequence_length=config['sequence_length'],
                max_normal_samples=config['synthetic_normal_limit'],
                max_abnormal_samples=config['synthetic_abnormal_limit']
            )

            scenario_results = run_scenario(
                config=cfg,
                scenario_name='scenario2_real_plus_synthetic',
                train_dataset=real_full_dataset,
                test_folds=test_folds,
                synthetic_dataset=synthetic_dataset_s2
            )
            all_scenario_results['scenario2'] = scenario_results

        # ------------------------------------------------------------------------
        # Scenario 3: Synthetic train, Synthetic test (all synthetic data)
        # ------------------------------------------------------------------------
        if 3 in config['run_scenarios']:
            print(f"\n{'='*80}")
            print("Scenario 3: Synthetic train, Synthetic test")
            print(f"{'='*80}")
            print("Loading all synthetic data...")

            # Load all synthetic data for this scenario
            synthetic_full_dataset_s3 = SyntheticDataset(
                config['synthetic_normal_folder'],
                config['synthetic_abnormal_folder'],
                sequence_length=config['sequence_length'],
                max_normal_samples=config['scenario3_synthetic_normal'],  # None = use all
                max_abnormal_samples=config['scenario3_synthetic_abnormal']  # None = use all
            )

            print(f"Loaded {len(synthetic_full_dataset_s3)} synthetic samples")

            # Create or load cached k-fold splits from synthetic data
            synth_cache_path = os.path.join(
                cfg['output_dir'],
                f"folds_synthetic_n{cfg['n_folds']}_t{cfg['test_samples_per_class']}_seed{cfg['random_seed']}.json"
            )
            if os.path.exists(synth_cache_path):
                with open(synth_cache_path, 'r') as f:
                    cached = json.load(f)
                synthetic_folds = [(fold['train_indices'], fold['test_indices']) for fold in cached.get('folds', [])]
                print(f"Loaded cached synthetic folds from {synth_cache_path}")
            else:
                synthetic_folds = create_stratified_kfold_splits(
                    synthetic_full_dataset_s3,
                    n_folds=cfg['n_folds'],
                    test_per_class=cfg['test_samples_per_class'],
                    seed=cfg['random_seed']
                )
                cache_payload = {
                    'n_folds': int(cfg['n_folds']),
                    'test_per_class': int(cfg['test_samples_per_class']),
                    'seed': int(cfg['random_seed']),
                    'timestamp': datetime.now().isoformat(),
                    'folds': [
                        {
                            'train_indices': list(map(int, tr)),
                            'test_indices': list(map(int, te))
                        }
                        for (tr, te) in synthetic_folds
                    ]
                }
                with open(synth_cache_path, 'w') as f:
                    json.dump(cache_payload, f, indent=4)
                print(f"Saved synthetic folds to {synth_cache_path}")

            scenario_results = run_scenario(
                config=cfg,
                scenario_name='scenario3_synthetic_only',
                train_dataset=synthetic_full_dataset_s3,
                test_folds=synthetic_folds,
                synthetic_dataset=None  # All data is synthetic
            )
            all_scenario_results['scenario3'] = scenario_results

        # ------------------------------------------------------------------------
        # Scenario 4: Synthetic train, Real test
        # ------------------------------------------------------------------------
        if 4 in config['run_scenarios']:
            print('='*80)
            print("Scenario 4: Synthetic train, Real test")
            print('='*80)
            print("Loading all synthetic data for training...")
            
            synthetic_full_dataset_s4 = SyntheticDataset(
                config['synthetic_normal_folder'],
                config['synthetic_abnormal_folder'],
                sequence_length=config['sequence_length'],
                max_normal_samples=None,  # Use all synthetic normal
                max_abnormal_samples=None  # Use all synthetic abnormal
            )
            print(f"Loaded {len(synthetic_full_dataset_s4)} synthetic samples for training")
            
            # Use run_scenario() just like other scenarios!
            scenario_results = run_scenario(
                config=cfg,
                scenario_name='scenario_4_synthetic_train_real_test',
                train_dataset=synthetic_full_dataset_s4,  # Train on synthetic data
                test_folds=test_folds,  # Use real data test splits (for test indices)
                synthetic_dataset=None,  # All training data is already synthetic
                test_dataset=real_full_dataset,  # Test on real data
                use_all_train=True  # Train on ALL synthetic samples per fold
            )
            all_scenario_results['scenario_4'] = scenario_results


        # ------------------------------------------------------------------------
        # Scenario 5: Real train, Synthetic test (NEW!)
        # ------------------------------------------------------------------------
        
        if 5 in config['run_scenarios']:
            print("="*80)
            print("Scenario 5: Real train, Synthetic test")
            print("="*80)
            print("Note: Training on real data, testing on synthetic data")
            print("This evaluates how well real-trained models recognize synthetic patterns")
            
            # Load synthetic dataset for testing
            print("Loading synthetic dataset for testing...")
            synthetic_full_dataset_s5 = SyntheticDataset(
                config['synthetic_normal_folder'],
                config['synthetic_abnormal_folder'],
                sequence_length=config['sequence_length'],
                max_normal_samples=None,  # Use all available
                max_abnormal_samples=None  # Use all available
            )
            print(f"Loaded {len(synthetic_full_dataset_s5)} synthetic samples for testing")
            
            # Create or load cached k-fold splits from synthetic data for testing
            synth_test_cache_path = os.path.join(
                cfg['output_dir'],
                f"folds_synthetic_test_n{cfg['n_folds']}_t{cfg['test_samples_per_class']}_seed{cfg['random_seed']}.json"
            )
            if os.path.exists(synth_test_cache_path):
                with open(synth_test_cache_path, 'r') as f:
                    cached = json.load(f)
                synthetic_test_folds = [(fold['train_indices'], fold['test_indices']) for fold in cached.get('folds', [])]
                print(f"Loaded cached synthetic test folds from {synth_test_cache_path}")
            else:
                synthetic_test_folds = create_stratified_kfold_splits(
                    synthetic_full_dataset_s5,
                    n_folds=cfg['n_folds'],
                    test_per_class=cfg['test_samples_per_class'],
                    seed=cfg['random_seed']
                )
                cache_payload = {
                    'n_folds': int(cfg['n_folds']),
                    'test_per_class': int(cfg['test_samples_per_class']),
                    'seed': int(cfg['random_seed']),
                    'timestamp': datetime.now().isoformat(),
                    'folds': [
                        {
                            'train_indices': list(map(int, tr)),
                            'test_indices': list(map(int, te))
                        }
                        for (tr, te) in synthetic_test_folds
                    ]
                }
                with open(synth_test_cache_path, 'w') as f:
                    json.dump(cache_payload, f, indent=4)
                print(f"Saved synthetic test folds to {synth_test_cache_path}")
            
            # Use run_scenario() just like other scenarios!
            scenario_results = run_scenario(
                config=cfg,
                scenario_name='scenario_5_real_train_synthetic_test',
                train_dataset=real_full_dataset,           # Train on real data
                test_folds=test_folds,                     # Use real data train splits
                synthetic_dataset=None,                    # No synthetic in training
                test_dataset=synthetic_full_dataset_s5,    # Test on synthetic data
                alt_test_folds=synthetic_test_folds        # Use synthetic test indices
            )
            all_scenario_results['scenario5'] = scenario_results


            # ------------------------------------------------------------------------
            # Summary of all scenarios
            # ------------------------------------------------------------------------
            print(f"\n{'='*80}")
            print("SUMMARY OF ALL SCENARIOS")
            print(f"{'='*80}")

            summary_data = []
            for scenario_name, results in all_scenario_results.items():
                summary_data.append({
                    'Scenario': scenario_name,
                    'Accuracy': f"{results['overall_accuracy']:.4f}",
                    'Balanced Acc': f"{results['overall_balanced_accuracy']:.4f}",
                    'F1-Score': f"{results['overall_f1_score']:.4f}",
                    'Sensitivity': f"{results['detailed_metrics']['sensitivity']:.4f}",
                    'Specificity': f"{results['detailed_metrics']['specificity']:.4f}"
                })

            summary_df = pd.DataFrame(summary_data)
            print("\n" + summary_df.to_string(index=False))

            # Save summary
            summary_df.to_csv(os.path.join(config['output_dir'], 'scenarios_summary.csv'), index=False)

            with open(os.path.join(config['output_dir'], 'all_scenarios_results.json'), 'w') as f:
                json.dump(all_scenario_results, f, indent=4)

            print(f"\nAll results saved to {config['output_dir']}")
            print("\nK-Fold Cross-Validation completed successfully!")
            print("\nAll 5 Scenarios:")
            print("  1. Real train → Real test")
            print("  2. Real+Synthetic train → Real test")
            print("  3. Synthetic train → Synthetic test")
            print("  4. Synthetic train → Real test")
            print("  5. Real train → Synthetic test (Evaluates real-to-synthetic generalization)")


if __name__ == '__main__':
    main()

