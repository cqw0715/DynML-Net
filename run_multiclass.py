import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

# Sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss, 
                             average_precision_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize, StandardScaler
from collections import Counter

# Imbalanced-Learn imports
from imblearn.over_sampling import SMOTE, RandomOverSampler

# PyTorch imports
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Import modules
from modules import (
    BEST_PARAMS, ESMFeatureExtractor, load_data,
    CNNBranch, TransformerBranch, MambaBranch, DynML_Net
)

def train_multiclass_task(csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = BEST_PARAMS

    # 1. Data Loading and Feature Extraction
    seqs, labels = load_data(csv_path)
    num_classes = len(np.unique(labels))

    extractor = ESMFeatureExtractor()
    # Multiclass cache path
    X_raw = extractor.extract_features(seqs, cache_path="esm_650m_features_mean_multiclass.pkl")
    y_raw = labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    history = {name: [] for name in ['DynML_Net']}

    print(f" Starting training DynML-Net (Multiclass Task, Classes: {num_classes})...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_raw), 1):
        print(f"\n--- Fold {fold}/5 ---")

        X_train_fold, y_train_fold = X_raw[train_idx], y_raw[train_idx]
        X_val_fold, y_val_fold = X_raw[val_idx], y_raw[val_idx]

        # 2. Dynamic Imbalanced Sampling (SMOTE / RandomOverSampler)
        counts = Counter(y_train_fold)
        min_samples = min(counts.values())
        k_neighbors = min(5, min_samples - 1)

        if k_neighbors < 1:
            print(" Samples too few, using Random OverSampler")
            sampler = RandomOverSampler(random_state=42)
        else:
            print(f" Using SMOTE (k={k_neighbors})")
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)

        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_fold, y_train_fold)

        # 3. Data Standardization and DataLoader
        scaler = StandardScaler()
        X_train = torch.FloatTensor(scaler.fit_transform(X_train_resampled))
        y_train = torch.LongTensor(y_train_resampled)
        X_val = torch.FloatTensor(scaler.transform(X_val_fold))
        y_val = torch.LongTensor(y_val_fold)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=p['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=p['batch_size'])

        # 4. Model Initialization
        cnn_branch = CNNBranch(
            X_raw.shape[1], 
            num_classes, 
            dropout_rate=p['cnn_dropout']
        ).to(device)
        
        trans_branch = TransformerBranch(
            X_raw.shape[1], 
            num_classes=num_classes, 
            num_layers=p['transformer_layers'], 
            dropout_rate=p['trans_dropout'] 
        ).to(device)
        
        mamba_branch = MambaBranch(
            X_raw.shape[1], 
            num_classes, 
            num_blocks=p['mamba_blocks'], 
            dropout_rate=p['mamba_dropout']
        ).to(device)

        model = DynML_Net(
            input_dim=X_raw.shape[1], 
            num_classes=num_classes, 
            embed_dim=p['embed_dim'],
            gate_dropout1=p['gate_dropout1'],
            gate_dropout2=p['gate_dropout2'],
            refine_dropout=p['refine_dropout'],
            attn_dropout=p['attn_dropout']
        ).to(device)
        model.set_branches(cnn_branch, trans_branch, mamba_branch)

        optimizer = optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

        # 5. Training Loop (with Early Stopping)
        best_val_loss, patience, counter = float('inf'), 5, 0
        best_model_state = None

        for epoch in range(50):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                o1, o2, o3, o_fused = model(batch_x)

                # Cross Entropy Loss (keep original logic to ensure training effectiveness)
                loss_ce = (criterion(o1, batch_y) + criterion(o2, batch_y) + 
                           criterion(o3, batch_y) + criterion(o_fused, batch_y)) / 4

                # KL Divergence Loss (keep original logic to ensure training effectiveness)
                probs = [F.softmax(o, dim=1) for o in [o1, o2, o3]]
                log_probs = [F.log_softmax(o, dim=1) for o in [o1, o2, o3]]

                loss_kl = (
                    kl_loss_fn(log_probs[0], probs[1].detach()) + kl_loss_fn(log_probs[0], probs[2].detach()) +
                    kl_loss_fn(log_probs[1], probs[0].detach()) + kl_loss_fn(log_probs[1], probs[2].detach()) +
                    kl_loss_fn(log_probs[2], probs[0].detach()) + kl_loss_fn(log_probs[2], probs[1].detach())
                ) / 6

                total_loss = loss_ce + p['kl_weight'] * loss_kl
                total_loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss_sum = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    _, _, _, o_fused = model(batch_x.to(device))
                    val_loss_sum += criterion(o_fused, batch_y.to(device)).item()
            
            avg_val_loss = val_loss_sum / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience:
                    break

        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
            model.eval()

        # 6. Evaluation and Metrics Collection
        fold_results = {name: {'probs': [], 'preds': []} for name in history.keys()}
        true_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x.to(device))
                true_labels.extend(batch_y.numpy())

                prob_tensor = F.softmax(outputs[3], dim=1).cpu()
                fold_results['DynML_Net']['probs'].extend(prob_tensor.numpy())
                fold_results['DynML_Net']['preds'].extend(prob_tensor.argmax(dim=1).numpy())

        # Calculate metrics
        y_true_onehot = label_binarize(true_labels, classes=range(num_classes))

        for name in history.keys():
            preds = np.array(fold_results[name]['preds'])
            probs = np.array(fold_results[name]['probs'])

            cm = confusion_matrix(true_labels, preds, labels=range(num_classes))
            # Prevent division by zero
            class_sums = cm.sum(axis=1)
            class_sums[class_sums == 0] = 1

            metrics = {
                'Accuracy': accuracy_score(true_labels, preds),
                'F1-Macro': f1_score(true_labels, preds, average='macro'),
                'HammingLoss': hamming_loss(true_labels, preds),
                'MAP': average_precision_score(y_true_onehot, probs, average='macro'),
                'Class_Acc': cm.diagonal() / class_sums,
                'Class_AUC': roc_auc_score(y_true_onehot, probs, average=None, multi_class='ovr')
            }
            history[name].append(metrics)

    # ==========================================
    # Results Report
    # ==========================================
    print("\n" + "="*60)
    print("Final Performance Evaluation Report (5-Fold Cross-Validation)")
    print("="*60)

    for name in history.keys():
        print(f"\n>>> {name} Branch <<<")
        # Scalar metrics
        scalar_metrics = ['Accuracy', 'F1-Macro', 'HammingLoss', 'MAP']
        for m_name in scalar_metrics:
            vals = [h[m_name] for h in history[name]]
            mean_val, std_val = np.mean(vals), np.std(vals)
            print(f"{m_name:<15}: {mean_val:.4f} ± {std_val:.4f}")

        # Class-level metrics
        all_class_acc_list = [h['Class_Acc'] for h in history[name]]
        all_class_auc_list = [h['Class_AUC'] for h in history[name]]

        mean_class_acc = np.mean(all_class_acc_list, axis=0)
        std_class_acc = np.std(all_class_acc_list, axis=0)
        mean_class_auc = np.mean(all_class_auc_list, axis=0)
        std_class_auc = np.std(all_class_auc_list, axis=0)

        print(f"\nClass Detailed Metrics (Mean ± Std):")
        for c in range(num_classes):
            acc_str = f"{mean_class_acc[c]:.4f} ± {std_class_acc[c]:.4f}"
            auc_str = f"{mean_class_auc[c]:.4f} ± {std_class_auc[c]:.4f}"
            print(f" Class {c+1}: Acc={acc_str}, AUC={auc_str}")

if __name__ == "__main__":
    # Please replace with your multiclass data file path
    train_multiclass_task("90%_1587.csv")