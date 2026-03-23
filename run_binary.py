import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os

# Sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# PyTorch imports
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Import modules
from modules import (
    BEST_PARAMS, ESMFeatureExtractor, load_data,
    CNNBranch, TransformerBranch, MambaBranch, DynML_Net
)

def train_binary_task(csv_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p = BEST_PARAMS

    extractor = ESMFeatureExtractor()
    seqs, labels = load_data(csv_path)
    # Binary classification cache path
    X_raw = extractor.extract_features(seqs, cache_path="esm_650m_features_mean_binary.pkl")
    y_raw = labels

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    fused_history = []

    print(f" Starting DynML-Net training (Binary Classification Task)...")
    print(f" LR: {p['lr']}, WD: {p['weight_decay']}, KL: {p['kl_weight']}")
    print("-" * 50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_raw), 1):
        scaler = StandardScaler() 
        X_train = torch.FloatTensor(scaler.fit_transform(X_raw[train_idx]))
        X_val = torch.FloatTensor(scaler.transform(X_raw[val_idx]))
        y_train = torch.LongTensor(y_raw[train_idx])
        y_val = torch.LongTensor(y_raw[val_idx])

        cw = compute_class_weight('balanced', classes=np.unique(y_raw), y=y_raw[train_idx])
        alpha = torch.FloatTensor(cw).to(device)

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=p['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=p['batch_size'])

        # Initialize branches (num_classes=2)
        num_classes = 2
        cnn_branch = CNNBranch(input_dim=X_raw.shape[1], num_classes=num_classes, dropout_rate=p['cnn_dropout']).to(device)
        trans_branch = TransformerBranch(input_dim=X_raw.shape[1], d_model=256, nhead=8, num_layers=p['transformer_layers'], num_classes=num_classes, dropout_rate=p['trans_dropout']).to(device)
        mamba_branch = MambaBranch(input_dim=X_raw.shape[1], num_classes=num_classes, num_blocks=p['mamba_blocks'], dropout_rate=p['mamba_dropout']).to(device)

        # Initialize DynML-Net
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
        criterion = nn.CrossEntropyLoss(weight=alpha)
        kl_loss = nn.KLDivLoss(reduction='batchmean')

        best_val_loss, counter, patience = float('inf'), 0, 5
        best_model_state = None

        for epoch in range(50):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                o1, o2, o3, o_fused = model(bx)

                loss_ce = (criterion(o1, by) + criterion(o2, by) + criterion(o3, by) + criterion(o_fused, by)) / 4
                p1, p2, p3 = F.softmax(o1, dim=1), F.softmax(o2, dim=1), F.softmax(o3, dim=1) 
                log_p1, log_p2, log_p3 = F.log_softmax(o1, dim=1), F.log_softmax(o2, dim=1), F.log_softmax(o3, dim=1)

                loss_kl = (kl_loss(log_p1, p2.detach()) + kl_loss(log_p1, p3.detach()) +
                           kl_loss(log_p2, p1.detach()) + kl_loss(log_p2, p3.detach()) +
                           kl_loss(log_p3, p1.detach()) + kl_loss(log_p3, p2.detach())) / 6
                
                total_loss = loss_ce + p['kl_weight'] * loss_kl
                total_loss.backward()
                optimizer.step()

            model.eval()
            val_l = 0.0
            with torch.no_grad():
                for bx, by in val_loader:
                    _, _, _, o_f = model(bx.to(device))
                    val_l += criterion(o_f, by.to(device)).item()
            
            avg_val_loss = val_l / len(val_loader)
            if avg_val_loss < best_val_loss:
                best_val_loss, counter = avg_val_loss, 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                counter += 1
                if counter >= patience: break

        if best_model_state: model.load_state_dict(best_model_state)
        model.eval()

        true_y = []
        fused_probs = []
        fused_preds = []
        with torch.no_grad():
            for bx, by in val_loader:
                outputs = model(bx.to(device))
                true_y.extend(by.numpy())
                prob = F.softmax(outputs[3], dim=1)
                fused_probs.extend(prob[:, 1].cpu().numpy())
                fused_preds.extend(torch.argmax(prob, dim=1).cpu().numpy())

        preds, probs = np.array(fused_preds), np.array(fused_probs)
        fold_metrics = {
            'Acc': accuracy_score(true_y, preds),
            'Sn': recall_score(true_y, preds),
            'Sp': recall_score(true_y, preds, pos_label=0),
            'MCC': matthews_corrcoef(true_y, preds),
            'F1': f1_score(true_y, preds),
            'AUC': roc_auc_score(true_y, probs)
        }
        fused_history.append(fold_metrics)
        print(f"Fold {fold} - DynML-Net AUC: {fold_metrics['AUC']:.4f}")

    # Print final average performance
    print("\n" + "="*40)
    print("=== DynML-Net (Binary Classification) Final Average Performance Metrics ===")
    print("="*40)
    metrics_names = ['Acc', 'Sn', 'Sp', 'MCC', 'F1', 'AUC']
    for m_name in metrics_names:
        vals = [h[m_name] for h in fused_history]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"{m_name}: {mean_val:.4f} ± {std_val:.4f}")
    print("="*40)

if __name__ == "__main__":
    # Please replace with your binary classification data file path
    train_binary_task("tra_pos1587_neg1589.csv")