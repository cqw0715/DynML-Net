import os
import pickle
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import esm
from mamba_ssm import Mamba 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# Best Hyperparameter Configuration (from Optuna)
# ==========================================
BEST_PARAMS = {
    'lr': 0.00015659813436174998,
    'weight_decay': 5.201826253047156e-05,
    'kl_weight': 0.4,
    'cnn_dropout': 0.2,
    'trans_dropout': 0.2,
    'mamba_dropout': 0.4,
    'gate_dropout1': 0.4,
    'gate_dropout2': 0.3,
    'refine_dropout': 0.15,
    'attn_dropout': 0.2,
    'embed_dim': 128,
    'transformer_layers': 4,
    'mamba_blocks': 5,
    'batch_size': 16
}

# ==========================================
# 1. Data Preprocessing and Feature Extraction Module
# ==========================================
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    sequences = data['Sequence'].values
    labels = data['Label'].values
    if labels.min() == 1:
        labels = labels - 1
    return sequences, labels

class ESMFeatureExtractor:
    """ESM-2 650M Feature Extractor, supporting automatic GPU/CPU switching and checkpoint resumption"""
    def __init__(self):
        self.gpu_model = None
        self.gpu_batch_converter = None
        self.cpu_model = None
        self.cpu_batch_converter = None
        self.device = None
        self._initialize_models()

    def _initialize_models(self):
        try:
            if torch.cuda.is_available():
                print("Loading GPU model (ESM-2 650M)...")
                self.gpu_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.gpu_device = torch.device('cuda')
                self.gpu_model = self.gpu_model.to(self.gpu_device)
                self.gpu_batch_converter = alphabet.get_batch_converter()
                self.device = self.gpu_device
                print("GPU model loaded successfully")
            else:
                print("CUDA not available, using CPU directly")
        except Exception as e:
            print(f"GPU model loading failed: {e}")

        try:
            print("Loading CPU model as backup...")
            self.cpu_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.cpu_device = torch.device('cpu')
            self.cpu_model = self.cpu_model.to(self.cpu_device)
            self.cpu_batch_converter = alphabet.get_batch_converter()
            if self.device is None:
                self.device = self.cpu_device
            print("CPU model loaded successfully")
        except Exception as e:
            print(f"CPU model loading failed: {e}")
            raise

    def _extract_batch_features(self, batch_data, use_gpu=True):
        try:
            if use_gpu and self.gpu_model is not None:
                model, batch_converter, device = self.gpu_model, self.gpu_batch_converter, self.gpu_device
            else:
                model, batch_converter, device = self.cpu_model, self.cpu_batch_converter, self.cpu_device

            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                seq_lengths = (batch_tokens != model.alphabet.padding_idx).sum(1)

            batch_features = []
            for seq_idx in range(token_representations.size(0)):
                seq_len = seq_lengths[seq_idx].item()
                seq_rep = token_representations[seq_idx, :seq_len]
                batch_features.append(seq_rep.mean(0).cpu().numpy())

            del batch_tokens, results, token_representations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return batch_features
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and use_gpu:
                print(f"GPU out of memory, switching to CPU to retry...")
                return self._extract_batch_features(batch_data, use_gpu=False)
            raise

    def extract_features(self, sequences, cache_path=None, batch_size=1):
        progress_file = None
        if cache_path:
            progress_file = cache_path.replace('.pkl', '_progress.pkl')
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)

        if cache_path and os.path.exists(cache_path):
            print(f"Loading features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        start_idx = 0
        features = []

        if progress_file and os.path.exists(progress_file):
            try:
                with open(progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                    features = progress_data['features']
                    start_idx = progress_data['last_index'] + 1
                    print(f"Resuming extraction: Completed {len(features)}, starting from index {start_idx}")
            except Exception:
                start_idx = 0
                features = []

        if start_idx >= len(sequences):
            if cache_path:
                with open(cache_path, 'wb') as f:
                    pickle.dump(np.array(features), f)
                if os.path.exists(progress_file):
                    os.remove(progress_file)
            return np.array(features)

        print(f"Starting feature extraction (ESM-2 650M)... (Progress: {start_idx}/{len(sequences)})")
        i = start_idx
        use_gpu = (self.gpu_model is not None)

        while i < len(sequences):
            batch_end = min(i + batch_size, len(sequences))
            batch = sequences[i:batch_end]
            batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]

            try:
                batch_features = self._extract_batch_features(batch_data, use_gpu=use_gpu)
                features.extend(batch_features)

                if progress_file:
                    with open(progress_file, 'wb') as f:
                        pickle.dump({'features': features, 'last_index': batch_end - 1}, f)

                if (i // batch_size) % 10 == 0:
                    mode = "GPU" if use_gpu and self.gpu_model else "CPU"
                    print(f"{mode} Mode - Progress: {batch_end}/{len(sequences)}")
                
                i = batch_end
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and use_gpu:
                    print("💥 GPU Out of Memory, permanently switching to CPU mode")
                    use_gpu = False
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                i = batch_end
            except Exception:
                i = batch_end

        features_array = np.array(features)
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(features_array, f)
            if progress_file and os.path.exists(progress_file):
                os.remove(progress_file)
            print(f"Features saved: {cache_path}")

        print(f"Feature extraction complete! Dimensions: {features_array.shape}")
        return features_array

# ==========================================
# 2. Heterogeneous Model Branch Definitions
# ==========================================
class CNNBranch(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Unflatten(1, (1, 256)),
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveMaxPool1d(1)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        return self.classifier(feat)

class TransformerBranch(nn.Module):
    def __init__(self, input_dim=1280, d_model=256, nhead=8, num_layers=4, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return self.classifier(x)

class MambaBranch(nn.Module):
    def __init__(self, input_dim, num_classes, num_blocks=5, dropout_rate=0.2):
        super().__init__()
        self.preprocess = nn.Linear(input_dim, 256)
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=256, d_state=16, d_conv=4, expand=2) for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(1) 
        for block in self.mamba_blocks:
            x = x + block(x)
        x = self.norm(x).squeeze(1)
        x = self.dropout(x)
        return self.classifier(x)

# ==========================================
# 3. Core Fusion Network (DynML-Net)
# ==========================================
class DynML_Net(nn.Module):
    def __init__(self, input_dim, num_classes=2, embed_dim=128, 
                 gate_dropout1=0.3, gate_dropout2=0.2, refine_dropout=0.1,
                 attn_dropout=0.2):
        super().__init__()
        self.cnn = None
        self.trans = None
        self.mamba = None

        self.logits_norm = nn.LayerNorm(num_classes)
        self.feature_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Attention Block 1
        self.attn1 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=attn_dropout, batch_first=True)
        self.attn_norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm1 = nn.LayerNorm(embed_dim)

        # Attention Block 2
        self.attn2 = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=attn_dropout, batch_first=True)
        self.attn_norm2 = nn.LayerNorm(embed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm2 = nn.LayerNorm(embed_dim)

        # Gating Mechanism
        total_gate_dim = embed_dim * 3 + num_classes * 3
        self.gate = nn.Sequential(
            nn.Linear(total_gate_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(gate_dropout1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(gate_dropout2),
            nn.Linear(128, 3)
        )
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.8 + 1e-6)))

        self.refine = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LayerNorm(num_classes),
            nn.GELU(),
            nn.Dropout(refine_dropout)
        )

    def set_branches(self, cnn, trans, mamba):
        self.cnn = cnn
        self.trans = trans
        self.mamba = mamba

    def forward(self, x):
        o1, o2, o3 = self.cnn(x), self.trans(x), self.mamba(x)
        branches = torch.stack([o1, o2, o3], dim=1)

        branches_norm = self.logits_norm(branches)
        x_proj = self.feature_proj(branches_norm)

        # Block 1
        attn_out, _ = self.attn1(x_proj, x_proj, x_proj)
        x = self.attn_norm1(x_proj + attn_out)
        x = self.ffn_norm1(x + self.ffn1(x))

        # Block 2
        attn_out, _ = self.attn2(x, x, x)
        x = self.attn_norm2(x + attn_out)
        x = self.ffn_norm2(x + self.ffn2(x))

        raw_logits = branches.flatten(1)
        fused_proj = x.flatten(1)
        combined_feat = torch.cat([fused_proj, raw_logits], dim=1)
        gate_scores = self.gate(combined_feat)

        temp = F.softplus(self.log_temp) + 1e-4
        weights = F.softmax(gate_scores / temp, dim=1).unsqueeze(-1)
        o_fused = (branches * weights).sum(dim=1)
        o_fused = o_fused + self.refine(o_fused)

        return o1, o2, o3, o_fused