import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
from mamba_ssm import Mamba
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import warnings
import pickle 
warnings.filterwarnings('ignore')

# ==========================================
# 0 Best Hyperparameter Configuration (from Optuna)
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
# Core Model Architecture 
# ==========================================
class CNNBranch(nn.Module):
    def __init__(self, input_dim=1280, num_classes=8):
        super().__init__()
        dropout_rate = BEST_PARAMS['cnn_dropout']
        
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
        return self.classifier(self.net(x).flatten(1))

class TransformerBranch(nn.Module):
    def __init__(self, input_dim=1280, d_model=256, nhead=8, num_classes=8):
        super().__init__()
        dropout_rate = BEST_PARAMS['trans_dropout']
        num_layers = BEST_PARAMS['transformer_layers']
        
        self.embedding = nn.Linear(input_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dropout=dropout_rate
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        return self.classifier(self.transformer(x).squeeze(1))

class MambaBranch(nn.Module):
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        dropout_rate = BEST_PARAMS['mamba_dropout']
        num_blocks = BEST_PARAMS['mamba_blocks']
        
        self.preprocess = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(dropout_rate)
        # Use the number of blocks from configuration
        self.mamba_blocks = nn.ModuleList([
            Mamba(d_model=256, d_state=16, d_conv=4, expand=2) 
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.preprocess(x).unsqueeze(1)
        x = self.dropout(x) 
        for block in self.mamba_blocks:
            x = x + block(x)
        return self.classifier(self.norm(x).squeeze(1))

class DynML_Net(nn.Module):
    def __init__(self, input_dim, num_classes=8, embed_dim=None):
        super().__init__()

        if embed_dim is None:
            embed_dim = BEST_PARAMS['embed_dim']
        
        self.cnn = CNNBranch(input_dim, num_classes)
        self.trans = TransformerBranch(input_dim, num_classes=num_classes)
        self.mamba = MambaBranch(input_dim, num_classes)
        
        self.logits_norm = nn.LayerNorm(num_classes)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
        attn_dropout = BEST_PARAMS['attn_dropout']
        
        # Attention Block 1
        self.attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=8, 
            dropout=attn_dropout, 
            batch_first=True
        )
        self.attn_norm1 = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(attn_dropout), 
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm1 = nn.LayerNorm(embed_dim)
        
        # Attention Block 2
        self.attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=8, 
            dropout=attn_dropout, 
            batch_first=True
        )
        self.attn_norm2 = nn.LayerNorm(embed_dim)
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(attn_dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm2 = nn.LayerNorm(embed_dim)
        
        # Gate Network with specific dropouts from BEST_PARAMS
        total_gate_dim = embed_dim * 3 + num_classes * 3
        self.gate = nn.Sequential(
            nn.Linear(total_gate_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(BEST_PARAMS['gate_dropout1']),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(BEST_PARAMS['gate_dropout2']),
            nn.Linear(128, 3)
        )
        
        self.log_temp = nn.Parameter(torch.tensor(np.log(0.8 + 1e-6)))
        
        # Refine Network
        self.refine = nn.Sequential(
            nn.Linear(num_classes, num_classes),
            nn.LayerNorm(num_classes),
            nn.GELU(),
            nn.Dropout(BEST_PARAMS['refine_dropout'])
        )

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

# ==========================================
# ESM Feature Extractor
# ==========================================
class ESMFeatureExtractor:
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
                print("🚀 Attempting to load GPU model (ESM-2 650M)...")
                self.gpu_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
                self.gpu_device = torch.device('cuda')
                self.gpu_model = self.gpu_model.to(self.gpu_device)
                self.gpu_batch_converter = alphabet.get_batch_converter()
                self.device = self.gpu_device
                print("✅ GPU model loaded successfully")
        except Exception as e:
            print(f"❌ GPU model loading failed: {e}")
        
        try:
            print("🖥️ Loading CPU model as backup...")
            self.cpu_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.cpu_device = torch.device('cpu')
            self.cpu_model = self.cpu_model.to(self.cpu_device)
            self.cpu_batch_converter = alphabet.get_batch_converter()
            if self.device is None: self.device = self.cpu_device
            print("✅ CPU model loaded successfully")
        except Exception as e:
            print(f"❌ CPU model loading failed: {e}")
            raise

    def _extract_batch_features(self, batch_data, use_gpu=True):
        try:
            model = self.gpu_model if use_gpu and self.gpu_model else self.cpu_model
            batch_converter = self.gpu_batch_converter if use_gpu and self.gpu_model else self.cpu_batch_converter
            device = self.gpu_device if use_gpu and self.gpu_model else self.cpu_device
                
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                seq_lengths = (batch_tokens != model.alphabet.padding_idx).sum(1)
                batch_features = [token_representations[i, :seq_lengths[i]].mean(0).cpu().numpy() 
                                 for i in range(token_representations.size(0))]
            
            del batch_tokens, results
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return batch_features
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and use_gpu:
                return self._extract_batch_features(batch_data, use_gpu=False)
            raise

    def extract_features(self, sequences, cache_path=None, batch_size=1):
        if cache_path and os.path.exists(cache_path):
            print(f"📂 Loading features from cache: {cache_path}")
            with open(cache_path, 'rb') as f: return pickle.load(f)
        
        features = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_data = [(str(idx), seq) for idx, seq in enumerate(batch)]
            features.extend(self._extract_batch_features(batch_data))
            if (i // batch_size) % 10 == 0: print(f"📊 Progress: {min(i+batch_size, len(sequences))}/{len(sequences)}")
            
        features_array = np.array(features)
        if cache_path:
            with open(cache_path, 'wb') as f: pickle.dump(features_array, f)
        return features_array

# ==========================================
# CSV Processing Utility Functions
# ==========================================
def validate_sequence(seq):
    """Validate protein sequence"""
    seq = seq.strip().upper()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWYBX")
    invalid_chars = [c for c in seq if c not in valid_aa]
    
    if invalid_chars:
        return False, f"Invalid characters: {', '.join(set(invalid_chars))}"
    if len(seq) < 10:
        return False, "Sequence too short (minimum 10 amino acids required)"
    if len(seq) > 10000:
        return False, "Sequence too long (maximum 10,000 amino acids)"
    
    return True, ""

def validate_csv_sequences(sequences, seq_names):
    """Validate sequences in CSV, return valid indices and error messages"""
    valid_indices = []
    errors = []
    
    for i, seq in enumerate(sequences):
        is_valid, message = validate_sequence(seq)
        if is_valid:
            valid_indices.append(i)
        else:
            errors.append((seq_names[i], message))
    
    return valid_indices, errors

def parse_csv_sequences(uploaded_file):
    """
    Parse uploaded CSV file, intelligently identify sequence and name columns
    Returns: (list of sequence names, list of sequences, original DataFrame, sequence column name, name column name)
    """
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully read CSV file: {len(df)} rows, {len(df.columns)} columns")
        
        seq_col = None
        name_col = None
        possible_seq_cols = ['sequence', 'seq', 'protein_sequence', 'aa_sequence', 'peptide', 'protein']
        possible_name_cols = ['name', 'id', 'protein_id', 'identifier', 'accession', 'entry']
        
        for col in df.columns:
            if col.lower() in possible_seq_cols:
                seq_col = col
                break
        
        if seq_col is None:
            for col in df.columns:
                if 'seq' in col.lower() or 'sequence' in col.lower():
                    seq_col = col
                    break
        
        for col in df.columns:
            if col.lower() in possible_name_cols:
                name_col = col
                break
        
        if seq_col is None:
            st.error("❌ No sequence column detected. Please ensure the CSV contains one of the following column names: 'Sequence', 'Seq', 'Protein_Sequence', etc.")
            return None, None, None, None, None
        
        sequences = []
        for idx, seq in enumerate(df[seq_col]):
            if pd.isna(seq) or str(seq).strip() == "":
                st.warning(f"⚠️ Sequence at row {idx+1} is empty, will be skipped")
                sequences.append(None)
            else:
                sequences.append(str(seq).strip().upper())
        
        if name_col is not None:
            seq_names = []
            for idx, name in enumerate(df[name_col]):
                if pd.isna(name) or str(name).strip() == "":
                    seq_names.append(f"Seq_{idx+1}")
                else:
                    seq_names.append(str(name).strip())
        else:
            seq_names = [f"Seq_{i+1}" for i in range(len(sequences))]
        
        valid_indices = [i for i, seq in enumerate(sequences) if seq is not None and len(seq.strip()) > 0]
        filtered_names = [seq_names[i] for i in valid_indices]
        filtered_seqs = [sequences[i] for i in valid_indices]
        
        return filtered_names, filtered_seqs, df, seq_col, name_col
    
    except Exception as e:
        st.error(f"❌ Failed to parse CSV file: {str(e)}")
        return None, None, None, None, None

# ==========================================
# Model Loading
# ==========================================
@st.cache_resource
def load_model_and_scaler():
    """Load model and scaler"""
    import numpy as np
    import numpy.core.multiarray
    import sklearn.preprocessing._data
    
    safe_globals = [
        np.core.multiarray.scalar,
        np.dtype,
        np.ndarray,
        StandardScaler,
        sklearn.preprocessing._data.StandardScaler
    ]
    
    for obj in safe_globals:
        try:
            torch.serialization.add_safe_globals([obj])
        except Exception:
            pass
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = "best_multiclass_model.pth"
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found!")
        st.stop()
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Virus name mapping
    virus_map = {
        0: "PEDV",
        1: "TGEV",
        2: "PoRV",
        3: "PDCoV",
        4: "PSV",
        5: "PAstV",
        6: "PoNoV",
        7: "SADS-Cov"
    }
    
    # Instantiate model, input dimension fixed to 1280 (ESM-2 650M layer 33 output dim), 8 classes
    model = DynML_Net(input_dim=1280, num_classes=8).to(device)
    
    # Load state dictionary
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        st.error(f"Model weight loading failed, possibly due to mismatch between hyperparameter config and weight file: {str(e)}")
        st.info("Please confirm that 'best_multiclass_model.pth' was trained using the current BEST_PARAMS configuration.")
        st.stop()
    
    model.eval()
    
    scaler = checkpoint['scaler']
    
    return model, scaler, virus_map, device

# ==========================================
# Prediction and Visualization Functions
# ==========================================
def predict(model, scaler, sequences, device, virus_map):
    """Perform prediction"""
    extractor = ESMFeatureExtractor()
    st.info("🧬 Extracting ESM-2 features, please wait...")
    features = extractor.extract_features(sequences)
    
    st.info("⚖️ Standardizing features...")
    scaled_features = scaler.transform(features)
    
    st.info("🧠 Performing prediction...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for i in range(len(scaled_features)):
            x = torch.FloatTensor(scaled_features[i:i+1]).to(device)
            _, _, _, fused_output = model(x)
            probs = F.softmax(fused_output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            results.append({
                'probabilities': probs,
                'predicted_class': virus_map[pred_idx],
                'confidence': probs[pred_idx]
            })
    
    return results

def create_probability_chart(probs, virus_map, title="Prediction Probability Distribution"):
    """Create probability distribution chart with English labels to avoid encoding issues"""
    fig, ax = plt.subplots(figsize=(10, 5))
    # Arrange virus names by index order
    viruses = [virus_map[i] for i in range(len(probs))]
    colors = ['red' if i == np.argmax(probs) else 'steelblue' for i in range(len(probs))]
    
    bars = ax.bar(viruses, probs, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xlabel('Virus Class', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.02,
            f'{prob:.2f}',
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig

# ==========================================
# Streamlit Application Main Function
# ==========================================
def main():
    st.set_page_config(
        page_title="Porcine Intestinal Virus Protein Classifier", 
        page_icon="🦠", 
        layout="wide"
    )
    
    st.title("🦠 Porcine Intestinal Virus Protein Sequence Multi-Classification System") 
    st.markdown("""
    This system uses the DynML_Net model to classify **porcine intestinal virus** protein sequences, supporting the identification of **8** common porcine intestinal virus subtypes.
    """)
    
    
    with st.spinner("⏳ Loading model and related components..."):
        try:
            model, scaler, virus_map, device = load_model_and_scaler()
        except Exception as e:
            st.error(f"Critical error occurred while loading model: {str(e)}")
            st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    tab1, tab2, tab3 = st.tabs(["🔬 Single Sequence Prediction", "📁 Batch Prediction (CSV)", "ℹ️ About Virus Types"])
    
    with tab1:
        st.header("Single Sequence Prediction")
        sequence_input = st.text_area(
            "Enter Protein Sequence (Amino Acid Sequence)",
            height=150,
            placeholder="e.g., MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHF..."
        )
        
        if st.button("🚀 Predict", type="primary", use_container_width=True):
            if not sequence_input.strip():
                st.warning("⚠️ Please enter a valid protein sequence")
            else:
                is_valid, message = validate_sequence(sequence_input)
                if not is_valid:
                    st.error(f"❌ Invalid sequence: {message}")
                else:
                    with st.spinner("⏳ Processing..."):
                        start_time = time.time()
                        results = predict(model, scaler, [sequence_input], device, virus_map)
                        elapsed_time = time.time() - start_time
                    
                    res = results[0]
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.metric(
                            "Predicted Virus Family", 
                            res['predicted_class'],
                            delta=f"{res['confidence']:.1%} Confidence"
                        )
                        st.caption(f"⏱️ Processing Time: {elapsed_time:.2f} seconds")
                    
                    with col2:
                        fig = create_probability_chart(
                            res['probabilities'], 
                            virus_map,
                            f"Probability Distribution (Conf: {res['confidence']:.1%})"
                        )
                        st.pyplot(fig)
    
    with tab2:
        st.header("Batch Prediction (CSV Format)")
        uploaded_file = st.file_uploader(
            "📤 Upload CSV File (must contain 'Sequence' column)", 
            type=["csv"]
        )
        
        if uploaded_file is not None:
            seq_names, sequences, raw_df, seq_col, name_col = parse_csv_sequences(uploaded_file)
            
            if sequences is None or len(sequences) == 0:
                st.stop()
            
            if st.button("🚀 Start Batch Prediction", type="primary", use_container_width=True):
                valid_indices, errors = validate_csv_sequences(sequences, seq_names)
                
                if errors:
                    st.error(f"❌ Found {len(errors)} invalid sequences")
                    st.error(f"❌ Details: {str(errors)}")
                    st.stop()
                
                with st.spinner(f"⏳ Predicting {len(valid_indices)} sequences..."):
                    valid_seqs = [sequences[i] for i in valid_indices]
                    valid_names = [seq_names[i] for i in valid_indices]
                    results = predict(model, scaler, valid_seqs, device, virus_map)
                
                results_data = []
                for i, (name, res) in enumerate(zip(valid_names, results)):
                    row = {
                        'Sequence name': name,
                        'Predicted virus': res['predicted_class'],
                        'Confidence level': res['confidence']
                    }
                    for j in range(8):
                        row[virus_map[j]] = res['probabilities'][j] 
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Prediction Results (CSV)",
                    data=csv,
                    file_name="virus_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with tab3:
        st.header("ℹ️ About Virus Types")
        st.markdown("""
        ### 🐷 Supported 8 Common Porcine Intestinal Virus Types
        - **PEDV**: Porcine Epidemic Diarrhea Virus
        - **TGEV**: Transmissible Gastroenteritis Virus
        - **PoRV**: Porcine Rotavirus
        - **PDCoV**: Porcine Delta Coronavirus
        - **PSV**: Porcine Sapelo virus
        - **PAstV**: Porcine Astrovirus
        - **PoNoV**: Porcine Norovirus
        - **SADS-Cov**: Swine Acute Diarrhea Syndrome Coronavirus
        """)

if __name__ == "__main__":
    main()