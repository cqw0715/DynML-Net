import streamlit as st
import importlib.util
import sys
import os

original_set_page_config = st.set_page_config

def dummy_set_page_config(*args, **kwargs):
    pass

# Set main page configuration
original_set_page_config(
    page_title="Virus Prediction and Identification System Integration Platform",
    page_icon="🦠",
    layout="wide"
)

# Webpage Title
st.title("🐷 Porcine Intestinal Virus Prediction and Identification System V1.0")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3>🔬 System Introduction</h3>
    <p>This platform integrates two types of prediction and identification tasks:</p>
    <ul>
        <li><b>1. Porcine Intestinal Virus Binary Classification Model</b>: Identifies whether a protein sequence is a porcine intestinal virus</li>
        <li><b>2. Porcine Intestinal Virus Multi-classification Model</b>: Identifies 8 different common types of porcine intestinal viruses</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color: #f0f2f6; padding: 15px; border-radius: 20px; margin-bottom: 30px; width: 100%; max-width: 1200px; margin-left: auto; margin-right: auto;">
    <p><center>
    Please select a tab to switch to different porcine intestinal virus prediction task modules
    </center></p>
</div>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2 = st.tabs(["🦠 Porcine Intestinal Virus Binary Classification Model", "🦠 Porcine Intestinal Virus Multi-classification Model"])

# Function to load and run the model
def run_model(tab, model_file, model_name):
    with tab:

        st.set_page_config = dummy_set_page_config
        
        try:
            # Check if file exists
            if not os.path.exists(model_file):
                st.error(f"❌ Model file not found: {model_file}")
                st.info("Please ensure the model file is in the same directory as this application")
                return
                
            # Dynamically load module
            spec = importlib.util.spec_from_file_location(model_name, model_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[model_name] = module
            spec.loader.exec_module(module)
            
            # Run the main function of the model
            if hasattr(module, 'main'):
                module.main()
            else:
                st.error(f"❌ Model {model_name} does not define a main() function")
        except Exception as e:
            st.error(f"❌ Error loading model {model_name}: {str(e)}")
            st.exception(e)
        finally:

            st.set_page_config = original_set_page_config

# Run models in their respective tabs
run_model(tab1, "1.py", "model_pev")
run_model(tab2, "2.py", "model_multiclass")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    <p>Porcine Intestinal Virus Prediction and Identification System &copy; 2026 | School of Artificial Intelligence, Anhui Agricultural University</p>
</div>
""", unsafe_allow_html=True)