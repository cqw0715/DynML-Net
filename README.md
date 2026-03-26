# DynML-Net: A porcine enteric viruses identification network based on protein language models and a dynamic heterogeneous multi-branch architecture

## Dataset
### Data source (PEV)
NCBI: https://www.ncbi.nlm.nih.gov
<br>UniProt: https://www.uniprot.org
<br>VirusDIP: https://db.cngb.org/virusdip

### Data source (Non_PEV)
[1] NCBI, UniProt, VirusDIP<br>
[2] Yuxuan, P., Lantian, Y., JhihHua, J., Zhuo, W., & TzongYi, L. (2021). AVPIden: a new scheme for identification and functional prediction of antiviral peptides based on machine learning approaches. Briefings in bioinformatics, 22(6). doi:10.1093/bib/bbab263
<br>github: https://github.com/BiOmicsLab/AVPIden.git

## Model execution
### Core Dependencies
The main environment for the model operation is as follows：<br>
| Library          | Version  |
|------------------|----------|
| transformers     | 4.40.0   |
| torch            | 2.4.0    |
| fair-esm         | 2.0.0    |
| numpy            | 2.4.3    |<br>

I have uploaded the complete environment for running the model locally to the "0-requirements.txt" file. You can refer to this file for installation.<br>
The corresponding model runtime environment can be installed directly via pip in the terminal.
```bash
pip install transformers==4.40.0
pip install torch==2.4.0
pip install fair-esm==2.0.0
pip install numpy==2.4.3
pip install imblearn
pip install pandas
pip install xxx
```
### Model Training
Please use a single CSV file to store the dataset for model execution. The file must include the following two columns:<br>
1. 'Sequence': for storing protein sequences.<br>
2. 'Label': for recording the class label of the protein sequence.<br>

Then, input your dataset name into the code：
```bash
    train_binary_task("tra_pos1587_neg1589.csv")     ##153
    train_multiclass_task("90%_1587.csv")            ##233
```
Finally, run the following command and wait for the model execution to complete.
```bash
    python run_binary.py
    python run_multiclass.py
```

## System deployment
Before deploying the system, please download all files in the "3-System deployment" folder. Installation Requirements for the Operating Environment Corresponding to the System：
```bash
pip install -r requirements.txt
```
After installation, you can run the following command:
```bash
streamlit run main.py
```
If you are redirected to the following page, the deployment is successful!!!
<img width="1918" height="891" alt="0-主页面" src="https://github.com/user-attachments/assets/13ca0e34-f1e3-4b8a-961c-6fc35647924f" />
