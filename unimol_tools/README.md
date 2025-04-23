# Modified UniMol-Tools for AdaQSPR

Original [UniMol-Tools Project](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_tools).

## install
 - Notice: [Uni-Core](https://github.com/dptech-corp/Uni-Core) is needed, please install it first. Current Uni-Core requires torch>=2.0.0 by default, if you want to install other version, please check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
```python
## unicore and other dependencies installation
pip install -r requirements.txt
## clone repo
git clone https://github.com/dptech-corp/Uni-Mol.git
cd Uni-Mol/unimol_tools/unimol_tools

## download pretrained weights
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mof_pre_no_h_CORE_MAP_20230505.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mp_all_h_230313.pt
wget https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/oled_pre_no_h_230101.pt

# TODO: Add weights for AdaQSPR

mkdir -p weights
mv *.pt weights/

## install
cd ..
python setup.py install
```
