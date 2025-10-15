# AdaQSPR: A prediction model for epoxy resin comprehensive properties

## Project Introduction

AdaQSPR is a transfer learning-based 3D molecular representation learning framework for predicting the comprehensive properties of epoxy resins. This framework can efficiently predict key performance indicators such as glass transition temperature (Tg), initial thermal decomposition temperature (Td5%), tensile strength (TS), relative permittivity (ε), dielectric loss (tanδ), and electrical breakdown strength (Eb).

## Project Structure

```
.
├── unimol_tools_task_specific/  # Core model code directory
│   ├── models/                 # Model definitions
│   ├── utils/                  # Utility functions
│   ├── tasks/                  # Task definitions
│   ├── config/                 # Configuration files
│   ├── data/                   # Data processing
│   ├── train.py                # Training script
│   ├── predict.py              # Prediction script
├── data/                       # Dataset directory
│   ├── results-candidate/      # Results of prediction of candidate materials data
│   ├── train/                  # Training data
│   │   ├── Tg.xlsx             # Glass transition temperature data
│   │   ├── Td.xlsx             # Thermal decomposition temperature data
│   │   ├── TS.xlsx             # Tensile strength data
│   │   ├── epsilon.xlsx        # Relative permittivity data
│   │   ├── tan_delta.xlsx      # Dielectric loss data
│   │   ├── Eb.xlsx             # Electrical breakdown strength data
│   │   └── organic_molecular_structure-property_dataset.xlsx # Organic molecular structure-property dataset
│   └── candidate/              # Candidate materials data
├── Tg/                         # Tg prediction related files
├── Td/                         # Td prediction related files
├── TS/                         # TS prediction related files
├── e/                          # ε prediction related files
├── tan/                        # tanδ prediction related files
├── Eb/                         # Eb prediction related files
└── paper.txt                   # Research paper
```

## Research Background

Vitrimer is a type of covalent adaptable network that can change its topology through exchange reactions, offering excellent properties such as reprocessability, repairability, and degradability. However, compared to traditional thermosetting materials, the thermal, mechanical, and electrical properties of vitrimers may be compromised. This research aims to construct a Quantitative Structure-Property Relationship (QSPR) model to guide the design of disulfide epoxy vitrimers with both excellent comprehensive properties and eco-friendly characteristics.

## Datasets

This project contains two datasets:
1. Organic molecular structure-property dataset: Contains property data of over 100,000 organic molecules
2. Epoxy resin structure-property dataset: Contains experimental data of over 1,700 epoxy resin macroscopic properties

## Model Framework

The AdaQSPR model is based on transfer learning and domain-specific adaptation design, mainly consisting of two parts:
1. Domain-specific Adaptation: Based on the Uni-Mol framework, pretrained on the organic molecular structure-property dataset
2. Task-specific Adaptation: Fine-tuned on the epoxy resin structure-property dataset to predict six macroscopic properties of epoxy resins

## Application Case

The research team used the AdaQSPR model to screen 216 different dynamic disulfide epoxy vitrimers and ultimately selected 5 promising candidates for experimental verification. The experimental results proved that these 5 disulfide epoxy vitrimers have excellent comprehensive properties, including:
- Glass transition temperature Tg > 146°C
- Electrical breakdown strength Eb > 34.8 kV/mm
- Four of the materials have tensile strength TS ≥ 60 MPa
- Good repairability and degradability

## Weights 

We open-source the weights of the AdaQSPR model for reproducibility.

Link: https://disk.pku.edu.cn/link/AA547E8FBF09114E7981DA845D7DAD5CCB
Expire time: 2035-08-01 15:53

## Usage/Installation

- **Step1: Install the [UniMol framework](https://github.com/deepmodeling/Uni-Mol)**
```
pip install unimol==1.0.0
```

- **Step2: Install the UniMol-Tools**
```
cd unimol_tools
```
Then, please follow the instructions in the [UniMol-Tools README](unimol_tools/README.md) to install the UniMol-Tools.

- **Step3(Optional): Train the AdaQSPR model**
```
cd Tg # Take Tg as an example
python trainpoly.py
```

- **Step4(Optional): Predict the comprehensive properties of epoxy resins**
```
cd Tg # Take Tg as an example
python pred_poly.py
```

## Acknowledgements

We would like to thank the following projects for their contributions to this work:
- [Uni-Mol](https://github.com/deepmodeling/Uni-Mol)




