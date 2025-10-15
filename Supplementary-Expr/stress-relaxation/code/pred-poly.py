from unimol_tools import MolTrain2, MolPredict2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import freeze_support
import joblib
from rdkit import Chem

def main():
    # read from excel
    df = pd.read_excel('data/PredData/t_train.xlsx', sheet_name='Sheet1')  #change the file name #'data/PredData/candidate_disulfide_vitrimers.xlsx'

    cate = 't'
    tcol = 'logt*'
    gcol = 'catalyst-0without1added'

    print(f'Data Length: {len(df)}')

    # 定义一个函数，将smiles转换为标准值
    def Standardized_SMILES(value):
        if pd.notna(value):
            #print(value)
            smiles = value
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:  # 确保SMILES有效
                new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            else:
                new_smiles = smiles  # 如果无效，保留原SMILES
            value = new_smiles
            #print(value)
            return value
        else:
            return value

    data_col_names = ['monomer1', 'monomer2', 'monomer3', 'monomer4', 'curing_agent_1', 'curing_agent_2', 'curing_agent_3', 'curing_agent_4']
    for col in data_col_names:
        smiles_i = df[col]
        smiles_i = smiles_i.apply(Standardized_SMILES)
        df[col] = smiles_i
        #print(df[col])
        df[col].fillna('C', inplace=True)

    mol_col_names = [d for d in df.keys() if 'ratios_mol' in d]
    assert len(mol_col_names) == len(data_col_names)
    for col in mol_col_names:
        df[col].fillna(0, inplace=True)

    jituan_col_names = [d for d in df.keys() if 'functionality' in d]
    assert len(jituan_col_names) == len(data_col_names)
    for col in jituan_col_names:
        df[col].fillna(0, inplace=True)

    qiangji_col_names = [d for d in df.keys() if 'num_of_hydroxyl' in d]
    assert len(qiangji_col_names) == len(data_col_names)
    for col in qiangji_col_names:
        df[col].fillna(0, inplace=True)

    data = []

    for idx, dcol in enumerate(data_col_names):
        smiles = df[dcol]
        mask_feat = np.array([v != 'C' for v in smiles]).astype(int)
        global_feat = df[gcol]
        mol_feat = df[mol_col_names[idx]]
        jituan_feat = df[jituan_col_names[idx]]
        qiangji_feat = df[qiangji_col_names[idx]]

        feat = np.stack([mask_feat, global_feat, mol_feat, jituan_feat, qiangji_feat], axis=1)
        data.append({
            'SMILES': smiles,
            'target': df[tcol],
            'feat': feat,
        })

    clf = MolPredict2(load_model=f'./exp/exp-poly-{cate}')
    predict = clf.predict(data)
    sc = joblib.load(f'./exp/exp-poly-{cate}/target_scaler.ss')
    p_inverse = sc.inverse_transform(predict)
    print(p_inverse.max())
    df[f'{tcol}_pred'] = np.array(list(p_inverse))
    df.to_excel(f'data/PredData/prediction_train_{cate}.xlsx', index=False)   #change the file name
    df.keys()


if __name__ == "__main__":
    freeze_support()  #add this if you using windows system
    main()
