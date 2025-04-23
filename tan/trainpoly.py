import os
from rdkit import Chem
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from unimol_tools import MolTrain2
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

def main():
    # read from excel
    def norm_smiles(raw_smiles):
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is not None:
            new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return new_smiles
        else:
            return raw_smiles

    cate = 'tan_delta'
    tcol = 'log_tan_delta'
    gcol = 'catalyst-0without1added'

    df = pd.read_excel(f'../data/train/{cate}.xlsx', sheet_name='Sheet1')
    print(f'Data Length: {len(df)}')

    data_col_names = ['monomer1', 'monomer2', 'monomer3', 'monomer4', 'curing_agent_1', 'curing_agent_2', 'curing_agent_3', 'curing_agent_4']
    for col in data_col_names:
        df[col].fillna('C', inplace=True)
        df[col] = df[col].apply(norm_smiles)

    mol_col_names = [d for d in df.keys() if 'ratios_mol%' in d]
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

    clf = MolTrain2(task='regression',
                    data_type='molecule',
                    epochs=80,
                    learning_rate=0.0001,
                    batch_size=8,
                    early_stopping=20,
                    metrics='mae',
                    split='random',
                    save_path=f'./exp/exp-poly-{cate}',
                    )

    clf.fit(data)



if __name__ == "__main__":
    ## add this if you using windows system
    # freeze_support() 
    main()