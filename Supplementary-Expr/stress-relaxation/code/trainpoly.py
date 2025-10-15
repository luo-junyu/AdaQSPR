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
        if mol is not None:  # 确保SMILES有效
            new_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            return new_smiles
        else:
            return raw_smiles

    cate = 't'
    tcol = 'logt*'
    gcol = 'catalyst-0without1added'

    df = pd.read_excel(f'data/{cate}.xlsx', sheet_name='Sheet1')
    print(f'Data Length: {len(df)}')

    data_col_names = ['monomer1', 'monomer2', 'monomer3', 'monomer4', 'curing_agent_1', 'curing_agent_2', 'curing_agent_3', 'curing_agent_4']
    for col in data_col_names:
        df[col].fillna('C', inplace=True) #缺失值替换为C
        df[col] = df[col].apply(norm_smiles) #标准化SMILES

    mol_col_names = [d for d in df.keys() if 'ratios_mol' in d]
    assert len(mol_col_names) == len(data_col_names) #检查 mol_col_names 和 data_col_names 的长度是否相等
    for col in mol_col_names:
        df[col].fillna(0, inplace=True) #缺失值填充为0

    jituan_col_names = [d for d in df.keys() if 'functionality' in d]
    assert len(jituan_col_names) == len(data_col_names)
    for col in jituan_col_names:
        df[col].fillna(0, inplace=True)

    qiangji_col_names = [d for d in df.keys() if 'num_of_hydroxyl' in d]
    assert len(qiangji_col_names) == len(data_col_names)
    for col in qiangji_col_names:
        df[col].fillna(0, inplace=True)

    data = []

    for idx, dcol in enumerate(data_col_names): #使用 enumerate 函数同时获取 data_col_names 列表中元素的索引 idx 和元素值 dcol。enumerate 函数会将可迭代对象（这里是 data_col_names 列表）组合为一个索引序列
        smiles = df[dcol] #从数据框 df 中选取列名为 dcol 的列数据，并将其赋值给变量 smiles；pandas.Series 是 Pandas 库中一个重要的数据结构，它是一维带标签的数组，可以存储任意数据类型（如整数、浮点数、字符串、Python 对象等）。
        mask_feat = np.array([v != 'C' for v in smiles]).astype(int) #这行代码做了两件事。首先，使用列表推导式 [v != 'C' for v in smiles] 遍历 smiles 中的每个元素 v，判断其是否不等于 'C'，并将结果存储为布尔值列表。然后，使用 np.array() 将这个布尔值列表转换为 numpy 数组，并使用 .astype(int) 将布尔值数组转换为整数数组（True 转换为 1，False 转换为 0）。
        global_feat = df[gcol]
        mol_feat = df[mol_col_names[idx]]
        jituan_feat = df[jituan_col_names[idx]]
        qiangji_feat = df[qiangji_col_names[idx]]

        feat = np.stack([mask_feat, global_feat, mol_feat, jituan_feat, qiangji_feat], axis=1) #使用 np.stack() 函数将 mask_feat、global_feat、mol_feat、jituan_feat 和 qiangji_feat 这几个数组沿着 axis=1（即列方向）进行堆叠，生成一个新的二维 numpy 数组 feat。
        data.append({
            'SMILES': smiles,
            'target': df[tcol],
            'feat': feat,
        })
        #创建一个字典，包含三个键值对：'SMILES' 对应之前选取的 smiles 数据，'target' 对应从数据框 df 中选取的列名为 tcol 的数据，'feat' 对应前面生成的特征矩阵 feat。然后将这个字典添加到列表 data 中。

    clf = MolTrain2(task='regression',
                    data_type='molecule',
                    epochs=100,
                    learning_rate=0.00008,
                    batch_size=16,
                    early_stopping=20,
                    metrics='mae',
                    split='random',
                    save_path=f'./exp/exp-poly-{cate}',
                    )

    clf.fit(data)



if __name__ == "__main__":
    freeze_support()  # add this if you using windows system
    main()