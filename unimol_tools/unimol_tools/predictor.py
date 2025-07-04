# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import pandas as pd
import numpy as np
import torch
import argparse
import joblib
from torch.utils.data import DataLoader, Dataset
from .data import MOFReader, DataHub
from .models import UniMolModel
from .tasks import Trainer
from rdkit import Chem


class MolDataset(Dataset):
    """
    A :class:`MolDataset` class is responsible for interface of molecular dataset.
    """
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
    

class UniMolRepr(object):
    """
    A :class:`UniMolRepr` class is responsible for interface of molecular representation by unimol
    """
    def __init__(self, data_type='molecule', 
                 remove_hs=False, 
                 use_gpu=True):
        """
        Initialize a :class:`UniMolRepr` class.

        :param data_type: str, default='molecule', currently support molecule, oled.
        :param remove_hs: bool, default=False, whether to remove hydrogens in molecular.
        :param use_gpu: bool, default=True, whether to use gpu.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = UniMolModel(output_dim=1, data_type=data_type, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.params = {'data_type': data_type, 'remove_hs': remove_hs}
   
    def get_repr(self, data=None, return_atomic_reprs=False):
        """
        Get molecular representation by unimol.

        :param data: str, dict or list, default=None, input data for unimol. 

            - str: smiles string or path to a smiles file.

            - dict: custom conformers, should take atoms and coordinates as input.

            - list: list of smiles strings.

        :param return_atomic_reprs: bool, default=False, whether to return atomic representations.

        :return: dict of molecular representation.
        """

        if isinstance(data, str):
            # single smiles string.
            data = [data]
        elif isinstance(data, dict):
            # custom conformers, should take atoms and coordinates as input.
            assert 'atoms' in data and 'coordinates' in data
        elif isinstance(data, list):
            # list of smiles strings.
            assert isinstance(data[-1], str)
        else:
            raise ValueError('Unknown data type: {}'.format(type(data)))
        datahub = DataHub(data=data, 
                         task='repr', 
                         is_train=False, 
                         **self.params,
                        )
        dataset = MolDataset(datahub.data['unimol_input'])
        self.trainer = Trainer(task='repr')
        repr_output = self.trainer.inference(self.model, 
                                             return_repr=True, 
                                             return_atomic_reprs=return_atomic_reprs, 
                                             dataset=dataset)
        return repr_output



scaler = {'CoRE_MAP': [1.318703908155812, 1.657051374039756,'log1p_standardization']}
class MOFDataset(Dataset):
    def __init__(self, mof_data, aux_data):
        self.mof_data = mof_data
        self.aux_data = aux_data

    def __len__(self):
        return len(self.aux_data)

    def __getitem__(self, idx):
        d = copy.deepcopy(self.mof_data)
        for k in self.aux_data[idx]:
            d[k] = self.aux_data[idx][k]
        return d


class MOFPredictor(object):
    def __init__(self, use_gpu=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = UniMolModel(output_dim=1, data_type='mof').to(self.device).half()
        self.model.eval()

    def single_predict(self, cif_path='1.cif', gas='CH4', pressure=10000, temperature=100):
        d = MOFReader().read_with_gas(cif_path=cif_path, gas=gas)
        d['pressure'] = np.log10(pressure)
        d['temperature'] = temperature

        dd = self.model.batch_collate_fn_mof([d])
        for k in dd:
            dd[k] = dd[k].to(self.device)

        with torch.no_grad():
            predict = self.model(**dd).detach().cpu().numpy()[0][0]
            predict = np.expm1(scaler['CoRE_MAP'][0] + scaler['CoRE_MAP'][1] * predict)
            predict = np.clip(predict, 0, None)
        return predict

    def predict_grid(self, cif_path='1.cif', gas='CH4', temperature_list=[168,298], pressure_bins=100):
        mof = MOFReader().read_with_gas(cif_path=cif_path, gas=gas)
        dd = []
        pressure_list = np.logspace(0, 5.0, pressure_bins)
        for temperature in temperature_list:
            for pressure in pressure_list:
                dd.append({'temperature':temperature, 'pressure':np.log10(pressure)})
        dataloader = DataLoader(dataset=MOFDataset(mof, dd),
                                batch_size=8,
                                shuffle=False,
                                collate_fn=self.model.batch_collate_fn_mof,
                                drop_last=False)

        predict_list = []
        with torch.no_grad():
            for dd in dataloader:
                for k in dd:
                    dd[k] = dd[k].to(self.device)
                _predict = self.model(**dd).detach().cpu().numpy()[:,0]
                _predict = np.expm1(scaler['CoRE_MAP'][0] + scaler['CoRE_MAP'][1] * _predict)
                _predict = np.clip(_predict, 0, None)
                predict_list.extend(list(_predict))

        idx = pd.MultiIndex.from_product([temperature_list, pressure_list], names=['temperature','pressure'])
        grid_df = pd.DataFrame({'absorp_prediction': predict_list}, index=idx).reset_index()
        grid_df['gas'] = gas
        grid_df['mof'] = cif_path.split('/')[-1]
        return grid_df