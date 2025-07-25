# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torch.utils.data import Dataset
import numpy as np
from ..utils import logger
from .unimol import UniMolModel
from .loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss, MAEwithNan


NNMODEL_REGISTER = {
    'unimolv1': UniMolModel,
}

LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': MAEwithNan,
}
ACTIVATION_FN = {
    # predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    # softmax is used for multiclass classification
    'multiclass': lambda x: F.softmax(x, dim=-1),
    'regression': lambda x: x,
    # sigmoid is used for multilabel classification
    'multilabel_classification': lambda x: F.sigmoid(x),
    # no activation function is used for multilabel regression
    'multilabel_regression': lambda x: x,
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}


class NNModel2(object):
    """A :class:`NNModel` class is responsible for initializing the model"""    
    def __init__(self, data, trainer, **params):
        """
        Initializes the neural network model with the given data and parameters.

        :param data: (dict) Contains the dataset information, including features and target scaling.
        :param trainer: (object) An instance of a training class, responsible for managing training processes.
        :param params: Various additional parameters used for model configuration.

        The model is configured based on the task type and specific parameters provided.
        """


        # Target Data
        self.data1 = data[0]
        self.num_classes = self.data1['num_classes']
        self.target_scaler = self.data1['target_scaler']

        # Features
        self.data_feats = [d['unimol_input'] for d in data]

        self.data_mask = [d['feat'] for d in data]

        # self.data_mask = [[v != 'C' for v in d['smiles']] for d in data]
        # self.features1 = data1['unimol_input']
        # self.features2 = data2['unimol_input']
        # self.num_classes = self.data1['num_classes']
        # self.target_scaler = self.data1['target_scaler']

        self.model_name = params.get('model_name', 'unimolv1')
        self.data_type = params.get('data_type', 'molecule')
        self.loss_key = params.get('loss_key', None)
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params['task']
        if self.task in OUTPUT_DIM:
            self.model_params['output_dim'] = OUTPUT_DIM[self.task]
        elif self.task == 'multiclass':
            self.model_params['output_dim'] = self.data['multiclass_cnt']
        else:
            self.model_params['output_dim'] = self.num_classes
        self.model_params['device'] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if self.task == 'multilabel_classification':
            if self.loss_key is None:
                self.loss_key = 'focal'
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.task]
        self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(**self.model_params)

    def _init_model(self, model_name, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration.

        :return: An instance of the specified neural network model.
        :raises ValueError: If the model name is not recognized.
        """
        freeze_layers = params.get('freeze_layers', None)
        freeze_layers_reversed = params.get('freeze_layers_reversed', False)
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
            if isinstance(freeze_layers, str):
                freeze_layers = freeze_layers.replace(' ', '').split(',')
            if isinstance(freeze_layers, list):
                for layer_name, layer_param in model.named_parameters():
                    should_freeze = any(layer_name.startswith(freeze_layer) for freeze_layer in freeze_layers)
                    layer_param.requires_grad = not (freeze_layers_reversed ^ should_freeze)
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model

    def collect_data(self, X, y, idx):
        """
        Collects and formats the training or validation data.

        :param X: (np.ndarray or dict) The input features, either as a numpy array or a dictionary of tensors.
        :param y: (np.ndarray) The target values as a numpy array.
        :param idx: Indices to select the specific data samples.

        :return: A tuple containing processed input data and target values.
        :raises ValueError: If X is neither a numpy array nor a dictionary.
        """
        assert isinstance(y, np.ndarray), 'y must be numpy array'
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError('X must be numpy array or dict')

    def run(self):
        """
        Executes the training process of the model. This involves data preparation, 
        model training, validation, and computing metrics for each fold in cross-validation.
        """
        logger.info("start training Uni-Mol:{}".format(self.model_name))

        # self.data1_feats = [d['unimol_input'] for d in data1]
        # self.data2_feats = [d['unimol_input'] for d in data2]
        # self.data1_mask = [d['smiles'] != 'C' for d in data1]
        # self.data2_mask = [d['smiles'] != 'C' for d in data2]

        Xs = np.asarray(self.data_feats).T
        Xmasks = np.asarray(self.data_mask).transpose(1, 0, 2)
        y = np.asarray(self.data1['target'])
        group = np.asarray(self.data1['group']) if self.data1['group'] is not None else None
        if self.task == 'classification':
            y_pred = np.zeros_like(
                y.reshape(y.shape[0], self.num_classes)).astype(float)
        else:
            y_pred = np.zeros((y.shape[0], self.model_params['output_dim']))
        for fold, (tr_idx, te_idx) in enumerate(self.splitter.split(Xs, y, group)):
            X_train, Xm_train, y_train = Xs[tr_idx], Xmasks[tr_idx], y[tr_idx]
            X_valid, Xm_valid, y_valid = Xs[te_idx], Xmasks[te_idx], y[te_idx]

            # X_train_1, X_train_2, y_train = X1[tr_idx], X2[tr_idx], y[tr_idx]
            # X_valid_1, X_valid_2, y_valid = X1[te_idx], X2[te_idx], y[te_idx]

            # traindataset = NNDataset(X_train_1, y_train)
            # validdataset = NNDataset(X_valid_1, y_valid)

            traindataset = MyDataset(X_train, Xm_train, y_train)
            validdataset = MyDataset(X_valid, Xm_valid, y_valid)
            if fold > 0:
                # need to initalize model for next fold training
                self.model = self._init_model(**self.model_params)
            _y_pred = self.trainer.fit_predict(
                self.model, traindataset, validdataset, self.loss_func, self.activation_fn, self.save_path, fold, self.target_scaler)
            y_pred[te_idx] = _y_pred

            if 'multiclass_cnt' in self.data1:
                label_cnt = self.data1['multiclass_cnt']
            else:
                label_cnt = None

            logger.info("fold {0}, result {1}".format(
                    fold,
                    self.metrics.cal_metric(
                        self.data1['target_scaler'].inverse_transform(y_valid),
                        self.data1['target_scaler'].inverse_transform(_y_pred),
                        label_cnt=label_cnt
                    )
                )
            )

        self.cv['pred'] = y_pred
        self.cv['metric'] = self.metrics.cal_metric(self.data1['target_scaler'].inverse_transform(
            y), self.data1['target_scaler'].inverse_transform(self.cv['pred']))
        self.dump(self.cv['pred'], self.save_path, 'cv.data')
        self.dump(self.cv['metric'], self.save_path, 'metric.result')
        logger.info("Uni-Mol metrics score: \n{}".format(self.cv['metric']))
        logger.info("Uni-Mol & Metric result saved!")

    def dump(self, data, dir, name):
        """
        Saves the specified data to a file.

        :param data: The data to be saved.
        :param dir: (str) The directory where the data will be saved.
        :param name: (str) The name of the file to save the data.
        """
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)

    def evaluate(self, trainer=None,  checkpoints_path=None):
        """
        Evaluates the model by making predictions on the test set and averaging the results.

        :param trainer: An optional trainer instance to use for prediction.
        :param checkpoints_path: (str) The path to the saved model checkpoints.
        """
        logger.info("start predict NNModel:{}".format(self.model_name))

        Xs = np.asarray(self.data_feats).T
        Xmasks = np.asarray(self.data_mask).transpose(1, 0, 2)
        # y = np.asarray(self.data1['target'])

        # X1 = np.asarray(self.features1)
        # X2 = np.asarray(self.features2)
        # # X = np.concatenate((X1, X2), axis=1)
        # y = np.asarray(self.data1['target'])

        # testdataset = NNDataset(self.features, np.asarray(self.data1['target']))
        testdataset = MyDataset(Xs, Xmasks, np.asarray(self.data1['target']))
        
        for fold in range(self.splitter.n_splits):
            model_path = os.path.join(checkpoints_path, f'model_{fold}.pth')
            self.model.load_state_dict(torch.load(
                model_path, map_location=self.trainer.device)['model_state_dict'])
            _y_pred, _, __ = trainer.predict(self.model, testdataset, self.loss_func, self.activation_fn,
                                             self.save_path, fold, self.target_scaler, epoch=1, load_model=True)
            if fold == 0:
                y_pred = np.zeros_like(_y_pred)
            y_pred += _y_pred
        y_pred /= self.splitter.n_splits
        self.cv['test_pred'] = y_pred

    def count_parameters(self, model):
        """
        Counts the number of trainable parameters in the model.

        :param model: The model whose parameters are to be counted.

        :return: (int) The number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def NNDataset(data, label=None):
    """
    Creates a dataset suitable for use with PyTorch models.

    :param data: The input data.
    :param label: Optional labels corresponding to the input data.

    :return: An instance of TorchDataset.
    """
    return TorchDataset(data, label)


class TorchDataset(Dataset):
    """
    A custom dataset class for PyTorch that handles data and labels. This class is compatible with PyTorch's Dataset interface
    and can be used with a DataLoader for efficient batch processing. It's designed to work with both numpy arrays and PyTorch tensors. """
    def __init__(self, data, label=None):
        """
        Initializes the dataset with data and labels.

        :param data: The input data.
        :param label: The target labels for the input data.
        """
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        """
        Retrieves the data item and its corresponding label at the specified index.

        :param idx: (int) The index of the data item to retrieve.

        :return: A tuple containing the data item and its label.
        """
        return self.data[idx], self.label[idx]

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        :return: (int) The size of the dataset.
        """
        return len(self.data)

class MyDataset(TorchDataset):
    def __init__(self, data, mask, label=None):
        self.data = data
        self.mask = mask
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx], self.label[idx]

    def __len__(self):
        return len(self.data)