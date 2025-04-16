"""多层感知机数据加载模块"""
import numpy as np
import pandas as pd
from mindspore import dataset as ds
from thyassist.machine_learning.configuration import MlpModelConfig

config = MlpModelConfig()

TITLE = ["Area", "Mean", "StdDev", "Mode", "Min", "Max", "Rerim.",
         "Circ.", "Feret", "Median", "FeretX", "FeretY", "FeretAngle",
         "MinFeret", "AR", "Round", "Solidity"]

class SyntheticData:
    def __init__(self, features, labels):
        self.features, self.labels = features, labels

    def __getitem__(self, index):   # __getitem__(self, index) 一般用来迭代序列(常见序列如：列表、元组、字符串)
        return self.features[index], self.labels[index]

    def __len__(self):
        return self.features.shape[0]


def create_mlp_dataset(cancer_cell_data_path, not_cancer_cell_data_path):
    cancer_data = pd.read_csv(cancer_cell_data_path)
    not_cancer_data = pd.read_csv(not_cancer_cell_data_path)

    cancer_data['label'] = 'cancer_cell'
    not_cancer_data['label'] = 'not_cancer_cell'

    all_data = pd.concat((cancer_data.iloc[:, 1:], not_cancer_data.iloc[:, 1:]))
    features_index = all_data.dtypes[all_data.dtypes != 'object'].index

    all_data[features_index] = all_data[features_index].apply(
        lambda x: (x - x.mean()) / (x.std())
    )
    all_data = pd.get_dummies(all_data, columns=['label'])
    features = all_data.iloc[:, :-2].values.astype(np.float32)
    labels = all_data.iloc[:, -2:].values.astype(np.float32)

    # 按照列来拼接并打乱顺序
    features_and_labels = np.hstack((features, labels))
    shuffled_features_and_labels = features_and_labels[np.random.permutation(features_and_labels.shape[0]), :]

    train_data = shuffled_features_and_labels[:525, :]
    test_data = shuffled_features_and_labels[525:, :]

    train_features = train_data[:, :-2]
    train_labels = train_data[:, -2:]
    eval_features = test_data[:, :-2]
    eval_labels = test_data[:, -2:]

    train_dataset = ds.GeneratorDataset(source=SyntheticData(train_features, train_labels),
                                        column_names=['features', 'label'],
                                        python_multiprocessing=False)
    eval_dataset = ds.GeneratorDataset(source=SyntheticData(eval_features, eval_labels),
                                       column_names=['features', 'label'],
                                       python_multiprocessing=False)

    train_dataset = train_dataset.batch(batch_size=config.train_batch_size,
                                        drop_remainder=True)
    eval_dataset = eval_dataset.batch(batch_size=config.eval_batch_size,
                                      drop_remainder=True)
    return train_dataset, eval_dataset


__all__ = ["TITLE",
           "create_mlp_dataset"]
