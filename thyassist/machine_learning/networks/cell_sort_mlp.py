"""多层感知机"""
from mindspore import nn
from thyassist.machine_learning.configuration import MlpModelConfig

config = MlpModelConfig()

class CellSortMlp(nn.Cell):
    def __init__(self):
        super(CellSortMlp, self).__init__()
        self.fc1 = nn.Dense(config.num_features, 64)
        self.fc2 = nn.Dense(64, 32)
        self.fc3 = nn.Dense(32, 8)
        self.fc4 = nn.Dense(8, config.num_classes)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x


__all__ = ['CellSortMlp']
