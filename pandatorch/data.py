from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class DataFrame(Dataset):
    def __init__(self, X, y):
        self.features = np.array(X)
        self.target = y
        self.df = pd.concat([X, y], axis=1)
        self.__convert_target_type()

    def __convert_target_type(self):
        self.classes = self.target.unique()
        self.class_to_idx = dict()
        if self.target.dtype != "object":
            return
        else:
            for i, val in enumerate(self.classes):
                self.class_to_idx[val] = i
            self.target.replace(self.class_to_idx, inplace=True)
            self.target = np.array(self.target)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.get_number_of_columns(self.target) > 1:
            return self.features[idx, :], self.target[idx, :]

        elif self.get_number_of_columns(self.target) == 1:
            return self.features[idx, :], self.target[idx]

    def get_number_of_columns(self, data):
        if(len(data.shape) > 1):
            return data.shape[1]
        elif len(data.shape) == 1:
            return 1
