# -*- coding: utf-8 -*- 


import pandas as pd
import numpy as np


from typing import Union


class DictDataset(object):
    """Custom data structure / handler for supervised learning"""

    def __init__(
        self, x: Union[pd.Series, pd.DataFrame], y: Union[pd.Series, pd.DataFrame]
    ):
        assert x.shape[0] == y.shape[0], "x and y must have the same first dimension"

        self.x: np.array = np.array(x)
        self.y: np.array = np.array(y).squeeze()

    def __getitem__(self, idx: int):
        x = self.x[idx]
        y = self.y[idx]

        return {"x": x, "y": y}

    def __len__(self):
        return self.x.shape[0]

