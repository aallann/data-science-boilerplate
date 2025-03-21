# -*- codign: utf-8 -*-


import pandas as pd
import numpy as np


import abc


from typing import List, Union, Mapping



class Transform(abc.ABC):
    ''' Transform base class '''

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def transform(self, data_frame: pd.DataFrame):
        '''Forward transform'''
        pass

    @abc.abstractmethod
    def inverse_transform(self, data_frame: pd.DataFrame):
        '''Inverse transform'''
        pass


class Composite(Transform):
    ''' --[[ Composite transform for multiprocesses.

        Transformations are at this stage assumed to be bijective,
        i.e., the inverse mapping

                    [ Tform.inverse_transform(·) ]

        exists and is well defined such that

            [ Tform.inverse_transform(Trans.transform(·)) ]

        forms an identity function for each Tform in the pipeline;
        but not commutative; that is why we traverse the pipeline 
        in reverse when applying such inverse mappings. ]]--

    Args
    ----
        :param transforms: transformations pipeline. Expects a 
            list of instantiated transforms.
    '''

    def __init__(self, transforms: List[Transform]):
        assert(len(Transforms > 0), 'must provide target transformations')
        self.transforms = transforms

    def transform(self,_frame: pd.DataFrame) -> pd.DataFrame:
        ''' Forward transforms '''
        for tform in self.transforms:
            data_frame = tform.transform(data_frame)

        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        ''' Inverse transforms '''
        for tform in reversed(self.transforms):
            data_frame = tform.inverse_transform(data_frame)

        return data_frame


class UnitGaussianNormalizer(Transform):
    ''' Normalizes data to unit (standard) Gaussian, i.e., zero
        mean, unit variance.

    Args
    ----
        :param dims: dimensional discriminator; since we use a
            dataframe dims corresponds to a list of target columns.
    '''

        EPS = float(1e-6)  # static data member. get away.

    def __init__(self, dims: list = None):
        super().__init__()
        self.dims = dims
        self.mean: Mapping[str, float] = {}
        self.std: Mapping[str, float] = {}
        

    def transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        ''' Forward transform '''
        if not self.dims:
            self.dims = data_frame.columns
        for dim in self.dims:
            values: np.array = np.array(data_frame[dim].values)
            self.mean[dim] = np.mean(values)
            self.std[dim] = np.std(values)
            data_frame[dim] = (values - self.mean[dim]) / (self.std[dim] + EPS)
        return data_frame

    def inverse_transform(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        ''' Inverse transform '''
        for dim in self.dims:
            values: np.array = np.array(data_frame[dim].values)
            data_frame[dim] = values * (self.std[dim] + EPS) + self.mean[dim]
        return data_frame


class LogarithmicScaler:
    pass

class ExponentialScaler:
    pass

class CategoricalEncoder:
    pass
