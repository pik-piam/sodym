from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List
import yaml
import os

from .named_dim_arrays import Parameter
from .mfa_definition import DimensionDefinition
from .dimensions import DimensionSet, Dimension


class DataReader(ABC):

    def read_dimensions(self, dimension_definitions: List[DimensionDefinition]) -> DimensionSet:
        dimensions = [self.read_dimension(definition) for definition in dimension_definitions]
        return DimensionSet(dimensions=dimensions)

    @abstractmethod
    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        pass

    @abstractmethod
    def read_scalar_data(self, parameters: List[str]) -> dict:
        pass

    @abstractmethod
    def read_parameter_values(self, parameter: str, dims: DimensionSet) -> Parameter:
        pass


class JakobsDataReader(DataReader):
    def __init__(self, input_data_path):
        self.input_data_path = input_data_path

    def read_scalar_data(self, parameters: List[str]):
        path = os.path.join(self.input_data_path, 'scalar_parameters.yml')
        try:
            with open(path, 'r') as stream:
                data = yaml.safe_load(stream)
        except FileNotFoundError:
            return {}
        return {name: data[name] for name in data if name in parameters}

    def read_parameter_values(self, parameter: str, dims):
        data = self.read_data_to_df(type='dataset', name=parameter)
        values = self.get_np_from_df(data, dims.names)
        return Parameter(dims=dims, values=values)
    
    def read_dimension(self, definition: DimensionDefinition):
        data = self.read_data_to_list("dimension", definition.filename, definition.dtype)
        return Dimension(name=definition.name, letter=definition.letter, items=data)

    def read_data_to_df(self, type: str, name: str):
        if type != 'dataset':
            raise RuntimeError(f"Invalid type {type}.")
        datasets_path = os.path.join(self.input_data_path, 'datasets', f'{name}.csv')
        data = pd.read_csv(datasets_path)
        return data

    def read_data_to_list(self, type: str, name: str, dtype: type):
        if type != 'dimension':
            raise RuntimeError(f"Invalid type {type}.")
        path = os.path.join(self.input_data_path, 'dimensions', f'{name}.csv')
        data = np.loadtxt(path, dtype=dtype, delimiter=';').tolist()
        # catch size one lists, which are transformed to scalar by np.ndarray.tolist()
        data = data if isinstance(data, list) else [data]
        return data

    @staticmethod
    def get_np_from_df(df_in: pd.DataFrame, dims: tuple):
        df = df_in.copy()
        dim_columns = [d for d in dims if d in df.columns]
        value_cols = np.setdiff1d(df.columns, dim_columns)
        df.set_index(dim_columns, inplace=True)
        df = df.sort_values(by=dim_columns)

        # check for sparsity
        if df.index.has_duplicates:
            raise Exception("Double entry in df!")
        shape_out = df.index.shape if len(dim_columns) == 1 else df.index.levshape
        if np.prod(shape_out) != df.index.size:
            raise Exception("Dataframe is missing values!")

        if np.any(value_cols != 'value'):
            out = {vc: df[vc].values.reshape(shape_out) for vc in value_cols}
        else:
            out = df["value"].values.reshape(shape_out)
        return out
