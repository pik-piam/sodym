import numpy as np
import pandas as pd
import yaml
from .paths import dimensions_path, datasets_path, scalar_parameters_path


def read_data_to_df(type: str, name: str):
    if type == 'dataset':
        path = datasets_path(f"{name}.csv")
    else:
        raise RuntimeError(f"Invalid type {type}.")
    data = pd.read_csv(path)
    return data


def read_scalar_data(name:str):
    path = scalar_parameters_path()
    with open(path, 'r') as stream:
        parameters = yaml.safe_load(stream)
    return parameters[name]


def read_data_to_list(type: str, name: str, dtype: type):
    if type == 'dimension':
        path = dimensions_path(f"{name}.csv")
    else:
        raise RuntimeError(f"Invalid type {type}.")
    data = np.loadtxt(path, dtype=dtype, delimiter=';').tolist()
    # catch size one lists, which are transformed to scalar by np.ndarray.tolist()
    data = data if isinstance(data, list) else [data]
    return data


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
