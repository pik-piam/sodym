import logging
import numpy as np
import pandas as pd
from typing import Literal, Optional, TYPE_CHECKING, Iterable
from pydantic import BaseModel as PydanticBaseModel

if TYPE_CHECKING:
    from .named_dim_arrays import NamedDimArray


class NDADataFormat(PydanticBaseModel):

    type: Literal["long", "wide"]
    value_column: str = "value"
    columns_dim: Optional[str] = None


class DataFrameToNDAConverter:

    def __init__(self, df: pd.DataFrame, nda: 'NamedDimArray'):
        self.df = df.copy()
        self.nda = nda
        self.nda_values = self.get_nda_values()

    def get_nda_values(self) -> np.ndarray:

        logging.debug(f"Start setting values for NamedDimArray {self.nda.name} with dimensions {self.nda.dims.names} from dataframe.")

        logging.debug("Dropping index. If index is needed, please apply df.reset_index() before passing df to nda.")
        self.df.reset_index(inplace=True, drop=True)

        self._determine_format()

        self._df_to_long_format()

        self._check_missing_dim_columns()

        self.df.set_index(self.dim_columns, inplace=True)
        self.df = self.df.sort_values(by=self.dim_columns)

        self._check_data_complete()

        return self.df[self.format.value_column].values.reshape(self.nda.shape)

    def _determine_format(self):
        self.dim_columns = [c for c in self.df.columns if c in self.nda.dims.names]
        logging.debug(f"Recognized index columns by name: {self.dim_columns}")

        self._check_for_dim_columns_by_items()
        self._check_value_columns()

    def _check_for_dim_columns_by_items(self):
        for cn in self.df.columns:
            if cn in self.dim_columns:
                continue
            found = self._check_if_dim_column_by_items(cn)
            if not found:
                logging.debug(f"Could not find dimension with same items as column {cn}. "
                              "Assuming this is the first value column; Won't look further.")
                return

    def _check_if_dim_column_by_items(self, column_name: str) -> bool:
        logging.debug(f"Checking if {column_name} is a dimension by comparing items with dim items")
        for dim in self.nda.dims:
            if self.same_items(self.df[column_name].unique(), dim.items):
                logging.debug(f"{column_name} is dimension {dim.name}.")
                self.df.rename(columns={column_name: dim.name}, inplace=True)
                self.dim_columns.append(dim.name)
                return True
        return False

    def _check_value_columns(self):
        value_cols = np.setdiff1d(self.df.columns, self.dim_columns)
        logging.debug(f"Assumed value columns: {value_cols}")
        logging.debug("Trying to match set of value column names with items of dimension.")
        value_cols_are_dim = self._check_if_value_columns_match_dim_items(value_cols)
        if not value_cols_are_dim:
            self._check_if_valid_long_format(value_cols)

    def _check_if_value_columns_match_dim_items(self, value_cols: list[str]) -> bool:
        for dim in self.nda.dims:
            if self.same_items(value_cols, dim.items):
                logging.debug(f"Value columns match dimension items of {dim.name}.")
                self.format = NDADataFormat(type="wide", columns_dim=dim.name)
                return True
        return False

    def _check_if_valid_long_format(self, value_cols: list[str]):
        logging.debug("Could not find dimension with same item set as value column names. Assuming long format, i.e. one value column.")
        if len(value_cols) == 1:
            self.format = NDADataFormat(type="long", value_column=value_cols[0])
            logging.debug(f"Value column name is {value_cols[0]}.")
        else:
            raise ValueError("More than one value columns. Could not find a dimension the items of which match the set of value column names. "
                                f"Value columns: {value_cols}. Please check input data for format, typos, data types and missing items.")

    def _df_to_long_format(self):
        if self.format.type != "wide":
            return
        logging.debug("Converting wide format to long format.")
        value_cols = self.dims[self.format.columns_dim].items
        self.df = self.df.melt(
            id_vars=[c for c in self.df.columns if c not in value_cols],
            value_vars=value_cols,
            var_name=self.format.columns_dim,
            value_name=self.format.value_column)
        self.format = NDADataFormat(type="long", value_column=self.format.value_column)
        self.dim_columns.append(self.format.columns_dim)

    def _check_missing_dim_columns(self):
        missing_dim_columns = np.setdiff1d(self.nda.dims.names, self.dim_columns)
        for c in missing_dim_columns:
            if len(self.nda.dims[c].items) == 1:
                self.df[c] = self.nda.dims[c].items[0]
                self.dim_columns.append(c)
            else:
                raise ValueError(f"Dimension {c} from array has more than one item, but is not found in df. Please specify column in dataframe.")

    def _check_data_complete(self):
        if self.df.index.has_duplicates:
            raise Exception("Double entry in df!")
        for dim in self.nda.dims:
            df_dim_items = self.df.index.get_level_values(dim.name).unique()
            if not self.same_items(dim.items, df_dim_items):
                raise Exception(f"Missing items in index for dimension {dim.name}! NamedDimArray items: {set(dim.items)}, df items: {set(df_dim_items)}")
        if np.prod(self.nda.shape) != self.df.index.size:
            raise Exception(f"Dataframe is missing items! NamedDimArray size: {np.prod(self.shape)}, df size: {self.df.index.size}")

    @staticmethod
    def same_items(arr1: Iterable, arr2: Iterable) -> bool:
        return len(set(arr1).symmetric_difference(set(arr2))) == 0
