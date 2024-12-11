from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict
import yaml

from .named_dim_arrays import Parameter
from .mfa_definition import DimensionDefinition, ParameterDefinition
from .dimensions import DimensionSet, Dimension


class DataReader:
    """Template for creating a data reader, showing required methods and data formats needed for
    use in the MFASystem model.
    """

    def read_dimensions(self, dimension_definitions: List[DimensionDefinition]) -> DimensionSet:
        dimensions = [self.read_dimension(definition) for definition in dimension_definitions]
        return DimensionSet(dim_list=dimensions)

    @abstractmethod
    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        pass

    def read_scalar_data(self, parameters: List[str]) -> dict:
        """Optional addition method if additional scalar parameters are required."""
        pass

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        pass

    def read_parameters(
        self, parameter_definitions: List[ParameterDefinition], dims: DimensionSet
    ) -> Dict[str, Parameter]:
        parameters = {}
        for parameter_definition in parameter_definitions:
            dim_subset = dims.get_subset(parameter_definition.dim_letters)
            parameters[parameter_definition.name] = self.read_parameter_values(
                parameter_name=parameter_definition.name,
                dims=dim_subset,
            )
        return parameters


class DimensionReader(ABC):

    read_dimensions = DataReader.read_dimensions

    @abstractmethod
    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        pass


class CSVDimensionReader(DimensionReader):
    """Expects a single row or single columns csv file with no header containing the dimension items.

    Args:
        dimension_files (dict): {dimension_name: file_path, ...}
        **read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv. The default is {"header": None}. Not encouraged to use, since it may not lead to the intended DataFrame format. Sticking to recommended csv file format is preferred.
    """

    def __init__(
        self,
        dimension_files: dict = None,
        **read_csv_kwargs,
    ):
        self.dimension_files = dimension_files  # {dimension_name: file_path, ...}
        self.read_csv_kwargs = read_csv_kwargs

    def read_dimension(self, definition: DimensionDefinition):
        if self.dimension_files is None:
            raise ValueError("No dimension files specified.")
        path = self.dimension_files[definition.name]
        if "header" not in self.read_csv_kwargs:
            self.read_csv_kwargs["header"] = None
        df = pd.read_csv(path, **self.read_csv_kwargs)
        return Dimension.from_df(df, definition)


class ExcelDimensionReader(DimensionReader):
    """Expects a single row or single columns excel sheet with no header containing the dimension items.

    Args:
        dimension_files (dict): {dimension_name: file_path, ...}
        dimension_sheets (dict): {dimension_name: sheet_name, ...}
        **read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel. The default is {"header": None}. Not encouraged to use, since it may not lead to the intended DataFrame format. Sticking to recommended excel file format is preferred.
    """
    def __init__(
        self,
        dimension_files: dict = None,
        dimension_sheets: dict = None,
        **read_excel_kwargs,
    ):
        self.dimension_files = dimension_files  # {dimension_name: file_path, ...}
        self.dimension_sheets = dimension_sheets
        self.read_excel_kwargs = read_excel_kwargs

    def read_dimension(self, definition: DimensionDefinition):
        if self.dimension_files is None:
            raise ValueError("No dimension files specified.")
        path = self.dimension_files[definition.name]
        # load data from excel
        if self.dimension_sheets is None:
            sheet_name = None
        else:
            sheet_name = self.dimension_sheets[definition.name]
        # default for header is None
        if "header" not in self.read_excel_kwargs:
            self.read_excel_kwargs["header"] = None
        df = pd.read_excel(path, sheet_name=sheet_name, **self.read_excel_kwargs)
        return Dimension.from_df(df, definition)


class ParameterReader(ABC):

    @abstractmethod
    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        pass

    read_parameters = DataReader.read_parameters


class CSVParameterReader(ParameterReader):
    """For expected format, see :py:class:`sodym.df_to_nda.DataFrameToNDADataConverter`

    Args:
        parameter_files (dict): {parameter_name: file_path, ...}
        **read_csv_kwargs: Additional keyword arguments passed to pandas.read_csv. Not encouraged to use, since it may not lead to the intended DataFrame format. Sticking to recommended csv file format is preferred
    """
    def __init__(
        self,
        parameter_files: dict = None,
        **read_csv_kwargs,
    ):
        self.parameter_filenames = parameter_files  # {parameter_name: file_path, ...}
        self.read_csv_kwargs = read_csv_kwargs

    def read_parameter_values(self, parameter_name: str, dims):
        if self.parameter_filenames is None:
            raise ValueError("No parameter files specified.")
        datasets_path = self.parameter_filenames[parameter_name]
        data = pd.read_csv(datasets_path, **self.read_csv_kwargs)
        return Parameter.from_df(dims=dims, name=parameter_name, df=data)


class ExcelParameterReader(ParameterReader):
    """For expected format, see :py:class:`sodym.df_to_nda.DataFrameToNDADataConverter`

    Args:
        parameter_files (dict): {parameter_name: file_path, ...}
        parameter_sheets (dict): {parameter_name: sheet_name, ...}
        **read_excel_kwargs: Additional keyword arguments passed to pandas.read_excel. Not encouraged to use, since it may not lead to the intended DataFrame format. Sticking to recommended excel file format is preferred
    """
    def __init__(
        self,
        parameter_files: dict = None,
        parameter_sheets: dict = None,
        **read_excel_kwargs,
    ):
        self.parameter_files = parameter_files  # {parameter_name: file_path, ...}
        self.parameter_sheets = parameter_sheets  # {parameter_name: sheet_name, ...}
        self.read_excel_kwargs = read_excel_kwargs

    def read_parameter_values(self, parameter_name: str, dims):
        if self.parameter_files is None:
            raise ValueError("No parameter files specified.")
        datasets_path = self.parameter_files[parameter_name]
        if self.parameter_sheets is None:
            sheet_name = None
        else:
            sheet_name = self.parameter_sheets[parameter_name]
        data = pd.read_excel(datasets_path, sheet_name=sheet_name, **self.read_excel_kwargs)
        return Parameter.from_df(dims=dims, name=parameter_name, df=data)


class ScalarDataReader(ABC):
    def read_scalar_data(self, parameters: List[str], **read_scalar_kwargs) -> dict:
        """Optional addition method if additional scalar parameters are required."""
        raise NotImplementedError("No scalar data reader specified.")


class EmptyScalarDataReader(ScalarDataReader):

    def read_scalar_data(self, parameters: List[str]):
        if parameters:
            raise ValueError("List of scalar parameters is not empty, but no real reader for scalar data is specified.")
        return {}


class CheckedScalarDataReader(ScalarDataReader):
    """Parent class for scalar data readers that ensures that the read data and requested parameters
    match.
    """

    def __init__(self, scalar_file: str = None, **read_scalar_kwargs):
        self.scalar_file = scalar_file
        self.read_scalar_kwargs = read_scalar_kwargs

    def read_scalar_data(self, parameters: List[str]):
        if not parameters:
            return {}
        if self.scalar_file is None:
            raise ValueError("No scalar data file specified.")
        data = self._get_dict()
        if not set(parameters) == set(data.keys()):
            raise ValueError(
                f"Parameter names in yaml file do not match requested parameters. Unexpected parameters: {set(data.keys()) - set(parameters)}; Missing parameters: {set(parameters) - set(data.keys())}."
            )
        return data

    def add_pd_reader_defaults(self):
        if "header" not in self.read_scalar_kwargs:
            self.read_scalar_kwargs["header"] = None
        if "index_col" not in self.read_scalar_kwargs:
            self.read_scalar_kwargs["index_col"] = 0

    def _get_dict(self) -> dict:
        raise NotImplementedError(f"Method not implemented. Choose a subclass of {self.__class__.__name__}.")


class YamlScalarDataReader(CheckedScalarDataReader):

    def _get_dict(self) -> dict:
        with open(self.scalar_file, "r") as stream:
            return yaml.safe_load(stream)


class CSVScalarDataReader(CheckedScalarDataReader):

    def _get_dict(self) -> dict:
        self.add_pd_reader_defaults()
        df = pd.read_csv(self.scalar_file, **self.read_scalar_kwargs)
        dict = df.to_dict()[df.columns[0]]
        return dict


class ExcelScalarDataReader(CheckedScalarDataReader):

    def __init__(self, scalar_file: str = None, scalar_sheet: str = None, **read_scalar_kwargs):
        self.sheet_name = scalar_sheet
        super().__init__(scalar_file, **read_scalar_kwargs)

    def _get_dict(self) -> dict:
        self.add_pd_reader_defaults()
        df= pd.read_excel(self.scalar_file, sheet_name=self.sheet_name, **self.read_scalar_kwargs)
        dict = df.to_dict()[df.columns[0]]
        return dict


class CompoundDataReader(DataReader):

    def __init__(
        self,
        dimension_reader: DimensionReader,
        parameter_reader: ParameterReader,
        scalar_data_reader: ScalarDataReader = EmptyScalarDataReader(),
    ):
        self.dimension_reader = dimension_reader
        self.parameter_reader = parameter_reader
        self.scalar_data_reader = scalar_data_reader

    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        return self.dimension_reader.read_dimension(dimension_definition)

    def read_parameter_values(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        return self.parameter_reader.read_parameter_values(parameter_name, dims)

    def read_scalar_data(self, parameters: List[str]) -> dict:
        return self.scalar_data_reader.read_scalar_data(parameters)
