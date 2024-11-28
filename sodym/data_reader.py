from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict
import yaml

from .named_dim_arrays import Parameter
from .mfa_definition import DimensionDefinition, ParameterDefinition
from .dimensions import DimensionSet, Dimension


class DimensionReader(ABC):
    def read_all(self, dimension_definitions: List[DimensionDefinition]) -> DimensionSet:
        dimensions = [self.read_single(definition) for definition in dimension_definitions]
        return DimensionSet(dim_list=dimensions)

    @abstractmethod
    def read_single(self, dimension_definition: DimensionDefinition) -> Dimension:
        pass


class CSVDimensionReader(DimensionReader):
    def __init__(
        self,
        dimension_files: dict = None,
    ):
        self.dimension_files = dimension_files  # {dimension_name: file_path, ...}

    def read_single(self, definition: DimensionDefinition):
        if self.dimension_files is None:
            raise ValueError("No dimension files specified.")
        path = self.dimension_files[definition.name]
        data = np.loadtxt(path, dtype=definition.dtype, delimiter=";").tolist()
        # catch size one lists, which are transformed to scalar by np.ndarray.tolist()
        data = data if isinstance(data, list) else [data]
        return Dimension(name=definition.name, letter=definition.letter, items=data)


class ExcelDimensionReader(DimensionReader):
    def __init__(
        self,
        dimension_files: dict = None,
        dimension_sheets: dict = None,
    ):
        self.dimension_files = dimension_files  # {dimension_name: file_path, ...}
        self.dimension_sheets = dimension_sheets

    def read_single(self, definition: DimensionDefinition):
        if self.dimension_files is None:
            raise ValueError("No dimension files specified.")
        path = self.dimension_files[definition.name]
        # load data from excel
        if self.dimension_sheets is None:
            sheet_name = None
        else:
            sheet_name = self.dimension_sheets[definition.name]
        data = pd.read_excel(path, sheet_name=sheet_name, header=None).to_numpy()
        if not np.min(data.shape) == 1:
            raise ValueError(
                f"Dimension data for {definition.name} must have only one row or column."
            )
        data = data.flatten().tolist()
        # delete header for items if present
        if data[0] == definition.name:
            data = data[1:]
        return Dimension(name=definition.name, letter=definition.letter, items=data)


class ParameterReader(ABC):
    @abstractmethod
    def read_single(self, parameter_name: str, dims: DimensionSet) -> Parameter:
        pass

    def read_all(
        self, parameter_definitions: List[ParameterDefinition], dims: DimensionSet
    ) -> Dict[str, Parameter]:
        parameters = {}
        for parameter in parameter_definitions:
            dim_subset = dims.get_subset(parameter.dim_letters)
            parameters[parameter.name] = self.read_single(
                parameter_name=parameter.name,
                dims=dim_subset,
            )
        return parameters


class CSVParameterReader(ParameterReader):
    def __init__(
        self,
        parameter_files: dict = None,
    ):
        self.parameter_filenames = parameter_files  # {parameter_name: file_path, ...}

    def read_single(self, parameter_name: str, dims):
        if self.parameter_filenames is None:
            raise ValueError("No parameter files specified.")
        datasets_path = self.parameter_filenames[parameter_name]
        data = pd.read_csv(datasets_path)
        return Parameter.from_df(dims=dims, name=parameter_name, df=data)


class ExcelParameterReader(ParameterReader):
    def __init__(
        self,
        parameter_files: dict = None,
        parameter_sheets: dict = None,
    ):
        self.parameter_files = parameter_files  # {parameter_name: file_path, ...}
        self.parameter_sheets = parameter_sheets  # {parameter_name: sheet_name, ...}

    def read_single(self, parameter_name: str, dims):
        if self.parameter_files is None:
            raise ValueError("No parameter files specified.")
        datasets_path = self.parameter_files[parameter_name]
        if self.parameter_sheets is None:
            sheet_name = None
        else:
            sheet_name = self.parameter_sheets[parameter_name]
        data = pd.read_excel(datasets_path, sheet_name=sheet_name)
        return Parameter.from_df(dims=dims, name=parameter_name, df=data)


class ScalarDataReader(ABC):
    def read(self, parameters: List[str]) -> dict:
        """Optional addition method if additional scalar parameters are required."""
        raise NotImplementedError("No scalar data reader specified.")


class EmptyScalarDataReader(ScalarDataReader):
    def read(self, parameters: List[str]):
        return None


class YamlScalarDataReader(ScalarDataReader):
    def __init__(self, scalar_data_yaml_file: str = None):
        self.scalar_data_yaml_file = scalar_data_yaml_file

    def read(self, parameters: List[str]):
        if self.scalar_data_yaml_file is None:
            raise ValueError("No scalar data file specified.")
        with open(self.scalar_data_yaml_file, "r") as stream:
            data = yaml.safe_load(stream)
        if not set(parameters) == set(data.keys()):
            raise ValueError(
                f"Parameter names in yaml file do not match requested parameters. Unexpected parameters: {set(data.keys()) - set(parameters)}; Missing parameters: {set(parameters) - set(data.keys())}."
            )
        return data


class DataReader:
    """Template for creating a data reader, showing required methods and data formats needed for
    use in the MFASystem model.
    """

    def __init__(
        self,
        dimension_reader: DimensionReader,
        parameter_reader: ParameterReader,
        scalar_data_reader: ScalarDataReader = EmptyScalarDataReader(),
    ):
        self.dimension_reader = dimension_reader
        self.parameter_reader = parameter_reader
        self.scalar_data_reader = scalar_data_reader
