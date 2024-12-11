import numpy as np
import pytest

from sodym.data_reader import CSVDimensionReader, ExcelDimensionReader, CSVParameterReader, ExcelParameterReader, CSVScalarDataReader, ExcelScalarDataReader, YamlScalarDataReader
from sodym.mfa_definition import DimensionDefinition, ParameterDefinition, MFADefinition
from sodym.mfa_system import MFASystem


csv_dimension_files = {
    "animals": "tests/tests_data/dimension_animals_horizontal.csv",
    "time": "tests/tests_data/dimension_time_vertical.csv",
    "region": "tests/tests_data/dimension_region_single.csv",
}
excel_dimension_file = "tests/tests_data/dimensions.xlsx"
excel_dimension_files = {a: excel_dimension_file for a in csv_dimension_files.keys()}
dimension_sheet_names = {a: a for a in csv_dimension_files.keys()}
dimension_definitions = [
    DimensionDefinition(name="animals", letter="a", dtype=str),
    DimensionDefinition(name="time", letter="t", dtype=int),
    DimensionDefinition(name="region", letter="r", dtype=str),
]
dimension_items_expected = {
    "animals": ["dog", "cat"],
    "time": [1990, 2000, 2010],
    "region": ["world"],
}
parameter_definitions = [
    ParameterDefinition(name="p1", dim_letters=["t", "r"]),
    ParameterDefinition(name="p2", dim_letters=["t", "r"]),
    ParameterDefinition(name="p3", dim_letters=["t", "r"]),
    ParameterDefinition(name="p4", dim_letters=["t", "r"]),
    ParameterDefinition(name="p5", dim_letters=["t", "r", "a"]),
    ParameterDefinition(name="p6", dim_letters=["t", "r", "a"]),
]
csv_parameter_files = {a.name: "tests/tests_data/parameter_" + a.name + ".csv" for a in parameter_definitions}
excel_parameter_file = "tests/tests_data/parameters.xlsx"
excel_parameter_files = {a.name: excel_parameter_file for a in parameter_definitions}
parameter_sheet_names = {a.name: a.name for a in parameter_definitions}


def test_dimensions():

    csv_reader = CSVDimensionReader(csv_dimension_files)
    excel_reader = ExcelDimensionReader(excel_dimension_files, dimension_sheet_names)

    for reader in [csv_reader, excel_reader]:
        dimension_set = reader.read_dimensions(dimension_definitions)

        for dim in dimension_set:
            assert dim.items == dimension_items_expected[dim.name]

    return


def test_valid_parameter_reader():

    dims = CSVDimensionReader(csv_dimension_files).read_dimensions(dimension_definitions)

    csv_reader = CSVParameterReader(csv_parameter_files)
    excel_reader = ExcelParameterReader(excel_parameter_files, parameter_sheet_names)

    for reader in [csv_reader, excel_reader]:
    # for reader in [excel_reader]:
        parameters = reader.read_parameters(parameter_definitions, dims)
        for pn, p in parameters.items():
            if pn in ["p1", "p2", "p3", "p4"]:
                assert np.array_equal(p.values, [[1], [2], [3]])
            elif pn in ["p5", "p6"]:
                assert np.array_equal(p.values, np.array([[[1, 4]], [[2, 5]], [[3, 6]]]))


def test_wrong_parameter_reader():

    parameter_definitions = [
        # missing row
        ParameterDefinition(name="e1", dim_letters=["t", "a"]),
        # wrong dim item
        ParameterDefinition(name="e2", dim_letters=["t"]),
        # additional dim item
        ParameterDefinition(name="e3", dim_letters=["t"]),
        # empty cell
        ParameterDefinition(name="e4", dim_letters=["t"]),
        # missing dim
        ParameterDefinition(name="e5", dim_letters=["t", "a"]),
    ]

    csv_parameter_files = {a.name: "tests/tests_data/parameter_" + a.name + ".csv" for a in parameter_definitions}
    excel_parameter_file = "tests/tests_data/parameters.xlsx"
    excel_parameter_files = {a.name: excel_parameter_file for a in parameter_definitions}
    sheet_names = {a.name: a.name for a in parameter_definitions}

    wrong_csv_reader = CSVParameterReader(csv_parameter_files)
    wrong_excel_reader = ExcelParameterReader(excel_parameter_files, sheet_names)

    dims = CSVDimensionReader(csv_dimension_files).read_dimensions(dimension_definitions)

    for reader in [wrong_csv_reader, wrong_excel_reader]:
        for prm_def in parameter_definitions:
            with pytest.raises(ValueError):
                reader.read_parameter_values(prm_def.name, dims.get_subset(prm_def.dim_letters))

    return


def test_scalar_reader():

    test_prms = {
        "v1": ["a"],
        "v2": ["a", "b", "c"],
    }
    prm_values = {
        "v1": [0.5],
        "v2": [0.5, 0.6, 0.7],
    }
    base_name = "tests/tests_data/scalars"

    for test_name, prm_names in test_prms.items():

        readers = [
            YamlScalarDataReader(scalar_file = f"{base_name}_{test_name}.yml"),
            CSVScalarDataReader(scalar_file = f"{base_name}_{test_name}.csv"),
            ExcelScalarDataReader(scalar_file = f"{base_name}.xlsx", scalar_sheet = test_name)
        ]

        for reader in readers:
            data = reader.read_scalar_data(prm_names)
            for i_prm, prm in enumerate(prm_names):
                assert data[prm] == prm_values[test_name][i_prm]


def test_build_mfa_system():

    definition_ws = MFADefinition(
        dimensions=dimension_definitions,
        processes=[],
        stocks=[],
        flows=[],
        parameters=parameter_definitions,
        scalar_parameters=["a", "b", "c"],
    )
    definition_ns = MFADefinition(
        dimensions=dimension_definitions,
        processes=[],
        stocks=[],
        flows=[],
        parameters=parameter_definitions,
        scalar_parameters=None,
    )
    class MinimalMFASystem(MFASystem):
        def compute(self):
            pass

    mfa = MinimalMFASystem.from_csv(
        definition_ws,
        csv_dimension_files,
        csv_parameter_files,
        scalar_file = "tests/tests_data/scalars_v2.csv",
        )
    assert mfa.scalar_parameters["a"] == 0.5
    mfa = MinimalMFASystem.from_csv(
        definition_ns,
        csv_dimension_files,
        csv_parameter_files,
        )

    mfa = MinimalMFASystem.from_excel(
        definition_ws,
        excel_dimension_files,
        excel_parameter_files,
        dimension_sheets=dimension_sheet_names,
        parameter_sheets=parameter_sheet_names,
        scalar_file = "tests/tests_data/scalars.xlsx",
        scalar_sheet="v2",
        )
    assert mfa.scalar_parameters["b"] == 0.6
    mfa = MinimalMFASystem.from_excel(
        definition_ns,
        excel_dimension_files,
        excel_parameter_files,
        dimension_sheets=dimension_sheet_names,
        parameter_sheets=parameter_sheet_names,
        )

    with pytest.raises(ValueError):
        # no scalars in definition, scalar file specified
        mfa = MinimalMFASystem.from_csv(
            definition_ns,
            csv_dimension_files,
            csv_parameter_files,
            scalar_file = "tests/tests_data/scalars_v2.csv",
            )
        # scalars in definition, no scalar file specified
        mfa = MinimalMFASystem.from_csv(
            definition_ws,
            csv_dimension_files,
            csv_parameter_files,
            )
        # same for excel
        mfa = MinimalMFASystem.from_excel(
            definition_ns,
            excel_dimension_files,
            excel_parameter_files,
            dimension_sheets=dimension_sheet_names,
            parameter_sheets=parameter_sheet_names,
            scalar_file = "tests/tests_data/scalars.xlsx",
            scalar_sheet="v2",
            )
        mfa = MinimalMFASystem.from_excel(
            definition_ws,
            excel_dimension_files,
            excel_parameter_files,
            dimension_sheets=dimension_sheet_names,
            parameter_sheets=parameter_sheet_names,
            )

