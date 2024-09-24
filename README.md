# SODYM
The SODYM package provides key functionality for material flow analysis, including
- the class `MFASystem` acting as a template (parent class) for users to create their own material flow models
- the class `NamedDimArray` handling mathematical operations between multi-dimensional arrays
- different classes like `DynamicStockModel` representing stocks accumulation, in- and outflows based on age cohort tracking and lifetime distributions. Those can be integrated in the `MFASystem`.
- different options for data input and export, as well as visualization

## Thanks

SODYM is an adaptation of:

ODYM<br>
Copyright (c) 2018 Industrial Ecology<br>
author: Stefan Pauliuk, Uni Freiburg, Germany<br>
https://github.com/IndEcol/ODYM<br>

## Why choose SODYM?

MFA models mainly consist on mathematical operations on different multi-dimensional arrays.

For example, the generation of different waste types `waste` might be a 3D-array defined over the dimensions time $t$, region $r$ and waste type $w$, and might be calculated from multiplying `end_of_life_products` (defined over time, region, and product type $p$) with a `waste_share` mapping from product type to waste type.
In numpy, the according matrix multiplication can be carried out with the `einsum` function, were an index string indicates the involved dimensions:

```
waste = np.einsum('trw,pw->trp', end_of_life_products, waste_share)
```

SODYM introduces a data type `NamedDimArray`, which stores the dimensions of the array and internally manages the dimensions of different arrays involved in mathematical operations.

With this, the above example reduces to

```
waste[...] = end_of_life_products * waste_share
```

This gives a SODYM-based MFA models the following properties:

- **Simplicity:** Since dimensions are automatically managed by the user, coding array operations becomes much easier. No knowledge about the einsum function, about the dimensions of each involved array or their order are required.
- **Sustainability:** When changing the dimensionality of any array in your code, you only have to apply the change once, where the array is defined, instead of adapting every operation involving it. This also allows, for example, to add or remove an entire dimension from your model with minimal effort.
- **Versatility:** We offer different levels of SODYM use: Users can choose to use the standard methods implemented for data read-in, system setup and visualization, or only use only some of the data types like `NamedDimArray`, and custom methods for the rest.
- **Robustness:** Through the use of [Pydantic](https://docs.pydantic.dev/latest/), the setup of the system and data read-in are type-checked, highlighting errors early-on.
- **Performance:** The use of numpy ndarrays ensures low model runtimes compared with dimension matching through pandas dataframes.

## Installation

SODYM dependencies are managed with [poetry](https://python-poetry.org/), which creates a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and installs dependencies automatically. The process is quite simple:

1. Clone the SODYM repository using git.
2. [Install poetry](https://python-poetry.org/docs/#installation)
   - *Optional*: Configure poetry to create your virtual environment in the project folder via `poetry config virtualenvs.in-project true`
3. From the project main directory, run 'python -m poetry install' to obtain all the necessary dependencies.

To execute python commands using the virtual environment of this project, use `poetry run <command>`, or activate the environment via `[/path/to/].venv/Scripts/activate`.
Further information can be found in the documentations of poetry and virtual environments linked above.

## Documentation

To build and view the documentation, follow these steps:

1. From the main directory, run `poetry install --with docs`
2. Navigate to the `docs` subdirectory, and run `poetry run make html`.
3. Open the file `docs/build/html/index.html` to view the documentation.

## Examples

The notebooks in the [examples](examples) folder provide usage examples of the code.

