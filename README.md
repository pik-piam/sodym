# sodym
The sodym package provides key functionality for material flow analysis, with the class `MFASystem` acting as a template (parent class) for users to create their own material flow models.

The concepts used in sodym are based on:
ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

## Install
Clone the sodym repository using git. Activate the virtual environment and install poetry, and then run 'python -m poetry install' to obtain all the necessary dependencies.

## Build your own MFA system
The functionality provided by sodym is general compared to specific MFA systems,
therefore the user is required to build upon this for their specific use-case.

First of all, the user needs to decide what they want to model:
which processes, flows and stocks are important and which dimensions do they want to consider
(time, regions, different materials etc.).
```
from sodym import DimensionDefinition, FlowDefinition, MFADefinition
dimensions = [DimensionDefinition(name='Time', dim_letter='t', dtype=int)]
processes = ['sysenv', 'use']
flows = [
    FlowDefinition(from_process='sysenv', to_process='use', dim_letters=('t', )),
    FlowDefinition(from_process='use', to_process='sysenv', dim_letters=('t', )),
]
stocks = [StockDefinition(name='use', dim_letters=('t', ))]
parameters = [ParameterDefinition(name='production_rate', dim_letters=('t', ))]
mfa_definition = MFADefinition(dimensions=dimensions, processes=processes, flows=flows, stocks=stocks, parameters=parameters)
```
The user also needs to specify how the different flows and stocks are related to each other.
For this, they need to create their own MFA system,
which can be done using python inheritance
(i.e. defining a new class that inherits from `MFASystem`)
and defining the class method `compute`,
which provides equations linking the different model attributes.
```
from sodym import MFASystem

class MyMFA(MFASystem):
    def compute(self):
        self.flows['use => sysenv'] = self.stocks['use'].outflow
        ...
```
Then, the model requires some data, which must pass some validations.
For this, sodym has the `DataReader` class, which is a template specifying the methods and output types required, in order for the user to create their own data reader, depending on how they wish to store their model data.
An `ExampleDataReader` is also defined in sodym and provides functionality for reading dimension and parameter datasets from .csv files.

After the user has defined their MFA system class and their data reader, they can be put together and the model can be run.
```
my_data_reader = MyDataReader()
dims = my_data_reader.read_dimensions(mfa_definition.dimensions)
parameters = my_data_reader.read_parameters(mfa_definition.parameters)
...
```
Finally we are ready to initialise an instance of our MFA system with the data we have just read,
and then perform the computations.
```
my_mfa = MyMFA(dims=dims, parameters=parameters, ...)
my_mfa.compute()
```
Either after initialisation or after the computation, the user can access the attributes of the MFASystem instance, and e.g. write the results to .csv file.
```
results = my_mfa.flows['use => sysenv']
results.to_df().to_csv('file_path_to_store_my_results.csv')
```
sodym provides further data writing and plotting functionality for Sankey diagrams in the `DataWriter` class.

## Advantages of using sodym
* validation of the model definition, ensuring that it is sufficient and consistent
* validation of input data ensuring the size matches the specified dimensions
* automated re-ordering and/or reduction of dimensions, through the use of sodym's `NamedDimArray` objects, simplying the specification of equations in the user-defined `compute` method
* functionality to check that all flows and stocks are accounted for (mass balance)
* out of the box plotting functionality for Sankey diagrams
