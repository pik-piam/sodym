# sodym
The sodym package provides key functionality for material flow analysis, with the class `MFASystem` acting as a template (parent class) for users to create their own material flow models.

## Install
Clone the sodym repository using git. Activate the virtual environment and install poetry, and then run 'python -m poetry install' to obtain all the necessary dependencies.

## Build your own MFA system
The functionality provided by sodym is general compared to specific MFA systems, therefore the user is required to build upon this for their specific use-case.
This can be done using python inheritance (i.e. defining a new class that inherits from `MFASystem`) and defining class methods: `set_up_definition` which needs to return an `MFADefinition` object, and `compute`, which provides equations linking the different model attributes.
```
from sodym import MFASystem, MFADefinition, DimensionDefinition, FlowDefinition

class MyMFA(MFASystem):
    def set_up_definition(self):
        dimensions = [DimensionDefinition(name='Time', dim_letter='t', dtype=int)]
        processes = ['sysenv', 'use']
        flows = [
            FlowDefinition(from_process='sysenv', to_process='use', dim_letters=('t', )),
            FlowDefinition(from_process='use', to_process='sysenv', dim_letters=('t', )),
        ]
        ...
        return MFADefinition(dimensions=dimensions, processes=processes, ...)

    def compute(self):
        self.flows['use => sysenv'] = self.stocks['use'].outflow
        ...
```
As well as the system definition, described above, the model also requires some data, and this data must also pass some validations.
For this, sodym has the `DataReader` class, which is a template specifying the methods and output types required, in order for the user to create their own data reader, depending on how they wish to store their model data.
An `ExampleDataReader` is also defined in sodym and provides functionality for reading dimension and parameter datasets from .csv files.

After the user has defined their MFA system class and their data reader, they can be put together and the model can be run.
```
my_data_reader = MyDataReader()
my_mfa = MyMFA(data_reader=my_data_reader)
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
