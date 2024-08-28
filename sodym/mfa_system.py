from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel as PydanticBaseModel

from .mfa_definition import MFADefinition
from .dimensions import DimensionSet
from .named_dim_arrays import Flow, Process, Parameter, NamedDimArray
from .stocks import Stock
from .stock_helper import make_empty_stocks
from .flow_helper import make_empty_flows
from .data_reader import DataReader


class MFASystem(PydanticBaseModel):
    """An MFASystem class handles the definition, setup and calculation of a Material Flow Analysis system, which
    consists of a set of processes, flows, stocks defined over a set of dimensions. 
    For the concrete definition of the system, a subclass of MFASystem must be implemented.

    **Example**

    Define your MFA System:

    >>> from sodym import MFASystem
    >>> class CustomMFA(MFASystem):
    >>>     def set_up_definition(self) -> MFADefinition:
    >>>         # define the model here
    >>>     def compute(self):
    >>>         # do some computations on the CustomMFA attributes: stocks and flows

    Initialize and run your MFA System model:

    >>> from sodym import ExampleDataReader
    >>> data_reader = ExampleDataReader(...)
    >>> mfa = MFASystem(data_reader=data_reader)
    >>> mfa.compute()

    MFA flows, stocks and parameters are defined as instances of subclasses of :py:class:`sodym.named_dim_arrays.NamedDimArray`.
    Dimensions are managed with the :py:class:`sodym.dimensions.Dimension` and :py:class:`sodym.dimensions.DimensionSet`.
    """
    dims: DimensionSet
    parameters: Dict[str, Parameter]
    scalar_parameters: Optional[dict] = {}
    processes: Dict[str, Process]
    flows: Dict[str, Flow]
    stocks: Dict[str, Stock]
    mfa_cfg: Optional[dict] = {}

    @classmethod
    def from_data_reader(cls, definition: MFADefinition, data_reader: DataReader, mfa_cfg: dict={}):
        """Define and set up the MFA system and load all required data.
        Does not compute stocks or flows yet."""
        dims = data_reader.read_dimensions(definition.dimensions)
        parameters = data_reader.read_parameters(definition.parameters, dims=dims)
        scalar_parameters = data_reader.read_scalar_data(definition.scalar_parameters)
        processes = {
            name: Process(name=name, id=id) for id, name in enumerate(definition.processes)
        }
        flows = make_empty_flows(processes=processes, flow_definitions=definition.flows, dims=dims)
        stocks = make_empty_stocks(processes=processes, stock_definitions=definition.stocks, dims=dims)
        return cls(
            dims=dims, parameters=parameters, scalar_parameters=scalar_parameters,
            processes=processes, flows=flows, stocks=stocks, mfa_cfg=mfa_cfg,
        )

    @abstractmethod
    def compute(self):
        """Perform all computations for the MFA system."""
        pass

    def get_new_array(self, **kwargs) -> NamedDimArray:
        dims = self.dims.get_subset(kwargs["dim_letters"]) if "dim_letters" in kwargs else self.dims
        return NamedDimArray(dims=dims, **kwargs)

    def get_relative_mass_balance(self):
        """Determines a relative mass balance for each process of the MFA system.

        The mass balance of a process is calculated as the sum of
        - all flows entering subtracted by all flows leaving (-) the process
        - the stock change of the process

        The total mass of a process is caluclated as the sum of
        - all flows entering and leaving the process
        - the stock change of the process

        The process with ID 0 is the system boundary. Its mass balance serves as a mass balance of the whole system."""

        # start of with largest possible dimensionality;
        # addition and subtraction will automatically reduce to the maximum shape, i.e. the dimensions contained in all
        # flows to and from the process
        balance = [0.0 for _ in self.processes]
        total = [0.0 for _ in self.processes]

        # Add flows to mass balance
        for flow in (
            self.flows.values()
        ):  # values refers here to the values of the flows dictionary which are the Flows themselves
            balance[flow.from_process_id] -= flow  # Subtract flow from start process
            balance[flow.to_process_id] += flow  # Add flow to end process
            total[flow.from_process_id] += flow  # Add flow to total of start process
            total[flow.to_process_id] += flow  # Add flow to total of end process

        # Add stock changes to the mass balance
        for stock in self.stocks.values():
            if stock.process_id is None:  # not connected to a process
                continue
            balance[stock.process_id] -= stock.inflow
            balance[0] += (
                stock.inflow
            )  # subtract stock changes to process with number 0 (system boundary) for mass balance of whole system
            balance[stock.process_id] += stock.outflow
            balance[0] -= (
                stock.outflow
            )  # add stock changes to process with number 0 (system boundary) for mass balance of whole system

            total[flow.from_process_id] += stock.inflow
            total[flow.to_process_id] += stock.outflow

        relative_balance = [(b / (t + 1.0e-9)).values for b, t in zip(balance, total)]

        return relative_balance

    def check_mass_balance(self):
        """Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't."""

        print("Checking mass balance...")
        # returns array with dim [t, process, e]
        relative_balance = self.get_relative_mass_balance()  # assume no error if total sum is 0
        id_failed = [
            np.any(balance_percentage > 0.1) for balance_percentage in relative_balance
        ]  # error is bigger than 0.1 %
        messages_failed = [
            f"{p.name} ({np.max(relative_balance[p.id])*100:.2f}% error)"
            for p in self.processes.values()
            if id_failed[p.id]
        ]
        if np.any(np.array(id_failed[1:])):
            raise RuntimeError(f"Error, Mass Balance fails for processes {', '.join(messages_failed)}")
        else:
            print("Success - Mass balance consistent!")
        return
