from abc import abstractmethod
from typing import Dict, Optional

import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict

from .mfa_definition import MFADefinition
from .dimensions import DimensionSet
from .named_dim_arrays import Flow, Process, Parameter, NamedDimArray
from .stocks import Stock
from .stock_helper import make_empty_stocks
from .flow_helper import make_empty_flows
from .data_reader import DataReader


class MFASystem(PydanticBaseModel):
    """An MFASystem class handles the calculation of a Material Flow Analysis system, which
    consists of a set of processes, flows, stocks defined over a set of dimensions. 
    For the concrete definition of the system, a subclass of MFASystem must be implemented.

    **Example**

    Define your MFA System:

    >>> from sodym import MFASystem
    >>> class CustomMFA(MFASystem):
    >>>     def compute(self):
    >>>         # do some computations on the CustomMFA attributes: stocks and flows

    Initialize and run your MFA System model:

    >>> from sodym import ExampleDataReader
    >>> data_reader = ExampleDataReader(dimension_datasets={...}, ...)
    >>> dimension_definitions = [DimensionDefinition(name='time', letter='t', dtype=int), ...]
    >>> dims = data_reader.read_dimensions(dimension_definitions)
    >>> mfa = MFASystem(dim=dims, ...)
    >>> mfa.compute()

    MFA flows, stocks and parameters are defined as instances of subclasses of :py:class:`sodym.named_dim_arrays.NamedDimArray`.
    Dimensions are managed with the :py:class:`sodym.dimensions.Dimension` and :py:class:`sodym.dimensions.DimensionSet`.
    """
    model_config = ConfigDict(protected_namespaces=())

    dims: DimensionSet
    parameters: Dict[str, Parameter]
    scalar_parameters: Optional[dict] = {}
    processes: Dict[str, Process]
    flows: Dict[str, Flow]
    stocks: Dict[str, Stock]

    @classmethod
    def from_data_reader(cls, definition: MFADefinition, data_reader: DataReader):
        """Define and set up the MFA system and load all required data.
        Initialises stocks and flows with all zero values."""
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
            processes=processes, flows=flows, stocks=stocks,
        )

    @abstractmethod
    def compute(self):
        """Perform all computations for the MFA system."""
        pass

    def get_new_array(self, **kwargs) -> NamedDimArray:
        dims = self.dims.get_subset(kwargs["dim_letters"]) if "dim_letters" in kwargs else self.dims
        return NamedDimArray(dims=dims, **kwargs)

    def get_mass_balance(self):
        """The mass balance of a process is calculated as the sum of
        - all flows entering subtracted by all flows leaving (-) the process
        - the stock change of the process
        Start with minimum possible dimensionality;
        addition and subtraction will automatically reduce to the maximum shape,
        i.e. the dimensions contained in all flows to and from the process.
        """
        balance = {p : 0.0 for p in self.processes.keys()}

        # Add flows to mass balance
        for flow in self.flows.values():
            balance[flow.from_process.name] -= flow  # Subtract flow from start process
            balance[flow.to_process.name] += flow  # Add flow to end process

        # Add stock changes to the mass balance
        for stock in self.stocks.values():
            if stock.process_id is None:  # not connected to a process
                continue
            # add/subtract stock changes to processes
            balance[stock.process.name] -= stock.inflow
            balance[stock.process.name] += stock.outflow
            # add/subtract stock changes in system boundary for mass balance of whole system
            balance['sysenv'] += stock.inflow
            balance['sysenv'] -= stock.outflow

        return balance

    def get_mass_totals(self):
        """The total mass of a process is caluclated as the sum of
        - all flows entering and leaving the process
        - the stock change of the process
        """
        totals = {p : 0.0 for p in self.processes.keys()}

        for flow in self.flows.values():
            totals[flow.from_process.name] += flow  # Add flow to total of start process
            totals[flow.to_process.name] += flow  # Add flow to total of end process

        for stock in self.stocks.values():
            if stock.process_id is None:  # not connected to a process
                continue
            totals[stock.process.name] += stock.inflow
            totals[stock.process.name] += stock.outflow

        return totals

    def get_relative_mass_balance(self, epsilon=1e-9):
        """Determines a relative mass balance for each process of the MFA system,
        by dividing the mass balances by the mass totals.
        """
        balances = self.get_mass_balance()
        totals = self.get_mass_totals()
        relative_balance = {
            p_name : (balances[p_name] / (totals[p_name] + epsilon)).values
            for p_name in self.processes
        }
        return relative_balance

    def check_mass_balance(self):
        """Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't."""

        print("Checking mass balance...")
        # returns array with dim [t, process, e]
        relative_balance = self.get_relative_mass_balance()  # assume no error if total sum is 0
        id_failed = {
            p_name : np.any(balance_percentage > 0.1) for p_name, balance_percentage in relative_balance.items()
        }  # error is bigger than 0.1 %
        messages_failed = [
            f"{p_name} ({np.max(relative_balance[p_name])*100:.2f}% error)"
            for p_name in self.processes.keys()
            if id_failed[p_name] and p_name!='sysenv'
        ]
        if any(id_failed.values()):
            raise RuntimeError(f"Error, Mass Balance fails for processes {', '.join(messages_failed)}")
        else:
            print("Success - Mass balance consistent!")
        return
