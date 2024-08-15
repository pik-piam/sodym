from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
from .mfa_definition import MFADefinition
from .named_dim_arrays import Flow, Process, Parameter, NamedDimArray
from .stocks import Stock
from .data_reader import DataReader


class MFASystem(ABC):
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

    def __init__(self, data_reader: DataReader, model_cfg: dict={}):
        """Define and set up the MFA system and load all required data.
        Does not compute stocks or flows yet."""
        for k, v in model_cfg.items():  # Allows other model properties to be set as attributes,
            # which can be acccessed in other class methods.
            setattr(self, k, v)
        self.definition = self.set_up_definition()
        self.dims = data_reader.read_dimensions(self.definition.dimensions)
        self.parameters = self.read_parameters(data_reader=data_reader)
        self.scalar_parameters = data_reader.read_scalar_data(self.definition.scalar_parameters)
        self.processes = {
            name: Process(name=name, id=id) for id, name in enumerate(self.definition.processes)
        }
        self.flows = self.initialize_flows(processes=self.processes)
        self.stocks = self.initialize_stocks(processes=self.processes)

    @abstractmethod
    def compute(self):
        """Perform all computations for the MFA system."""
        pass

    @abstractmethod
    def set_up_definition(self) -> MFADefinition:
        """Wrapper for the fill_definition routine defined in the subclass."""
        pass

    def initialize_flows(self, processes: Dict[str, Process]) -> Dict[str, Flow]:
        """Initialize all defined flows with zero values."""
        flows = {}
        for flow_definition in self.definition.flows:
            try:
                from_process = processes[flow_definition.from_process_name]
                to_process = processes[flow_definition.to_process_name]
            except KeyError:
                raise KeyError(f"Missing process required by flow definition {flow_definition}.")
            dims = self.dims.get_subset(flow_definition.dim_letters)
            flow = Flow(from_process=from_process, to_process=to_process, dims=dims)
            flows[flow.name] = flow
        return flows

    def initialize_stocks(self, processes: Dict[str, Process]) -> Dict[str, Stock]:
        """Initialize all defined stocks with zero values."""
        stocks = {}
        for stock_definition in self.definition.stocks:
            dims = self.dims.get_subset(stock_definition.dim_letters)
            try:
                process = processes[stock_definition.process_name]
            except KeyError:
                raise KeyError(f"Missing process required by stock definition {stock_definition}.")
            stock = Stock.from_definition(stock_definition, dims=dims, process=process)
            stocks[stock.name] = stock
        return stocks

    def read_parameters(self, data_reader: DataReader) -> Dict[str, Parameter]:
        """Use the data_reader (DataReader object) to obtain data for all the parameters
        in the MFA system definition."""
        parameters = {}
        for parameter in self.definition.parameters:
            dims = self.dims.get_subset(parameter.dim_letters)
            parameters[parameter.name] = data_reader.read_parameter_values(
                parameter=parameter.name, dims=dims
            )
        return parameters

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
