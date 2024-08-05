"""
Concepts based on:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Re-written for use in simson project
"""

import numpy as np
from abc import ABC, abstractmethod
from .mfa_definition import MFADefinition, StockDefinition
from .named_dim_arrays import Flow, Parameter, Process, NamedDimArray
from .stocks_in_mfa import Stock, StockWithDSM
from .dimensions import Dimension, DimensionSet
from ..tools.read_data import read_scalar_data


class MFASystem(ABC):
    """
    An MFASystem class handles the definition, setup and calculation of a Material Flow Analysis system,
    which consists of a set of processes, flows, stocks defined over a set of dimensions.
    For the concrete definition of the system, a subclass of MFASystem must be implemented.

    MFA flows, stocks and parameters are defined as instances of subclasses of NamedDimArray.
    Dimensions are managed with the Dimension and DimensionSet .
    Please refer to these classes for further information.
    """

    def __init__(self):
        """
        Define and set up the MFA system and load all required data.
        Does not compute stocks or flows yet.
        """
        self.set_up_definition()
        self.set_up_dimensions()
        self.initialize_processes()
        self.initialize_flows()
        self.initialize_stocks()
        self.initialize_parameters()
        self.initialize_scalar_parameters()

    @abstractmethod
    def compute(self):
        """
        Perform all computations for the MFA system.
        """
        pass

    @abstractmethod
    def set_up_definition(self):
        """
        Wrapper for the fill_definition routine defined in the subclass
        """
        self.definition = MFADefinition()

    def set_up_dimensions(self):
        """
        Given the dimension definition in the subclass,
        which includes file names for loading of a list of items along each dimension,
        this function loads a DimensionSet object, which includes loading of the items along each dimension.
        The mandatory Time dimension gets additional special treatment, to handle past and future.
        """
        dim_constructor_args = [dict(d) | {'do_load': True} for d in self.definition.dimensions]
        self.dims = DimensionSet(arg_dicts_for_dim_constructors=dim_constructor_args)
        self.set_up_years()

    def set_up_years(self):
        """
        Load historic years from file, and deduct future years as non-historic years from the Time dimension.
        Get indices for all historic and future years for array slicing.
        """
        self.years = self.dims._dict['Time']
        self.historic_years = Dimension(name='historic_years')
        self.historic_years.load_items(dtype=int)
        self.future_years = Dimension(name='future_years')
        self.future_years.set_items([y for y in self.dims['Time'].items if y not in self.historic_years.items])
        self.i_historic = np.arange(self.historic_years.len)
        self.i_future = np.arange(self.historic_years.len, self.dims['Time'].len)

    def initialize_processes(self):
        """Convert the process definition list to dict of Process objects, indexed by name."""
        self.processes = {name: Process(name, id) for id, name in enumerate(self.definition.processes)}

    def initialize_flows(self):
        """
        Convert the flow definition list to dict of Process objects initialized with value 0., indexed by name.
        Flow names are deducted from the processes they connect.
        """
        flow_list = [Flow(flow_definition) for flow_definition in self.definition.flows]
        self.flows = {f.name: f for f in flow_list}
        for f in self.flows.values():
            f.init_dimensions(self.dims)
            f.attach_to_processes(self.processes)

    def initialize_stocks(self):
        self.stocks = {sd.name: Stock(sd) for sd in self.definition.stocks}
        for s in self.stocks.values():
            s.init_dimensions(self.dims)
            s.init_arrays()
            s.attach_to_process(self.processes)

    def initialize_parameters(self):
        self.parameters = {prd.name: Parameter(prd) for prd in self.definition.parameters}
        for p in self.parameters.values():
            p.init_dimensions(self.dims)
            p.load_values()

    def initialize_scalar_parameters(self):
        self.scalar_parameters = {spd['name']: read_scalar_data(spd['name']) for spd in self.definition.scalar_parameters}

    def get_new_stock(self, with_dsm: bool=False, **kwargs):
        stock_definition = StockDefinition(**kwargs)
        if with_dsm:
            return StockWithDSM(parent_alldims=self.dims, stock_definition=stock_definition)
        else:
            return Stock(parent_alldims=self.dims, stock_definition=stock_definition)

    def get_new_array(self, **kwargs):
        return NamedDimArray(parent_alldims=self.dims, **kwargs)

    def get_subset_transformer(self, dim_letters: tuple):
        """
        Get a Parameter/NamedDimArray which transforms between two dimensions, one of which is a subset of the other.
        """
        assert len(dim_letters) == 2, "Only two dimensions are allowed"
        dims = self.dims.get_subset(dim_letters)
        assert set(dims[0].items).issubset(set(dims[1].items)) or set(dims[1].items).issubset(set(dims[0].items)), \
            f"Dimensions '{dims[0].name}' and '{dims[1].name}' are not subset and superset or vice versa."
        out = NamedDimArray(name=f'transform_{dims[0].letter}_<->_{dims[1].letter}', parent_alldims=dims)
        # set all values to 1 if first axis item equals second axis item
        for i, item in enumerate(dims[0].items):
            if item in dims[1].items:
                out.values[i, dims[1].index(item)] = 1
        return out

    def get_relative_mass_balance(self):
        """
        Determines a relative mass balance for each process of the MFA system.

        The mass balance of a process is calculated as the sum of
        - all flows entering subtracted by all flows leaving (-) the process
        - the stock change of the process

        The total mass of a process is caluclated as the sum of
        - all flows entering and leaving the process
        - the stock change of the process



        The process with ID 0 is the system boundary. Its mass balance serves as a mass balance of the whole system.
        """

        # start of with largest possible dimensionality;
        # addition and subtraction will automatically reduce to the maximum shape, i.e. the dimensions contained in all flows to and from the process
        balance = [0. for _ in self.processes]
        total = [0. for _ in self.processes]

        # Add flows to mass balance
        for flow in self.flows.values():  # values refers here to the values of the flows dictionary which are the Flows themselves
            balance[flow.from_process_id] -= flow # Subtract flow from start process
            balance[flow.to_process_id]   += flow # Add flow to end process
            total[flow.from_process_id] += flow # Add flow to total of start process
            total[flow.to_process_id] += flow # Add flow to total of end process

        # Add stock changes to the mass balance
        for stock in self.stocks.values():
            if stock.process_id is None: # not connected to a process
                continue
            balance[stock.process_id] -= stock.inflow
            balance[0] += stock.inflow # subtract stock changes to process with number 0 (system boundary) for mass balance of whole system
            balance[stock.process_id] += stock.outflow
            balance[0] -= stock.outflow # add stock changes to process with number 0 (system boundary) for mass balance of whole system

            total[flow.from_process_id] += stock.inflow
            total[flow.to_process_id] += stock.outflow

        relative_balance = [(b / (t + 1.e-9)).values for b, t in zip(balance, total)]


        return relative_balance


    def check_mass_balance(self):
        """
        Compute mass balance, and check whether it is within a certain tolerance.
        Throw an error if it isn't.
        """

        print("Checking mass balance...")
        # returns array with dim [t, process, e]
        relative_balance = self.get_relative_mass_balance()  # assume no error if total sum is 0
        id_failed = [np.any(balance_percentage > 0.1) for balance_percentage in relative_balance]# error is bigger than 0.1 %
        messages_failed = [f'{p.name} ({np.max(relative_balance[p.id])*100:.2f}% error)' for p in self.processes.values() if id_failed[p.id]]
        if np.any(np.array(id_failed[1:])):
                raise RuntimeError(f"Error, Mass Balance fails for processes {', '.join(messages_failed)}")
        else:
            print("Success - Mass balance consistent!")
        return

    @property
    @abstractmethod
    def display_names(self):
        """
        Dictionary to change the string that variables are displayed with in figures.
        """
        return {}
