"""
Concepts based on:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Re-written for use in simson project
"""

import warnings
import numpy as np
from .dynamic_stock_model import DynamicStockModel
from .named_dim_arrays import StockArray, Parameter
from .dimensions import DimensionSet


class Stock:
    """
    Stock objects are components of an MFASystem, where materials can accumulate over time.
    They consist of three NamedDimArrays: stock (the accumulation), inflow, outflow.

    The base class only allows to compute the stock from known inflow and outflow.
    The subclass StockWithDSM allows computations using a lifetime distribution function, which is necessary if not both inflow and outflow are known.
    """

    def __init__(self, name=None, process=None, dim_letters=None, parent_alldims: DimensionSet=None):
        self.name = name
        self._process_name = process
        self.dim_letters = dim_letters

        self.dims = None
        self.process = None

        self.stock = None
        self.inflow = None
        self.outflow = None

        if parent_alldims is not None:
            self.init_dimensions(parent_alldims)
            self.init_arrays()
        return

    def init_dimensions(self, parent_alldims: DimensionSet):
        self.dims = parent_alldims.get_subset(self.dim_letters)
        if self.dims.names[0] not in ['time', 'Time', 'years', 'Years']:
            warnings.warn(f"The DynamicStockModel class expects time as the first dimension. The name of the first dimension ({self.dims[0].name}) is not recognized as such. Please be sure that it is.")

    def init_arrays(self):
        assert self.dims is not None, "Dimensions must be initialized before arrays"
        self.stock = StockArray(f"{self.name}_stock", self.dim_letters, parent_alldims=self.dims)
        self.inflow = StockArray(f"{self.name}_inflow", self.dim_letters, parent_alldims=self.dims)
        self.outflow = StockArray(f"{self.name}_outflow", self.dim_letters, parent_alldims=self.dims)

    def attach_to_process(self, processes: dict):
        assert self._process_name is not None, "Process name must be set before attaching"
        self.process = processes[self._process_name]
        self.process_id = self.process.id

    def compute_stock(self):
        self.stock.values[...] = np.cumsum(self.inflow.values - self.outflow.values, axis=self.dims.index('t'))


class StockWithDSM(Stock):
    """
    Computes stocks, inflows and outflows based on a lifetime distribution function.
    It does so by interfacing
    the Stock class, which is based on NamedDimArray objects
    with
    the DynamicStockModel class, which contains the number crunching and takes numpy arrays as input.
    """

    def __init__(self, name=None, process=None, dim_letters=None, parent_alldims: DimensionSet=None):
        super().__init__(name, process, dim_letters, parent_alldims)
        self.dsm = None
        self.ldf_type = None
        self.lifetime_mean = None
        self.lifetime_std = None
        return

    def set_lifetime(self, ldf_type, lifetime_mean: Parameter, lifetime_std: Parameter):
        assert self.dims is not None, "Dimensions must be initialized before arrays"
        self.ldf_type = ldf_type
        self.lifetime_mean = StockArray(f"{self.name}_lifetime_mean", self.dim_letters, parent_alldims=self.dims)
        self.lifetime_std = StockArray(f"{self.name}_lifetime_std", self.dim_letters, parent_alldims=self.dims)
        self.lifetime_mean.values[...] = lifetime_mean.cast_values_to(self.dims)
        self.lifetime_std.values[...]  = lifetime_std.cast_values_to(self.dims)

    def compute_inflow_driven(self):
        assert self.ldf_type is not None, "lifetime not yet set"
        assert self.inflow is not None, "inflow not yet set"
        self.dsm = DynamicStockModel(shape=self.dims.shape(),
                                     inflow=self.inflow.values,
                                     ldf_type=self.ldf_type,
                                     lifetime_mean=self.lifetime_mean.values,
                                     lifetime_std=self.lifetime_std.values)
        self.dsm.compute_inflow_driven()
        self.outflow.values[...] = self.dsm.outflow
        self.stock.values[...] = self.dsm.stock

    def compute_stock_driven(self):
        assert self.ldf_type is not None, "lifetime not yet set"
        assert self.stock is not None, "stock arry not yet set"
        self.dsm = DynamicStockModel(shape=self.dims.shape(),
                                     stock=self.stock.values,
                                     ldf_type=self.ldf_type,
                                     lifetime_mean=self.lifetime_mean.values,
                                     lifetime_std=self.lifetime_std.values)
        self.dsm.compute_stock_driven()
        self.inflow.values[...] = self.dsm.inflow
        self.outflow.values[...] = self.dsm.outflow
