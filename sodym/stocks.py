import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional
from .dynamic_stock_model import DynamicStockModel
from .named_dim_arrays import StockArray, Parameter, Process
from .dimensions import DimensionSet
from .mfa_definition import StockDefinition


class Stock(PydanticBaseModel):
    """Stock objects are components of an MFASystem, where materials can accumulate over time.
    They consist of three :py:class:`sodym.named_dim_arrays.NamedDimArrays` : stock (the accumulation), inflow, outflow.

    The base class only allows to compute the stock from known inflow and outflow.
    The subclass StockWithDSM allows computations using a lifetime distribution function, which is necessary if not both
    inflow and outflow are known.
    """
    model_config = ConfigDict(protected_namespaces=())

    stock: StockArray
    inflow: StockArray
    outflow: StockArray
    name: str
    process_name: Optional[str] = None
    process: Process

    @model_validator(mode="after")
    def check_process_names(self):
        if self.process_name and self.process.name != self.process_name:
            raise ValueError("Missmatching process names in Stock object")
        self.process_name = self.process.name
        return self

    @classmethod
    def from_definition(cls, stock_definition: StockDefinition, dims: DimensionSet, process: Process):
        name = stock_definition.name
        stock = StockArray(dims=dims, name=f"{name}_stock")
        inflow = StockArray(dims=dims, name=f"{name}_inflow")
        outflow = StockArray(dims=dims, name=f"{name}_outflow")
        return cls(name=name, stock=stock, inflow=inflow, outflow=outflow, process=process)

    @property
    def process_id(self):
        return self.process.id

    def compute_stock(self):
        self.stock.values[...] = np.cumsum(self.inflow.values - self.outflow.values, axis=self.stock.dims.index("t"))


class StockWithDSM(Stock):
    """Computes stocks, inflows and outflows based on a lifetime distribution function.

    It does so by interfacing the Stock class, which is based on NamedDimArray objects with the DynamicStockModel class,
    which contains the number crunching and takes numpy arrays as input.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    dsm: Optional[DynamicStockModel] = None
    ldf_type: Optional[str] = None
    lifetime_mean: Optional[StockArray] = None
    lifetime_std: Optional[StockArray] = None

    def set_lifetime(self, ldf_type, lifetime_mean: Parameter, lifetime_std: Parameter):
        self.ldf_type = ldf_type
        lifetime_mean_values = lifetime_mean.cast_values_to(self.stock.dims)
        self.lifetime_mean = StockArray(
            name=f"{self.name}_lifetime_mean", dims=self.stock.dims, values=lifetime_mean_values
        )
        lifetime_std_values = lifetime_std.cast_values_to(self.stock.dims)
        self.lifetime_std = StockArray(
            name=f"{self.name}_lifetime_std", dims=self.stock.dims, values=lifetime_std_values
        )

    def compute_inflow_driven(self):
        assert self.ldf_type is not None, "lifetime not yet set"
        assert self.inflow is not None, "inflow not yet set"
        self.dsm = DynamicStockModel(
            shape=self.stock.dims.shape(),
            inflow=self.inflow.values,
            ldf_type=self.ldf_type,
            lifetime_mean=self.lifetime_mean.values,
            lifetime_std=self.lifetime_std.values,
        )
        self.dsm.compute_inflow_driven()
        self.outflow.values[...] = self.dsm.outflow
        self.stock.values[...] = self.dsm.stock

    def compute_stock_driven(self):
        assert self.ldf_type is not None, "lifetime not yet set"
        assert self.stock is not None, "stock arry not yet set"
        self.dsm = DynamicStockModel(
            shape=self.stock.dims.shape(),
            stock=self.stock.values,
            ldf_type=self.ldf_type,
            lifetime_mean=self.lifetime_mean.values,
            lifetime_std=self.lifetime_std.values,
        )
        self.dsm.compute_stock_driven()
        self.inflow.values[...] = self.dsm.inflow
        self.outflow.values[...] = self.dsm.outflow
