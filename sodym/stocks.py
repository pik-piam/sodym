import numpy as np
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional
from .dynamic_stock_model import (
    DynamicStockModel, InflowDrivenDSM, StockDrivenDSM, SurvivalModel,
    FixedSurvival, FoldedNormalSurvival, NormalSurvival, LogNormalSurvival, WeibullSurvival,
)
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

    def check_stock_balance(self):
        balance = self.get_stock_balance()
        balance = np.max(np.abs(balance).sum(axis=0))
        if balance > 1:  # 1 tonne accuracy
            raise RuntimeError("Stock balance for dynamic stock model is too high: " + str(balance))
        elif balance > 0.001:
            print("Stock balance for model dynamic stock model is noteworthy: " + str(balance))

    def get_stock_balance(self):
        """Check whether inflow, outflow, and stock are balanced.
        If possible, the method returns the vector 'Balance', where Balance = inflow - outflow - stock_change
        """
        dsdt = np.diff(self.stock.values, axis=0, prepend=0)  #stock_change(t) = stock(t) - stock(t-1)
        return self.inflow.values - self.outflow.values - dsdt


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

    @property
    def shape(self):
        return self.stock.dims.shape()

    @property
    def survival_model(self) -> SurvivalModel:
        survival_map = {
            'Fixed': FixedSurvival,
            'Normal': NormalSurvival,
            'FoldedNormal': FoldedNormalSurvival,
            'LogNormal': LogNormalSurvival,
            'Weibull': WeibullSurvival,
        }
        if self.ldf_type not in survival_map:
            raise ValueError(f'ldf_type must be one of {list(survival_map.keys())}.')
        return survival_map[self.ldf_type]

    def compute_inflow_driven(self):
        self.dsm = InflowDrivenDSM(
            shape=self.shape,
            inflow=self.inflow.values,
            survival_model=self.survival_model(
                shape=self.shape,
                lifetime_mean=self.lifetime_mean.values,
                lifetime_std=self.lifetime_std.values,
            )
        )
        self.dsm.compute()
        self.outflow.values[...] = self.dsm.outflow
        self.stock.values[...] = self.dsm.stock
        self.check_stock_balance()

    def compute_stock_driven(self):
        self.dsm = StockDrivenDSM(
            shape=self.shape,
            stock=self.stock.values,
            survival_model=self.survival_model(
                shape=self.shape,
                lifetime_mean=self.lifetime_mean.values,
                lifetime_std=self.lifetime_std.values,
            )
        )
        self.dsm.compute()
        self.inflow.values[...] = self.dsm.inflow
        self.outflow.values[...] = self.dsm.outflow
        self.check_stock_balance()
