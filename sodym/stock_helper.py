import logging
from typing import Optional
from .survival_functions import (
    FixedSurvival, FoldedNormalSurvival, NormalSurvival, LogNormalSurvival, WeibullSurvival,
)
from .named_dim_arrays import StockArray, Parameter, Process
from .dimensions import DimensionSet
from .mfa_definition import StockDefinition
from .stocks import DynamicStockModel, InflowDrivenDSM, StockDrivenDSM, FlowDrivenStock


def make_empty_stock(stock_definition: StockDefinition, dims: DimensionSet, process: Process):
    name = stock_definition.name
    stock = StockArray(dims=dims, name=f"{name}_stock")
    inflow = StockArray(dims=dims, name=f"{name}_inflow")
    outflow = StockArray(dims=dims, name=f"{name}_outflow")
    return FlowDrivenStock(name=name, stock=stock, inflow=inflow, outflow=outflow, process=process)


def create_dynamic_stock(
    name: str, process: Process,
    time_letter: str='t',
    stock: Optional[StockArray] = None,
    inflow: Optional[StockArray] = None,
    process_name: Optional[str] = None,
    ldf_type: Optional[str] = None,
    lifetime_mean: Optional[Parameter] = None,
    lifetime_std: Optional[Parameter] = None,
) -> DynamicStockModel:
    if stock is None and inflow is None:
        raise ValueError('Either stock or inflow must be passed to create a dynamic stock object.')
    dims = stock.dims if stock is not None else inflow.dims
    lifetime_mean, lifetime_std = cast_lifetime(lifetime_mean, lifetime_std, dims)
    survival_model = get_survival_model(ldf_type)(
        dims=dims, lifetime_mean=lifetime_mean, lifetime_std=lifetime_std, time_letter=time_letter
    )
    if stock is not None:
        logging.info('Creating StockDrivenDSM object')
        return StockDrivenDSM(
            name=name, process=process, process_name=process_name,
            stock=stock, survival_model=survival_model,
        )
    elif inflow is not None:
        logging.info('Creating InflowDrivenDSM object')
        return InflowDrivenDSM(
            name=name, process=process, process_name=process_name,
            inflow=inflow, survival_model=survival_model
        )


def cast_lifetime(lifetime_mean: Parameter, lifetime_std: Parameter, dims: DimensionSet):
    lifetime_mean_values = lifetime_mean.cast_values_to(dims)
    lifetime_std_values = lifetime_std.cast_values_to(dims)
    return lifetime_mean_values, lifetime_std_values


def get_survival_model(ldf_type):
    survival_map = {
        'Fixed': FixedSurvival,
        'Normal': NormalSurvival,
        'FoldedNormal': FoldedNormalSurvival,
        'LogNormal': LogNormalSurvival,
        'Weibull': WeibullSurvival,
    }
    if ldf_type not in survival_map:
        raise ValueError(f'ldf_type must be one of {list(survival_map.keys())}.')
    return survival_map[ldf_type]
