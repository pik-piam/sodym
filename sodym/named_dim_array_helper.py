import numpy as np

from .named_dim_arrays import Flow, NamedDimArray
from .dimensions import Dimension, DimensionSet


def named_dim_array_stack(named_dim_arrays: list[NamedDimArray], dimension: Dimension) -> NamedDimArray:
    """Stack a list of NamedDimArray objects using a new dimension.
    Like numpy.stack with axis=-1, but for `NamedDimArray`s.
    Method can be applied to `NamedDimArray`s, `StockArray`s, `Parameter`s and `Flow`s.
    """
    named_dim_array0 = named_dim_arrays[0]
    if len(named_dim_arrays) != dimension.len:
        raise ValueError(
            'Number of objects to stack must be equal to length of new dimension'
        )
    for named_dim_array in named_dim_arrays[1:]:
        if named_dim_array.dims != named_dim_array0.dims:
            raise ValueError(
                'Existing dimensions must be identical in all objects to be stacked.'
            )
    axis = len(named_dim_array0.dims.dimensions)
    extended_values = np.stack(
        [named_dim_array.values for named_dim_array in named_dim_arrays], axis=axis
    )
    extended_dimensions = DimensionSet(dimensions=named_dim_array0.dims.dimensions+[dimension])
    if isinstance(named_dim_array0, Flow):
        return Flow(
            values=extended_values,
            dims=extended_dimensions,
            from_process=named_dim_array0.from_process,
            to_process=named_dim_array0.to_process,
            )
    # otherwise, for NamedDimArray, StockArray or Parameter
    return named_dim_array0.__class__(values=extended_values, dims=extended_dimensions)
