from einops import rearrange
import numpy as np

from .named_dim_arrays import Flow, NamedDimArray
from .dimensions import Dimension, DimensionSet


def ndarray_stack(ndarrays: list[NamedDimArray], dimension: Dimension) -> NamedDimArray:
    """Stack a list of NamedDimArray objects using a new dimension.
    Like numpy.stack with axis=-1, but for `NamedDimArray`s.
    Method can be applied to `NamedDimArray`s, `StockArray`s, `Parameter`s and `Flow`s.
    """
    ndarray0 = ndarrays[0]
    for ndarray in ndarrays[1:]:
        if ndarray.dims != ndarray0.dims:
            raise ValueError(
                'Existing dimensions must be identical in all objects to be stacked.'
            )
    axis = len(ndarray0.dims.dimensions)
    extended_values = np.stack([ndarray.values for ndarray in ndarrays], axis=axis)
    extended_dimensions = DimensionSet(dimensions=ndarray0.dims.dimensions+[dimension])
    if isinstance(ndarray0, Flow):
        return Flow(
            values=extended_values,
            dims=extended_dimensions,
            from_process=ndarray0.from_process,
            to_process=ndarray0.to_process,
            )
    # otherwise, for NamedDimArray, StockArray or Parameter
    return ndarray0.__class__(values=extended_values, dims=extended_dimensions)


def ndarray_split(ndarray: NamedDimArray, dim_letter: str) -> dict[NamedDimArray]:
    """Reverse the ndarray_stack, returns a dictionary of `NamedDimArray`s
    associated with the item in the dimension that has been split.
    Method can be applied to `NamedDimArray`s, `StockArray`s, `Parameter`s and `Flow`s.
    """
    if dim_letter not in ndarray.dims.letters:
        raise ValueError('Dimension to split on must exist in the ndarray')
    smaller_dimensions = ndarray.dims.drop(dim_letter, inplace=False)
    extracted_dimension = ndarray.dims[dim_letter]
    reorganised_values = rearrange(
        ndarray.values, f'{ndarray.dims.spaced_string} -> {dim_letter} {smaller_dimensions.spaced_string}'
    )
    kwargs = {}
    if isinstance(ndarray, Flow):
        kwargs = {"from_process": ndarray.from_process, "to_process": ndarray.to_process}
    return {
        item: ndarray.__class__(
            values=reorganised_values[i],
            dims=smaller_dimensions,
            **kwargs
        ) for i, item in enumerate(extracted_dimension.items)
    }

