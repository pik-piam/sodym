"""
Concepts based on:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Re-written for use in simson project
"""

from copy import copy
import numpy as np
import pandas as pd
from pydantic import BaseModel as PydanticBaseModel, ConfigDict, model_validator
from typing import Optional

from .dimensions import DimensionSet
from .mfa_definition import DefinitionWithDimLetters


class NamedDimArray(PydanticBaseModel):
    """ "Parent class for an array with pre-defined dimensions, which are addressed by name. Operations between
    different multi-dimensional arrays can than be performed conveniently, as the dimensions are automatically matched.

    In order to 'fix' the dimensions of the array, the array has to be 'declared' by calling the NamedDimArray object
    constructor with a set of dimensions before working with it.
    Basic mathematical operations between NamedDimArrays are defined, which return a NamedDimArray object as a result.

    In order to set the values of a NamedDimArray object to that of another one, the ellipsis slice ('[...]') can be
    used, e.g.
    foo[...] = bar.
    This ensures that the dimensionality of the array (foo) is not changed, and that the dimensionality of the
    right-hand side NamedDimArray (bar) is consistent.
    While the syntaxes like of 'foo = bar + baz' are also possible (where 'bar' and 'baz' are NamedDimArrays),
    it is not recommended, as it provides no control over the dimensionality of 'foo'. Use foo[...] = bar + baz instead.

    The values of the NamedDimArray object are stored in a numpy array, and can be accessed directly via the 'values'
    attribute.
    So if type(bar) is np.ndarray, the operation
    foo.values[...] = bar
    is also possible.
    It is not recommended to use 'foo.values = bar' without the slice, as this might change the dimensionality of
    foo.values.

    Subsets of arrays can be set or retrieved.
    Here, slicing information is passed instead of the ellipsis to the square brackets of the NamedDimArray, i.e.
    foo[keys] = bar or foo = bar[keys]. For details on the allowed values of 'keys', see the docstring of the
    SubArrayHandler class.

    The dimensions of a NamedDimArray stored as a DimensionSet object in the 'dims' attribute."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dims: DimensionSet
    values: Optional[np.ndarray] = None
    name: Optional[str] = "unnamed"

    @model_validator(mode="after")
    def fill_values(self):
        if self.values is None:
            self.values = np.zeros(self.dims.shape())
        elif self.values.shape != self.dims.shape():
            raise ValueError("Values passed to {self.__cls__.__name__} must have the same shape as the DimensionSet.")
        return self

    @classmethod
    def from_definition_and_parent_alldims(cls, definition: DefinitionWithDimLetters, parent_alldims: DimensionSet):
        dims = parent_alldims.get_subset(definition.dim_letters)
        return cls(dims=dims, **dict(definition))

    @classmethod
    def from_args(
        cls, parent_alldims: DimensionSet, name: str = "unnamed", dim_letters: tuple = None, values: np.ndarray = None
    ):
        """
        - dimensions are set in the form of a DimensionSet object, which is derived as a subset from a parent
          DimensionSet object.
        - values can be initialized directly (usually done for parameters, but not for flows and stocks, which are only
          computed later) or otherwise are filled with zeros
        """
        dims = parent_alldims.get_subset(dim_letters)
        return cls(dims=dims, name=name, values=values)

    def sub_array_handler(self, definition):
        return SubArrayHandler(self, definition)

    @property
    def shape(self):
        return self.dims.shape()

    def sum_values(self):
        return np.sum(self.values)

    def sum_values_over(self, sum_over_dims: tuple = ()):
        result_dims = (o for o in self.dims.letters if o not in sum_over_dims)
        return np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)

    def cast_values_to(self, target_dims: DimensionSet):
        assert all([d in target_dims.letters for d in self.dims.letters]), (
            "Target of cast must contain all " \
            f"dimensions of the object! Source dims '{self.dims.string}' are not all contained in target dims " \
            f"'{target_dims.string}'. Maybe use sum_values_to() before casting"
        )
        # safety procedure: order dimensions
        values = np.einsum(
            f"{self.dims.string}->{''.join([d for d in target_dims.letters if d in self.dims.letters])}", self.values
        )
        index = tuple([slice(None) if d in self.dims.letters else np.newaxis for d in target_dims.letters])
        multiple = tuple([1 if d.letter in self.dims.letters else d.len for d in target_dims])
        values = values[index]
        values = np.tile(values, multiple)
        return values

    def cast_to(self, target_dims: DimensionSet):
        return NamedDimArray(dims=target_dims, values=self.cast_values_to(target_dims), name=self.name)

    def sum_values_to(self, result_dims: tuple = ()):
        return np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)

    def sum_nda_to(self, result_dims: tuple = ()):
        return NamedDimArray(
            dims=self.dims.get_subset(result_dims), values=self.sum_values_to(result_dims), name=self.name
        )

    def sum_nda_over(self, sum_over_dims: tuple = ()):
        result_dims = tuple([d for d in self.dims.letters if d not in sum_over_dims])
        return NamedDimArray(
            dims=self.dims.get_subset(result_dims), values=self.sum_values_over(sum_over_dims), name=self.name
        )

    def _prepare_other(self, other):
        assert isinstance(other, (NamedDimArray, int, float)), (
            "Can only perform operations between two " "NamedDimArrays or NamedDimArray and scalar."
        )
        if isinstance(other, (int, float)):
            other = NamedDimArray(dims=self.dims, values=other * np.ones(self.shape))
        return other

    def intersect_dims_with(self, other):
        matching_dims = []
        for dim in self.dims.dimensions:
            if dim.letter in other.dims.letters:
                matching_dims.append(dim)
        return DimensionSet(dimensions=matching_dims)

    def union_dims_with(self, other):
        all_dims = copy(self.dims.dimensions)
        letters_self = self.dims.letters
        for dim in other.dims.dimensions:
            if dim.letter not in letters_self:
                all_dims.append(dim)
        return DimensionSet(dimensions=all_dims)

    def __add__(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        return NamedDimArray(
            dims=dims_out, values=self.sum_values_to(dims_out.letters) + other.sum_values_to(dims_out.letters)
        )

    def __sub__(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        return NamedDimArray(
            dims=dims_out, values=self.sum_values_to(dims_out.letters) - other.sum_values_to(dims_out.letters)
        )

    def __mul__(self, other):
        other = self._prepare_other(other)
        dims_out = self.union_dims_with(other)
        values_out = np.einsum(f"{self.dims.string},{other.dims.string}->{dims_out.string}", self.values, other.values)
        return NamedDimArray(dims=dims_out, values=values_out)

    def __truediv__(self, other):
        other = self._prepare_other(other)
        dims_out = self.union_dims_with(other)
        values_out = np.einsum(
            f"{self.dims.string},{other.dims.string}->{dims_out.string}", self.values, 1.0 / other.values
        )
        return NamedDimArray(dims=dims_out, values=values_out)

    def minimum(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        values_out = np.minimum(self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters))
        return NamedDimArray(dims=dims_out, values=values_out)

    def maximum(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        values_out = np.maximum(self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters))
        return NamedDimArray(dims=dims_out, values=values_out)

    def __neg__(self):
        return NamedDimArray(dims=self.dims, values=-self.values)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        inv_self = NamedDimArray(dims=self.dims, values=1 / self.values)
        return inv_self * other

    def __getitem__(self, keys):
        """Defines what is returned when the object with square brackets stands on the right-hand side of an assignment,
        e.g. foo = foo = bar[{'e': 'C'}] Here, it is solely used for slicing, the the input tot the square brackets must
        be a dictionary defining the slice."""
        return self.sub_array_handler(keys).to_nda()

    def __setitem__(self, keys, item):
        """Defines what is returned when the object with square brackets stands on the left-hand side of an assignment,
        i.e. 'foo[bar] = baz' For allowed values in the square brackets (bar), see the docstring of the SubArrayHandler
        class.

        The RHS (baz) is required here to be a NamedDimArray.
        If you want to set the values of a NamedDimArray object directly to a numpy array, use the syntax
        'foo.values[...] = bar'."""
        assert isinstance(item, NamedDimArray), "Item on RHS of assignment must be a NamedDimArray"
        slice_obj = self.sub_array_handler(keys)
        slice_obj.values_pointer[...] = item.sum_values_to(slice_obj.dim_letters)

    def to_df(self):
        index = pd.MultiIndex.from_product([d.items for d in self.dims], names=self.dims.names)
        df = index.to_frame(index=False)
        df["value"] = self.values.flatten()
        return df


class SubArrayHandler:
    """This class handles subsets of the 'values' numpy array of a NamedDimArray object, created by slicing along one or
    several dimensions. It specifies the behavior of foo[definition] = bar and foo = bar[definition], where foo and bar
    are NamedDimArray objects. This is done via the __getitem__ and __setitem__ methods of the NamedDimArray class.

    It returns either
    - a new NamedDimArray object (via the to_nda() function), or
    - a pointer to a subset of the values array of the parent NamedDimArray object, via the values_pointer attribute.

    There are several possible syntaxes for the definition of the subset:
    - An ellipsis slice '...' can be used to address all the values of the original NamedDimArray object
        Example: foo[...] addresses all values of the NamedDimArray object foo.
    - A dictionary can be used to define a subset along one or several dimensions.
      The dictionary has the form {'dim_letter': 'item_name'}.
        Example: foo[{'e': 'C'}] addresses all values where the element is carbon,
      Instead of a single 'item_name', a list of 'item_names' can be passed.
        Example: foo[{'e': 'C', 'r': ['EUR', 'USA']}] addresses all values where the element is carbon and the region is
        Europe or the USA.
    - Instead of a dictionary, an item name can be passed directly. In this case, the dimension is inferred from the
      item name.
      Throws an error if the item name is not unique, i.e. occurs in more than one dimension.
        Example: foo['C'] addresses all values where the element is carbon
      Several comma-separated item names can be passed, which appear in __getitem__ and __setitem__ methods as a tuple.
      Those can either be in the same dimension or in different dimensions.
        Example: foo['C', 'EUR', 'USA'] addresses all values where the element is carbon and the region is Europe or the
        USA.

    Note that does not inherit from NamedDimArray, so it is not a NamedDimArray object itself.
    However, one can use it to create a NamedDimArray object with the to_nda() method.
    """

    def __init__(self, named_dim_array: NamedDimArray, definition):
        self.nda = named_dim_array
        self._get_def_dict(definition)
        self.has_dim_with_several_items = any(isinstance(v, (tuple, list, np.ndarray)) for v in self.def_dict.values())
        self._init_ids()

    def _get_def_dict(self, definition):
        if isinstance(definition, type(Ellipsis)):
            self.def_dict = {}
        elif isinstance(definition, dict):
            self.def_dict = definition
        elif isinstance(definition, tuple):
            self.def_dict = self.to_dict_tuple(definition)
        else:
            self.def_dict = self.to_dict_single_item(definition)

    def to_dict_single_item(self, item):
        if isinstance(item, slice):
            raise ValueError(
                "Numpy indexing of NamedDimArrays is not supported. Details are given in the NamedDimArray class "
                "docstring."
            )
        dict_out = None
        for d in self.nda.dims:
            if item in d.items:
                if dict_out is not None:
                    raise ValueError(
                        f"Ambiguous slicing: Item '{item}' is found in multiple dimensions. Please specify the "
                        "dimension by using a slicing dict instead."
                    )
                dict_out = {d.letter: item}
        if dict_out is None:
            raise ValueError(f"Slicing item '{item}' not found in any dimension.")
        return dict_out

    def to_dict_tuple(self, slice_def):
        dict_out = {}
        for item in slice_def:
            key, value = self.to_dict_single_item(item)
            if key not in dict_out:  # if key does not exist, add it
                dict_out[key] = [value]
            else:
                dict_out[key].append(value)
        # if there is only one item along a dimension, convert list to single item
        return {k: v if len(v) > 1 else v[0] for k, v in dict_out.items()}

    @property
    def ids(self):
        """Indices used for slicing the values array."""
        return tuple(self._id_list)

    @property
    def values_pointer(self):
        """Pointer to the subset of the values array of the parent NamedDimArray object."""
        return self.nda.values[self.ids]

    @property
    def dim_letters(self):
        """Updated dimension letters, where sliced dimensions with only one item along that direction are removed."""
        all_letters = self.nda.dims.letters
        # remove the dimensions along which there is only one item
        letters_removed = [d for d, items in self.def_dict.items() if isinstance(items, str)]
        return tuple([d for d in all_letters if d not in letters_removed])

    def to_nda(self):
        """Return a NamedDimArray object that is a slice of the original NamedDimArray object.

        Attention: This creates a new NamedDimArray object, which is not linked to the original one.
        """
        assert (
            not self.has_dim_with_several_items
        ), "Cannot convert to NamedDimArray if there are dimensions with several items"
        dims = self.nda.dims.get_subset(self.dim_letters)
        return NamedDimArray(dims=dims, values=self.values_pointer, name=self.nda.name)

    def _init_ids(self):
        """
        - Init the internal list of index slices to slice(None) (i.e. no slicing, keep all items along that dimension)
        - For each dimension that is sliced, get the corresponding item IDs and set the index slice to these IDs.
        """
        self._id_list = [slice(None) for _ in self.nda.dims.letters]
        for dim_letter, item_or_items in self.def_dict.items():
            item_ids_singledim = self._get_items_ids(dim_letter, item_or_items)
            self._set_ids_singledim(dim_letter, item_ids_singledim)

    def _get_items_ids(self, dim_letter, item_or_items):
        """Given either a single item name or a list of item names, return the corresponding item IDs, along one
        dimension 'dim_letter'."""
        if isinstance(item_or_items, str):  # single item
            return self._get_single_item_id(dim_letter, item_or_items)
        elif isinstance(item_or_items, (tuple, list, np.ndarray)):  # list of items
            return [self._get_single_item_id(dim_letter, item) for item in item_or_items]

    def _get_single_item_id(self, dim_letter, item_name):
        return self.nda.dims[dim_letter].items.index(item_name)

    def _set_ids_singledim(self, dim_letter, ids):
        self._id_list[self.nda.dims.index(dim_letter)] = ids


class Process(PydanticBaseModel):
    """Processes serve as nodes for the MFA system layout definition. Flows are defined between two processes. Stocks
    are connected to a process. Processes do not contain values themselves.

    Processes get an ID by the order they are defined in  in the MFA system definition. The process with ID 0
    necessarily contains everything outside the system boundary.
    """

    name: str
    id: int

    @model_validator(mode="after")
    def check_id0(self):
        if self.id == 0 and self.name != "sysenv":
            raise ValueError(
                "The process with ID 0 must be named 'sysenv', as it contains everything outside the system boundary."
            )
        return self


class Flow(NamedDimArray):
    """The values of Flow objects are the main computed outcome of the MFA system. A flow connects two processes. Its
    name is set as a combination of the names of the two processes it connects.

    Note that it is a subclass of NamedDimArray, so most of the methods are defined in the NamedDimArray class.
    """
    model_config = ConfigDict(protected_namespaces=())

    from_process: Process
    to_process: Process
    from_process_name: Optional[str] = None
    to_process_name: Optional[str] = None

    @model_validator(mode="after")
    def check_process_names(self):
        if self.from_process_name and self.from_process.name != self.from_process_name:
            raise ValueError("Missmatching process names in Flow object")
        self.from_process_name = self.from_process.name
        if self.to_process_name and self.to_process.name != self.to_process_name:
            raise ValueError("Missmatching process names in Flow object")
        self.to_process_name = self.to_process.name
        return self

    @model_validator(mode="after")
    def flow_name_related_to_proccesses(self):
        self.name = f"{self.from_process_name} => {self.to_process_name}"
        return self

    @property
    def from_process_id(self):
        return self.from_process.id

    @property
    def to_process_id(self):
        return self.to_process.id


class StockArray(NamedDimArray):
    """Stocks allow accumulation of material at a process, i.e. between two flows.

    As Stock contains NamedDimArrays for its stock value, inflow and outflow. For details, see the Stock class.
    """

    pass


class Parameter(NamedDimArray):
    """Parameters are used for example to define the share of flows that go into one branch when the flow splits at a
    process.

    All methods are defined in the NamedDimArray parent class.
    """

    pass
