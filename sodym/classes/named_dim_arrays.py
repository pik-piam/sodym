"""
Concepts based on:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Re-written for use in simson project
"""

import numpy as np
import pandas as pd
from ..tools.read_data import read_data_to_df, get_np_from_df
from .dimensions import DimensionSet


class NamedDimArray(object):
    """"
    Parent class for an array with pre-defined dimensions, which are addressed by name.
    Operations between different multi-dimensional arrays can than be performed conveniently, as the dimensions are automatically matched.

    In order to 'fix' the dimensions of the array, the array has to be 'declared' by calling the NamedDimArray object constructor with a set of dimensions before working with it.
    Basic mathematical operations between NamedDimArrays are defined, which return a NamedDimArray object as a result.

    In order to set the values of a NamedDimArray object to that of another one, the ellipsis slice ('[...]') can be used, e.g.
    foo[...] = bar.
    This ensures that the dimensionality of the array (foo) is not changed, and that the dimensionality of the right-hand side NamedDimArray (bar) is consistent.
    While the syntaxes like of 'foo = bar + baz' are also possible (where 'bar' and 'baz' are NamedDimArrays),
    it is not recommended, as it provides no control over the dimensionality of 'foo'. Use foo[...] = bar + baz instead.

    The values of the NamedDimArray object are stored in a numpy array, and can be accessed directly via the 'values' attribute.
    So if type(bar) is np.ndarray, the operation
    foo.values[...] = bar
    is also possible.
    It is not recommended to use 'foo.values = bar' without the slice, as this might change the dimensionality of foo.values.

    Subsets of arrays can be set or retrieved.
    Here, slicing information is passed instead of the ellipsis to the square brackets of the NamedDimArray, i.e.
    foo[keys] = bar or foo = bar[keys]. For details on the allowed values of 'keys', see the docstring of the SubArrayHandler class.

    The dimensions of a NamedDimArray stored as a DimensionSet object in the 'dims' attribute.
    """

    def __init__(self, name: str = 'unnamed', dim_letters: tuple = None, parent_alldims: DimensionSet = None, values: np.ndarray = None):
        """
        The minimal initialization sets only the name and the dimension letters.
        Optionally,
        - ...dimensions can be set in the form of a DimensionSet object, which is derived as a subset from a parent DimensionSet object.
        - ...values can be initialized directly (usually done for parameters, but not for flows and stocks, which are only computed later)
        """
        self.name   = name # object name
        assert type(dim_letters) == tuple or dim_letters is None, "dim_letters must be a tuple, if given"
        self._dim_letters = dim_letters

        self.dims = None
        self.values = None

        if parent_alldims is not None:
            self.init_dimensions(parent_alldims)
        if values is not None:
            self.set_values(values)

    def init_dimensions(self, parent_alldims: DimensionSet):
        """
        Get a DimensionSet object of the dimensions that the the array is defined over, by selecting the required subset of the parent_alldims.
        After defining the dimensions, the shape of the value array is known and the array can be initialized.
        """
        self.dims = parent_alldims.get_subset(self._dim_letters) # object name
        self.init_values()

    def init_values(self):
        self.values = np.zeros(self.dims.shape())

    def load_values(self):
        data = self.load_data()
        self.set_values(data)

    def load_data(self):
        data = read_data_to_df(type='dataset', name=self.name)
        data = get_np_from_df(data, self.dims.names)
        return data

    def set_values(self, values: np.ndarray):
        assert self.values is not None, "Values not yet initialized"
        assert values.shape == self.values.shape, "Shape of 'values' input array does not match dimensions of NamedDimArray object"
        self.values[...] = values

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
        assert all([d in target_dims.letters for d in self.dims.letters]), f"Target of cast must contain all dimensions of the object! Source dims '{self.dims.string}' are not all contained in target dims '{target_dims.string}'. Maybe use sum_values_to() before casting"
        # safety procedure: order dimensions
        values = np.einsum(f"{self.dims.string}->{''.join([d for d in target_dims.letters if d in self.dims.letters])}", self.values)
        index = tuple([slice(None) if d in self.dims.letters else np.newaxis for d in target_dims.letters])
        multiple = tuple([1 if d.letter in self.dims.letters else d.len for d in target_dims])
        values = values[index]
        values = np.tile(values, multiple)
        return values

    def cast_to(self, target_dims: DimensionSet):
        return NamedDimArray(dim_letters=target_dims.letters,
                             parent_alldims=target_dims,
                             values=self.cast_values_to(target_dims))

    def sum_values_to(self, result_dims: tuple = ()):
        result = np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)
        return np.einsum(f"{self.dims.string}->{''.join(result_dims)}", self.values)

    def sum_nda_to(self, result_dims: tuple = ()):
        return NamedDimArray(dim_letters=result_dims,
                             parent_alldims=self.dims,
                             values=self.sum_values_to(result_dims))

    def sum_nda_over(self, sum_over_dims: tuple = ()):
        result_dims = tuple([d for d in self.dims.letters if d not in sum_over_dims])
        return NamedDimArray(dim_letters=result_dims,
                             parent_alldims=self.dims,
                             values=self.sum_values_over(sum_over_dims))

    def _prepare_other(self, other):
        assert isinstance(other, (NamedDimArray, int, float)), "Can only perform operations between two NamedDimArrays or NamedDimArray and scalar."
        if isinstance(other, (int, float)):
            other = NamedDimArray(dim_letters=self.dims.letters,
                                  parent_alldims=self.dims,
                                  values=other * np.ones(self.shape))
        return other

    def intersect_dims_with(self, other):
        return DimensionSet(dimensions=list(set(self.dims).intersection(set(other.dims))))

    def union_dims_with(self, other):
        return DimensionSet(dimensions=list(set(self.dims).union(set(other.dims))))

    def __add__(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        return NamedDimArray(dim_letters=dims_out.letters,
                             parent_alldims=dims_out,
                             values=self.sum_values_to(dims_out.letters) + other.sum_values_to(dims_out.letters))

    def __sub__(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        return NamedDimArray(dim_letters=dims_out.letters,
                                parent_alldims=dims_out,
                                values=self.sum_values_to(dims_out.letters) - other.sum_values_to(dims_out.letters))

    def __mul__(self, other):
        other = self._prepare_other(other)
        dims_out = self.union_dims_with(other)
        return NamedDimArray(dim_letters=dims_out.letters,
                             parent_alldims=dims_out,
                             values=np.einsum(f"{self.dims.string},{other.dims.string}->{dims_out.string}", self.values, other.values))

    def __truediv__(self, other):
        other = self._prepare_other(other)
        dims_out = self.union_dims_with(other)
        return NamedDimArray(dim_letters=dims_out.letters,
                             parent_alldims=dims_out,
                             values=np.einsum(f"{self.dims.string},{other.dims.string}->{dims_out.string}", self.values, 1./other.values))

    def minimum(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        return NamedDimArray(dim_letters=dims_out.letters,
                             parent_alldims=dims_out,
                             values=np.minimum(self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters)))

    def maximum(self, other):
        other = self._prepare_other(other)
        dims_out = self.intersect_dims_with(other)
        return NamedDimArray(dim_letters=dims_out.letters,
                             parent_alldims=dims_out,
                             values=np.maximum(self.sum_values_to(dims_out.letters), other.sum_values_to(dims_out.letters)))

    def __neg__(self):
        result = NamedDimArray(dim_letters=self.dims.letters,
                               parent_alldims=self.dims,
                               values=-self.values)
        return result

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        inv_self = NamedDimArray(dim_letters=self.dims.letters,
                                 parent_alldims=self.dims,
                                 values=1/self.values)
        return inv_self * other

    def __getitem__(self, keys):
        """
        Defines what is returned when the object with square brackets stands on the right-hand side of an assignment, e.g. foo = foo = bar[{'e': 'C'}]
        Here, it is solely used for slicing, the the input tot the square brackets must be a dictionary defining the slice.
        """
        return self.sub_array_handler(keys).to_nda()

    def __setitem__(self, keys, item):
        """
        Defines what is returned when the object with square brackets stands on the left-hand side of an assignment, i.e. 'foo[bar] = baz'
        For allowed values in the square brackets (bar), see the docstring of the SubArrayHandler class.

        The RHS (baz) is required here to be a NamedDimArray.
        If you want to set the values of a NamedDimArray object directly to a numpy array, use the syntax 'foo.values[...] = bar'.
        """
        assert isinstance(item, NamedDimArray), "Item on RHS of assignment must be a NamedDimArray"
        slice_obj = self.sub_array_handler(keys)
        slice_obj.values_pointer[...] = item.sum_values_to(slice_obj.dim_letters)

    def to_df(self):
        index = pd.MultiIndex.from_product([d.items for d in self.dims], names=self.dims.names)
        df = index.to_frame(index=False)
        df['value'] = self.values.flatten()
        return df


class SubArrayHandler():
    """
    This class handles subsets of the 'values' numpy array of a NamedDimArray object, created by slicing along one or several dimensions.
    It specifies the behavior of foo[definition] = bar and foo = bar[definition], where foo and bar are NamedDimArray objects.
    This is done via the __getitem__ and __setitem__ methods of the NamedDimArray class.
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
        Example: foo[{'e': 'C', 'r': ['EUR', 'USA']}] addresses all values where the element is carbon and the region is Europe or the USA.
    - Instead of a dictionary, an item name can be passed directly. In this case, the dimension is inferred from the item name.
      Throws an error if the item name is not unique, i.e. occurs in more than one dimension.
        Example: foo['C'] addresses all values where the element is carbon
      Several comma-separated item names can be passed, which appear in __getitem__ and __setitem__ methods as a tuple. Those can either be in the same dimension or in different dimensions.
        Example: foo['C', 'EUR', 'USA'] addresses all values where the element is carbon and the region is Europe or the USA.

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
            raise ValueError("Numpy indexing of NamedDimArrays is not supported. Details are given in the NamedDimArray class docstring.")
        dict_out = None
        for d in self.nda.dims:
            if item in d.items:
                if dict_out is not None:
                    raise ValueError(f"Ambiguous slicing: Item '{item}' is found in multiple dimensions. Please specify the dimension by using a slicing dict instead.")
                dict_out = {d.letter: item}
        if dict_out is None:
            raise ValueError(f"Slicing item '{item}' not found in any dimension.")
        return dict_out

    def to_dict_tuple(self, slice_def):
        dict_out = {}
        for item in slice_def:
            key, value = self.to_dict_single_item(item)
            if key not in dict_out: # if key does not exist, add it
                dict_out[key] = [value]
            else:
                dict_out[key].append(value)
        # if there is only one item along a dimension, convert list to single item
        return {k: v if len(v) > 1 else v[0] for k, v in dict_out.items()}

    @property
    def ids(self):
        """
        Indices used for slicing the values array
        """
        return tuple(self._id_list)

    @property
    def values_pointer(self):
        """
        Pointer to the subset of the values array of the parent NamedDimArray object.
        """
        return self.nda.values[self.ids]

    @property
    def dim_letters(self):
        """
        Updated dimension letters, where sliced dimensions with only one item along that direction are removed.
        """
        all_letters = self.nda.dims.letters
        # remove the dimensions along which there is only one item
        letters_removed = [d for d, items in self.def_dict.items() if isinstance(items, str)]
        return tuple([d for d in all_letters if d not in letters_removed])

    def to_nda(self):
        """
        Return a NamedDimArray object that is a slice of the original NamedDimArray object.
        Attention: This creates a new NamedDimArray object, which is not linked to the original one.
        """
        assert not self.has_dim_with_several_items, "Cannot convert to NamedDimArray if there are dimensions with several items"
        return NamedDimArray(dim_letters=self.dim_letters,
                             parent_alldims=self.nda.dims,
                             values=self.values_pointer)

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
        """
        Given either a single item name or a list of item names, return the corresponding item IDs, along one dimension 'dim_letter'.
        """
        if isinstance(item_or_items, str): # single item
            return self._get_single_item_id(dim_letter, item_or_items)
        elif isinstance(item_or_items, (tuple, list, np.ndarray)): # list of items
            return [self._get_single_item_id(dim_letter, item) for item in item_or_items]

    def _get_single_item_id(self, dim_letter, item_name):
        return self.nda.dims[dim_letter].items.index(item_name)

    def _set_ids_singledim(self, dim_letter, ids):
        self._id_list[self.nda.dims.index(dim_letter)] = ids



class Process():
    """
    Processes serve as nodes for the MFA system layout definition.
    Flows are defined between two processes. Stocks are connected to a process.
    Processes do not contain values themselves.

    Processes get an ID by the order they are defined in  in the MFA system definition.
    The process with ID 0 necessarily contains everything outside the system boundary.
    """

    def __init__(self, name: str, id: int):
        if id == 0:
            assert name == 'sysenv', "The process with ID 0 must be named 'sysenv', as it contains everything outside the system boundary."
        self.name = name
        self.id = id


class Flow(NamedDimArray):
    """
    The values of Flow objects are the main computed outcome of the MFA system.
    A flow connects two processes.
    Its name is set as a combination of the names of the two processes it connects.

    Note that it is a subclass of NamedDimArray, so most of the methods are defined in the NamedDimArray class.
    """

    def __init__(self, from_process: str, to_process: str, dim_letters: tuple):
        """
        Wrapper for the NamedDimArray constructor (without initialization of dimensions and values).
        Important: The flow name is defined here as a combination of the names of the two processes it connects.
        """
        name = f"{from_process} => {to_process}"
        super().__init__(name, dim_letters)
        self._from_process_name = from_process
        self._to_process_name = to_process

    def attach_to_processes(self, processes: dict):
        """
        Store links to the Process objects the Flow connects, and their IDs.
        (To set up the links, the names given in the Flow definition dict are used)
        """
        self.from_process = processes[self._from_process_name]
        self.to_process = processes[self._to_process_name]
        self.from_process_id = self.from_process.id
        self.to_process_id = self.to_process.id


class StockArray(NamedDimArray):
    """
    Stocks allow accumulation of material at a process, i.e. between two flows.
    As Stock contains NamedDimArrays for its stock value, inflow and outflow.
    For details, see the Stock class.
    """
    pass


class Parameter(NamedDimArray):
    """
    Parameters are used for example to define the share of flows that go into one branch when the flow splits at a process.

    All methods are defined in the NamedDimArray parent class.
    """
    pass
