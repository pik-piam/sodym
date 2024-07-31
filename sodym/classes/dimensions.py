"""
Concepts based on:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Re-written for use in simson project
"""

from copy import copy
from ..tools.read_data import read_data_to_list


class Dimension(object):
    """
    One of multiple dimensions over which MFA arrays are defined.
    Defined by a name, a letter for shorter addressing, and a list of items.
    For example, the dimension 'Region' could have letter 'r' and a country list as items.
    The list of items can be loaded from a csv file, or set directly,
    for example if a subset of an existing dimension is formed.
    """

    def __init__(self, name: str, dim_letter: str = None, do_load: bool = False, dtype: type = None, filename: str = None, items: list = None):
        self.name = name
        if dim_letter:
            self.letter = dim_letter
        assert not (do_load and items), "You can't both load and set items"
        if items:
            self.set_items(items)
        if do_load:
            self.load_items(dtype, filename)

    def set_items(self, items: list):
        self.items = items

    def load_items(self, dtype=str, filename: str = None):
        filename = filename if filename else self.name
        data = read_data_to_list("dimension", filename, dtype)
        self.set_items(data)

    @property
    def len(self):
        assert self.items, "Items not loaded yet"
        return len(self.items)

    def index(self, item):
        return self.items.index(item)


class DimensionSet(object):
    """
    A set of Dimension objects which MFA arrays are defined over.
    The objects are stored in the internal _list, but can be accessed via __getitem__ with either the name or the letter.
    """

    def __init__(self, arg_dicts_for_dim_constructors: list=None, dimensions: list=None):
        """
        The entries of arg_dicts_for_dim_constructors serve as arguments of the constructor for each Dimension object.
        Alternatively, a list of Dimension objects can be provided directly.
        """
        assert bool(arg_dicts_for_dim_constructors) != bool(dimensions), "Either defdicts or dimensions must be provided"
        if dimensions is not None:
            self._list = dimensions
        elif arg_dicts_for_dim_constructors is not None:
            self._list = [Dimension(**arg_dict) for arg_dict in arg_dicts_for_dim_constructors]

    @property
    def _dict(self):
        """
        contains mappings

        letter --> dim object and
        name --> dim object
        """
        return {dim.name: dim for dim in self._list} | {dim.letter: dim for dim in self._list}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, int):
            return self._list[key]
        else:
            raise TypeError("Key must be string or int")

    def __iter__(self):
        return iter(self._list)

    def size(self, key: str):
        return self._dict[key].len

    def shape(self, keys: tuple = None):
        keys = keys if keys else self.letters
        return tuple(self.size(key) for key in keys)

    def get_subset(self, dims: tuple = None):
        """
        returns a copy if dims are not given
        """
        subset = copy(self)
        if dims is not None:
            subset._list = [self._dict[dim_key] for dim_key in dims]
        return subset

    @property
    def names(self):
        return tuple([dim.name for dim in self._list])

    @property
    def letters(self):
        return tuple([dim.letter for dim in self._list])

    @property
    def string(self):
        return "".join(self.letters)

    def index(self, key):
        return [d.letter for d in self._list].index(key)
