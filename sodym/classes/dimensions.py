"""
Concepts based on:

ODYM
Copyright (c) 2018 Industrial Ecology
author: Stefan Pauliuk, Uni Freiburg, Germany
https://github.com/IndEcol/ODYM

Re-written for use in simson project
"""

from copy import copy
from pydantic import BaseModel as PydanticBaseModel, Field, AliasChoices


class Dimension(PydanticBaseModel):
    """One of multiple dimensions over which MFA arrays are defined.

    Defined by a name, a letter for shorter addressing, and a list of items. For example, the dimension 'Region' could
    have letter 'r' and a country list as items. The list of items can be loaded from a csv file, or set directly, for
    example if a subset of an existing dimension is formed.
    """

    name: str = Field(..., min_length=2)
    letter: str = Field(..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter"))
    items: list

    @property
    def len(self):
        return len(self.items)

    def index(self, item):
        return self.items.index(item)


class DimensionSet(PydanticBaseModel):
    """A set of Dimension objects which MFA arrays are defined over.

    The objects are stored in the internal _list, but can be accessed via __getitem__ with either the name or the
    letter.
    """

    dimensions: list[Dimension]

    @property
    def _dict(self):
        """Contains mappings.

        letter --> dim object and name --> dim object
        """
        return {dim.name: dim for dim in self.dimensions} | {dim.letter: dim for dim in self.dimensions}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._dict[key]
        elif isinstance(key, int):
            return self.dimensions[key]
        else:
            raise TypeError("Key must be string or int")

    def __iter__(self):
        return iter(self.dimensions)

    def size(self, key: str):
        return self._dict[key].len

    def shape(self, keys: tuple = None):
        keys = keys if keys else self.letters
        return tuple(self.size(key) for key in keys)

    def get_subset(self, dims: tuple = None):
        """Returns a copy if dims are not given."""
        subset = copy(self)
        if dims is not None:
            subset.dimensions = [self._dict[dim_key] for dim_key in dims]
        return subset

    @property
    def names(self):
        return tuple([dim.name for dim in self.dimensions])

    @property
    def letters(self):
        return tuple([dim.letter for dim in self.dimensions])

    @property
    def string(self):
        return "".join(self.letters)

    def index(self, key):
        return [d.letter for d in self.dimensions].index(key)
