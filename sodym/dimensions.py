from copy import copy
from pydantic import BaseModel as PydanticBaseModel, Field, AliasChoices, model_validator
from typing import Dict


class Dimension(PydanticBaseModel):
    """One of multiple dimensions over which MFA arrays are defined.

    Defined by a name, a letter for shorter addressing, and a list of items.

    **Example**

        >>> from sodym import Dimension
        >>> regions = Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])

    The list of items can be loaded using a :py:class:`sodym.data_reader.DataReader` object,
    or set directly, for example if a subset of an existing dimension is formed.
    """

    name: str = Field(..., min_length=2)
    letter: str = Field(..., min_length=1, max_length=1, validation_alias=AliasChoices("letter", "dim_letter"))
    items: list

    @property
    def len(self) -> int:
        return len(self.items)

    def index(self, item) -> int:
        return self.items.index(item)


class DimensionSet(PydanticBaseModel):
    """A set of Dimension objects which MFA arrays are defined over.

    **Example**

        >>> from sodym import Dimension, DimensionSet
        >>> regions = Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])
        >>> time = Dimension(name='Time', letter='t', items=[1990, 2000, 2010, 2020, 2030])
        >>> dimensions = DimensionSet([regions, time])

    It is expected that DimensionSet instances are created via the :py:class:`sodym.data_reader.DataReader`.

        >>> from sodym import DataReader, DimensionDefinition, Dimension
        >>> class MyDataReader(DataReader):
        >>>    def read_dimension(self, dimension_definition: DimensionDefinition) -> Dimension:
        >>>        if dimension_definition.letter == 't':
        >>>            return Dimension(name='Time', letter='t', items=[1990, 2000, 2010, 2020, 2030])
        >>>        elif dimension_definition.letter == 'r':
        >>>            return Dimension(name='Region', letter='r', items=['Earth', 'Moon', 'Sun'])
        >>>        raise ValueError('No data available for desired dimension')
        >>> data_reader = MyDataReader()
        >>> time_definition = DimensionDefinition(name='Time', letter='t', dtype=int)
        >>> region_definition = DimensionDefinition(name='Region', letter='r', dtype=str)
        >>> definitions = [time_definition, region_definition]
        >>> dimensions = data_reader.read_dimensions(dimension_definitions=definitions)

    """

    dimensions: list[Dimension]

    @model_validator(mode='after')
    def no_repeated_dimensions(self):
        letters = self.letters
        if len(letters) != len(set(letters)):
            raise ValueError('Dimensions must have unique letters in DimensionSet.')
        return self

    def drop(self, key: str, inplace: bool=False):
        dim_to_drop = self._dict[key]
        if not inplace:
            dimensions = copy(self.dimensions)
            dimensions.remove(dim_to_drop)
            return DimensionSet(dimensions=dimensions)
        self.dimensions.remove(dim_to_drop)

    @property
    def _dict(self) -> Dict[str, Dimension]:
        """Contains mappings.

        letter --> dim object and name --> dim object
        """
        return {dim.name: dim for dim in self.dimensions} | {dim.letter: dim for dim in self.dimensions}

    def __getitem__(self, key) -> Dimension:
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

    def get_subset(self, dims: tuple = None) -> 'DimensionSet':
        """Selects :py:class:`Dimension` objects from the object attribute dimensions,
        according to the dims passed, which can be either letters or names.
        Returns a copy if dims are not given.
        """
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

    @property
    def spaced_string(self):
        return " ".join(self.letters)

    def index(self, key):
        return [d.letter for d in self.dimensions].index(key)
