import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal
from pydantic_core import ValidationError
import pytest

from sodym import NamedDimArray, DimensionSet


dimensions = [
        {'name': 'place', 'letter': 'p', 'items': ['Earth', 'Sun', 'Moon', 'Venus']},
        {'name': 'time', 'letter': 't', 'items': [1990, 2000, 2010]},
    ]
dims = DimensionSet(dim_list=dimensions)
values = np.random.rand(4, 3)
numbers = NamedDimArray(name='two', dims=dims, values=values)

animals = {'name': 'animal', 'letter': 'a', 'items': ['cat', 'mouse']}
dims_incl_animals = DimensionSet(dim_list=dimensions+[animals])
animal_values = np.random.rand(4, 3, 2)
space_animals = NamedDimArray(name='space_animals', dims=dims_incl_animals, values=animal_values)


def test_named_dim_array_validations():
    dimensions = [
        {'name': 'place', 'letter': 'p', 'items': ['World', ]},
        {'name': 'time', 'letter': 't', 'items': [1990, 2000, 2010]},
    ]
    dims = DimensionSet(dim_list=dimensions)

    # example with values with the correct shape
    NamedDimArray(name='numbers', dims=dims, values=np.array([[1, 2, 3], ]))

    # example with dimensions reversed
    with pytest.raises(ValidationError):
        NamedDimArray(name='numbers', dims=dims, values=np.array([[1], [2], [3], ]))

    # example with too many values
    with pytest.raises(ValidationError):
        NamedDimArray(name='numbers', dims=dims, values=np.array([[1, 2, 3, 4], ]))

    # example with no values passed -> filled with zeros
    zero_values = NamedDimArray(name='numbers', dims=dims)
    assert zero_values.values.shape == (1, 3)
    assert np.all([zero_values.values == 0])


def test_cast_to():
    # example of duplicating values along new axis (e.g. same number of cats and mice)
    casted_named_dim_array = numbers.cast_to(target_dims=dims_incl_animals)
    assert casted_named_dim_array.dims == dims_incl_animals
    assert casted_named_dim_array.values.shape == (4, 3, 2)
    assert_almost_equal(np.sum(casted_named_dim_array.values), 2 * np.sum(values))

    # example with differently ordered dimensions
    target_dims = DimensionSet(dim_list=[animals]+dimensions[::-1])
    casted_named_dim_array = numbers.cast_to(target_dims=target_dims)
    assert casted_named_dim_array.values.shape == (2, 3, 4)


def test_sum_nda_to():
    # sum over one dimension
    summed_named_dim_array = space_animals.sum_nda_to(result_dims=('p', 't'))
    assert summed_named_dim_array.dims == DimensionSet(dim_list=dimensions)
    assert_array_almost_equal(summed_named_dim_array.values, np.sum(animal_values, axis=2))

    # sum over two dimensions
    summed_named_dim_array = space_animals.sum_nda_to(result_dims=('t'))
    assert_array_almost_equal(summed_named_dim_array.values, np.sum(np.sum(animal_values, axis=2), axis=0))

    # example attempt to get a resulting dimension that does not exist
    with pytest.raises(KeyError):
        space_animals.sum_nda_to(result_dims=('s'))

    # example where dimensions to sum over are specified rather than the remaining dimensions
    summed_over = space_animals.sum_nda_over(sum_over_dims=('p', 'a'))
    assert_array_almost_equal(summed_over.values, summed_named_dim_array.values)

    # example sum over dimension that doesn't exist
    nothing_changes = space_animals.sum_nda_over(sum_over_dims=('s'))
    assert_array_almost_equal(nothing_changes.values, space_animals.values)
    assert nothing_changes.dims == space_animals.dims

def test_maths():
    # test minimum
    minimum = space_animals.minimum(numbers)
    assert minimum.dims == dims
    assert_array_almost_equal(minimum.values, np.minimum(values, animal_values.sum(axis=2)))

    # test maximum
    maximum = space_animals.maximum(numbers)
    assert maximum.dims == dims
    assert_array_almost_equal(maximum.values, np.maximum(values, animal_values.sum(axis=2)))

    # test sum
    summed = space_animals + numbers
    assert summed.dims == dims
    assert_array_almost_equal(summed.values, animal_values.sum(axis=2) + values)

    # test minus
    subtracted = space_animals - numbers
    assert subtracted.dims == dims
    assert_array_almost_equal(subtracted.values, animal_values.sum(axis=2) - values)
    subtracted_flipped = numbers - space_animals
    assert subtracted_flipped.dims == dims
    assert_array_almost_equal(subtracted_flipped.values, values - animal_values.sum(axis=2))

    # test multiply
    multiplied = numbers * space_animals
    assert multiplied.dims == dims_incl_animals  # different from behaviour of above methods
    assert_array_almost_equal(multiplied.values[:, :, 0], values * animal_values[:, :, 0])
    assert_array_almost_equal(multiplied.values[:, :, 1], values * animal_values[:, :, 1])

    # test divide
    divided = space_animals / numbers
    assert divided.dims == dims_incl_animals
    assert_array_almost_equal(divided.values[:, :, 0], animal_values[:, :, 0]/values)
    assert_array_almost_equal(divided.values[:, :, 1], animal_values[:, :, 1]/values)
    divided_flipped = numbers / space_animals
    assert divided_flipped.dims == dims_incl_animals
    assert_array_almost_equal(divided_flipped.values[:, :, 0], values/(animal_values[:, :, 0]))
    assert_array_almost_equal(divided_flipped.values[:, :, 1], values/(animal_values[:, :, 1]))


def test_get_item():
    cats_on_the_moon = space_animals['Moon']['cat']
    assert(isinstance(cats_on_the_moon, NamedDimArray))
    assert_array_almost_equal(cats_on_the_moon.values, space_animals.values[2, :, 0])
    # note that this does not work for the time dimension (not strings)
    # and also assumes that no item appears in more than one dimension


def test_sub_array_handler():
    space_cat = space_animals['cat']  # space cat from str
    another_space_cat = space_animals[{'a': 'cat'}]  # space cat from dict
    assert_array_equal(space_cat.values, another_space_cat.values)

    space_1990 = space_animals[{'t': 1990}]  # space animals in 1990
    assert space_1990.values.shape == (4, 2)
    assert space_1990.dims.letters == ('p', 'a')

    with pytest.raises(ValueError):
        space_animals[{'a': 'dog'}]  # there isn't a dog in space_animals
