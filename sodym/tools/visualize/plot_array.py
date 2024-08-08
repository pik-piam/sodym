from matplotlib import pyplot as plt
import numpy as np
from ...classes.named_dim_arrays import NamedDimArray


def visualize_array(
    array: NamedDimArray,
    intra_line_dim,
    x_array: NamedDimArray = None,
    tile_dim=None,
    linecolor_dim=None,
    slice_dict=None,
    summed_dims=None,
    fig_ax=None,
    title=None,
    label_in=None,
):
    assert not (
        linecolor_dim is not None and label_in is not None
    ), "Either dim_lines or label_in can be given, but not both."

    fig, ax, nx, ny = get_fig_ax(array, tile_dim, fig_ax)

    fig.suptitle(title if title is not None else array.name)

    array_reduced = sum_and_slice(array, slice_dict, summed_dims)
    arrays_tile = list_of_slices(array_reduced, tile_dim)

    if x_array is not None:
        x_array = x_array.cast_to(array.dims)
        x_array = sum_and_slice(x_array, slice_dict, summed_dims)
    x_tiles = list_of_slices(x_array, tile_dim, len(arrays_tile))

    for i_tile, (array_tile, x_tile) in enumerate(zip(arrays_tile, x_tiles)):
        ax_tile = ax[i_tile // nx, i_tile % nx]
        item_tile = dim_item_name_by_index(array, tile_dim, i_tile)
        plot_tile(ax_tile, array_tile, x_tile, intra_line_dim, linecolor_dim, label_in, tile_dim, item_tile)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center")
    return fig, ax


def plot_tile(ax_tile, array_tile, x_tile, intra_line_dim, linecolor_dim, label_in, tile_dim, item_tile):
    if tile_dim is not None:
        ax_tile.set_title(f"{tile_dim}={item_tile}")

    arrays_line = list_of_slices(array_tile, linecolor_dim)
    x_lines = list_of_slices(x_tile, linecolor_dim)
    for j, (array_line, x_line) in enumerate(zip(arrays_line, x_lines)):
        label = get_label(array_line, linecolor_dim, j, label_in)
        assert array_line.dims.names == (intra_line_dim,), (
            "All dimensions of array must be given exactly once. Either as x_dim / tile_dim / linecolor_dim, or in "
            "slice_dict or summed_dims."
        )
        if x_line is not None:
            x = x_line.values
        else:
            x = array_line.dims[intra_line_dim].items
        ax_tile.plot(x, array_line.values, label=label)
    ax_tile.set_xlabel(intra_line_dim)


def dim_item_name_by_index(array: NamedDimArray, dim_name, i_item):
    if dim_name is None:
        return None
    else:
        return array.dims[dim_name].items[i_item]


def sum_and_slice(array: NamedDimArray, slice_dict, summed_dims):
    array = array.sub_array_handler(slice_dict).to_nda()
    if summed_dims is not None:
        array = array.sum_nda_over(summed_dims)
    return array


def list_of_slices(array, dim_to_slice, n_return_none=1):
    if array is None:
        return [None] * n_return_none
    elif dim_to_slice is not None:
        arrays_tile = [
            array.sub_array_handler({array.dims[dim_to_slice].letter: item}).to_nda()
            for item in array.dims[dim_to_slice].items
        ]
    else:
        arrays_tile = [array]
    return arrays_tile


def get_label(array: NamedDimArray, linecolor_dim, j, label_in):
    if label_in is not None:
        label = label_in
    else:
        label = dim_item_name_by_index(array, linecolor_dim, j)
    return label


def get_fig_ax(array, dim_tiles, fig_ax):
    if fig_ax is None:
        if dim_tiles is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 9))
            nx, ny = 1, 1
        else:
            nx, ny = get_tiles(array, dim_tiles)
            fig, ax = plt.subplots(nx, ny, figsize=(10, 9))
    else:
        fig, ax = fig_ax
        nx, ny = ax.shape
    return fig, ax, nx, ny


def get_tiles(array, dim_tiles):
    n_tiles = array.dims[dim_tiles].len
    nx = int(np.ceil(np.sqrt(n_tiles)))
    ny = int(np.ceil(n_tiles / nx))
    return nx, ny
