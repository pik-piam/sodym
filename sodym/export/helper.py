from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel as PydanticBaseModel
import re
import unicodedata

from ..named_dim_arrays import NamedDimArray
from ..dimensions import DimensionSet


class ArrayPlotter(PydanticBaseModel):

    array: NamedDimArray
    intra_line_dim: str
    x_array: NamedDimArray = None
    subplot_dim: str = None
    linecolor_dim: str = None
    fig_ax: tuple = None
    line_label: str = None
    xlabel: str = None
    ylabel: str = None

    # TODO: make validator
    # assert not (
    #     linecolor_dim is not None and line_label is not None
    # ), "Either dim_lines or line_label can be given, but not both."

    # TODO: make validator
    # dims = linecolor_dim + subplot_dim + intra_line_dim should be the same as array.dims.names
    # should be the same as the dims of the array
    # x_array should have a subset of the dims of the array

    def plot(self, save_path: str=None, do_show: bool = False):
        self.fill_fig_ax()
        subplotlist_array, subplotlist_x_array, subplotlist_name = self.prepare_arrays()
        self.plot_all_subplots(subplotlist_array, subplotlist_x_array, subplotlist_name)
        self.plot_legend()
        if save_path is not None:
            plt.savefig(save_path)
        if do_show:
            plt.show()
        return self.fig_ax

    def prepare_arrays(self):
        self.get_x_array_like_value_array()
        subplotlist_array, subplotlist_name = list_of_slices(self.array, self.subplot_dim)
        subplotlist_x_array, _ = list_of_slices(self.x_array, self.subplot_dim)
        return subplotlist_array, subplotlist_x_array, subplotlist_name

    @property
    def dims_after_slice(self):
        original_dims = self.array.dims.letters
        dims_removed = [d for d, v in self.slice_dict.items() if not isinstance(v, (list, tuple))]
        return [d for d in original_dims if d not in dims_removed]

    @property
    def dims_after_slice_sum(self):
        return [d for d in self.dims_after_slice if d not in self.summed_dims]

    def plot_legend(self):
        handles, labels = self.ax[0, 0].get_legend_handles_labels()
        self.fig.legend(handles, labels, loc="lower center")

    def plot_all_subplots(self, subplotlist_array, subplotlist_x_array, subplotlist_name):
        for i_subplot, (array_subplot, x_array_subplot, name_subplot) in enumerate(zip(subplotlist_array, subplotlist_x_array, subplotlist_name)):
            i, j = i_subplot // self.nx, i_subplot % self.nx
            self.plot_subplot( ax=self.ax[i, j], array=array_subplot, x_array=x_array_subplot)
            self.label_subplot(ax=self.ax[i, j], name_subplot_item=name_subplot)

    def fill_fig_ax(self):
        if self.fig_ax is not None:  # already filled from input argument
            return
        if self.subplot_dim is None:
            nx, ny = 1, 1
        else:
            n_subplots = self.array.dims[self.subplot_dim].len
            nx = int(np.ceil(np.sqrt(n_subplots)))
            ny = int(np.ceil(n_subplots / nx))
        self.fig_ax = plt.subplots(nx, ny, figsize=(10, 9), squeeze=False)

    def get_x_array_like_value_array(self):
        if self.x_array is None:
            x_dim_obj = self.array.dims[self.intra_line_dim]
            x_dimset = DimensionSet(dimensions=[x_dim_obj])
            self.x_array = NamedDimArray(dims=x_dimset, values=np.array(x_dim_obj.items), name=self.intra_line_dim)
        self.x_array = self.x_array.cast_to(self.array.dims)

    @property
    def fig(self) -> plt.Figure:
        return self.fig_ax[0]

    @property
    def ax(self) -> plt.Axes:
        return self.fig_ax[1]

    @property
    def nx(self):
        return self.ax.shape[0]

    @property
    def ny(self):
        return self.ax.shape[1]

    def plot_subplot(self, ax: plt.Axes, array: NamedDimArray, x_array: NamedDimArray):
        linelist_array, linelist_name = list_of_slices(array, self.linecolor_dim)
        linelist_x_array, _ = list_of_slices(x_array, self.linecolor_dim)
        for (array_line, x_array_line, name_line) in zip(linelist_array, linelist_x_array, linelist_name):
            label = self.line_label if self.line_label is not None else name_line
            assert array_line.dims.names == (self.intra_line_dim,), (
                "All dimensions of array must be given exactly once. Either as x_dim / subplot_dim / linecolor_dim, or in "
                "slice_dict or summed_dims."
            )
            ax.plot(x_array_line.values, array_line.values, label=label)

    def label_subplot(self, ax: plt.Axes, name_subplot_item: str):
        if self.subplot_dim is not None:
            title = f"{self.subplot_dim}={name_subplot_item}"
            ax.set_title(title)
        xlabel = self.xlabel if self.xlabel is not None else self.x_array.name
        if xlabel != "unnamed":
            ax.set_xlabel(xlabel)
        ylabel = self.ylabel if self.ylabel is not None else self.array.name
        if ylabel != "unnamed":
            ax.set_ylabel(ylabel)


def list_of_slices(array: NamedDimArray, dim_name_to_slice) -> tuple[list[NamedDimArray], list[str]]:
    if dim_name_to_slice is not None:
        dim_to_slice = array.dims[dim_name_to_slice]
        list_array = [
            array[{dim_to_slice.letter: item}]
            for item in dim_to_slice.items
        ]
        list_name = dim_to_slice.items
    else:
        list_array = [array]
        list_name = [None]
    return list_array, list_name


def to_valid_file_name(value: str) -> str:
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII. Convert spaces or repeated dashes to single dashes.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]', '_', value).strip('-_')
