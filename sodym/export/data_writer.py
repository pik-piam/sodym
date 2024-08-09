import logging
import os
import pickle
from typing import Optional

from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel as PydanticBaseModel, ConfigDict

from ..classes.mfa_system import MFASystem


class DataWriter(PydanticBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mfa: MFASystem
    output_path: str
    do_save_figs: bool = True
    do_show_figs: bool = True
    do_export: dict = {'pickle': True, 'csv': True}
    sankey: dict = {'do_visualize': True, 'color_scheme': 'blueish'}
    display_names: Optional[dict] = {}

    def export(self):
        if self.do_export.get("pickle", False):
            self._export_to_pickle()
        if self.do_export.get("csv", False):
            self._export_to_csv()

    def visualize_results(self):
        if self.sankey['do_visualize']:
            self.visualize_sankey()

    def export_path(self, filename: str = None):
        path_tuple = (self.output_path, 'export')
        if filename is not None:
            path_tuple += (filename,)
        return os.path.join(*path_tuple)

    def figure_path(self, filename: str):
        return os.path.join(self.output_path, 'figures', filename)

    def _export_to_pickle(self):
        dict_out = self._convert_to_dict(self.mfa)
        pickle.dump(dict_out, open(self.export_path("mfa.pickle"), "wb"))
        logging.info(f'Data saved to {self.export_path("mfa.picke")}')

    def _export_to_csv(self):
        dir_out = os.path.join(self.export_path(), "flows")
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        for flow_name, flow in self.mfa.flows.items():
            df = flow.to_df()
            path_out = os.path.join(dir_out, f'{flow_name.replace(" => ", "__2__")}.csv')
            df.to_csv(path_out, index=False)
        logging.info(f'Data saved in directory {dir_out}')

    @staticmethod
    def _convert_to_dict(mfa: MFASystem):
        dict_out = {}
        dict_out["dimension_names"] = {d.letter: d.name for d in mfa.dims}
        dict_out["dimension_items"] = {d.name: d.items for d in mfa.dims}
        dict_out["flows"] = {n: f.values for n, f in mfa.flows.items()}
        dict_out["flow_dimensions"] = {n: f.dims.letters for n, f in mfa.flows.items()}
        return dict_out

    def _display_name(self, name):
        return self.display_names[name] if name in self.display_names else name

    def _show_and_save_pyplot(self, fig, name):
        if self.do_save_figs:
            plt.savefig(self.figure_path(f"{name}.png"))
        if self.do_show_figs:
            plt.show()

    def _show_and_save_plotly(self, fig: go.Figure, name):
        if self.do_save_figs:
            fig.write_image(self.figure_path(f"{name}.png"))
        if self.do_show_figs:
            fig.show()

    def visualize_sankey(self):
        mfa = self.mfa
        # exclude_nodes = ['sysenv', 'atmosphere', 'emission', 'captured']
        exclude_nodes = ["sysenv"]
        exclude_flows = []
        year = 2050
        region_id = 0
        carbon_only = True

        nodes = [p for p in mfa.processes.values() if p.name not in exclude_nodes]
        ids_in_sankey = {p.id: i for i, p in enumerate(nodes)}
        exclude_node_ids = [p.id for p in mfa.processes.values() if p.name in exclude_nodes]

        if self.sankey["color_scheme"] == "blueish":
            material_colors = [f"hsv({10 * i + 200},40,150)" for i in range(mfa.dims[mfa.product_dimension_name].len)]
#        elif color_scheme == "antique":
#            material_colors = pl.colors.qualitative.Antique[: mfa.dims[cfg.product_dimension_name].len]
#        elif color_scheme == "viridis":
#            material_colors = pl.colors.sample_colorscale(
#                "Viridis", mfa.dims[cfg.product_dimension_name].len + 1, colortype="rgb"
#            )
        else:
            raise Exception("invalid color scheme")

        link_dict = {"label": [], "source": [], "target": [], "color": [], "value": []}

        def add_link(**kwargs):
            for key, value in kwargs.items():
                link_dict[key].append(value)

        product_dim_letter = mfa.product_dimension_name[0].lower()

        for f in mfa.flows.values():
            if (
                (f.name in exclude_flows)
                or (f.from_process_id in exclude_node_ids)
                or (f.to_process_id in exclude_node_ids)
            ):
                continue
            source = ids_in_sankey[f.from_process_id]
            target = ids_in_sankey[f.to_process_id]
            label = self._display_name(f.name)

            id_orig = f.dims.string
            has_materials = product_dim_letter in id_orig
            id_target = f"ter{product_dim_letter if has_materials else ''}{'s' if mfa.has_scenarios else ''}"
            values = np.einsum(f"{id_orig}->{id_target}", f.values)

            if carbon_only:
                values = values[:, 0, ...]
            else:
                values = np.sum(values, axis=1)

            if mfa.has_scenarios:
                try:
                    values = values[mfa.dims["Time"].index(year), region_id, ..., 1]
                except IndexError:
                    pass
            # choose SSP2 as default scenario
            # TODO: Implement Scenario switch
            else:  # MFA doesn't use scenarios
                values = values[mfa.dims["Time"].index(year), region_id, ...]

            if has_materials:
                for im, c in enumerate(material_colors):
                    try:
                        add_link(label=label, source=source, target=target, color=c, value=values[im])
                    except IndexError:
                        pass
            else:
                add_link(label=label, source=source, target=target, color="hsl(230,20,70)",    value=values)

        fig = go.Figure(
            go.Sankey(
                arrangement="snap",
                node={
                    "label": [self._display_name(p.name) for p in nodes],
                    "color": ["gray" for p in nodes],  # 'rgb(50, 50, 50)'
                    # "x": [0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
                    # "y": [0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
                    "pad": 10,
                },  # 10 Pixels
                link=link_dict,
            )
        )
        self._show_and_save_plotly(fig, "sankey")
