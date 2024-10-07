import logging
import os
import pickle
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel as PydanticBaseModel

from ..mfa_system import MFASystem
from .helper import to_valid_file_name


class DataWriter(PydanticBaseModel):

    sankey: dict = {'do_visualize': True, 'color_scheme': 'blueish'}
    display_names: Optional[dict] = {}

    def visualize_results(self, mfa: MFASystem):
        if self.sankey['do_visualize']:
            self.visualize_sankey(mfa=mfa)

    def export_mfa_to_pickle(self, mfa: MFASystem, export_path: str):
        dict_out = self._convert_to_dict(mfa)
        pickle.dump(dict_out, open(export_path, "wb"))
        logging.info(f'Data saved to {export_path}')

    def export_mfa_flows_to_csv(self, mfa: MFASystem, export_directory: str):
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
        for flow_name, flow in mfa.flows.items():
            path_out = os.path.join(export_directory, f'{to_valid_file_name(flow_name)}.csv')
            flow.to_df().to_csv(path_out)
        logging.info(f'Data saved in directory {export_directory}')

    def export_mfa_stocks_to_csv(self, mfa: MFASystem, export_directory: str):
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
        for stock_name, stock in mfa.stocks.items():
            df = stock.stock.to_df()
            path_out = os.path.join(export_directory, f'{to_valid_file_name(stock_name)}_stock.csv')
            df.to_csv(path_out, index=False)
        logging.info(f'Data saved in directory {export_directory}')

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

    def visualize_sankey(self, mfa: MFASystem):
        # exclude_nodes = ['sysenv', 'atmosphere', 'emission', 'captured']
        exclude_nodes = ["sysenv"]
        exclude_flows = []
        year = 2050
        region_id = 0
        carbon_only = True

        # Get product dim letter
        mfa_dim_letters = mfa.dims.letters
        if 'm' in mfa_dim_letters:
            product_dim_letter = 'm'
        elif 'g' in mfa_dim_letters:
            product_dim_letter = 'g'
        assert 'product_dim_letter' in locals()

        nodes = [p for p in mfa.processes.values() if p.name not in exclude_nodes]
        ids_in_sankey = {p.id: i for i, p in enumerate(nodes)}
        exclude_node_ids = [p.id for p in mfa.processes.values() if p.name in exclude_nodes]

        if self.sankey["color_scheme"] == "blueish":
            material_colors = [f"hsv({10 * i + 200},40,150)" for i in range(mfa.dims[product_dim_letter].len)]
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
            id_target = f"ter{product_dim_letter if has_materials else ''}{''}"
            values = np.einsum(f"{id_orig}->{id_target}", f.values)

            if carbon_only:
                values = values[:, 0, ...]
            else:
                values = np.sum(values, axis=1)

            values = values[mfa.dims["Time"].index(year), region_id, ...]

            if has_materials:
                for im, c in enumerate(material_colors):
                    try:
                        add_link(label=label, source=source, target=target, color=c, value=values[im])
                    except IndexError:
                        pass
            else:
                add_link(label=label, source=source, target=target, color="hsl(230,20,70)", value=values)

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
        return fig
