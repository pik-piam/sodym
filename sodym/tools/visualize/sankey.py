import numpy as np
import plotly as pl
import plotly.graph_objects as go
from ...classes.mfa_system import MFASystem
from ..config import cfg
from .vistools import visualize_if_set_in_config, show_and_save_plotly, dn


@visualize_if_set_in_config('sankey')
def visualize_mfa_sankey(mfa: MFASystem):
    # exclude_nodes = ['sysenv', 'atmosphere', 'emission', 'captured']
    exclude_nodes = ['sysenv']
    exclude_flows = []
    year = 2050
    region_id = 0
    carbon_only = True
    color_scheme = 'blueish'

    nodes = [p for p in mfa.processes.values() if p.name not in exclude_nodes]
    ids_in_sankey = {p.id: i for i, p in enumerate(nodes)}
    exclude_node_ids = [p.id for p in mfa.processes.values() if p.name in exclude_nodes]
    flows = {f for f in mfa.flows.values() if (f.name not in exclude_flows
                                               and f.from_process_id not in exclude_node_ids
                                               and f.to_process_id not in exclude_node_ids)}

    if color_scheme == 'antique':
        material_colors = pl.colors.qualitative.Antique[:mfa.dims[cfg.product_dimension_name].len]
    elif color_scheme == 'viridis':
        material_colors = pl.colors.sample_colorscale('Viridis', mfa.dims[cfg.product_dimension_name].len + 1, colortype='rgb')
    elif color_scheme == 'blueish':
        material_colors = [f'hsv({10 * i + 200},40,150)' for i in range(mfa.dims[cfg.product_dimension_name].len)]
    else:
        raise Exception('invalid color scheme')

    link_dict = {"label": [], "source": [], "target": [], "color": [], "value": []}

    def add_link(**kwargs):
        for key, value in kwargs.items():
            link_dict[key].append(value)

    product_dim_letter = cfg.product_dimension_name[0].lower()

    for f in flows:
        source = ids_in_sankey[f.from_process_id]
        target = ids_in_sankey[f.to_process_id]
        label = dn(f.name)

        id_orig = f.dims.string
        has_materials = product_dim_letter in id_orig
        id_target = f"ter{product_dim_letter if has_materials else ''}{'s' if cfg.has_scenarios else ''}"
        values = np.einsum(f"{id_orig}->{id_target}", f.values)

        if carbon_only:
            values = values[:,0,...]
        else:
            values = np.sum(values, axis = 1)

        time_index = mfa.dims['Time'].index(year)  # todo delete
        if cfg.has_scenarios:

            try:
                values = values[mfa.dims['Time'].index(year),region_id,..., 1]
            except IndexError:
                test = values[mfa.dims['Time'].index(year), region_id, ...]
                a=0
            # choose SSP2 as default scenario
            # TODO: Implement Scenario switch
        else:   # MFA doesn't use scenarios
            values = values[mfa.dims['Time'].index(year), region_id, ...]

        if has_materials:
            for im, c in enumerate(material_colors):
                try:
                    add_link(label=label, source=source, target=target, color=c, value=values[im])
                except IndexError:
                    a=0
        else:
            add_link(label=label, source=source, target=target, color='hsl(230,20,70)', value=values)


    fig = go.Figure(go.Sankey(
        arrangement = "snap",
        node = {
            "label": [dn(p.name) for p in nodes],
            "color": ['gray' for p in nodes], # 'rgb(50, 50, 50)'
            # "x": [0.2, 0.1, 0.5, 0.7, 0.3, 0.5],
            # "y": [0.7, 0.5, 0.2, 0.4, 0.2, 0.3],
            'pad':10},  # 10 Pixels
        link = link_dict ))

    show_and_save_plotly(fig, 'sankey')