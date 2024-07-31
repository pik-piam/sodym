import os
import pickle
from .config import cfg
from .paths import export_path
from ..classes.mfa_system import MFASystem


def export(mfa: MFASystem):
    if cfg.do_export.get('pickle', False):
        export_to_pickle(mfa)
    if cfg.do_export.get('csv', False):
        export_to_csv(mfa)


def export_to_pickle(mfa: MFASystem):
    dict_out = convert_to_dict(mfa)
    pickle.dump(dict_out, open(export_path('mfa.pickle'), "wb"))


def export_to_csv(mfa: MFASystem):
    dir_out = os.path.join(export_path(), 'flows')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for flow_name, flow in mfa.flows.items():
        df = flow.to_df()
        path_out = os.path.join(dir_out, f'{flow_name.replace(" => ", "__2__")}.csv')
        df.to_csv(path_out, index=False)


def convert_to_dict(mfa: MFASystem):
    dict_out = {}
    dict_out['dimension_names'] = {d.letter: d.name for d in mfa.dims}
    dict_out['dimension_items'] = {d.name: d.items for d in mfa.dims}
    dict_out['flows'] = {n: f.values for n, f in mfa.flows.items()}
    dict_out['flow_dimensions'] = {n: f.dims.letters for n, f in mfa.flows.items()}
    return dict_out
