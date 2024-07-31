import os
from .config import cfg

def dimensions_path(filename: str):
    return os.path.join(cfg.input_data_path, 'dimensions', filename)

def datasets_path(filename: str):
    return os.path.join(cfg.input_data_path, 'datasets', filename)

def scalar_parameters_path():
    return os.path.join(cfg.input_data_path, 'scalar_parameters.yml')

def export_path(filename: str = None):
    path_tuple = (cfg.output_path, 'export')
    if filename is not None:
        path_tuple += (filename,)
    return os.path.join(*path_tuple)

def figure_path(filename: str):
    return os.path.join(cfg.output_path, 'figures', filename)