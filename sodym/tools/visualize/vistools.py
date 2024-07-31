from ...classes.mfa_system import MFASystem
from ..config import cfg
from ..paths import figure_path
from matplotlib import pyplot as plt
import plotly.graph_objects as go

visualization_routines = {}

# Here, some general, non-model-specific display names can be set
display_names = {
    'my_variable': 'My Variable',
}

def visualize_if_set_in_config(name):
    def decorator(routine):
        def wrapper(mfa, *args, **kwargs):
            if cfg.visualize[name]['do_visualize']:
                update_display_names_mfa(mfa)
                routine(mfa, *args, **kwargs)
        return wrapper
    return decorator

def update_display_names_mfa(mfa: MFASystem):
    display_names.update(mfa.display_names)

def dn(st):
    return display_names[st] if st in display_names else st


def show_and_save_pyplot(fig, name):
    if cfg.do_save_figs:
        plt.savefig(figure_path(f"{name}.png"))
    if cfg.do_show_figs:
        plt.show()

def show_and_save_plotly(fig: go.Figure, name):
    if cfg.do_save_figs:
        fig.write_image(figure_path(f"{name}.png"))
    if cfg.do_show_figs:
        fig.show()