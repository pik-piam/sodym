"""Home for helper functions to the `Flow` class, including naming functions."""

from typing import Callable

from .named_dim_arrays import Process, Flow
from .dimensions import DimensionSet
from .mfa_definition import FlowDefinition


def process_names_with_arrow(from_process: Process, to_process: Process) -> str:
    return f"{from_process.name} => {to_process.name}"


def process_names_no_spaces(from_process: Process, to_process: Process) -> str:
    return f"{from_process.name}_to_{to_process.name}"


def process_ids(from_process: Process, to_process: Process) -> str:
    return f"F{from_process.id}_{to_process.id}"


def make_empty_flows(
    processes: dict[str, Process],
    flow_definitions: list[FlowDefinition],
    dims: DimensionSet,
    naming: Callable[[Process, Process], str] = process_names_with_arrow,
) -> dict[str, Flow]:
    """Initialize all defined flows with zero values."""
    flows = {}
    for flow_definition in flow_definitions:
        try:
            from_process = processes[flow_definition.from_process_name]
            to_process = processes[flow_definition.to_process_name]
        except KeyError:
            raise KeyError(f"Missing process required by flow definition {flow_definition}.")
        if flow_definition.name_override is not None:
            name = flow_definition.name_override
        else:
            name = naming(from_process, to_process)
        dim_subset = dims.get_subset(flow_definition.dim_letters)
        flow = Flow(from_process=from_process, to_process=to_process, name=name, dims=dim_subset)
        flows[flow.name] = flow
    return flows
