from .named_dim_arrays import Process, Flow
from .dimensions import DimensionSet
from .mfa_definition import FlowDefinition


def make_empty_flows(
        processes: dict[str, Process], flow_definitions: list[FlowDefinition],
        dims: DimensionSet,
        ) -> dict[str, Flow]:
    """Initialize all defined flows with zero values."""
    flows = {}
    for flow_definition in flow_definitions:
        try:
            from_process = processes[flow_definition.from_process_name]
            to_process = processes[flow_definition.to_process_name]
        except KeyError:
            raise KeyError(f"Missing process required by flow definition {flow_definition}.")
        dim_subset = dims.get_subset(flow_definition.dim_letters)
        flow = Flow(from_process=from_process, to_process=to_process, dims=dim_subset)
        flows[flow.name] = flow
    return flows
