from typing import List

from sodym.named_dim_arrays import Process

def make_processes(definitions: List[str]) -> dict[str, Process]:
    return {
        name: Process(name=name, id=id) for id, name in enumerate(definitions)
    }