from torch._inductor.custom_graph_pass import CustomGraphPass

import torch

class DA8W4ConcatLinearCPUPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph):  # -> None:
        ...
    def uuid(self):  # -> bytes:
        ...

def register_da8w4_concat_linear_cpu_pass():  # -> None:
    ...
