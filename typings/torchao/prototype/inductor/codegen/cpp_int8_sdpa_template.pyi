from torch._inductor import ir
from torch._inductor.codegen.cpp_flex_attention_template import CppFlexAttentionTemplate

USEFUL_FUNCTIONS = ...
ALLOCATE_BUFFER = ...
INT8_SDPA_ONE_LOOP_TEMPLATE = ...
INT8_SDPA_SEVERAL_LOOPS_TEMPLATE = ...

class CppInt8SdpaTemplate(CppFlexAttentionTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        scale,
        q_scale,
        q_zp,
        k_scale,
        k_zp,
        v_scale,
        v_zp,
        a_scale,
        a_zp,
        o_scale,
        o_zp,
    ) -> None: ...
    @staticmethod
    def add_choices(
        choices,
        input_nodes,
        layout,
        scale,
        q_scale,
        q_zp,
        k_scale,
        k_zp,
        v_scale,
        v_zp,
        a_scale,
        a_zp,
        o_scale,
        o_zp,
    ):  # -> DataProcessorTemplateWrapper:
        ...
    def reshape_attn_mask_to_4d(
        self, kernel, attn_mask: ir.Buffer, batchSize, num_head, qSize, kvSize
    ):  # -> Any:
        ...
    def get_options(
        self,
        query: ir.Buffer,
        key: ir.Buffer,
        value: ir.Buffer,
        qSize,
        kvSize,
        headSize,
        batchSize,
        num_head,
        num_threads,
    ):  # -> dict[str, int]:
        ...
    def render(
        self,
        kernel,
        template_buffer_node: ir.CppTemplateBuffer | None = ...,
        epilogue_nodes: list[ir.IRNode] | None = ...,
        **kwargs,
    ) -> str: ...
    def codegen_useful_function(self, kernel_name: str):  # -> Any:
        ...
    def codegen_allocate_buffer(
        self, buffer_name: str, buffer_dtype, buffer_size
    ):  # -> Any:
        ...
