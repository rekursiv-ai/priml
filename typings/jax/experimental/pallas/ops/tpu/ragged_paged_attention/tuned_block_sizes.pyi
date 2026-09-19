from _typeshed import Incomplete

MAX_PAGES_PER_SEQ: int
TUNED_BLOCK_SIZES: Incomplete

def next_power_of_2(x: int): ...
def simplify_key(key): ...
def get_tpu_version() -> int: ...
def get_device_name(num_devices: int | None = None): ...
def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    num_q_heads_per_blk,
    num_kv_heads_per_blk,
    head_dim,
    page_size,
    max_num_batched_tokens,
    pages_per_seq,
) -> tuple[int, int]: ...
def get_min_page_size(max_model_len, min_page_size: int = 16): ...
