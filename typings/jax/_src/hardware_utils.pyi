import enum

class TpuVersion(enum.IntEnum):
    v2 = 0
    v3 = 1
    plc = 2
    v4 = 3
    v5p = 4
    v5e = 5
    v6e = 6
    tpu7x = 7

def num_available_tpu_chips_and_device_id(): ...
def has_visible_nvidia_gpu() -> bool: ...
def transparent_hugepages_enabled() -> bool: ...
