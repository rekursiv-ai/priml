from torchao.quantization.granularity import PerRow, PerTensor

"""
Common types for float8 quantization
"""
type FP8Granularity = PerTensor | PerRow
