from collections.abc import Sequence

import enum

from _typeshed import Incomplete

__all__ = ["fft", "fft_p"]

class FftType(enum.IntEnum):
    FFT = 0
    IFFT = 1
    RFFT = 2
    IRFFT = 3

def fft(x, fft_type: FftType | str, fft_lengths: Sequence[int]): ...

fft_p: Incomplete
