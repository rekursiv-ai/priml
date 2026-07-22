from dataclasses import dataclass

@dataclass(frozen=True)
class Granularity: ...

@dataclass(frozen=True)
class PerTensor(Granularity): ...

@dataclass(frozen=True)
class PerAxis(Granularity):
    axis: int

@dataclass(frozen=True)
class PerGroup(Granularity):
    group_size: int

@dataclass(frozen=True)
class PerRow(Granularity): ...

@dataclass(frozen=True)
class PerToken(Granularity): ...

@dataclass(frozen=True)
class PerBlock(Granularity):
    block_size: tuple[int, ...]
