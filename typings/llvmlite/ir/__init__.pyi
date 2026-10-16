from collections.abc import Sequence
from typing import Self

class Type:
    def as_pointer(self, addrspace: int = ...) -> PointerType: ...

class PointerType(Type): ...

class IntType(Type):
    width: int
    def __new__(cls, bits: int) -> Self: ...

class VoidType(Type):
    def __init__(self) -> None: ...

class LiteralStructType(Type):
    elements: tuple[Type, ...]
    def __init__(self, elems: Sequence[Type], packed: bool = ...) -> None: ...

class FunctionType(Type):
    return_type: Type
    args: tuple[Type, ...]
    var_arg: bool
    def __init__(
        self,
        return_type: Type,
        args: Sequence[Type],
        var_arg: bool = ...,
    ) -> None: ...

class Value:
    type: Type

class Constant(Value):
    constant: object
    def __init__(self, typ: Type, constant: object) -> None: ...

class NamedValue(Value):
    name: str

class Module:
    name: str
    globals: dict[str, GlobalValue]
    def __init__(self, name: str = ...) -> None: ...
    def declare_intrinsic(
        self,
        intrinsic: str,
        tys: Sequence[Type] = ...,
        fnty: FunctionType | None = ...,
    ) -> Function: ...

class GlobalValue(NamedValue):
    module: Module

class Argument(NamedValue): ...
class Block(NamedValue): ...

class Function(GlobalValue):
    ftype: FunctionType
    args: tuple[Argument, ...]
    def __init__(self, module: Module, ftype: FunctionType, name: str) -> None: ...
    def append_basic_block(self, name: str = ...) -> Block: ...

class Instruction(NamedValue): ...
class CallInstr(Instruction): ...
class LoadAtomicInstr(Instruction): ...
class StoreAtomicInstr(Instruction): ...
class GEPInstr(Instruction): ...
class ExtractValue(Instruction): ...
class CastInstr(Instruction): ...

class IRBuilder:
    def __init__(self, block: Block | None = ...) -> None: ...
    @property
    def module(self) -> Module: ...
    def ret_void(self) -> Instruction: ...
    def call(
        self,
        fn: Value,
        args: Sequence[Value],
        name: str = ...,
        cconv: str | None = ...,
        tail: bool = ...,
        fastmath: Sequence[str] = ...,
        attrs: Sequence[str] = ...,
        arg_attrs: dict[int, Sequence[str]] | None = ...,
    ) -> CallInstr: ...
    def load_atomic(
        self,
        ptr: Value,
        ordering: str,
        align: int,
        name: str = ...,
        typ: Type | None = ...,
    ) -> LoadAtomicInstr: ...
    def store_atomic(
        self,
        value: Value,
        ptr: Value,
        ordering: str,
        align: int,
    ) -> StoreAtomicInstr: ...
    def gep(
        self,
        ptr: Value,
        indices: Sequence[Value],
        inbounds: bool = ...,
        name: str = ...,
        source_etype: Type | None = ...,
    ) -> GEPInstr: ...
    def extract_value(
        self,
        agg: Value,
        idx: int | Sequence[int],
        name: str = ...,
    ) -> ExtractValue: ...
    def bitcast(self, value: Value, typ: Type, name: str = ...) -> CastInstr: ...
