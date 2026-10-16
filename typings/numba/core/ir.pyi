from collections.abc import Iterator

class Inst: ...
class Stmt(Inst): ...

class Var:
    name: str

class Global:
    name: str
    value: object

class Assign(Stmt):
    target: Var
    value: object

class Block:
    body: list[Stmt]
    def find_insts[I: Inst](self, cls: type[I]) -> Iterator[I]: ...

class FunctionIR:
    blocks: dict[int, Block]
