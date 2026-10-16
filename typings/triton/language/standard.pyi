from triton.language import tensor
from triton.runtime.jit import JITFunction

sum: JITFunction[..., tensor]
