"""Gated delta-rule attention with explicit recurrent state.

The operation order follows the Transformers 5.17.0 PyTorch reference
(Apache-2.0). In particular, normalization and accumulation use float32.
"""

from torch import Tensor

import torch
import torch.nn.functional


def chunk_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Compute chunked gated delta-rule attention with fp32 accumulation.

    Args:
      query: Queries shaped ``[batch, sequence, heads, key_width]``.
      key: Keys with the same shape as query.
      value: Values shaped ``[batch, sequence, heads, value_width]``.
      g: Log-decays shaped ``[batch, sequence, heads]``.
      beta: Update strengths with the same shape as g.
      chunk_size: Positive number of positions in each triangular system.
      initial_state: Optional fp32 ``[batch, heads, key_width, value_width]``.
      output_final_state: Return the final state when true.
      use_qk_l2norm_in_kernel: Normalize queries and keys before the scan.

    Returns:
      output: Attention output in query's dtype and value's shape.
      state: Final fp32 state, or None. The input state is never mutated.

    References:
      https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
        PyTorch chunk_gated_delta_rule reference.

    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive; got {chunk_size}.")
    dtype = query.dtype
    batch, sequence, _, key_width = key.shape
    heads, value_width = value.shape[-2:]
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query, key = _norm_l2(query), _norm_l2(key)
    query = query * float(query.shape[-1] ** -0.5)
    padding = (chunk_size - sequence % chunk_size) % chunk_size
    query, key, value = (
        torch.nn.functional.pad(tensor, (0, 0, 0, padding))
        for tensor in (query, key, value)
    )
    beta, decay = (
        torch.nn.functional.pad(tensor, (0, padding)) for tensor in (beta, decay)
    )
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        tensor.reshape(batch, heads, -1, chunk_size, tensor.shape[-1])
        for tensor in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(batch, heads, -1, chunk_size)
    upper = torch.ones(
        chunk_size, chunk_size, dtype=torch.bool, device=query.device
    ).triu(1)
    cumulative = decay.cumsum(dim=3)
    pairwise = cumulative.unsqueeze(4) - cumulative.unsqueeze(3)
    pairwise = pairwise.masked_fill(upper, float("-inf")).exp()
    system = (k_beta @ key.transpose(-1, -2)) * pairwise
    attention = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        system, v_beta, upper=False, unitriangular=True
    )
    k_cumulative = torch.linalg.solve_triangular(
        system,
        decayed_k_beta,
        upper=False,
        unitriangular=True,
    )
    state: Tensor = (
        torch.zeros(
            batch,
            heads,
            key_width,
            value_width,
            dtype=new_values.dtype,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(new_values)
    )
    output = torch.zeros_like(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    for index in range((sequence + padding) // chunk_size):
        correction = new_values[:, :, index] - k_cumulative[:, :, index] @ state
        inter_chunk = query[:, :, index] @ state
        output[:, :, index] = inter_chunk + attention[:, :, index] @ correction
        state = (
            state * chunk_decay[:, :, index]
            + key[:, :, index].transpose(-1, -2) @ correction
        )
    output = output.reshape(batch, heads, -1, value_width)[:, :, :sequence]
    output = output.transpose(1, 2).to(dtype, memory_format=torch.contiguous_format)
    return output, state if output_final_state else None


def recurrent_gated_delta_rule(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Compute gated delta-rule attention one token at a time.

    Args:
      query: Queries shaped ``[batch, sequence, heads, key_width]``.
      key: Keys with the same shape as query.
      value: Values shaped ``[batch, sequence, heads, value_width]``.
      g: Log-decays shaped ``[batch, sequence, heads]``.
      beta: Update strengths with the same shape as g.
      initial_state: Optional ``[batch, heads, key_width, value_width]`` state.
      output_final_state: Return the final state when true.
      use_qk_l2norm_in_kernel: Normalize queries and keys before the scan.

    Returns:
      output: Attention output in query's dtype and value's shape.
      state: Final fp32 state, or None. The input state is never mutated.

    """
    dtype = query.dtype
    batch, sequence, _, key_width = key.shape
    heads, value_width = value.shape[-2:]
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for tensor in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query, key = _norm_l2(query), _norm_l2(key)
    query = query / float(query.shape[-1] ** 0.5)
    state = (
        torch.zeros(
            batch, heads, key_width, value_width, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.to(value)
    )
    output = torch.zeros_like(value)
    for index in range(sequence):
        q, k, v = query[:, :, index], key[:, :, index], value[:, :, index]
        state = state * decay[:, :, index].exp()[..., None, None]
        memory = (state * k.unsqueeze(-1)).sum(dim=-2)
        correction = (v - memory) * beta[:, :, index].unsqueeze(-1)
        state = state + k.unsqueeze(-1) * correction.unsqueeze(-2)
        output[:, :, index] = (state * q.unsqueeze(-1)).sum(dim=-2)
    output = output.transpose(1, 2).contiguous().to(dtype)
    return output, state if output_final_state else None


def _norm_l2(tensor: Tensor) -> Tensor:
    return tensor * torch.rsqrt((tensor * tensor).sum(dim=-1, keepdim=True) + 1e-6)
