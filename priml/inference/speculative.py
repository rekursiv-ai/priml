"""Speculative decoding implementations for FP4 quantized models.

Research basis:
- EAGLE (Li et al. 2024): AR head + KD + token tree → 2.39× FP16 on A100
- Medusa (Cai et al. 2024): FFN heads + Seq-KD → 2.42× FP16 on 7B
- Token tree verification (SpecInfer): boosts acceptance rate
- FP4 interaction: UNEXPLORED — this experiment measures it

Key finding from survey (arXiv:2401.07851):
- FP16→FP32 loses 27-61% speedup across all methods
- FP4 quantization effect on acceptance rate is UNKNOWN
- This experiment fills that gap
"""

from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn as nn
from priml.inference.kv_cache import FP4KVCache


@dataclass
class DrafterOutput:
    """Output from a drafter model."""
    draft_tokens: torch.Tensor      # [batch, spec_len] or [batch, tree_size]
    draft_probs: torch.Tensor       # [batch, spec_len, vocab] or [batch, tree_size, vocab]
    drafter_kv: Optional[FP4KVCache]
    tree_structure: Optional[dict] = None  # For token tree: parent indices, depths


class FP4Quantizer:
    """FP4 quantization utilities for e2m1 and NVFP4 formats."""

    # e2m1: 1 sign, 2 exponent, 1 mantissa → values: ±6, ±4, ±3, ±2, ±1.5, ±1, ±0.5, 0
    E2M1_VALUES = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6])

    @staticmethod
    def quantize_e2m1(weight: torch.Tensor, block_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize to e2m1 FP4.

        Args:
            weight: Input weight tensor
            block_size: Block size for scaling (1=per-tensor, 16=NVFP4 style)

        Returns:
            qweight: Quantized uint8 tensor (packed 2 values per byte)
            scale: Scale factors per block
        """
        if block_size == 1:
            # Per-tensor quantization
            scale = weight.abs().max() / 6.0
            qweight = (weight / scale).round().clamp(-6, 6)
            # Pack 2 values per byte
            qweight_packed = FP4Quantizer._pack_uint4(qweight)
            return qweight_packed, scale.unsqueeze(0)
        else:
            # Block quantization (NVFP4 style)
            orig_shape = weight.shape
            weight_flat = weight.flatten()
            n_blocks = (weight_flat.numel() + block_size - 1) // block_size
            padded = torch.zeros(n_blocks * block_size, device=weight.device, dtype=weight.dtype)
            padded[:weight_flat.numel()] = weight_flat
            blocks = padded.view(n_blocks, block_size)

            scales = blocks.abs().max(dim=1).values / 6.0
            scales = scales.clamp_min(1e-8)
            qblocks = (blocks / scales.unsqueeze(1)).round().clamp(-6, 6)
            qweight_packed = FP4Quantizer._pack_uint4(qblocks.flatten()[:weight_flat.numel()])

            return qweight_packed, scales

    @staticmethod
    def dequantize_e2m1(qweight: torch.Tensor, scale: torch.Tensor, block_size: int = 1, orig_shape=None) -> torch.Tensor:
        """Dequantize e2m1 FP4 back to float."""
        # Unpack
        unpacked = FP4Quantizer._unpack_uint4(qweight)
        if block_size == 1:
            return unpacked.to(torch.float) * scale
        else:
            n_blocks = scale.numel()
            blocks = unpacked[:n_blocks * block_size].view(n_blocks, block_size)
            dequant = blocks.to(torch.float) * scale.unsqueeze(1)
            if orig_shape:
                return dequant.flatten()[:orig_shape.numel()].view(orig_shape)
            return dequant.flatten()

    @staticmethod
    def _pack_uint4(tensor: torch.Tensor) -> torch.Tensor:
        """Pack two int4 values into one uint8."""
        tensor = tensor.to(torch.int8) + 8  # Shift to 0-15
        even = tensor[0::2]
        odd = tensor[1::2] if tensor.numel() > 1 else torch.zeros_like(even)
        return (even << 4) | odd

    @staticmethod
    def _unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
        """Unpack uint8 to two int4 values."""
        even = (packed >> 4) & 0xF
        odd = packed & 0xF
        result = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
        result[0::2] = even
        result[1::2] = odd
        return result - 8  # Shift back to -8..7


class EagleDrafter(nn.Module):
    """EAGLE: Autoregressive head on target hidden states.

    Architecture: Linear(hidden_dim) → LayerNorm → Linear(vocab)
    Training: Seq-KD from target logits + LM loss
    Inference: Token tree generation with verification

    Reference: Li et al. "EAGLE: Speculative Sampling Requires Rethinking
    Feature Uncertainty" (ICML 2024)
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_layers: int = 1,
        fp4_quantize: bool = False,
        fp4_format: Literal["e2m1", "nvfp4_block16"] = "e2m1",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fp4_quantize = fp4_quantize
        self.fp4_format = fp4_format
        self.block_size = 16 if fp4_format == "nvfp4_block16" else 1

        self.norm = nn.LayerNorm(hidden_dim)
        self.ar_head = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)

        if fp4_quantize:
            self._quantize_weights()

    def _quantize_weights(self):
        """Quantize drafter weights to FP4."""
        for module, name in [(self.ar_head, "ar_head"), (self.output_proj, "output_proj")]:
            weight = module.weight.data
            qweight, scale = FP4Quantizer.quantize_e2m1(weight, self.block_size)
            module.register_buffer(f"fp4_weight", qweight)
            module.register_buffer(f"fp4_scale", scale)
            # Store original shape for dequantization
            module.register_buffer(f"orig_shape", torch.tensor(weight.shape))

    def _dequantize_weight(self, module: nn.Linear) -> torch.Tensor:
        """Dequantize weight for computation."""
        if not self.fp4_quantize:
            return module.weight
        qweight = getattr(module, "fp4_weight")
        scale = getattr(module, "fp4_scale")
        orig_shape = getattr(module, "orig_shape").tolist()
        return FP4Quantizer.dequantize_e2m1(qweight, scale, self.block_size, tuple(orig_shape))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate next-token logits from target hidden states."""
        x = self.norm(hidden_states)

        if self.fp4_quantize:
            # Dequantize for compute
            ar_weight = self._dequantize_weight(self.ar_head)
            x = x @ ar_weight.t()
        else:
            x = self.ar_head(x)

        x = torch.nn.functional.silu(x)

        if self.fp4_quantize:
            out_weight = self._dequantize_weight(self.output_proj)
            return x @ out_weight.t()
        return self.output_proj(x)

    def generate_tree(
        self,
        hidden_states: torch.Tensor,
        tree_depth: int,
        branching: int,
        temperature: float = 1.0,
    ) -> DrafterOutput:
        """Generate token tree for verification.

        EAGLE-style token tree: each node expands to `branching` children.
        Total nodes = (branching^(depth+1) - 1) / (branching - 1)

        Returns all nodes in tree for single verification pass.
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # Root node: first token from AR head
        root_logits = self.forward(hidden_states[:, -1:])  # [batch, 1, vocab]
        root_probs = torch.softmax(root_logits / temperature, dim=-1)
        root_tokens = root_probs.argmax(dim=-1)  # [batch, 1]

        # Build tree breadth-first
        all_tokens = [root_tokens]
        all_probs = [root_probs]
        tree_parents = [torch.zeros(batch_size, 1, dtype=torch.long, device=device)]  # Root has no parent
        tree_depths = [torch.zeros(batch_size, 1, dtype=torch.long, device=device)]

        current_hidden = hidden_states

        for depth in range(1, tree_depth + 1):
            prev_tokens = all_tokens[-1]
            num_prev = prev_tokens.shape[1]

            # Expand each node to `branching` children
            # For simplicity, use same hidden states (in practice, would propagate through target)
            # This is a simplified version - full impl would run target model on each path

            new_tokens_list = []
            new_probs_list = []
            new_parents = []
            new_depths = []

            for b in range(batch_size):
                for i in range(num_prev):
                    # Get hidden state for this node (simplified: reuse last)
                    node_hidden = current_hidden[b:b+1, -1:]

                    # Generate branching children
                    child_logits = self.forward(node_hidden).repeat(branching, 1, 1)
                    child_probs = torch.softmax(child_logits / temperature, dim=-1)
                    child_tokens = child_probs.argmax(dim=-1)

                    new_tokens_list.append(child_tokens)
                    new_probs_list.append(child_probs)
                    new_parents.append(torch.full((branching,), len(all_tokens) - 1 + i, device=device))
                    new_depths.append(torch.full((branching,), depth, device=device))

            if new_tokens_list:
                all_tokens.append(torch.cat(new_tokens_list, dim=0).unsqueeze(0))
                all_probs.append(torch.cat(new_probs_list, dim=0).unsqueeze(0))
                tree_parents.append(torch.cat(new_parents))
                tree_depths.append(torch.cat(new_depths))

        # Flatten tree
        draft_tokens = torch.cat(all_tokens, dim=1)  # [batch, tree_size]
        draft_probs = torch.cat(all_probs, dim=1)
        tree_structure = {
            "parents": torch.cat(tree_parents),
            "depths": torch.cat(tree_depths),
            "num_nodes": draft_tokens.shape[1],
        }

        return DrafterOutput(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            drafter_kv=None,  # Drafter doesn't maintain KV in this design
            tree_structure=tree_structure,
        )


class MedusaDrafter(nn.Module):
    """Medusa: Multiple FFN heads for parallel draft tokens.

    Architecture: Multiple FFN(hidden_dim) heads → each predicts next token
    Training: Seq-KD + LM loss on each head
    Inference: Parallel generation, token tree verification

    Reference: Cai et al. "Medusa: Simple LLM Inference Acceleration
    Framework with Multiple Decoding Heads" (ICLR 2024)
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int = 5,
        fp4_quantize: bool = False,
        fp4_format: Literal["e2m1", "nvfp4_block16"] = "e2m1",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.fp4_quantize = fp4_quantize
        self.fp4_format = fp4_format
        self.block_size = 16 if fp4_format == "nvfp4_block16" else 1

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size)
            ) for _ in range(num_heads)
        ])

        if fp4_quantize:
            self._quantize_heads()

    def _quantize_heads(self):
        """Quantize all head weights to FP4."""
        for head_idx, head in enumerate(self.heads):
            for layer_idx, layer in enumerate(head):
                if isinstance(layer, nn.Linear):
                    weight = layer.weight.data
                    qweight, scale = FP4Quantizer.quantize_e2m1(weight, self.block_size)
                    layer.register_buffer(f"fp4_weight", qweight)
                    layer.register_buffer(f"fp4_scale", scale)
                    layer.register_buffer(f"orig_shape", torch.tensor(weight.shape))

    def _dequantize_layer(self, layer: nn.Linear) -> torch.Tensor:
        if not self.fp4_quantize:
            return layer.weight
        qweight = getattr(layer, "fp4_weight")
        scale = getattr(layer, "fp4_scale")
        orig_shape = getattr(layer, "orig_shape").tolist()
        return FP4Quantizer.dequantize_e2m1(qweight, scale, self.block_size, tuple(orig_shape))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate logits from all heads.

        Returns: [batch, num_heads, vocab] — each head predicts next token
        """
        batch_size = hidden_states.shape[0]
        last_hidden = hidden_states[:, -1:]  # [batch, 1, hidden]

        all_logits = []
        for head in self.heads:
            x = last_hidden
            for layer in head:
                if isinstance(layer, nn.Linear):
                    if self.fp4_quantize:
                        weight = self._dequantize_layer(layer)
                        x = x @ weight.t()
                        if layer.bias is not None:
                            x = x + layer.bias
                    else:
                        x = layer(x)
                else:
                    x = layer(x)
            all_logits.append(x)

        return torch.stack(all_logits, dim=1)  # [batch, num_heads, vocab]

    def generate_parallel(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> DrafterOutput:
        """Generate parallel drafts from all heads."""
        logits = self.forward(hidden_states)  # [batch, num_heads, vocab]
        probs = torch.softmax(logits / temperature, dim=-1)
        tokens = probs.argmax(dim=-1)  # [batch, num_heads]

        # Build tree structure: root (head 0) → children (heads 1..n)
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        tree_structure = {
            "parents": torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),  # Root
                torch.zeros(batch_size, self.num_heads - 1, dtype=torch.long, device=device)  # All from root
            ], dim=1),
            "depths": torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                torch.ones(batch_size, self.num_heads - 1, dtype=torch.long, device=device)
            ], dim=1),
            "num_nodes": self.num_heads,
        }

        return DrafterOutput(
            draft_tokens=tokens,
            draft_probs=probs,
            drafter_kv=None,
            tree_structure=tree_structure,
        )


class TokenTreeVerifier:
    """Token tree verification (SpecInfer / EAGLE style).

    Algorithm:
    1. Drafter generates tree of candidate tokens
    2. Target model verifies all nodes in single forward pass (using tree attention mask)
    3. Accept longest prefix where all tokens match target distribution
    4. If mismatch at node: reject subtree, continue from parent

    Key: Single target forward pass verifies entire tree.
    """

    def __init__(
        self,
        target_model: nn.Module,
        fp4_kv_cache: FP4KVCache,
        acceptance_threshold: float = 0.9,
        temperature: float = 1.0,
    ):
        self.target = target_model
        self.kv_cache = fp4_kv_cache
        self.threshold = acceptance_threshold
        self.temperature = temperature

    def _build_tree_attention_mask(
        self,
        tree_structure: dict,
        seq_len: int,
    ) -> torch.Tensor:
        """Build attention mask for token tree verification.

        Each node attends to its ancestors (path to root) + input sequence.
        """
        num_nodes = tree_structure["num_nodes"]
        parents = tree_structure["parents"]  # [num_nodes]
        depths = tree_structure["depths"]    # [num_nodes]

        # Total sequence = input + tree nodes
        total_len = seq_len + num_nodes
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Input attends to input (causal)
        mask[:seq_len, :seq_len] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        # Tree nodes attend to input + their ancestors
        for i in range(num_nodes):
            node_idx = seq_len + i
            # Attend to input
            mask[node_idx, :seq_len] = True
            # Attend to ancestors
            curr = i
            while curr >= 0:
                anc_idx = seq_len + curr
                mask[node_idx, anc_idx] = True
                if curr == 0:
                    break
                curr = parents[curr].item()

        return mask

    def verify(
        self,
        drafter_output: DrafterOutput,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Verify draft tree against target model.

        Args:
            drafter_output: DrafterOutput with tree tokens/probs
            input_ids: Input token IDs [batch, seq_len]

        Returns:
            accepted_tokens: [batch, accepted_len]
            stats: {accepted, spec_depth, acceptance_rate, target_calls, tree_size}
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tree_structure = drafter_output.tree_structure
        draft_tokens = drafter_output.draft_tokens  # [batch, tree_size]
        draft_probs = drafter_output.draft_probs    # [batch, tree_size, vocab]
        num_nodes = tree_structure["num_nodes"]

        # Build tree attention mask
        attn_mask = self._build_tree_attention_mask(tree_structure, seq_len).to(device)

        # Prepare input for target: input_ids + draft tokens
        target_input = torch.cat([input_ids, draft_tokens], dim=1)  # [batch, seq_len + tree_size]

        # Single target forward pass with tree mask
        with torch.no_grad():
            target_logits = self.target(
                target_input,
                attention_mask=attn_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1),
                past_key_values=self.kv_cache,
                use_cache=True,
            ).logits

        # Target probs for tree nodes only
        target_probs = torch.softmax(
            target_logits[:, seq_len-1:seq_len-1+num_nodes] / self.temperature, dim=-1
        )  # [batch, tree_size, vocab]

        # Verify each node: target_prob[draft_token] > threshold
        # Gather target probs at draft token positions
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_nodes)
        node_indices = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        target_at_draft = target_probs[batch_indices, node_indices, draft_tokens]

        # Accept if prob > threshold
        accepted_mask = target_at_draft >= self.threshold  # [batch, tree_size]

        # Tree acceptance: must accept all ancestors of a node
        parents = tree_structure["parents"].to(device)
        for i in range(num_nodes):
            if i == 0:
                continue
            parent_mask = accepted_mask[:, parents[i]]
            accepted_mask[:, i] = accepted_mask[:, i] & parent_mask

        # Find longest accepted path per batch
        accepted_tokens_list = []
        accepted_counts = []

        for b in range(batch_size):
            accepted_idx = torch.where(accepted_mask[b])[0]
            if len(accepted_idx) > 0:
                # Take longest contiguous prefix from root
                # For tree, find max depth where all ancestors accepted
                max_depth = 0
                for i in range(num_nodes):
                    if accepted_mask[b, i]:
                        depth = tree_structure["depths"][i].item()
                        if depth > max_depth:
                            max_depth = depth
                # Get all nodes up to max_depth
                depth_mask = tree_structure["depths"] <= max_depth
                accepted = draft_tokens[b, depth_mask]
            else:
                accepted = torch.empty(0, dtype=torch.long, device=device)

            accepted_tokens_list.append(accepted)
            accepted_counts.append(len(accepted))

        # Pad to same length
        max_accepted = max(accepted_counts) if accepted_counts else 0
        if max_accepted > 0:
            accepted_padded = torch.full((batch_size, max_accepted), -1, dtype=torch.long, device=device)
            for b, tokens in enumerate(accepted_tokens_list):
                if len(tokens) > 0:
                    accepted_padded[b, :len(tokens)] = tokens
        else:
            accepted_padded = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        stats = {
            "accepted": accepted_counts,
            "spec_depth": max_accepted,
            "acceptance_rate": sum(accepted_counts) / (batch_size * num_nodes) if num_nodes > 0 else 0,
            "target_calls": 1,
            "tree_size": num_nodes,
        }

        return accepted_padded, stats


def speculative_decode_step(
    target_model: nn.Module,
    drafter: EagleDrafter | MedusaDrafter,
    verifier: TokenTreeVerifier,
    input_ids: torch.Tensor,
    fp4_kv_cache: FP4KVCache,
    max_spec_tokens: int,
    acceptance_threshold: float,
    drafter_type: Literal["eagle", "medusa"] = "eagle",
    tree_depth: int = 3,
    tree_branching: int = 2,
) -> tuple[torch.Tensor, dict]:
    """One speculative decode step.

    Returns:
        next_tokens: Accepted tokens [batch, accepted_len]
        stats: {accepted, spec_depth, acceptance_rate, target_calls}
    """
    # 1. Target forward on input → get hidden states
    with torch.no_grad():
        target_output = target_model(
            input_ids,
            past_key_values=fp4_kv_cache,
            use_cache=True,
            output_hidden_states=True,
        )
    hidden_states = target_output.hidden_states[-1]  # Last layer hidden

    # 2. Drafter generates tree/parallel tokens
    if drafter_type == "eagle":
        drafter_output = drafter.generate_tree(
            hidden_states, tree_depth, tree_branching
        )
    else:
        drafter_output = drafter.generate_parallel(hidden_states)

    # 3. Verifier checks tree in single target forward
    accepted_tokens, stats = verifier.verify(drafter_output, input_ids)

    # 4. Limit to max_spec_tokens
    if accepted_tokens.shape[1] > max_spec_tokens:
        accepted_tokens = accepted_tokens[:, :max_spec_tokens]
        stats["accepted"] = [min(a, max_spec_tokens) for a in stats["accepted"]]

    return accepted_tokens, stats


class SpeculativeDecodeEngine:
    """Full speculative decoding loop for generation."""

    def __init__(
        self,
        target_model: nn.Module,
        drafter: EagleDrafter | MedusaDrafter,
        verifier: TokenTreeVerifier,
        fp4_kv_cache: FP4KVCache,
        config: dict,
    ):
        self.target = target_model
        self.drafter = drafter
        self.verifier = verifier
        self.kv_cache = fp4_kv_cache
        self.config = config
        self.stats_history = []

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, dict]:
        """Generate tokens with speculative decoding."""
        all_accepted = []
        total_target_calls = 0
        total_accepted = 0
        total_spec = 0

        curr_ids = input_ids
        remaining = max_new_tokens

        while remaining > 0:
            # One speculative step
            accepted, stats = speculative_decode_step(
                self.target,
                self.drafter,
                self.verifier,
                curr_ids,
                self.kv_cache,
                max_spec_tokens=min(self.config.get("max_spec_tokens", 8), remaining),
                acceptance_threshold=self.config.get("acceptance_threshold", 0.9),
                drafter_type=self.config.get("drafter_type", "eagle"),
                tree_depth=self.config.get("tree_depth", 3),
                tree_branching=self.config.get("tree_branching", 2),
            )

            if accepted.shape[1] == 0:
                # Fallback: single token from target
                with torch.no_grad():
                    logits = self.target(curr_ids, past_key_values=self.kv_cache, use_cache=True).logits
                next_token = logits[:, -1:].argmax(dim=-1)
                accepted = next_token
                stats["accepted"] = [1] * curr_ids.shape[0]
                stats["target_calls"] = 1

            all_accepted.append(accepted)
            total_accepted += sum(stats["accepted"])
            total_spec += stats.get("spec_depth", 0)
            total_target_calls += stats.get("target_calls", 1)

            # Update input for next step
            curr_ids = torch.cat([curr_ids, accepted], dim=1)
            remaining -= accepted.shape[1]

            self.stats_history.append(stats)

        generated = torch.cat(all_accepted, dim=1)

        total_stats = {
            "total_accepted": total_accepted,
            "total_target_calls": total_target_calls,
            "avg_acceptance_rate": total_accepted / max(total_spec, 1),
            "speedup_estimate": total_accepted / total_target_calls if total_target_calls > 0 else 1.0,
            "steps": len(self.stats_history),
        }

        return generated, total_stats