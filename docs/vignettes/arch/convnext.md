# ConvNeXt

Liu et al., "A ConvNet for the 2020s", arXiv:2201.03545.

## Notation

```
p                                              # pixel position on the feature grid
xₚ                                 ∈ ℝᶜ        # features at p
LN                                             # LayerNorm over channels
GELU(s) = s Φ(s)                               # Φ = standard normal CDF
```

## Block

```
Uδ                                 ∈ ℝᶜ        # depthwise 7×7: δ ∈ {−3..3}², one weight per channel
uₚ    = Σ_δ Uδ ⊙ xₚ₊δ              ∈ ℝᶜ        # spatial mix, each channel alone
E                                  ∈ ℝ⁴ᶜˣᶜ     # expand ×4 (inverted bottleneck)
eₚ    = E LN(uₚ)                   ∈ ℝ⁴ᶜ
Y                                  ∈ ℝᶜˣ⁴ᶜ
γ                                  ∈ ℝᶜ        # layer scale, init 10⁻⁶
yₚ    = xₚ + γ ⊙ Y GELU(eₚ)        ∈ ℝᶜ        # residual (resnet.md)
```

A transformer block ([attention.md](attention.md)) with attention swapped for
a fixed local filter: both mix positions per channel; attention's weights
depend on the input, the 7×7 kernel's do not.

## Net

- stem: 4×4 conv, stride 4 ("patchify"), then LN.
- 4 stages of (3, 3, 9, 3) blocks; c = 96, 192, 384, 768 for ConvNeXt-T.
- between stages: LN, then 2×2 conv, stride 2.

ResNet-50, but (ImageNet-1K top-1 where the paper gives it):

1. stages (3, 4, 6, 3) → (3, 3, 9, 3): 78.8 → 79.4.
2. stem → patchify.
3. 3×3 conv → depthwise, wider.
4. inverted bottleneck ×4.
5. 7×7 kernel, moved before the 1×1s.
6. ReLU → one GELU per block: 81.3.
7. BatchNorm → one LN per block.
8. separate downsampling layers with LN: 82.0 (Swin-T: 81.3).

ConvNeXt-XL, ImageNet-22K pretrained, 384²: 87.8%.
