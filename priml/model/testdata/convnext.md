ConvNeXt forward witness

Reference: timm 1.0.25, timm.models.convnext.ConvNeXt.
Wheel SHA-256: bef7f61dd717cb2dbbb7e326f143e13d660a47ecbd84116e6fe33732bed5c484.
Upstream: https://github.com/huggingface/pytorch-image-models/tree/v1.0.25

Configuration: RGB input, patch size 4, four stages, depths (1, 1, 1, 1),
widths (2, 3, 4, 5), depthwise kernel 3, zero classifier classes,
maximum drop-path probability 0.1. Input shape (2, 3, 64, 96), built by
arange / 101. Evaluation mode disables stochastic depth.

The source-only oracle ran in an isolated worktree after its full pre-commit
passed. The source model and arithmetic ran inside host_agnostic_numerics.
The record contains every initialized source parameter and buffer, the whole
raw spatial map, and the whole pooled output. The native replay loads those
complete weights through a strict bijection before comparing outputs.

This witness does not compare constructor RNG consumption, training,
stochastic-depth draws, mixed precision, compilation, or CUDA arithmetic.
The source-only minter and original implementation are outside this package.
