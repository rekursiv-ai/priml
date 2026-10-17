# Architecture Vignettes

Each vignette states one architecture in the style of [minGRU](mingru.md):
notation, the computation as aligned equations with shapes, then a
"X, but:" list against the model it modifies.

## Recurrent

- [Recurrent memory: one family](memory.md) (start here)
- Gated RNNs
  - [minGRU (and PufferLib's highway variant)](mingru.md)
  - [minLSTM](minlstm.md)
- State-space models
  - [Mamba and Mamba-2](mamba.md)
- Networks as memory
  - [TTT and Titans](ttt.md)
- Training
  - [Parallel scan](scan.md)

## Attention

- Softmax
  - [Attention and multi-head attention](attention.md)
  - [Multi-head latent attention (MLA)](mla.md)
  - [LoLA and sparse caches (H2O, EM-LLM)](lola.md)
- Linear
  - [Gated linear attention (GLA)](gla.md)
  - [Gated DeltaNet](deltanet.md)
  - [Kimi Delta Attention (KDA), Kimi Linear, and successors](kda.md)

## Depth

- Skip connections
  - [Highway connections and ResNet](resnet.md)
- Convolution
  - [ConvNeXt](convnext.md)
- Recursion
  - [Tiny Recursive Model (TRM)](trm.md)
