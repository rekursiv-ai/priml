## Glossary

See [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).


| Name       |      | Shard | Description  |
|------------|------|-------|--------------|
| Rank       |      |       | Single GPU   |
| Node       |      | Rank  | Group of GPUs with fast interconnection (same host; nvlink). |
| Cluster    |      | Node  | Group of Nodes. |
| DDP        | [Distributed Data Parallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) | Model-per-Rank  | Each rank ("GPU") owns a model replica and processes a microbatch of data. An all-reduce syncs gradients across ranks (forming a minibatch). |
| HSDP       | [Hybrid Sharded Data Parallel](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) | Model-per-Node  | Combination of FSDP and DDP. Model sharded using FSDP _within each node_ (e.g., between the 8 GPUs on a single host) and data sharded across nodes |
| FSDP2      | [Fully Sharded Data Parallel](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)   | Model-per-Cluster | Parameters, gradients, and optimizer states are sharded across all available GPUs in a cluster, regardless of which machine they are on. |
| TP         | [Tensor Parallel](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html) | |  |
| PP         | [Pipeline Parallel](https://docs.pytorch.org/docs/main/distributed.pipelining.html) | | Execution of a model to be partitioned such that multiple micro-batches can execute different parts of the model code concurrently. |

Rule-of-thumb: choose the smallest model sharding possible.



|             | DDP ([Distributed Data Parallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)) | TP ([Tensor Parallel](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)) | FSDP ([Fully Sharded Data Parallel](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)) | HSDP ([Hybrid Sharded Data Parallel](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)) | PP ([Pipeline Parallel](https://docs.pytorch.org/docs/main/distributed.pipelining.html)) |
|-------------|---|---|---|---|---|
Strategy      | Data Parallelism. Each device gets a different slice of the input data. | Model Parallelism. A single layer's tensors are split across multiple devices. | Data Parallelism with Sharding. Combines data parallelism with sharding of the model state. | Hybrid Data Parallelism. Combines data parallelism with intra-node sharding. | Model Parallelism. The model is split by layer and distributed across devices. |
|||||||
Model State   | The entire model (parameters, gradients, and optimizer states) is replicated on every device. | The model's weights and activations for a given layer are partitioned across a group of devices. | All model states (parameters, gradients, and optimizer states) are sharded across all devices. | Model parameters, gradients, and optimizer states are sharded within nodes but replicated between nodes. | Each device holds and computes only a contiguous sequence of model layers.
|||||||
Communication | Gradients are synchronized via an all-reduce operation after the backward pass. | Requires frequent intra-layer communication between devices to reassemble and compute split tensors. Best with high-speed interconnects like NVLink. | Requires communication (all-gather) to reconstruct parameters before computation and communication (reduce-scatter) to synchronize sharded gradients during the backward pass. | Balances communication costs by sharding across high-bandwidth intra-node connections while using replication across slower inter-node connections. | Activations are passed from one device to the next during the forward pass, and gradients are passed back during the backward pass.
|||||||
Advantage     | High computational efficiency and simple to implement, as it requires minimal model code changes. | Enables parallelization for single layers that are too large for a single GPU, even with a small batch size. | Enables training models that are too large for a single GPU by significantly reducing per-device memory usage. | More flexible and scalable than FSDP, offering a balance between memory efficiency and communication overhead. | Allows training of extremely deep models by reducing the memory required per device.
|||||||
Disadvantage  | Memory intensive, as every device must store a full copy of the model. | Adds communication overhead and can be complex to implement within the model architecture. | Higher communication overhead than DDP, and the all-gather operation can cause memory spikes. | Can be complex to configure optimally across different hardware topologies. | Introduces "pipeline bubbles"—periods of idle time—due to dependencies between stages.
|||||||
Best For      | Training models that comfortably fit in the memory of a single GPU, using a larger batch size. | Large models where individual layers (e.g., in Transformers) are too big for a single GPU. | Very large models that exceed single-GPU memory limits, where memory savings are critical. | Large models that need to be scaled across many nodes with slower inter-node communication. | Extremely deep models where layers can be arranged sequentially. Often combined with microbatching to reduce idle time.
|||||||
Flexibility   | Lowest flexibility. Scales only the data dimension. | Low flexibility. The model architecture must be explicitly modified to split tensors. | High flexibility. Can be applied at different levels of sharding and supports offloading to CPU for greater memory savings. | Medium-to-High flexibility. Offers a configurable sharding degree, adapting to different communication topologies. | Medium flexibility. Requires splitting the model layers into sequential stages.

Recommended strategy for large models and NVLink: HSDP with per-layer TP, i.e,
- One node is one microbatch.
- Shard across heads?
