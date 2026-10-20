# Research map

Research checked on **20 October 2026**. This map separates the implemented teaching profile from architecture-specific extensions. It is a component taxonomy and visualization plan, not a claim that every model contains every component.

## The first teaching profile

The app models a decoder-only, causal, pre-RMSNorm transformer with RoPE and SwiGLU. Its detailed flow uses separate matrix products, head reshapes, positional rotations, masked attention, head concatenation, output projection and residual additions. The main decoder repeats L times with distinct parameters. [Transformer foundation](https://arxiv.org/abs/1706.03762), [RMSNorm](https://arxiv.org/abs/1910.07467), [RoPE](https://arxiv.org/abs/2104.09864), [GLU variants](https://arxiv.org/abs/2002.05202).

The [tensor laboratory](numerical-laboratory.md) now executes a small separate version of this profile with actual values from seeded, untrained parameters. It includes scalar provenance, parameter inspection, independent heads, vocabulary selection and cached decoding for full, sliding and fixed-anchor sparse retention. The architecture study heatmaps and graphical MTP scenarios retain their explicit illustrative labels. Neither mode claims trained-model activations.

For row-vector notation, `X: [B,T,D]`, `Wq: [D,Hq·Dh]`, `Wk/Wv: [D,Hkv·Dh]`, and `Wo: [Hq·Dh,D]`. The query-head output contracts the sequence axis when averaging values. Residual sums and SwiGLU's Hadamard product do not contract axes. This notation is our presentation convention; it may transpose weight matrices relative to a source implementation.

## Component taxonomy and interaction design

| Area                   | Components to represent                                                                                      | Appropriate visualization                                         | Current status                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| Input formatting       | Chat template, role/control IDs, special tokens, normalization, pre-tokenization, Unicode and bytes          | Text/byte span alignment; explicit added tokens                   | UTF-8 and demonstration BPE implemented; chat templates deferred |
| Tokenization           | Byte BPE, merge ranks, vocabulary IDs, Unigram/SentencePiece alternatives, byte fallback                     | Step-by-step merge trace or algorithm-specific segmentation view  | Real teaching BPE trace; adapter interface ready                 |
| Embeddings             | Token gather, optional learned positional embeddings, tied/untied output matrix                              | Selected ID → weight row; axis-labelled vector strip              | Gather and matrix shapes implemented                             |
| Residual stream        | Block inputs, additions, sequential/parallel branches, repeated layers                                       | Branch edges with explicit element-wise sums and layer selector   | Pre-norm sequential block implemented                            |
| Normalization          | RMSNorm, LayerNorm, learned scale, bias, epsilon; pre/post positions                                         | Feature axis reduction and normalization equation                 | RMSNorm implemented; alternatives need separate content          |
| Attention projections  | Q/K/V, reshape/transpose, query/key normalization, RoPE/other position methods                               | Tensor axes before/after each operation                           | Standard Q/K/V and RoPE implemented                              |
| Attention weighting    | QK transpose, feature contraction, scale, additive mask, softmax, weighted V, concat, output map             | Clickable contraction edges and per-head heatmap                  | Dense causal synthetic example implemented                       |
| KV state               | Per-layer K/V, append offset, query-to-KV-group mapping, prefix reads                                        | Selected tensor slice with explicit five axes; bytes vs sequence  | Implemented for MHA/GQA/MQA                                      |
| Feed-forward           | Gate/up/down matrices, activations, Hadamard gate; dense or routed expert replacement                        | Split projections → activation → gate → contraction               | Dense SwiGLU implemented                                         |
| Expert routing         | Router scores, top-k, normalized routing weights, dispatch, experts, combine, shared experts                 | Per-token route selection and active branch highlighting          | Extension plan; no simulated MoE routing yet                     |
| Conditional memory     | Causal n-gram windows, token compression, hashing, learned memory lookup, contextual gating                  | Token n-gram → explicit address → vector → residual fusion        | Illustrative n=2/3/4 lookup lab; full architecture deferred      |
| Output                 | Final norm, vocabulary logits, temperature, softmax, filtering, choice, ID-to-byte decoding                  | Hidden width → vocabulary width; inspect selected token and bytes | Structural graph implemented; no language generation             |
| Training               | Shifted labels, cross-entropy, optional router and MTP losses; dropout and gradients                         | A separate training/objective branch with valid-position masks    | Sequential MTP alignment lab; no backprop execution              |
| Multi-token prediction | Independent heads or sequential modules, future target offsets, shared projections, speculative verification | Depth slider with input/target token alignment                    | Sequential explanatory graph implemented                         |
| Other model families   | Encoder–decoder cross-attention, multimodal encoders/projectors, state-space or hybrid blocks                | New graph/state profiles with their own shape conventions         | Deferred; do not force them into the dense decoder profile       |

The [official tiktoken educational code](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py) provides a primary BPE reference. Our independent, deliberately small byte-BPE vocabulary has no normalization, pre-tokenization or special-token rules. It preserves bytes and handles split UTF-8 spans; it does not reproduce any production tokenizer.

## Attention architectures and execution are separate axes

| Family                  | Query / KV relationship                                          | What the explorer must show                                                                   |
| ----------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| MHA                     | One K/V head per query head                                      | Independent per-head K/V projections and cached state                                         |
| GQA                     | Several query heads share one K/V pair                           | The group mapping; K/V are shared logically, not necessarily copied physically                |
| MQA                     | All query heads share one K/V pair                               | One KV head with multiple query heads                                                         |
| MLA                     | Joint compressed K/V latent state and a positional-key component | Latent widths and state, up/down projections and architecture-specific computation            |
| Sliding-window / sparse | Only selected keys are visible per query                         | Actual visibility pattern, absolute positions, eviction policy and any retained global tokens |
| Cross-attention         | Q comes from one stream; K/V from another                        | Independent query/key lengths and the provenance of both streams                              |

[GQA](https://arxiv.org/abs/2305.13245) connects MHA and MQA through KV-head grouping. In the current viewer, zero-based query head h uses `g(h) = floor(h / (Hq/Hkv))` for contiguous groups. The UI updates projection sizes and cache accounting with the sharing choice.

MLA stores a different representation from dense K/V. Its low-rank latent and positional key state require explicit dimensions and dedicated cache accounting; reducing `Hkv` does not implement MLA. The [DeepSeek-V3 report](https://arxiv.org/html/2412.19437v2) and [official inference implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py) are starting references for that future profile.

[FlashAttention](https://arxiv.org/abs/2205.14135) is an exact, IO-aware implementation of attention that avoids materializing the entire score/probability matrices. A logical attention heatmap should therefore not imply that a kernel allocated that matrix. [PagedAttention](https://arxiv.org/abs/2309.06180) concerns physical KV-memory management. Head sharing, visibility masks, numerical kernels and physical allocation should be independent concepts in future model descriptions.

The [Mistral 7B paper](https://arxiv.org/abs/2310.06825) is the primary source for the workbench’s sliding-window and rolling-buffer explanations. [Mixtral of Experts](https://arxiv.org/abs/2401.04088) provides a concrete routed-expert model for a later per-token dispatch view. Neither is currently presented as an executable model preset.

## KV cache conventions

The bundled graph uses a logical all-layer view of K and V, each `[L,B,Hkv,R,Dh]`, with an absolute-position map. R equals the processed extent S under full retention and can be smaller under sliding or sparse retention. Implementations can store one tensor per layer, transpose axes, use paged blocks, shard heads or quantize storage. These choices must be recorded independently from semantic axes. Our view uses one selected layer/head plane; it is not a representation of nested tables.

Prefill computes multiple queries while applying a causal mask. Ordinary decode computes one new query and appends that layer's new K/V at the proper absolute position. Earlier layer-specific K/V are reused. The dense logical byte formula excludes physical allocation overhead. Sliding-window eviction, prefix sharing, padding, quantization metadata and device distribution require separate accounting.

## N-gram memory

The memory lab exposes causal bigrams, trigrams and four-grams over token IDs. Its one-head FNV-style address calculation is an illustrative mechanism, not the complete Engram algorithm. Learned conditional memory differs from dynamic attention state. Full [Engram](https://arxiv.org/html/2601.07372v1) includes tokenizer compression, multiple hash heads, contextual gating and short convolution; its [official repository](https://github.com/deepseek-ai/Engram) should guide an exact future implementation.

## MTP and output semantics

[Independent multi-token heads](https://arxiv.org/abs/2404.19737) and [sequential MTP modules](https://arxiv.org/html/2412.19437v2) are distinct designs. The current study illustrates the latter: previous-depth state plus a shifted embedding, then a depth-specific block and shared output projection. The alignment lab labels teacher-forced inputs and targets, and excludes unavailable targets. Auxiliary training modules need not run during ordinary inference. Speculative proposals require verification before tokens are accepted.

Output token IDs are **decoded** back to bytes/text; they are not re-encoded. A generated ID is appended to the model's sequence for another inference step. The graphical simulator explains that loop with abstract packets and scripted choices. The numerical laboratory executes seeded untrained weights and samples opaque vocabulary IDs locally; it has no language tokenizer or trained language knowledge.

## Recommended implementation sequence

1. **This foundation:** reliable graph interaction, mathematics, tokenizer trace, tensor shapes, logical cache view and portable document contract.
2. **Exact tokenizer lab:** user-specified tokenizer adapter, original ranks/config, precise span alignment and specialized operations. Preserve the current adapter boundary.
3. **Named model documents:** pinned official model revision/config, real parameter dimensions, documented discrepancies, verified per-model sources.
4. **New architecture profiles:** MLA, MoE routing and additional sparse/linear patterns with proper state schemas and masks. Migrate versioned documents where necessary.
5. **Measured trace import:** extend existing graphical and numerical playback with separately imported checkpoint traces; validate model identity, provenance, shapes and absolute positions.
6. **Comparisons and scale:** graph diff, reusable subgraphs, port-aware validation, large-model level-of-detail rendering, shared layout libraries and version migrations.

This sequence is an engineering recommendation based on the user's scope and the boundaries identified above; it is not a claim made by the cited papers.
