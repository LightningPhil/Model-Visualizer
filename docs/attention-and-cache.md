# Attention heads and KV retention

Choose **Split attention heads** in the attention study or whole-model toolbar. The study expands all query heads; the whole-model variant does so in every main decoder layer. The default configuration has 54 components / 80 connections in the head study and 420 / 610 in the split whole model. Batched and split whole-model layouts are saved independently. Existing component IDs, notes and layouts are retained.

## Independent heads and shared state

The batched projections produce `Q[B,Hq,T,Dh]`, `Knew[B,Hkv,T,Dh]` and `Vnew[B,Hkv,T,Dh]`. Splitting selects slices; it does not repeat the projection. Each query head has its own dot products, mask, softmax and value contraction. For the contiguous groups used by this teaching profile, query head `h` reads KV head `floor(h / (Hq/Hkv))`. MHA has Hkv=Hq, MQA has Hkv=1, and GQA uses an intermediate divisor. See [Attention Is All You Need, §3.2](https://arxiv.org/abs/1706.03762) and the [GQA paper, §2](https://arxiv.org/abs/2305.13245).

For one query head, the query slice is `[B,T,Dh]`. Working keys/values are `[B,U,Dh]`, with an absolute position for each slot. The dot product contracts Dh to give `[B,T,U]`; the mask and softmax preserve that shape. The weighted value sum contracts U to give `[B,T,Dh]`. Concatenation preserves head order to form `[B,T,Hq*Dh]`; Wo projects back to D. Shared K/V does not mean shared probabilities, averaged heads or copied cache allocations.

## Policy and position conventions

These workbench policies apply uniformly to all main layers and heads. Head sharing and retention are independent controls. They describe architecture and logical state, not a particular kernel, sparse matrix format or allocation strategy.

| Policy                  | Keys visible to absolute query q             | Persistent state after positions [0,S)                                 |
| ----------------------- | -------------------------------------------- | ---------------------------------------------------------------------- |
| Full                    | 0 ≤ k ≤ q                                    | Every processed position                                               |
| Sliding window          | max(0,q−W+1) ≤ k ≤ q                         | Latest min(S,W) positions                                              |
| Sparse teaching pattern | k ≤ q and either k mod stride = 0 or k > q−W | Fixed absolute stride anchors ∪ latest W positions, without duplicates |

W **includes the current position**. All future keys are excluded. Sliding retention has a bounded sequence axis; sparse anchors continue to grow with context. This fixed-anchor sparse policy is illustrative. It is not learned top-k selection, a reproduction of Sparse Transformer's factorized patterns, or a claim that sparse attention always reduces KV storage. [Sparse Transformers](https://arxiv.org/abs/1904.10509) provides primary background on structured masks; this workbench explicitly defines and tests its own pattern above.

[Mistral 7B, §2](https://arxiv.org/abs/2310.06825) describes sliding-window attention, rolling buffers and chunked prefill. Window conventions vary between implementations; use the explicit inequality above here. The cache lab's optional ring-buffer example maps absolute position p to slot `p mod W`. Slot indices wrap; RoPE position IDs do not. Paging, sharding, quantization, padding and allocator overhead remain separate implementation details.

## Persistent and working dimensions

- **S**: absolute processed sequence extent, including the current call's queries.
- **T**: query positions in the current call.
- **R**: positions retained after the call under the policy.
- **P**: retained prefix positions before the call, based on extent S−T. P can be zero for initial prefill.
- **U = P+T**: working key slots, combining retained prefix state and this call's new K/V.

Persistent K and V each have logical axes `[L,B,Hkv,R,Dh]` plus a position map. A single layer/head slice is `[B,R,Dh]`. Equal shapes do not imply equal contents across heads or layers. Logical scalar payload is `2*L*B*Hkv*R*Dh*scalarBytes`; the workbench fixes B=1 and excludes position-map metadata from its displayed byte totals. Sparse state is not represented as a dense S-long cache filled with zeros.

Prefill and verification can have more working keys than the final retained cache. Earlier queries in the batch still need their causal keys. Read the pre-update cache snapshot, combine it with all new keys, apply visibility separately for each absolute query, then commit retention. The graph explicitly shows both prior-state and current-call paths. Logical score views do not imply that a kernel materializes a complete matrix; [FlashAttention](https://arxiv.org/abs/2205.14135) distinguishes the attention computation from its memory-efficient execution.

## Simulation and verification transactions

Frames contain per-layer processed extents, actual retained position IDs and evicted IDs. This is graphical, illustrative execution with scripted proposals, without model weights or activations. It updates position accounting using the selected policy.

During an MTP verification batch, keep the pre-round cache checkpoint and pin trial positions. This temporary state may exceed the configured window. Do not overwrite required checkpoint entries irreversibly. At a mismatch, remove rejected trial state, retain the accepted prefix according to policy, and append the selected correction as an uncached output. At a full match, commit retention only after every verification query finishes. The next decoder call processes the uncached correction/bonus. Unit tests check that every rollback position was available in pinned state. A real execution adapter must implement the same transactional guarantee using actual tensors; a ring buffer alone cannot restore overwritten values.

## Portable format

`config.kvCache` is an optional additive v1 field. Omission retains full-prefix semantics, so existing documents round-trip without inserting defaults. Valid examples:

```json
{ "mode": "full" }
{ "mode": "sliding-window", "windowSize": 4 }
{ "mode": "sparse", "windowSize": 3, "stride": 4 }
```

Window size must be positive; stride must be at least 2; both are bounded by context capacity. The parser rejects unknown fields and invalid values. Export includes the policy with layouts and notes.

When layer or head counts change, the app archives derived-view layouts by `layers/queryHeads/kvHeads` in the versioned `model-visualizer.configuration-layouts` document extension. A new topology receives an automatic arrangement; returning to a previous configuration restores its saved geometry. Unknown versions of that extension are preserved without reinterpretation. Cache-policy edits do not change graph topology or move nodes.

`attention-heads` contains one `attention-expansion` component with the reserved `model-visualizer.heads` extension `{ "version": 1, "graph": "attention" }`. Its compiled instance IDs own saved positions. `whole-model-heads` contains an expansion variant `{ "version": 1, "baseGraph": "whole-model", "attentionGraph": "attention-heads" }`, referencing the original recipe rather than duplicating it. Recursive variants are rejected. These compilers implement the foundations profile; an unrelated architecture needs explicit semantics and its own validated recipe.

Earlier bundled foundations workspaces gain policy-aware R/P/U tensor axes and the missing transient current-call K/V paths through a deterministic migration. It preserves component IDs, notebook content, authored geometry and supported extension metadata. A version marker prevents repeated edits; import limits and semantic validation still apply.
