# The tensor laboratory

Open **Tensor laboratory → Run the mathematics** to execute a small transformer in the browser. Its parameters are seeded and **untrained**. The output consists of opaque vocabulary IDs, not meaningful generated language. No text, weights or prompts are sent to a service.

The architecture explorer and graphical MTP simulator remain available. The numerical laboratory is a separate reference profile; it does not execute an arbitrary imported architecture or a trained checkpoint.

## Explore one value

For actual trained weights, choose **Local trained model**. It runs the pinned Stories260K checkpoint and matching tokenizer in a worker, reusing this canvas and microscope with its own saved session. See [the trained profile and verification](local-trained-model.md).

1. Choose a question, a chapter, a decoder call or an individual operation. The graph follows the active component when **Follow flow** is enabled.
   In the architecture inspector, **Try a numerical example** opens the corresponding foundations operation directly. Large-model layer/head numbers map to the small profile; the panel labels it as a separate untrained example.
2. Click a component to inspect its recorded outputs, inputs and parameter matrices. Click a connection to inspect the available source payload and destination operands. Its architecture explanation and primary sources remain accessible.
3. Select a tensor, a head slice and a cell. The microscope explains the cell's computation, including contraction summands, RMS statistics, rotary coefficients, visibility decisions, softmax exponentials or element-wise operands.
4. Follow an operand to its parent tensor. **Back through the calculation** returns to the selected result. Transpose view only changes the display; it does not change execution or tensor axis order.
5. Open the **Attention head observatory** to compare independent query-head distributions over absolute key positions. Shared K/V is not shared attention probability.

**All parameter tensors** opens a searchable library of every embedding, projection and gain tensor, with exact shapes, element counts and Float32 payload bytes. Parameters are available before the first operation and stay fixed across decoder calls. This byte count excludes allocator overhead and stored trace snapshots. The cache strip likewise reports the logical resident K/V payload, independently from the trace's full history.

Play, pause, step or scrub the trace. Left/right arrows step and Space toggles playback when focus is outside interactive controls. The current input is a batch of query positions during prefill, then one new position per decode call. The confirmed output updates at commit. Selecting a token does not immediately put its K/V in the cache; the next forward pass consumes it.

The timeline exposes tensors only after their recorded operation. Parameters are available from the beginning. Inspecting a component that has not executed offers **Run to this operation**. If the most recent available binding belongs to an earlier decoder call, the inspector labels that call explicitly. The trace is calculated ahead of playback, so this is an inspection boundary, not a debugger that pauses the numerical computation itself.

## Reference profile and arithmetic

The profile has batch size 1, two decoder layers, residual width 16, four query heads, head width 4, intermediate width 32 and vocabulary size 256. MHA has four KV heads, GQA two, and MQA one. Its declared capacity is 64 positions; the interactive experiment is bounded to 1–8 input IDs and 1–8 new output IDs.

Each layer performs:

```text
X
├──────────────────────────────────────────┐
└─ RMSNorm → Q/K/V projections             │
             → head reshape               │
             → RoPE(Q,K)                   │
             → working K/V + cache update  │
             → each head: QKᵀ/√Dh + mask   │
                        → softmax → P·V    │
             → concatenate → Wo ────────── + → Y
Y
├──────────────────────────────────────────┐
└─ RMSNorm ─┬─ Wgate → SiLU ─┐             │
           └─ Wup ────────── ⊙ → Wdown ── + → next layer
```

The final RMSNorm feeds an independent vocabulary matrix. Embedding and vocabulary weights are **not tied**. There are no biases, dropout, training updates, numerical MTP, n-gram memory, quantization or fused GPU kernels in this profile.

Parameters use deterministic named pseudorandom streams. Each ordinary matrix element is sampled uniformly from `[-a,a)`, with `a = 0.7 sqrt(3 / firstAxisSize)`; RMS gains use `[0.9,1.1)`. This is an explicit teaching initialization, not a claim about a trained model or a particular library's initialization. Changing a seed changes parameters reproducibly.

Activations, parameters and K/V are stored in `Float32Array`; IDs use `Int32Array`. JavaScript `Number` arithmetic accumulates sums, and each operation's output rounds to Float32. This differs from the accumulation order and rounding of a GPU or BF16/FP16 kernel. Rearrangements copy values in this reference implementation; an optimized implementation can use views, strides or fused operations.

RMSNorm uses `sqrt(mean(x²) + 1e-5)` and a per-feature gain. RoPE uses adjacent feature pairs and angle `position × 10000^(-2j / Dh)`, with zero-based pair index `j`. Other implementations can permute paired features or scale frequencies differently. Neither V nor compact cache-slot indices are rotated. These implementation choices are displayed explicitly.

Attention and vocabulary softmax subtract the row maximum before exponentiation. Masked entries are negative infinity and contribute exactly zero. Temperature applies to vocabulary selection; attention uses temperature 1 after the usual `1/sqrt(Dh)` scaling. Greedy chooses a maximum, breaking ties by lowest ID. Seeded categorical sampling uses a separate reproducible random stream and normalizes the stored FP32 probability row by its actual sum.

## Cache semantics

Each decoder layer owns separate K/V tensors `[B,Hkv,R,Dh]` with an explicit mapping from retained slot to **absolute sequence position**. These are mathematical tensors with position metadata, not nested tables or a specified physical allocator.

- **Full:** retain every processed position.
- **Sliding window W:** a query at absolute position `q` can see keys `max(0,q-W+1)…q`. Persist the latest W positions after a call.
- **Sparse:** this teaching pattern permits causal fixed anchors `position % stride == 0` and the latest W positions. Persist their union. It does not stand for all sparse-attention architectures.

For a batched prefill, the attention operation uses working tensors `[B,Hkv,U,Dh]`, where `U = retainedPrefixCount + queryCount`. Final persistent retention must not remove a key needed by an earlier query in the same batch. The working tensors remain available until those queries finish. Their mask uses absolute query/key positions.

The **STORED K/V** strip reports the state in each recorded frame. **Target R** in the call's shape summary describes the expected retained extent after the layer processes that call. These can differ while playback is between layers. The last output ID can be confirmed and still uncached when the output budget ends.

The head observatory and cell grid show absolute positions even when retained slots are compacted. No ring-buffer, paged-memory or physical-stride implementation is implied by this numerical storage. See [attention and cache](attention-and-cache.md) for the architecture/storage distinction and the separate ring-buffer illustration.

## Experiments and saving

Edit input IDs, parameter seed, output budget, head sharing, retention, sampling and temperature in **Run setup**. The main model configuration is unchanged. A changed numerical setting resets playback and calculates a fresh trace.

**Pin baseline** records the current experiment settings. After a change, **Δ baseline** compares corresponding cells. Subtraction requires matching shapes, semantic axes, storage types and absolute positions. Masked-score tensors also require identical visibility patterns; when masks differ, compare the finite attention probabilities instead. Under identical masks, excluded cells retain their exclusion marker instead of displaying an undefined `−∞ − −∞`. A baseline is recomputed from its saved settings; it is not a second set of saved tensor arrays.

Bookmark operations, annotate them, and keep a laboratory notebook. The versioned `model-visualizer.numerical-lab` document extension stores settings, the active layout, separate layouts for each head-sharing topology, bookmarks, notes and optional baseline settings. **Export model** includes this extension alongside the existing main layouts and notebook. **Open** and browser-local saves preserve it. Unknown versions or unsupported fields are retained, and the temporary laboratory session does not overwrite them.

**Export trace** creates a separate readable JSON artifact containing the actual parameters, activations, K/V, shapes, axes, position maps, operation recipes, frame bindings and provenance. JSON represents masked negative infinity as the string `"-Infinity"`. The workspace does not contain those large arrays. Numerical trace import is not implemented; workspace import is the supported route for reopening experiment settings and notes.

## Implementation and extension boundary

`src/model/numerics/tensor.ts` implements tensor primitives. `engine.ts` calculates a renderer-independent immutable-snapshot trace through `ExecutionAdapter`. `explain.ts` provides per-cell provenance. `session.ts` validates the laboratory extension and derives a graph from the existing authored foundations templates. React components render this data without owning execution semantics.

The numerical graph owns independent geometry. It uses the same obstacle-aware router, drag behavior, automatic layouts and layout undo as the main explorer. It omits inactive MTP and n-gram paths and changes the generation labels to match numerical autoregressive behavior. The authored model JSON remains the single source of architectural templates; no generated duplicate is saved.

The runtime graph and primary references come from the bundled authored **foundations** document, even when a different or edited architecture is open. The current workspace owns only the experiment extension and theme. Editing or importing architecture content does not silently change the numerical runtime's mathematics or substitute an imported source for its references.

Additional trained local runtimes or imported measured traces should supply an explicit model identity, architecture compatibility, dtype, absolute positions, parameter provenance and sampling policy. It must distinguish logical tensors from kernel intermediates and physical storage. Large arrays should stay in separate trace artifacts. Network inference would additionally need a user-requested design and explicit visible controls; there is no service integration here.

## Primary references

The profile combines selected established components; none of these sources describes this exact untrained initialization or small model.

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762): scaled dot-product attention, independent heads, concatenation and output projection.
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245): query-head groups sharing K/V.
- Shazeer, [Fast Transformer Decoding: One Write-Head Is All You Need](https://arxiv.org/abs/1911.02150): multi-query attention and decoding state.
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467): RMS normalization with per-feature gain.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864): position-dependent pair rotations of Q/K.
- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): SwiGLU and its separate gate/up projections.
- Beltagy et al., [Longformer](https://arxiv.org/abs/2004.05150), and Jiang et al., [Mistral 7B](https://arxiv.org/abs/2310.06825): local-window attention context. The laboratory's fixed-stride anchor pattern is its own labelled example.

Verification includes hand-calculated primitives, causal independence, per-cell provenance and LaTeX, and cached-decode equivalence to full causal recomputation for all three head-sharing modes and all three retention policies. Browser checks cover actual values, slicing, transpose display, operand navigation, confirmed IDs, trace export, notebook/layout round-trips, themes and narrow screens. Routing tests check the numerical graph in both directions and all head-sharing modes.
