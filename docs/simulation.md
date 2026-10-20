# Graphical execution

Open **Whole model** to see the entire default inference architecture on one canvas: 252 components and 358 connections. **Run simulation** uses the same canvas with active operations and moving packet glyphs. Input/output are abstract packets; no text is needed. Every main decoder layer, all query-head results, and each K/V head slice is instantiated explicitly. The two additional MTP decoder modules are identified as separate inference components; the dedicated block/attention views explain their decoder operations. The optional n-gram extension remains visible but inactive. Training losses remain in the MTP study view.

The simulator is an **illustrative control-flow model**, with no weights, logits, measured attention values, model downloads or inference service. Packet IDs `P0`, `G0` and `D1.2` are visual identities, not vocabulary token IDs. Its arithmetic tracks query counts, sequence positions and logical KV size from the selected architecture.

## Playback

Choose input packet count, output budget, autoregressive or MTP mode, and a verification scenario. Play/pause, step backward/forward, reset, scrub the timeline or run to the final frame. Per-stage playback skips to milestones; every-operation playback exposes each attention, normalization, projection, residual and feed-forward stage. Camera follow can be disabled. Jump-to-stage and Fit graph work on the complete map. Playback is transient; exported JSON preserves model content, layouts and notebook notes.

Prefill supplies all prompt query positions, writes each layer's own K/V state, and uses a causal mask. Decode reuses that prefix and processes one new query token at a time. The display shows each layer's sequence count independently, so state grows as the pass reaches that layer. A selected final output token need not be cached when no subsequent forward pass is required.

The cache strip represents the sequence axis, not a physical allocation map or a table hierarchy. Each layer contains separate logical K and V tensors `[B,Hkv,R,Dh]` plus absolute position IDs; all-layer state is indexed by `[L,B,Hkv,R,Dh]`. R follows full, sparse or sliding-window retention. Current-call working keys can be larger than final retained state. Logical byte totals exclude allocator overhead, padding, sharding and quantization metadata.

## MTP proposal and verification

Two sequential MTP depths are an illustrative configuration inspired by the [DeepSeek-V3 report, §2.2](https://arxiv.org/html/2412.19437v2#S2.SS2). They are not a claim that every MTP model has two modules, uses this dense attention profile, or predicts independent future tokens.

1. The base decoder supplies the first next-token proposal and final hidden state.
2. Each MTP depth separately normalizes the previous-depth representation and shifted proposed-token embedding, concatenates/project them, runs an additional causal decoder module, and applies the shared vocabulary projection to propose another token.
3. The main decoder verifies the proposal block in a causal batch with tentative KV writes. Existing prefix logits verify the first candidate; batch row `j` predicts the following candidate. Later candidates cannot influence earlier positions through the mask.
4. Greedy verification accepts the longest matching prefix. At the first mismatch, discard that candidate and all later candidates. Roll back every main decoder layer's tentative K/V sequence axis to the accepted prefix.
5. Commit the matching prefix and a main-model correction at a mismatch, or a bonus token after a fully matching proposal block, subject to the remaining output budget. Process the uncached correction/bonus on the next iteration; reuse accepted K/V state.

The **mixed**, **all match**, and **first MTP draft mismatches** scenarios supply scripted reference choices to make these branches visible. Accepted packets use a check mark; rejected/discarded packets use a cross and pattern/opacity, alongside cyan/violet/amber colors. Nothing is claimed about trained-head acceptance rates or speedups. The feedback arrow stands for the next iteration's token IDs; generated text is not re-tokenized.

This is greedy prefix verification. [Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html) also describes stochastic speculative decoding, whose probability-ratio acceptance and residual distribution are different. The simulator does not present its simple equality check as that sampling algorithm.

## Keeping numerical execution possible

`src/model/simulation.ts` contains the renderer-neutral `ExecutionAdapter`, `SimulationTrace` and `SimulationFrame` contracts. Graph instance IDs connect execution to model components. Frame snapshots record phases, active components, packet state, query/context counts, per-layer KV lengths, tentative state and acceptance outcomes. Earlier frames are immutable snapshots, so scrubbing never reruns or changes a decision. `useSimulation.ts` owns playback; the React Flow renderer consumes active/completed IDs and does not compute inference.

Trace provenance distinguishes `illustrative`, `numerical`, and `measured` execution and can identify a model revision. A separate [numerical laboratory](numerical-laboratory.md) now implements local reference execution with seeded untrained parameters, actual tensors, scalar provenance and autoregressive decoding. Its settings and geometry do not replace the graphical simulation or main model. A trained/measured adapter must supply its actual tokenizer/weights/configuration provenance, real choices and cache behavior, and extend the trace explicitly for its tensor payloads, sampling distributions and physical storage details. Keep large tensors in a separate trace artifact. Do not fabricate measurements or silently reinterpret an illustrative run as inference. Any future network inference needs a user-requested design and explicit visible controls.

Cache policies and split head playback are described in [attention and cache](attention-and-cache.md). Trial writes pin the retained checkpoint until verification commits; rollback never assumes that an overwritten ring-buffer entry can be recovered.
