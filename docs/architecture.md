# Software architecture

## Layers of responsibility

| Location                         | Responsibility                                                                                                |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `examples/*.model.json`          | Authored model semantics, nested graph references, equations, sources, notes, positions and viewports         |
| `src/model/schema.ts`            | Runtime schema, reference validation, restricted dimension parser, cache accounting and types                 |
| `src/model/tokenizer.ts`         | Tokenizer adapter interface, real demonstration byte-BPE, labelled synthetic attention probabilities          |
| `src/model/layout.ts`            | Lazy-loaded ELK layout with known node bounds                                                                 |
| `src/model/storage.ts`           | Transactional import, export and browser-local persistence                                                    |
| `src/components/GraphCanvas.tsx` | React Flow rendering, node dragging, edge inspection, layout and viewport controls                            |
| `src/components/Inspector.tsx`   | Explanations, tensors, equations, primary references and notebook                                             |
| `src/components/Labs.tsx`        | BPE trace, head sharing/heatmap, dimensioned KV slice, illustrative n-gram addresses and MTP target alignment |
| `src/App.tsx`                    | Workspace state, navigation, configuration and document lifecycle                                             |

React Flow is the viewport/interaction layer; it does not own the portable document contract. ELK handles graph arrangement and loads only when needed. KaTeX renders untrusted equations with trust disabled and bounded expansion. Fonts are bundled, and the running app makes no inference or font-service requests.

## Document lifecycle

1. Read a valid locally saved v1 document, otherwise load the authored teaching document.
2. Resolve tensor expressions against model configuration and the current query/prefix lengths.
3. Render the active graph. Graph navigation restores its independent layout and viewport.
4. Persist committed node drags, layouts, viewports, notes, prompt and theme after a short debounce.
5. Export a complete JSON document. Import is size-bounded, parsed and semantically validated before replacing state.

Importing a document remounts graph state even if the document and graph IDs match the previous document. Layout completions are scoped to the graph that requested them and invalidated on unmount, preventing late results from contaminating a different graph.

## Graph hierarchy and branching

Each graph is a flat set of nodes and directed connections. A node's `childGraph` navigates to an independent detailed graph; `repeat` describes repeated blocks without drawing hundreds of copies. Each decoder layer has separate conceptual parameters/state. The selected layer is explicit in block/attention/feed-forward views and connected to the cache slice selector.

Connections may represent flow, residual additions, state access or auxiliary paths. ELK can arrange branches and residual DAGs. Auxiliary links are excluded from the ranking calculation so optional studies do not force the main compute path into misleading extra stages. Edges remain rendered and inspectable. Cycles can be described, but time-step/state semantics must be explained explicitly; this is not a recurrent execution engine.

Default overview positions are hand composed. Auto-layout deliberately replaces positions and side overrides for the active graph, then fits the result. Manual dragging does not automatically re-layout the graph.

## Shapes are descriptions, not execution

Expressions use a small arithmetic parser, never `eval`. The v1 teaching profile expects positive integer axes, even RoPE feature pairs, `D = Hq × Dh` and conventional MHA/GQA/MQA sharing. It exposes named tensors, rather than inferred port-level tensor typing. Import checks identifiers and expressions, but does **not** prove that arbitrary imported equations or matrix contractions are mathematically correct.

Prefill uses `T = S = prompt token count`. Decode describes a hypothetical next token: `T = 1`, `S = prompt token count + 1`. It does not sample or fabricate that token. Empty prompts use an explicitly labelled one-position reference shape. Attention labs are synthetic, capped at eight positions and labelled separately from full graph dimensions.

The logical dense KV size is `2 × L × B × Hkv × S × Dh × scalarBytes`, with `B = 1` in this version. The cache lab can vary S independently to study scaling. It excludes allocator overhead, padding, sharding and quantization metadata. State width in bytes is not a numerical quantization implementation.

## Extension boundaries

**New model content:** add a model JSON document, preserve its source references and validate it. Unknown component `kind` values receive a generic renderer. Node categories and existing symbolic axes form the v1 vocabulary.

**Tokenizer:** implement `TokenizerAdapter` to return IDs, byte spans and optional merge steps. Add an explicit adapter registry/capability description before supporting multiple executable tokenizers. Imported unknown adapter IDs currently retain graph inspection using reference shapes and disable token labs. Do not silently fall back to the demo tokenizer.

**Attention architecture:** MHA/GQA/MQA use the same dense logical state representation. MLA requires a new cache representation and configuration profile, not relabelled GQA. Windowed/sparse/linear attention similarly needs explicit mask/state semantics. Version the document contract when necessary.

**Actual traces:** introduce an independent trace schema keyed by document/model hash, graph/component ID, layer, head, step and tensor name. Include shape, dtype, positions, provenance and sampling/downsampling metadata. Keep large arrays out of layout files. Never replace synthetic examples silently with inferred data.

**MoE:** add router, scores, top-k selection, per-expert FFNs, dispatch/combine and optional shared experts. Distinguish total parameters from activated parameters. A static branching diagram alone is insufficient to demonstrate token-specific routing.

**Editing:** an agent can author arbitrary graph bifurcations in JSON now. A general visual graph editor, port-aware validation, undo history and comparison of model versions are future work.

## Verification

Core tests exercise format integrity, rejected references/expressions, all authored LaTeX, Unicode BPE round-trips, causal row normalization, KV byte scaling and overlap-free ELK placement in both directions. Browser tests cover navigation, edge mathematics, head sharing, prefill/decode, dragging and restoration, import/export, failed imports, search, labs and narrow screens. Visual captures supplement these behavioral checks; they are not pixel snapshots.
