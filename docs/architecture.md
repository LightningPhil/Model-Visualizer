# Software architecture

## Layers of responsibility

| Location                                                 | Responsibility                                                                                                |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `examples/*.model.json`                                  | Authored model semantics, nested graph references, equations, sources, notes, positions and viewports         |
| `src/model/schema.ts`                                    | Runtime schema, reference validation, restricted dimension parser, cache accounting and types                 |
| `src/model/tokenizer.ts`                                 | Tokenizer adapter interface, real demonstration byte-BPE, labelled synthetic attention probabilities          |
| `src/model/whole-model.ts`                               | Deterministic whole-model expansion from authored recipes and source templates                                |
| `src/model/graph-compiler.ts`                            | Runtime semantic instance cache, invalidated by authored content or configuration changes                     |
| `src/model/group-layout.ts`                              | Stage-local layout/routing plus fixed-obstacle inter-stage routing for expanded models                        |
| `src/model/simulation.ts`                                | Renderer-neutral graphical execution trace, causal verification scenarios and KV-state accounting             |
| `src/model/numerics/`                                    | Local numerical reference operations, actual tensors and scalar provenance; independent laboratory sessions   |
| `src/model/checkpoints/`, `examples/*.checkpoint.json`   | Pinned trained manifests, binary/tokenizer validation, local inference worker and separate studies            |
| `src/components/numerical/`                              | Numerical playback, tensor slicing, operation diagrams, head comparisons and experiment notebook              |
| `src/components/SimulationPanel.tsx`, `useSimulation.ts` | Packet/state visualization, playback, timeline, step detail and camera follow                                 |
| `src/model/layout.ts`                                    | ELK arrangements scored using the complete routed diagram                                                     |
| `src/model/routing.ts`, `geometry.ts`                    | Distinct ports, obstacle-aware libavoid routing, label placement, geometry checks and bounds                  |
| `src/model/diagram.worker.ts`, `layout-client.ts`        | Worker protocol, WASM initialization and coalescing of drag requests                                          |
| `src/model/storage.ts`                                   | Transactional import, export and browser-local persistence                                                    |
| `src/components/GraphCanvas.tsx`                         | React Flow rendering, node dragging, edge inspection, layout and viewport controls                            |
| `src/components/overview/`                               | Bounded OffscreenCanvas worker projection for distant views; SVG interaction and detail remain available      |
| `src/model/numerics/trace.worker.ts`                     | Background untrained runs, baseline captures and trace JSON preparation                                       |
| `src/components/Inspector.tsx`                           | Explanations, tensors, equations, primary references and notebook                                             |
| `src/components/Labs.tsx`                                | BPE trace, head sharing/heatmap, dimensioned KV slice, illustrative n-gram addresses and MTP target alignment |
| `src/App.tsx`                                            | Workspace state, navigation, configuration and document lifecycle                                             |

React Flow is the viewport/interaction layer; it does not own the portable document contract. ELK handles graph arrangement; libavoid handles orthogonal routing. Both execute outside the rendering thread. KaTeX renders untrusted equations with trust disabled and bounded expansion. Fonts and the routing WASM are bundled, and the running app makes no inference or font-service requests.

Camera subscriptions are isolated from graph rendering. Large distant maps use a bounded canvas worker backed by the same validated scene; close inspection and fallback use SVG. The decorative grid maintains useful screen spacing at small zooms. Semantic instances, unchanged node data and measured ports survive ordinary camera/layout edits. Reference runs, baselines and trace serialization also use workers, while numerical arithmetic and provenance remain unchanged. See [performance measurements, rendering boundaries and reproduction commands](performance.md).

## Document lifecycle

1. Read a valid locally saved v1 document, otherwise load the authored teaching document.
2. Resolve tensor expressions against model configuration and the current query/prefix lengths.
3. Render the active graph. Graph navigation restores its independent layout and viewport.
4. Persist committed node drags, layouts, viewports, notes, prompt and theme after a short debounce.
5. Export a complete JSON document. Import is size-bounded, parsed and semantically validated before replacing state.

Importing a document remounts graph state even if the document and graph IDs match the previous document. Layout completions are scoped to the graph that requested them and invalidated on unmount, preventing late results from contaminating a different graph.

## Explanatory help

The educational workflow also uses renderer-independent `src/model/study/` modules for a small trainable network, controlled investigations, run comparisons and a separate versioned investigation artifact. The shared reader can dock nonmodally within its originating numerical dialog or alongside the architecture canvas; selection following is opt-in and actual numerical cell context updates independently. Investigation records never become a second model definition. Their recipes reopen existing supported execution paths, with explicit checkpoint loading/inference. See [investigations, training and replay](help-wiki.md#investigations-training-and-replay).

`src/help/catalog.ts` owns detailed control explanations, enum alternatives and named primary-source attributions. `src/components/help/` resolves dynamic controls and manages one shared hover/focus card. Native `title` hints are replaced so they cannot bypass the configured delay. Mathematical references remain linked in the model wiki and inspector; tooltip content distinguishes architectural studies, scripted simulation, untrained reference arrays and learned checkpoint inference.

`src/help/wiki/` owns the renderer-independent educational library: concept guides, primary references, learning paths, semantic-kind associations, glossary, explicit worked-example arithmetic and search. `buildWikiEntries()` derives document entries directly from the current document and active compiled graph, preserving authored descriptions, equations, tensor axes, notes and references without writing a duplicate model. Numerical sessions pass their actual configuration and scope canvas navigation to their own runtime graph. The shared `WikiProvider` opens a lazily loaded reader above the current dialog. Back/forward navigation is session UI; bookmarks, recent reading and positions use a separate versioned device preference. Reading never writes model content, notebooks or layouts. Explicit canvas return selects an existing component/connection; explicit example actions open a supported foundations study or seek in the current run. Opening help pauses playback. See [the wiki’s content, evidence boundaries and extension workflow](help-wiki.md).

Each concept also has authored introductory and technical perspectives in renderer-independent content modules. The reader chooses sections without rewriting imported explanations, and tracks reading mode with device-library metadata. Twenty local worked-example types use explicit teaching calculations; their controls and pinned comparisons live in the reader and never alter model configuration or captured arrays. Search indexes both perspectives.

The default delay is 700 ms of continuous hover over one control, including its text/icon children. Crossing to another control starts a new timer. Leaving, clicking, editing or pressing Escape cancels pending help; scrolling starts any new hover from zero at the new position. Dragging and touch do not trigger automatic hover help. Keyboard focus provides the same explanations without taking focus away. Cards remain readable while hovered, fit the viewport and use non-focusable content with `role="tooltip"` and a temporary `aria-describedby` association. Escape dismisses visible help before a containing dialog. The behavior follows the [WAI tooltip pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tooltip/); native manual popovers keep cards above dialogs without clipping them to the dialog's scrolling area.

The question-mark control opens timing preferences from the main toolbar, model builder and numerical/trained workbenches. Delay (0–5 seconds) and enable/disable settings save separately under `model-visualizer.help-preferences.v1`. They are device preferences, not model content or another document format. Additional preference metadata is retained and future versions are not overwritten. Closed views, replaced controls, blur and unmount cannot leave a delayed ghost card behind. Browser tests cover exact timing, cancellation, preference persistence, nested dialogs, keyboard operation, enum details and light/narrow placement.

## Graph hierarchy and branching

Each graph is a flat set of nodes and directed connections. A node's `childGraph` navigates to an independent detailed graph; `repeat` describes repeated blocks without drawing hundreds of copies. Each decoder layer has separate conceptual parameters/state. The selected layer is explicit in block/attention/feed-forward views and connected to the cache slice selector.

The opt-in `model-visualizer.expansion` recipe creates an additional whole-model canvas from these same templates. It instantiates every main layer, explicit query-head results and per-KV-head state views, without storing duplicate generated model content. Stable instance IDs own saved layout positions; semantic validation expands the recipe to validate them. Compatible older workspaces are enriched without replacing notes or study layouts. See the model-format contract and [simulation design](simulation.md).

Connections may represent flow, residual additions, state access or auxiliary paths. All links participate in placement and routing, including optional branches. Cycles can be described, but time-step/state semantics must be explained explicitly; this is not a recurrent execution engine.

Default overview positions are hand composed. Auto-layout deliberately replaces positions and side overrides for the active graph, then fits the complete result, including labels and exterior routes. The undo button restores the previous positions, direction, side overrides and viewport (up to ten automatic layouts while the graph stays open). Manual dragging reroutes connections without rearranging other nodes; dropping onto another node moves the dragged node to nearby free space.

## Placement and routing

1. ELK reserves space for every node, edge label and branch. Three deterministic candidates vary spacing and node placement. The completed diagrams are scored by crossings, bends, path length and area. A rejected candidate does not overwrite the user's layout.
2. Every incident connection receives a separate port, ordered by the location of the connected component. Imported side overrides are honored until Auto-layout is requested.
3. libavoid performs joint orthogonal routing around fixed-size component obstacles. Labels are placed on long straight segments, away from nodes, other labels and bends. In cramped manual layouts, an annotation can move into a free gutter with one perpendicular leader, leaving the connection straight.
4. Labels become routing obstacles in a second transaction. An inline label has opposite entry/exit ports, so its connection runs straight through it. Other connections go around the label. The renderer rounds only right-angle corners; labels never create spline control points or a midpoint zigzag.
5. Validate endpoint attachment, orthogonality, reversals, bends at labels, node/label collisions and annotation leaders before displaying a result. Overlapping or impossible imported geometry produces a visible request for more room rather than an obstructed route. Auto-layout remains available, and unsuccessful layout preserves the current arrangement.

The worker coalesces drag updates and discards superseded results. Stale edges are withheld while moved geometry is being routed. React Flow handles are measured before their edges appear. The viewport fit uses route and label bounds as well as node bounds, with padding for canvas controls.

Expanded models are arranged and routed by stage first, then packed into adaptive serpentine bands using each stage's complete routed footprint. Band lengths follow the actual stage areas; short sections never inherit a decoder layer's maximum-sized grid cell. Translation offsets are snapped to whole canvas pixels to avoid translation-sensitive degeneracies in the native orthogonal router. Routing gutters widen only if the complete arrangement fails validation. A shared global port allocation is used for both local and inter-stage paths. The inter-stage router treats internal labels and annotation leaders as fixed obstacles and excludes fixed internal paths from label placement. The complete merged result still passes the same collision validator. This bounds layout complexity without hiding components or connections. Existing saved positions remain unchanged until Auto-layout is requested; undo restores the prior arrangement. Whole-map fit supports zooms down to `0.001`; stage navigation and search bring individual operations back into view. Node/port measurement is batched once per graph update, rather than forcing a render per component.

Only authored positions, direction and viewport are persisted; routes and ports are deterministic derived geometry. No document-version change or duplicate routing data is required. Keep the fixed node/label sizes shared between CSS, geometry and tests when changing the design.

The router uses [libavoid-js](https://github.com/Aksem/libavoid-js), the JavaScript/WASM port of [Adaptagrams libavoid](https://www.adaptagrams.org/documentation/libavoid.html), distributed under LGPL-2.1-or-later. The build ships its license, source/build links and a replaceable WASM asset; the application does not modify upstream source. See `public/THIRD_PARTY_NOTICES.txt`. ELK's [layered algorithm reference](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html) documents its placement, ports and edge-label options.

## Shapes are descriptions, not execution

Expressions use a small arithmetic parser, never `eval`. The v1 teaching profile expects positive integer axes, even RoPE feature pairs, `D = Hq × Dh` and conventional MHA/GQA/MQA sharing. It exposes named tensors, rather than inferred port-level tensor typing. Import checks identifiers and expressions, but does **not** prove that arbitrary imported equations or matrix contractions are mathematically correct.

Prefill uses `T = S = prompt token count`. Decode describes a hypothetical next token: `T = 1`, `S = prompt token count + 1`. It does not sample or fabricate that token. Empty prompts use an explicitly labelled one-position reference shape. Attention labs are synthetic, capped at eight positions and labelled separately from full graph dimensions.

The full-prefix logical KV size is `2 × L × B × Hkv × S × Dh × scalarBytes`, with `B = 1` in this version. The cache lab can vary S independently to study scaling. It excludes allocator overhead, padding, sharding and quantization metadata. State width in bytes is not a numerical quantization implementation.

The teaching-model builder exposes D explicitly. Changing Hq or Dh repartitions the same D; changing D preserves Hq when possible and otherwise selects the nearest compatible partition. Only integer head counts with even Dh, valid MHA/GQA/MQA sharing and an expanded graph within the view budget are offered. GQA's KV choices are proper divisors of Hq; MHA and MQA use their required endpoints. Edits are validated atomically before updating the document, and topology changes continue to archive/restore independent user layouts. Reducing context clamps its retention window/stride to capacity. These are the supported profile's constraints, not a universal rule for every transformer architecture. See the primary [GQA](https://arxiv.org/abs/2305.13245) and [RoPE](https://arxiv.org/abs/2104.09864) references and the [research conventions](research.md).

## Extension boundaries

**New model content:** add a model JSON document, preserve its source references and validate it. Unknown component `kind` values receive a generic renderer. Node categories and existing symbolic axes form the v1 vocabulary.

**Tokenizer:** implement `TokenizerAdapter` to return IDs, byte spans and optional merge steps. Add an explicit adapter registry/capability description before supporting multiple executable tokenizers. Imported unknown adapter IDs currently retain graph inspection using reference shapes and disable token labs. Do not silently fall back to the demo tokenizer.

**Attention architecture:** MHA/GQA/MQA use the same dense logical state representation. MLA requires a new cache representation and configuration profile, not relabelled GQA. Full, fixed-anchor sparse and sliding-window policies now have explicit masks, absolute positions and retention accounting in `attention.ts`. Other sparse/linear patterns still require their own state semantics. See [attention and cache](attention-and-cache.md).

**Actual traces:** the generic execution adapter now supports graphical and local numerical reference traces with explicit provenance and stable component IDs. The separate tensor laboratory executes a small, seeded **untrained** foundations profile, records actual arrays and absolute position maps, and explains individual cells through operation recipes. It reuses authored architecture templates and owns separate experiment geometry/settings in a versioned extension. It does not execute arbitrary imported models. External measured-trace import still needs a validated schema keyed by model identity, layer, head, step and tensor name, with dtype, positions and sampling/downsampling metadata. Keep large arrays in separate trace artifacts. See [the numerical laboratory](numerical-laboratory.md).

**MoE:** add router, scores, top-k selection, per-expert FFNs, dispatch/combine and optional shared experts. Distinguish total parameters from activated parameters. A static branching diagram alone is insufficient to demonstrate token-specific routing.

**Editing:** an agent can author arbitrary graph bifurcations in JSON now. Layout undo is implemented. A general visual graph editor, tensor typing between ports, general editing history and comparison of model versions are future work.

## Local trained checkpoint

The separate **Local trained model** entry loads bundled Stories260K weights and its matching tokenizer only on request. A worker verifies pinned SHA-256 hashes and runs the numerical primitives with learned parameters. Its actual configuration drives all five expanded layers and eight query heads. The shared workbench accepts this trace without substituting seeded tensors. Tied classifier weights are identified explicitly. Decoding, generation progress, separate geometry/notebook storage and trace export retain checkpoint and arithmetic provenance. This supports one validated checkpoint profile, not arbitrary architectures or model services. See [local trained-model inspection](local-trained-model.md).

## Verification

Core tests exercise format integrity, rejected references/expressions, all authored LaTeX, Unicode BPE round-trips, causal row normalization and KV byte scaling. Layout tests check every example in both directions, deterministic restoration, narrow label gutters, obstructed paths, fan-out/fan-in, residuals, cycles, self-loops and collision-free dragging. Browser tests sample the actual rounded SVG curves against rendered node/label boxes for every graph in both directions, and check full-diagram fitting and layout undo. Existing interaction, import/export, theme and responsive checks remain. Visual captures supplement these behavioral checks; they are not pixel snapshots.

## Split heads and cache policy

`heads.ts` compiles an opt-in head recipe from the authored attention template, exposing independent Q/score/mask/softmax/value paths with shared K/V groups. The whole-model split variant references the existing base recipe and saves its layout separately. `attention.ts` owns head mapping, visibility, retained absolute positions, working/persistent counts and synthetic masked probabilities. The graphical trace records per-layer position maps and evictions; speculative writes pin their checkpoint until verification commits. See [the semantics and primary sources](attention-and-cache.md).
