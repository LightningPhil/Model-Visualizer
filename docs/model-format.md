# Model document v1.0

Start from `examples/transformer-foundations.model.json` or an exported workspace. Its relative `$schema` points to `examples/model.schema.json`, generated from the runtime Zod contract. Schema validation alone cannot check references or attention-sharing invariants: use `parseDocument()` too.

## Top-level fields

| Field                       | Meaning                                                           |
| --------------------------- | ----------------------------------------------------------------- |
| `schemaVersion`             | Exactly `"1.0"`; reject unsupported versions rather than guessing |
| `id`, `name`, `description` | Stable machine identifier and human-facing model identity         |
| `documentation`             | Plain-text overview, conventions, caveats and provenance          |
| `config`                    | The supported dense decoder teaching profile                      |
| `tokenizer`                 | Executable adapter ID plus an honest description                  |
| `sources`                   | Stable source IDs, primary URLs, titles and relevance notes       |
| `graphs`                    | One or more independently laid-out graph views                    |
| `ui`                        | Active graph, theme, prompt and notebook                          |
| `extensions`                | Optional namespaced JSON metadata, preserved on round-trip        |

IDs use letters, numbers, dots, underscores and hyphens. Graph IDs are unique in the document; node and edge IDs are unique within each graph. All connections, child graphs, layout positions and citations must resolve. Source links must be HTTP(S).

## Components

Every component has `id`, `kind`, `title`, `subtitle`, `category`, `description`, `tensors`, `notes` and `sourceIds`. `equation` is optional LaTeX. `childGraph` opens a detailed graph; `repeat` is a supported dimension expression, such as `"L"`.

`kind` is an open string so new visual components can use the generic renderer. `category` is `data`, `compute`, `memory` or `output`. Category colour is reinforced by text and iconography. Put custom data in a namespaced `extensions` object, not arbitrary new standard fields.

A tensor contains a human-readable `name`, `role` (`input`, `output`, `weight` or `state`), ordered `shape` and an optional description. Each shape axis is an integer literal or restricted dimension expression.

```json
{
  "name": "Wq",
  "role": "weight",
  "shape": ["D", "Hq*Dh"],
  "description": "Query projection, using row-vector convention."
}
```

Tensor axis order is explicit and significant. Use row-vector notation for the bundled model's linear maps: `[B,T,D] × [D,F] → [B,T,F]`. Label transposes, reshapes, gathers, broadcasting, head sharing and contraction axes in explanations. Shape equality alone does not establish compatible semantics.

## Connections

Each connection describes `source`, `target`, `label`, `operation`, `description`, a carried/result `shape`, and optional LaTeX. Its `kind` is `flow`, `residual`, `state` or `auxiliary`. Source/target side overrides can be `top`, `right`, `bottom` or `left`. Auto-layout clears overrides to follow the selected direction.

Connections are explanations, not an executable autodifferentiation graph. Specify whether a shape represents the carried input or the operation result. For a complicated contraction, describe both operands in the equation or add explicit operation nodes with their own input/output tensors.

## Symbolic dimensions

| Symbol        | Meaning                                                                     |
| ------------- | --------------------------------------------------------------------------- |
| `B`           | Batch size; fixed to 1 in the current workbench                             |
| `T`           | Current query positions: prompt length for prefill, 1 for decode            |
| `S`           | Visible key positions: prompt length for prefill, prefix + 1 for decode     |
| `D`           | Hidden/residual width                                                       |
| `L`           | Number of decoder layers                                                    |
| `Hq`          | Query heads                                                                 |
| `Hkv`         | KV heads                                                                    |
| `Dh`          | Per-head feature width                                                      |
| `F`           | Feed-forward intermediate width                                             |
| `V`           | Vocabulary projection width                                                 |
| `C`           | Context capacity                                                            |
| `N`, `M`, `E` | Teaching memory constants: maximum n-gram order 4, 4096 slots, 128 features |

The parser accepts integer literals, known symbols, parentheses and `+ - * /`. Results must be positive safe integers. No functions, property access, JavaScript or implicit multiplication. Expressions are validated at a reference sequence length of 12; authors must also check their intended runtime lengths. Avoid expressions that become zero/negative for short prompts. If an imported expression is invalid at the current runtime length, the UI displays `?` for that axis rather than crashing.

MHA requires `Hkv = Hq`; MQA requires `Hkv = 1`; the GQA option uses an intermediate divisor `1 < Hkv < Hq`. The teaching profile uses even `Dh` and `D = Hq × Dh`. These constraints are intentionally scoped: a future model needing different projection widths, latent state or additional symbols requires a new profile/version, not inaccurate parameters squeezed into these fields.

## Layout and saving

```json
{
  "direction": "RIGHT",
  "positions": {
    "q": { "x": 80, "y": 120 },
    "scores": { "x": 420, "y": 120 }
  },
  "viewport": { "x": 24, "y": 50, "zoom": 0.85 }
}
```

Positions are canvas-space coordinates; viewport translation and zoom are independent. Direction is `RIGHT` or `DOWN`. Missing positions trigger auto-layout for the graph. Keep stable node IDs to retain user layouts across content edits; add a migration when IDs change. A file's incomplete layout is recomputed as a whole when first opened.

Routes, annotation bounds and per-connection ports are derived from the graph and saved positions; do not author or save a second set of bend points. Loading a complete layout preserves its node positions and reroutes the links. Auto-layout considers all connections (including auxiliary/state/residual paths), clears side overrides, and fits nodes, labels and outer tracks together. Its undo button restores the previous arrangement and viewport while the graph remains open. Leave space between imported components; impossible or overlapping geometry is reported visibly and can be repaired with Auto-layout.

Model imports are limited to 3 MiB, 30 graphs, 250 nodes per graph and 800 connections per graph. The format is for architecture and documentation. Large activation tensors belong in a separate trace artifact.

### Whole-model expansion recipe

The authored `whole-model` graph contains one `model-expansion` component with the reserved `model-visualizer.expansion` extension (recipe version 1). The recipe references input, block, attention, feed-forward, output, optional memory and generation study graphs and their entry/exit IDs. `src/model/whole-model.ts` instantiates those source components with stable prefixes (`tokens.`, `layer-0.attn.`, `layer-0.swiglu.`, `output.`, `generation.`), expands all main decoder layers, and adds explicit query-head results and KV-head views. The current recipe is a composition of the foundations profile; the role IDs such as `attn`, `weighted`, `concat`, `cachek`, `cachev`, `ids`, `context`, `hidden` and `sample` have defined meanings in that profile. A different architecture needs an explicit recipe/compiler extension.

This graph's saved `layout.positions` refer to its **expanded instance IDs**, which semantic validation checks against the compiler output. The source document retains only templates, the recipe, and saved positions; it never stores a second copy of the generated graph. Expanded views have separate limits of 4,096 components and 12,000 connections. The ordinary authored-graph limits remain unchanged. Unknown kinds continue to render generically unless they explicitly opt into the reserved expansion extension.

Very large maps can save viewport zooms down to `0.001`. This widens the accepted v1 viewport range without changing the meaning of existing files. Existing compatible workspaces gain the new authored generation graph and recipe while preserving their IDs, notes, metadata and saved study positions. Changes to layer/head counts keep surviving instance positions and remove obsolete IDs; an incomplete map is then laid out as a whole.

Unknown standard fields are rejected rather than silently dropped. Put additional metadata in `extensions`. This protects agent-authored content from disappearing during a round-trip.

## Agent authoring checklist

1. Read this contract and the source model before editing. Preserve user notes and graph positions.
2. Identify the exact architecture and revision. Prefer the author's paper, official config and implementation. Record discrepancies rather than inventing details.
3. Use accurate model-specific names and dimensions. Mark inferred or illustrative content explicitly.
4. Describe every branch, matrix product, element-wise operation, gather, reshape and state update. Add child graphs for detail.
5. Include citations close to the affected components via source IDs. Use LaTeX for equations; JSON encodes each backslash as `\\`.
6. Keep training objectives separate from inference. Distinguish static parameters, transient activations and persistent KV state.
7. Run semantic validation, render every equation and check dimensions across relevant prefill/decode lengths. Inspect auto-layout and both themes.
8. Export/import the file to verify that all explanations, references, layouts and metadata survive.

When extending the contract, update the runtime schema, generated JSON Schema, parser, examples, docs and tests together. Never quietly reinterpret existing fields. Version breaking changes and provide a deterministic migration that preserves layouts and notes.
