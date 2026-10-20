# Model Visualizer

An interactive atlas of language-model architecture, built in TypeScript. Follow a prompt through tokenization, transformer blocks, attention, KV state and output decoding. Inspect the mathematics on a component **or a connection**.

![The architecture workbench in dark mode](docs/images/workbench-dark.png)

## Run locally

Requires Node.js **24 or newer** and npm. Dependencies are pinned in `package-lock.json`.

```sh
npm ci
npm run dev
```

Open the Local URL printed by Vite. The app runs entirely in the browser, with bundled fonts, equations and research references. There are no API keys, accounts, model downloads or inference services.

```sh
npm run build       # TypeScript check and production build
npm test            # model contract, tokenizer, maths and layout checks
npm run test:e2e    # Chrome browser tests; needs an installed Chrome
npm run schema     # regenerate the distributable JSON Schema
```

The browser tests launch their own Vite server on port 5200. Set `PLAYWRIGHT_PREVIEW=1` to test the production build instead; build it first. `npm run preview` serves that build for manual inspection. CI installs Chromium and checks routing against both development and production servers; see the workflow.

## Explore

- **Nine graph views:** architecture overview, tokenizer, transformer block, attention, KV cache, feed-forward, output, n-gram memory and MTP.
- **60 components:** click to inspect tensor dimensions, rendered LaTeX, explanations and primary sources. Click an edge label to inspect its operation.
- **Real teaching BPE:** edit the prompt; see UTF-8 bytes, token IDs and every ranked merge. Unicode round-trips losslessly.
- **Attention:** switch MHA/GQA/MQA; inspect query-to-KV-head sharing and a causal heatmap. Prefill and decode use different query and key lengths.
- **KV cache:** select a layer, head, K/V plane and sequence length. A dimensioned tensor-slice visualization shows the logical axes. Byte counts follow the selected sharing and storage width.
- **Layouts:** drag nodes, pan/zoom, fit, or auto-layout horizontally/vertically. ELK compares complete arrangements; libavoid routes orthogonal connections around nodes and labels, with distinct ports and clean straight runs through annotations. Undo restores the previous layout. Every graph remembers its positions and viewport.
- **Portable workspaces:** automatic browser-local saves, editable notebook, and readable JSON import/export. Invalid imports leave the current workspace intact.
- **Appearance:** dark and light themes, cyan/violet/amber accents, category text and different legend shapes. No red/green status distinction.

Use `/` to search components. Open the **field guide** for explanations, the research library and a model-format introduction. On narrow screens, the menu opens navigation and the inspector button toggles its overlay.

## What this first version represents

The bundled **Transformer foundations** document is a teaching architecture: six distinct pre-norm decoder layers, RoPE, configurable dense MHA/GQA/MQA and SwiGLU. It is **not a named checkpoint**.

The tokenizer executes an explicit, small byte-BPE vocabulary. Tensor dimensions and logical KV bytes are calculated. Attention heatmaps are deterministic synthetic examples. There are **no trained weights or measured activations**, and the app does not generate language. The n-gram path is an illustrative conditional-memory extension; the MTP path illustrates sequential auxiliary training. These branches are labelled optional.

MLA, MoE, sparse/windowed attention, arbitrary tokenizer plugins and real activation traces are researched extension points, not completed inference features. The [research map](docs/research.md) distinguishes their semantics and the work needed to add them.

## Model documents and future development

The authored source of truth is [`examples/transformer-foundations.model.json`](examples/transformer-foundations.model.json). It keeps model semantics, tensor expressions, explanations, sources, notebook and layout in one versioned document. The app is a renderer and explorer of that document.

- [Model format and agent authoring instructions](docs/model-format.md)
- [Research map, component taxonomy and primary references](docs/research.md)
- [Software architecture and extension boundaries](docs/architecture.md)
- [Machine-readable JSON Schema](examples/model.schema.json)
- [Repository instructions for future agents](AGENTS.md)

The format is an architectural description, not a general tensor execution engine. Branches and subgraphs are supported; extending runtime mathematics or state representations requires explicit TypeScript support and appropriate validation. Breaking format changes require a version and migration.

## Saving

The last valid workspace saves under `model-visualizer.workspace.v1` in local storage. **Export model** downloads the full pretty-printed `.model.json` file, including every layout and notebook. **Open** validates a replacement before applying it. Browser-local saves are specific to an origin and device; export files provide portable copies.

The project is developed directly on `main`, without pull requests.
