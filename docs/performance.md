# Performance and rendering

Large maps use the same authored model, stable IDs and validated orthogonal routes at every zoom. Camera movement does not compile model instances, rebuild component data, reroute links or write the workspace on each frame. Committed camera changes still save normally.

## Measured result

On 21 October 2026, the production build was measured in an isolated Chrome 155 context on Windows with an NVIDIA RTX A5000 Laptop GPU. The workload is the default **Whole model · split heads** view: 420 components, 610 links, a 1440 × 960 viewport and 120 middle-button pan moves per run. Three runs were recorded before and after the changes. The table reports the median across runs; frame percentiles are computed within each run.

| Metric                             |    Before |     After |
| ---------------------------------- | --------: | --------: |
| Median frame interval              |   66.7 ms |   16.9 ms |
| 95th-percentile frame interval     |   83.5 ms |   50.3 ms |
| Main-thread JavaScript per gesture |  5,244 ms |    647 ms |
| Main-thread task time per gesture  |  8,014 ms |  1,579 ms |
| Long tasks per gesture             |       118 |         0 |
| Components / links present         | 420 / 610 | 420 / 610 |

These are local measurements, not a fixed frame-rate guarantee. Browser input-injection pacing is included; the first run also records a CPU profile. GPU availability, browser, display and hardware affect the result. The [raw measurements and baseline commit](benchmarks/2026-10-21.json) preserve all runs, including the slower frame tail. Run benchmarks separately from tests and other heavy work.

Start a production preview in one terminal:

```powershell
npm run build
npm run preview -- --port 5201 --strictPort
```

Run the benchmark in another:

```powershell
npm run benchmark -- --label current --profile
```

The benchmark creates a separate headless browser context with the authored example and writes JSON plus an optional CPU profile under ignored `verification.local/`. It leaves the user's saved workspace and browser tabs alone. `--svg` exercises the SVG fallback, `--detail` measures the close inspection view, and `--url http://127.0.0.1:5199/` targets another local server. Production is the useful comparison; development adds instrumentation overhead.

## Rendering and GPU work

`GraphCanvas` does not subscribe to camera movement. A small zoom control observes only the rounded zoom percentage. Node/edge objects, handlers, port signatures and stage bounds retain their identity when their inputs do not change. A dragged node no longer causes a quadratic search through all other node states. Collision-safe drops test candidate positions in distance order and stop at the nearest free one.

This follows [React Flow's performance guidance](https://reactflow.dev/learn/advanced-use/performance) on narrow store subscriptions and stable component, object and callback references.

The decorative grid adapts in powers of two to keep its screen spacing useful. A fixed world-space grid at whole-model zoom produced an extremely dense subpixel pattern; rasterizing it was a significant part of the measured slowdown.

For maps with at least 80 components below 18% zoom, `OverviewCanvas` sends a drawing-only scene to a dedicated worker. A transferred **OffscreenCanvas** draws the same component boxes, stage outlines, routes, labels' footprints, arrow direction and active execution packets. Tiny explanatory text returns automatically in the detailed SVG view. The bitmap is bounded by the viewport and device-pixel ratio, capped at 2; it is never a texture the size of the whole model. Path2D instances are cached per scene. Camera messages are coalesced to animation frames, and ordinary panning sends only the camera transform.

The browser can use its GPU for canvas rasterization and compositing. This uses the standard [OffscreenCanvas API](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas) and [dedicated-worker animation frames](https://developer.mozilla.org/en-US/docs/Web/API/DedicatedWorkerGlobalScope/requestAnimationFrame); it does not require WebGPU or change numerical arithmetic. In the measured browser, accelerated canvas, GPU compositing and rasterization were enabled. Forcing compositor promotion of the entire world-sized SVG was tested and made performance worse, so no blanket `will-change` hints are applied. [MDN describes their memory and rendering costs](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Properties/will-change).

React Flow retains its outer node and edge controls, transparent link hit targets, keyboard interaction and measured ports. The distant canvas has no independent model or hit-test geometry. Zooming in restores text, shapes and operation buttons immediately. Dark/light palettes come from the existing CSS variables. Geometry changes invalidate old worker acknowledgements before paint, so stale canvas paths cannot be restored over a moved component. Missing canvas capabilities, worker startup failures or context loss retain the complete SVG renderer.

## Workers and numerical inspection

ELK and libavoid already execute outside the UI thread. Their protocol now sends a graph's authored content once, then sends positions for subsequent routes. Drag updates are coalesced, briefly debounced while moving, and superseded replies are discarded. A validated layout result is used directly instead of immediately routing the entire model again. Existing handles stay mounted while new paths are pending; obstructed stale paths remain withheld.

The untrained numerical run and its pinned baseline use independent, short-lived workers alongside layout and drawing. Rapid setting edits cancel superseded workers. The UI labels pending work and retains the previous completed capture until the new one arrives. Controls describe the draft settings; graph geometry and tensor inspection describe the completed trace. Head-topology changes reject late geometry writes from the previous configuration.

Completed reference traces transfer their typed-array buffers to the UI. This preserves exact Float32/Int32 values without copying their payloads. [Transfer ownership detaches the sender's buffers](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects), so a completed worker does not retain a reusable trace. Export uses a separate serializer worker with a cloned snapshot: UI-owned buffers are never transferred or detached. The microscope remains readable while JSON is prepared. Trained checkpoint inference continues to use its existing local worker and pinned provenance.

The numerical kernels, scalar accumulation order, captured arrays, export contract and provenance are unchanged. GPU numerical kernels would need their own execution adapter, arithmetic description and accuracy tests. Accelerated drawing does not imply GPU inference.

Tensor colour scaling is a single streaming reduction, avoiding boxed full-array copies and JavaScript argument-count limits. Compatibility checks, scalar proofs and parameter inventories are memoized against their actual inputs. Tooltip lifecycle observers watch only the active control's ancestor chain, rather than every DOM change in the graph.

## Verification boundaries

Unit tests cover semantic cache invalidation/preservation, stable node and handle identities, million-value tensor scaling, worker message reuse and queued-job cancellation. Browser checks compare every overview path and ID with its SVG counterpart, verify no routing/scene recompilation during panning, exercise selection, zoom transitions, themes, resize bounds and unsupported/failed-worker fallback, and compare exported worker-produced arrays against the numerical reference.

The full routing suite still checks the actual SVG curve geometry against every relevant node and label, in both directions and after dragging/import/export. Routes, renderer projections and caches are derived runtime state. No model schema, authored example, notebook, layout ID or supported extension metadata changes are needed.
