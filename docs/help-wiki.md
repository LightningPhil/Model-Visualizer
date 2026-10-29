# The connected model wiki

The educational layer connects an operation in the diagram to an explanation of why it exists, what its axes mean, how its calculation works, and what can be concluded from the result. It replaces the short field-guide modal with a searchable local reference library. It does not change model execution or the model-document contract.

## Entry points and reading flow

- The sidebar’s **The model wiki** and the toolbar book icon open learning paths and the topic library.
- **Explain this view** opens the current graph’s authored description and its related concepts.
- The inspector’s **Explain this in the wiki** opens the selected component or connection, with immediate links to deeper guides.
- The simulator’s **Explain this step** uses the current frame’s actual component ID and pauses playback.
- The untrained/trained workbench’s **Explain this** uses the selected element, or the current frame’s component, and pauses playback. Closing help returns to the same workbench.

Concept guides include prerequisites, connected terminology, section navigation, equations, synthetic worked examples, relevant experiments, interpretation limits and primary references. Six learning paths organize the library around a token’s journey, attention arithmetic, generation state and investigation of actual arrays. Reading history supports back/forward navigation with saved reading positions during an open session.

All 49 concepts have two independently authored explanations. **Introduction** has no equations: purpose, an alternative mental picture, a sequence of operations and the picture’s limits. **Technical detail** preserves the complete original guide and adds a derivation or detailed investigation. Examples include normalization’s shared-denominator derivative, softmax sensitivity, relative rotations, cache reuse, and online softmax. These derivations are explanatory mathematics; they do not claim the app implements training or a fused kernel. **Interactive lab**, where available, offers controls and intermediate calculations. Arrow keys, Home and End navigate the explanation tabs. Switching modes preserves separate scroll positions and keeps the current article’s lab controls intact.

Inside the reader, `/` focuses wiki search. Keyboard events stay within the reader, so arrow keys and Space cannot step or restart the underlying numerical playback. Escape closes the wiki and returns to the originating study. The wiki uses explicit navigation rather than recursive hover tooltips on its own explanations.

Search matches titles, aliases, full explanations and tensor names/axes. A collection filter distinguishes editorial guides from model-document entries. Results identify their origin; document entries include their graph and element IDs. The active compiled graph supplies expanded instance IDs, while other document graphs supply their authored content. The wiki does not expand and duplicate every possible generated view just to index it.

**View on canvas** returns to an existing graph and selects the referenced component or connection. Numerical navigation is limited to the current numerical graph: it cannot open an architecture graph inside the runtime or replace a run. Topic browsing and explanations remain available on narrow screens through the **Topics** control.

## Interactive investigation

**Adam**, **AdamW**, and **Muon** extend the learning path with three full guides, their nonmathematical introductions and additional technical investigations. The Adam/AdamW shared lab compares separate scalar states on four prescribed gradients: raw/corrected moments, adaptive subtraction, decoupled shrinkage and resulting parameters. Prescribed gradients isolate update rules and are explicitly not a model-training quality comparison. Coupled L2 enters both moments; AdamW decay does not. Every supplied gradient, including zero, advances the step; skipped/missing gradients are documented as a separate framework policy.

The Muon lab uses a restricted 2×2 family with fixed singular-vector bases shared by the gradient and old momentum. It exposes EMA/lookahead conventions, Frobenius normalization, finite quintic iterations, signed factors/absolute singular magnitudes, matrices, and a learning-rate-scaled candidate delta. An ideal partial-polar target is shown separately; zero directions stay zero. Arithmetic uses binary64 rather than GPU bfloat16. The guide links an immutable author-code revision and states that eligibility, rectangular scaling, decay, precision and auxiliary optimizer groups need explicit recipes. Pure arithmetic lives in `optimizer-experiments.ts`; content and both perspectives in `optimizers.ts`; controls in `WikiOptimizerLab.tsx`. Tests independently compare the spectral recurrence with the full matrix polynomial and check bias correction, zero-gradient history, decay, zero/rank-deficient cases and workspace preservation.

[AdamW comparison](images/wiki-adamw-comparison.png) · [Narrow Muon matrices](images/wiki-muon-mobile.png)

The **How learning changes behaviour** path fills six additional areas: backpropagation/SGD, positional-method comparison, feature representations/superposition, controlled activation interventions, pretraining/SFT/preference objectives, and numerical precision. Every topic has an independently authored introductory explanation, a complete technical guide, a further derivation or investigation, and primary-source references. Existing RMSNorm and expert-routing guides also contain full LayerNorm/dense-versus-routed comparisons.

Seven new lab types provide local synthetic experiments: a scalar regression trajectory with analytic and finite-difference gradients; four positional entry points; a three-feature/two-coordinate mixture with a basis rotation; explicit clean/altered/ablation/patch/replacement arithmetic; demonstration versus reference-relative preference losses; binary32 cancellation plus stable softmax; and top-k affine experts. The existing normalization lab gains a shared-row LayerNorm/RMSNorm comparison. Reading-mode changes preserve controls; the global example reset restarts scratch values.

These are explicit teaching mechanisms, not transformer training, trained-feature discovery, checkpoint patching, an RLHF runtime, GPU FP16/BF16 emulation, or executable MoE in the numerical profile. Only selected toy expert branches are evaluated in the routed calculation; the separately labelled dense reference is its own calculation. The original architecture/document contract, checkpoint parameters and captured activations remain unchanged. New mathematical claims link to primary papers and the Goldberg floating-point tutorial. Tests check local derivatives, simultaneous updates, overshoot, rotary relative-shift identities, readout invariance, counterfactual metrics, DPO gap sensitivity, IEEE binary32 rounding, normalization centering and deterministic routing.

[Learning-loop preview](images/wiki-learning-loop.png) · [Narrow feature lab](images/wiki-features-mobile.png)

The **Why attention uses memory** path adds a comparison hub plus dedicated guides on full/local/sparse/cross-attention, normalized kernel-linear recurrence, KV quantization, prefix sharing and memory budgets. Together with existing head-sharing, MLA and kernel guides, it separates architectural representation, visibility, precision, temporary workspace and physical allocation. Each new concept has an introductory and technical treatment and primary references.

The comparison lab shows eight hypothetical state estimates: MHA, GQA, MQA, windowed GQA, MLA-style joint latent plus positional key, a normalized kernel-linear accumulator, mixed full/recurrent layers, and packed GQA with explicit group padding/metadata. Controls expose all dimensions rather than baking in claims about a named checkpoint. A shared-scale growth view compares each selected mechanism against MHA at six lengths. A separate one-head matrix estimate switches between fresh prefill and single-position decode. Neither panel is a peak-VRAM or latency benchmark. Recurrent state uses FP32 independently of KV precision; quantized groups are fictional and do not execute KIVI. Tests check exact small cases, growth laws, padding overhead, hybrid sums and workspace preservation.

[Comparison preview](images/wiki-memory-comparison.png) · [Narrow light preview](images/wiki-memory-mobile.png)

Twenty worked-example types expose the intermediate arithmetic: an attention row, RoPE, RMSNorm, SwiGLU, vocabulary probabilities, logical K/V accounting, head sharing, embedding lookup, matrix projection, residual addition and attention-state comparison. Attention supports editable query/key/value operands, five explanation stages, scenario presets and a pinned baseline with probability/output deltas. Projection exposes each product and output sum; embedding lookup edits fictional parameter rows; residual controls isolate additions from branch execution. Cache controls vary retention policy, window, sparse stride and bytes per scalar, showing P/U/R, retained absolute positions and payload. Storage-size changes estimate bytes without executing quantization. Every slider also accepts bounded, step-valid numeric input. Numerical toys are labelled **synthetic**; cache/head scratch calculations start from the active architecture or numerical configuration. Scratch controls leave model settings, parameters and recorded activations unchanged. Position previews use bounded indexed reads, even at large configured capacities.

Concept headers now lead back to execution. **Find this mechanism in your model** lists matching existing components/views, prioritizing the current selection and graph. **Show** selects an exact graph element. **Open numerical example** opens the separate seeded foundations profile at an explicit supported operation; **Inspect in this run** seeks within an existing untrained/trained capture. It never substitutes untrained parameters into a trained study or executes an arbitrary imported model.

Thirty-five expandable reasoning questions explain common confusions and suggest concrete observable checks. They connect the attention heatmap to value contributions, distinguish cache working sets from final retention, and explain normalization, gating and sampling assumptions. Learning-path navigation links the preceding and following guides without scores or completion claims.

## Glossary, search and reading places

**Glossary & current symbols** supplies fifty-seven definitions linked to detailed guides and their primary references. Symbol values come from the study that opened help; numerical help passes its runtime configuration explicitly. Tensor entries have clickable symbol definitions. Vocabulary-size V is distinguished from attention’s value tensor.

Search includes definitions, equations, source relevance notes and experiment instructions, with a separate glossary collection. Explicit term variants support questions about masking and British “normalisation”. Conservative **Did you mean** suggestions offer recognizable spelling corrections; they never silently alter the query or use a remote answer service.

**Save article**, **Saved & recent reading**, and **Continue reading** retain bookmarks and scroll positions under the independent device key `model-visualizer.wiki-reading.v1`. Concept places work across models; document places are scoped by model ID. Unavailable original-model entries remain visible without redirecting to another model. The reading library is outside model JSON, notebooks, workspace exports and traces. Supported serialization preserves additional metadata, future versions are not overwritten, and storage failures clearly apply only to the current session. Back/forward navigation history remains local to the open reader.

## Content ownership

| File                                                | Responsibility                                                                                 |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `src/help/wiki/types.ts`                            | Article/section types and editorial primary-source metadata                                    |
| `src/help/wiki/foundations.ts`                      | Reading the model, evidence, tensors, projections, input and decoder blocks                    |
| `src/help/wiki/attention.ts`                        | Attention arithmetic, masking, position encoding, head sharing, cache and storage distinctions |
| `src/help/wiki/practice.ts`                         | Output, training, MTP/verification, extension boundaries and workbench use                     |
| `src/help/wiki/catalog.ts`                          | Topic associations, learning paths, document-derived entries and full-text search              |
| `src/help/wiki/explorations.ts`                     | Renderer-independent teaching arithmetic and explicit numerical destinations                   |
| `src/help/wiki/perspectives*.ts`                    | Authored introductory explanations and additional technical derivations for all concepts       |
| `src/help/wiki/glossary.ts`                         | Definitions, symbol identities and guide links                                                 |
| `src/help/wiki/questions.ts`                        | Reasoning questions, explanations and observable checks                                        |
| `src/help/wiki/reading.ts`                          | Versioned device-library parsing and preservation                                              |
| `src/help/wiki/memory.ts`                           | Attention-pattern, recurrence, precision, sharing and budgeting guides                         |
| `src/help/wiki/memory-comparison.ts`                | Pure hypothetical state accounting, with representation-specific assumptions                   |
| `src/components/help/AttentionMemoryComparison.tsx` | Comparison controls, accessible table, growth bars and separate workspace estimate             |
| `src/components/help/WikiExploration.tsx`           | Interactive worked examples                                                                    |
| `src/components/help/WikiCollections.tsx`           | Glossary, reading library and concept-to-model navigation                                      |
| `src/components/help/WikiProvider.tsx`              | Shared session reader and lazy loading                                                         |
| `src/components/help/Wiki.tsx`                      | Accessible dialog, navigation, search, linked terminology and article rendering                |
| `src/components/help/wiki.css`                      | Dark/light reader and responsive browsing                                                      |

The concept library contains explanatory prose, not another model definition. Authored component explanations, notes, tensor shapes and equations continue to live in `examples/transformer-foundations.model.json` or the imported document. The wiki reads that content directly. It does not migrate documents, add model-document fields, store generated graph copies, change IDs or rewrite imported text. The reading-library device preference is independent.

The model documentation entry carries the document’s own overview and complete source library. Related concept guides describe the supported foundations profile and explicitly warn that an imported architecture can differ. Semantic kinds select related topics; an arbitrary unknown kind is not treated as executable mathematics inferred from its title.

The learning additions use `src/help/wiki/learning.ts` for full guides, `perspectives-learning.ts` for alternative explanations, and `learning-experiments.ts` for renderer-independent arithmetic. `src/components/help/WikiLearningLab.tsx` owns the corresponding scratch controls and visual rendering.

## Evidence and mathematical conventions

The reader identifies its originating workbench throughout browsing:

- Architecture: calculated shapes and explicitly labelled synthetic illustrations.
- Graphical simulation: scripted choices and calculated state accounting, without model activations.
- Numerical reference: actual arrays from seeded **untrained** parameters.
- Local trained checkpoint: actual learned parameters and captured reference activations.

Worked numbers in editorial articles are synthetic hand calculations and say so. The teaching byte-BPE is not a production tokenizer. Architecture, execution kernel and physical storage remain separate: GQA is head sharing, FlashAttention is an attention execution strategy, and PagedAttention concerns physical KV allocation. Cache entries explain K/V tensors with layer, batch, KV-head, retained-position and feature axes plus absolute position maps. MLA is described as a distinct unimplemented state representation.

Primary references link to papers or original implementations, with a note describing their relevance and limits. The sources establish the mechanisms, while application-specific claims come from the documented implementation. Mathematical claims must stay accurate for the explicitly identified profile. The reader uses the existing bounded, untrusted KaTeX renderer. Model-provided prose renders as React text rather than HTML.

## Investigations, training and replay

**Start with a question** opens four practical starting points: changing values without changing attention, shared K/V with independent queries, logical attention-memory scaling, and training a small network. **Investigations & records** is also available from wiki navigation. Guided experiments separate prediction, a one-factor edit, observable arithmetic, a hint and a mechanism explanation. They are synthetic examples, with links back to the full concept guides; no score or completion claim is inferred from opening them.

On screens at least 1100px wide, **Dock beside model** changes the reader to a nonmodal side panel and gives the originating workbench its own remaining width. **Expand reader** restores modal reading. Narrowing the window restores modal behavior automatically. The portal remains inside an originating numerical dialog so that browser modal inertness cannot disable its reader. **Follow selection** explicitly opts into following graph selections; otherwise the chosen article stays put. A numerical dock also shows the selected tensor cell, absolute coordinates, actual value and its calculation explanation. Closing the dock restores the original workbench width without rewriting saved geometry.

The trainable network is an independent 1 → 4 tanh → 1 regressor with 13 parameters, JavaScript Number arithmetic, five authored training observations and eight fixed noiseless holdout observations. All 13 analytic gradients use the same pre-update parameters and full-batch half-MSE. SGD, Adam and AdamW recompute gradients after every update. Adam variants expose raw moment buffers, bias correction, last-update deltas and newly recomputed gradients. This recipe applies AdamW decay to biases as well as weights. Seeds and all controls are explicit; changing assumptions restarts the trajectory. The curve, middle-only and noisy datasets teach fitting and limited held-out evaluation, not transformer training or a generalization guarantee. Plot clipping is stated separately from the unmodified numerical losses. The lab links the original backpropagation, Adam and AdamW papers.

**Record this investigation** captures a question, prediction, observation, conclusion, reproducible recipe and a bounded evidence snapshot. Numerical snapshots contain at most 64 consecutive selected cells, with tensor ID, shape and flat starting index, plus aligned baseline values when available. Full arrays remain in separate trace exports. Untrained recipes preserve current and baseline settings, operation, decoder call and selected cell; trained recipes preserve checkpoint revision, prompt/sampling settings, head replacement and donor prompt, selected operation/cell, and any saved comparison recipe. Reopening restores a setup; checkpoint loading and generation remain explicit. Unsupported checkpoint revisions do not fall back to another model.

Records use the independent device key `model-visualizer.investigations.v1`. JSON import/export uses its own version-1 envelope and validation, a 2 MB input limit and 100-record limit. Existing record IDs win on merge. Supported extra metadata is retained. Unsupported stored versions are preserved and permit only a temporary record with immediate export; storage failure is also visibly temporary. This artifact does not extend or rewrite model documents, layout IDs, freeform notebooks, reading bookmarks or checkpoint assets. Exporting a workspace does not implicitly export the investigation library.

`src/model/study/` owns pure training, controlled-investigation arithmetic, comparison and record validation. `TrainingLab`, `Investigations`, `StudyRecords` and `RunComparison` supply the interfaces. Tests use finite differences for every network parameter, independent first-step optimizer formulas, unchanged attention under value edits, exact controlled memory ratios, zero and self-patching identities, downstream recomputation and preservation/replay checks. Browser coverage includes both docking contexts, responsive reading, selected-cell updates, record exports/imports and actual trained interventions.

## Topic-by-topic editorial review

The 2026-10-29 review covered all 49 concept guides in catalog order, their introductions and technical perspectives, plus the 57 glossary definitions and companion questions. The target reader has a reasonable mathematics, physics and programming background, but may be new to model internals. Imported model-document explanations remain authored source material; this review does not silently rewrite a user’s imported document.

The review criteria were: state the prediction or mechanism before implementation details; define the indexes and locally reused symbols; explain the steps behind an equation; provide a reconstructible example where it clarifies the mechanism; identify the assumptions under which an identity holds; and distinguish arithmetic evidence, causal experiments and measured model behavior. Introductory sections remain equation-free without relying on anthropomorphic stories. Technical sections use the reader’s mathematical background rather than replacing a derivation with a warning or a source link. Sources establish the relevant mechanisms; worked examples and local derivations are labelled as such.

| Guide                                              | Review outcome                                                                                                                                                      |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| How to read a model                                | Separated conditional next-token prediction, layer depth and generation time before the diagram tour.                                                               |
| What the numbers actually mean                     | Separated reconstructing an operation from evaluating trained behavior. Checked reference arithmetic against the numerical implementation.                          |
| Parameters, activations and persistent state       | Added optimizer history as a distinct persistent training state, separate from inference K/V.                                                                       |
| Tensors, axes and shape notation                   | Added an explicit fixed-index/summed-index reading rule and clarified software tensor terminology.                                                                  |
| Linear projections and matrix multiplication       | Distinguished learned linear maps from orthogonal/idempotent projections and explained null-space information loss.                                                 |
| Text, UTF-8 and byte-pair encoding                 | Separated learning a vocabulary from applying fixed merge ranks to a prompt; retained the byte reconstruction and merge-order examples.                             |
| Embedding lookup and vocabulary identity           | Corrected batch/sequence wording and added a repeated-ID gather check; tying does not invert the lookup.                                                            |
| Residual stream and decoder blocks                 | Connected the scalar derivative to the vector Jacobian and explained possible cancellation of the identity contribution.                                            |
| RMSNorm, gain and epsilon                          | Corrected denominator wording, defined the derivative symbols and chain-rule step, and qualified final mean/scale after learned gain and bias.                      |
| SwiGLU and position-wise feed-forward networks     | Distinguished coordinate-wise gating from position-wise sublayers; expanded the SiLU derivative and operand-sign interpretation.                                    |
| What self-attention computes                       | Replaced the passage-reading analogy with a concrete weighted-read explanation and defined all indexes in the two contractions.                                     |
| Dot-product scores and √Dh scaling                 | Explained both steps of the variance argument and the independence assumptions supporting it.                                                                       |
| Causal masking and visible positions               | Added a compact-slot counterexample to the visual-triangle shortcut.                                                                                                |
| Softmax and stable probability calculations        | Defined the Kronecker delta, diagonal/off-diagonal derivatives and the domain of the odds-ratio identity.                                                           |
| RoPE: position as a rotation                       | Defined the frequency hidden by R(p) shorthand, pair indexes and angle units.                                                                                       |
| MHA, GQA and MQA                                   | Connected projection widths to split, transpose and concatenate operations without implying statistical independence.                                               |
| What an attention heatmap can tell you             | Added a numerical case where a smaller attention coefficient produces the larger value contribution.                                                                |
| KV cache: persistent attention state               | Added the cached-versus-recomputed comparison under matching causal policy and numerical tolerance.                                                                 |
| Prefill, decode and the generation loop            | Corrected prefill terminology and retained the explicit consumed-input/selected-output timeline.                                                                    |
| Full, sliding and sparse retention                 | Added an exact recent-window/anchor union with overlap counted once.                                                                                                |
| Calculate logical KV memory                        | Generalized the layerwise sum over widths and dtypes, explicitly retaining the equal-K/V-width assumption behind the factor of two.                                 |
| Architecture, FlashAttention and PagedAttention    | Defined the online normalization state and reconstructed a two-block attention result.                                                                              |
| From hidden state to vocabulary logits             | Added a two-logit odds/probability check and distinguished an individual score from a log-odds difference.                                                          |
| Greedy choice, sampling and temperature            | Added cumulative-interval sampling and primary-source-backed top-k/top-p definitions with explicit unsupported-mode boundaries.                                     |
| Next-token targets and cross-entropy               | Added a target-gradient example, nats-per-token units, perplexity interpretation and comparability conditions.                                                      |
| Multi-token prediction: objectives and designs     | Clarified how shifted teacher-forced inputs change the conditional task; defined objective indexes and loss weights.                                                |
| Draft, verify, accept and roll back                | Completed stochastic acceptance with the rejection distribution, mass-conservation argument and a two-candidate check; retained the scripted-simulator distinction. |
| N-gram lookup and conditional memory               | Explained deterministic addresses versus contextual use, and the published retrieval/gating stages with dimensions and a scoped equation.                           |
| Mixture-of-experts and routed computation          | Added router/expert accounting, corrected a stale statement about the standalone routing lab, and kept learned model execution distinct.                            |
| MLA and alternative attention state                | Rewrote the introductory picture and added dimensioned projection-associativity identities with a separate positional contribution.                                 |
| Learn with the tensor microscope                   | Added a concrete semantic-coordinate record and distinguished a scalar recipe from a behavioral description.                                                        |
| Study the local trained checkpoint                 | Clarified mechanism checks versus completion quality and why different-shaped trained/untrained runs are not a same-model intervention.                             |
| A practical way to investigate a model             | Added falsifiable expected observations and separated implementation invariants from empirical feature hypotheses.                                                  |
| Navigation, notes and portable model documents     | Added a recovery workflow and distinguished device reading preferences, portable study settings and tensor captures.                                                |
| Compare attention mechanisms and memory            | Explained when controlled scalar-count/precision ratios can be multiplied and what still must be added.                                                             |
| Full, local, sparse and cross-attention            | Distinguished source and target position spaces in encoder-decoder attention.                                                                                       |
| Linear attention and recurrent state               | Added a complete two-key accumulator/read example and disambiguated local feature-map width from FFN width.                                                         |
| KV cache quantization and precision                | Corrected introductory K/V wording, defined codebook symbols and separated rounding error from clipping error.                                                      |
| Shared prefixes, paging and cache reuse            | Replaced conflicting P/U notation and added a unique-physical-position count with an exact saving.                                                                  |
| Attention memory budgets and scaling               | Added matched full-cache and one-head-matrix byte checks with distinct lifetimes and scaling laws.                                                                  |
| How a model learns: gradients and updates          | Reconstructed the full default forward/backward/update example and clarified finite differences for a quadratic versus a general nonlinear loss.                    |
| Compare positional encoding methods                | Expanded additive-position cross terms to explain why a sinusoidal positional term alone is not the whole score.                                                    |
| Features, neurons and superposition                | Added two distinct nonnegative feature explanations producing the same toy hidden vector.                                                                           |
| Test a mechanism: ablation and activation patching | Replaced a loose necessity claim with a stated success criterion, background and intervention.                                                                      |
| From pretraining to assistant behaviour            | Defined policy/reference distributions, target masks, conditional KL, sequence log probabilities and the objective scale.                                           |
| Floating-point precision and numerical stability   | Connected the cancellation example to binary32 spacing at its actual exponent.                                                                                      |
| Adam: adaptive moments and optimizer state         | Corrected the stationary-moment condition, separated the raw second moment from variance/Hessian, and derived geometric history and half-lives.                     |
| AdamW: decoupled weight decay                      | Explained the coefficient range in which the isolated factor is actually sign-preserving shrinkage.                                                                 |
| Muon: matrix momentum and Newton–Schulz updates    | Introduced singular directions and the two norms, and stated the pinned implementation’s rectangular scaling convention.                                            |

The glossary review corrected LayerNorm’s square-root denominator, clarified that separate batch/head computations do not imply statistical independence, and removed a universal expansion assumption from intermediate width F. Existing IDs, section links, authored model content, layouts, notebooks and reading-library format are preserved. Validation checks equation rendering, source/link resolution, reader navigation and the interactive examples; passing these checks is not an automated proof of pedagogical quality or of every scientific statement.

## Adding or updating an explanation

1. Read the architecture/model-format contract and the relevant numerical or cache documentation.
2. Add a stable article ID, concise summary, aliases and a category to the relevant TypeScript content module.
3. Add both perspectives: a non-mathematical introduction with purpose, alternative picture, steps and limits, and an additional technical derivation. Keep the full technical guide. Label axes, reductions, ordering conventions and assumptions. Do not infer an imported model’s mathematics from a generic guide.
4. Include a small labelled synthetic worked example where it helps, plus a concrete workbench experiment where the mechanism is executable.
5. Distinguish what exists in this app from relevant background and future extensions. Do not claim arbitrary imported models run in the numerical engine.
6. Add appropriate primary references with scope notes, prerequisites and related article IDs. Add semantic kinds/view IDs only where the association is justified.
7. For terms that should link from prose, update the reader’s `termLinks` map. Avoid ambiguous single-letter automatic links.
8. Run formatting, build, unit tests and browser tests. Check both themes and narrow screens after changing the reader.

Tests validate unique IDs, resolving links/sources/learning paths, equation rendering, full-text and alias search, runtime navigation scoping, imported-document preservation, nested workbench dialogs, responsive layouts and context-to-canvas return. Browser tests use their own isolated contexts and do not touch a user’s live workspace.
