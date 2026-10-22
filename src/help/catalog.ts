export type HelpContent = {
  title: string
  paragraphs: string[]
  choices?: { label: string; description: string }[]
  sources?: string[]
}

const attention = [
  'Attention Is All You Need · Vaswani et al. (2017)',
  'GQA · Ainslie et al. (2023)',
]
const rope = ['RoFormer · Su et al. (2021)']
const cache = ['GQA · Ainslie et al. (2023)', 'Mistral 7B · Jiang et al. (2023)']
const entry = (title: string, ...paragraphs: string[]): HelpContent => ({ title, paragraphs })

/** Explanations of this application's supported profile, not universal model
 * constraints. Primary-source links live in the field guide and inspector. */
export const helpCatalog: Record<string, HelpContent> = {
  'Model width D': {
    ...entry(
      'Model width · D',
      'The number of features in each token’s residual-stream vector. In this teaching profile, D = Hq × Dh: all query heads together span exactly the model width.',
      'Changing D keeps the query-head count where possible, otherwise chooses a compatible partition. Projection shapes and logical cache estimates update together; existing trained weights are not resized.',
    ),
    sources: attention,
  },
  'Decoder layers': entry(
    'Decoder layers · L',
    'The number of successive transformer blocks. Each layer has its own attention and feed-forward parameters and its own K/V state; repeating a block in the diagram does not mean sharing its weights.',
    'More layers expand the whole-model chart and increase logical cache storage linearly. Choices that would exceed the expanded-view budget are omitted.',
  ),
  'Query heads': {
    ...entry(
      'Query heads · Hq',
      'Independent query-head paths compute attention using their assigned K/V head. Their results are concatenated before the attention output projection.',
      'Changing Hq repartitions the current D and automatically sets Dh = D / Hq. Only whole heads with even RoPE feature widths and compatible KV sharing are offered.',
    ),
    sources: attention,
  },
  'Features per head': {
    ...entry(
      'Features per head · Dh',
      'The feature dimension of each query, key and value head in this profile. Attention contracts matching Q/K features, and scores are scaled by 1 / √Dh.',
      'Changing Dh keeps D fixed and adjusts Hq = D / Dh. Dh must be even because this RoPE implementation rotates adjacent feature pairs.',
    ),
    sources: [...attention, ...rope],
  },
  'Attention architecture': {
    ...entry(
      'Attention architecture',
      'Choose how query heads share keys and values. This changes K/V projection widths and logical cache size, while query-head results still span D. Only compatible head partitions within the view budget are offered.',
    ),
    choices: [
      { label: 'MHA', description: 'One K/V pair per query head: Hkv = Hq.' },
      {
        label: 'GQA',
        description:
          'Equal-sized query groups share a K/V pair. Hkv is a proper divisor of Hq, strictly between 1 and Hq.',
      },
      { label: 'MQA', description: 'All query heads share a single K/V pair: Hkv = 1.' },
    ],
    sources: attention,
  },
  'KV heads': {
    ...entry(
      'Key/value heads · Hkv',
      'The number of distinct K/V head pairs per layer. The logical cache has layer, batch, KV-head, retained-position and feature axes; it is not indexed by query heads.',
      'In GQA, Hq / Hkv queries share each pair. MHA fixes Hkv to Hq; MQA fixes it to 1. A logical head slice is a view, not another allocation.',
    ),
    sources: cache,
  },
  'Feed-forward width': {
    ...entry(
      'Feed-forward width · F',
      'The intermediate feature count in this profile’s SwiGLU feed-forward network. Two D → F projections form the gate and value paths, an element-wise product combines them, and an F → D projection returns to the residual width.',
      'F is independently configurable; it does not need to equal D or the number of attention heads. This control describes matrix shapes, not a trained-model activation.',
    ),
    sources: ['GLU Variants Improve Transformer · Shazeer (2020)'],
  },
  'Context capacity': entry(
    'Context capacity · C',
    'The maximum sequence extent described by the teaching configuration. Increasing it changes capacity and cache estimates; it does not add decoder layers or enlarge each token vector.',
    'If capacity is reduced below the retention window or sparse stride, those bounds are reduced with it. The trained checkpoint has its own fixed capacity and a smaller inspection-capture limit.',
  ),
  'Cache scalar storage': {
    ...entry(
      'Bytes per cached scalar',
      'The logical bytes assigned to each K or V scalar for capacity estimates. Logical full-prefix state is 2 × L × B × Hkv × S × Dh × bytes per scalar.',
      '1 byte excludes quantization metadata; 2 bytes describes FP16/BF16-sized storage; 4 bytes describes FP32-sized storage. This setting does not execute quantization or change the numerical laboratory’s Float32 arrays.',
    ),
    sources: cache,
  },
  'KV cache retention': {
    ...entry(
      'KV cache retention',
      'Select which absolute positions remain in the logical K/V state. All policies remain causal: a query cannot read a future key. Retained-slot count can differ from the processed sequence extent.',
    ),
    choices: [
      {
        label: 'Full prefix',
        description: 'Retain every processed position up to the context capacity.',
      },
      {
        label: 'Sliding window',
        description:
          'Retain the latest W positions, including the current position. Absolute position IDs do not wrap.',
      },
      {
        label: 'Sparse · anchors + recent',
        description:
          'Teaching policy: retain positions divisible by the anchor stride plus the latest W positions. Anchors grow with context; this is not a fixed-size sliding cache.',
      },
    ],
    sources: cache,
  },
  'KV window size': {
    ...entry(
      'Retention window · W',
      'The number of most recent positions kept, including the current one. Under sliding retention this bounds persistent resident positions; during prefill, the working key set can be larger before retention is applied.',
      'In sparse retention this recent window is combined with absolute-position anchors. Position IDs stay absolute; any ring-buffer slot is a separate physical-storage example.',
    ),
    sources: cache,
  },
  'Sparse anchor stride': entry(
    'Sparse anchor stride',
    'Keep every absolute position divisible by this stride, in addition to the recent window. For stride 4, positions 0, 4, 8, … are anchors.',
    'A larger stride keeps fewer anchors. This is the application’s explicit teaching policy, not a claim about every sparse-attention architecture. It must be at least 2 and no larger than context capacity.',
  ),
  'Cached sequence length': entry(
    'Processed sequence extent',
    'The number of absolute positions used for this cache study. Full retention stores all of them; sliding and sparse policies keep their selected subset.',
    'The displayed bytes count logical K and V scalars only. Padding, allocator overhead, sharding and quantization metadata are outside this estimate.',
  ),
  'Cache layer': entry(
    'Inspect a decoder layer',
    'Choose which layer’s K/V state the slice displays. Each decoder layer owns distinct keys and values, even when its tensor dimensions match another layer.',
    'The selected layer changes the inspected slice, not the number of layers or the total cache estimate.',
  ),
  'Cache KV head': entry(
    'Inspect a KV head',
    'Choose a logical key/value head within the selected layer. With GQA or MQA, multiple query heads can read this same pair of cached head slices.',
    'This selects an axis of state; it does not duplicate storage or turn KV-head count into query-head count.',
  ),
  'Inspect query head': entry(
    'Inspect a query head',
    'Bring this head’s explicit path into view and open its explanation. Follow its Q projection through scores, masking, softmax and weighted values, with the correct shared K/V group.',
    'The selected head changes the inspected path; it does not change model topology or execute an extra attention computation.',
  ),
  'Active decoder layer': entry(
    'Active decoder layer',
    'Choose the layer represented by this reusable block/attention study. Parameters and state belong to that layer, although the same authored diagram template explains each block.',
    'The whole-model view shows all layers together. The layer selector changes the slice being studied rather than tying weights across layers.',
  ),
  'Jump to model stage': entry(
    'Jump to a model stage',
    'Move the camera to tokenization, a decoder layer, output or generation without rearranging the diagram. Useful when the complete model is too large to read at full-map zoom.',
    'Use Fit graph to return to the complete canvas. Dragging and Auto-layout remain independent of camera navigation.',
  ),
  'MTP prediction depth': entry(
    'Multi-token prediction depth',
    'The number of future-token offsets illustrated by the auxiliary MTP study. Additional heads predict farther-ahead targets rather than appending several confirmed tokens automatically.',
    'The graphical simulator uses scripted proposal/verification decisions. The numerical reference and the bundled trained checkpoint do not execute MTP heads.',
  ),
  'N-gram ending token': entry(
    'N-gram ending position',
    'Choose the token position where the illustrated bigram, trigram or four-token window ends. The diagram shows its IDs, teaching address and contextual combination.',
    'This memory is an explanatory extension; these addresses and vectors are not a trained retrieval-memory implementation.',
  ),
  'BPE merge step': entry(
    'Tokenizer merge step',
    'Scrub the real teaching byte-BPE merge history. Each step merges the highest-priority adjacent pair available in this deliberately small demonstration vocabulary.',
    'The pieces are byte spans: an intermediate piece can contain only part of a UTF-8 character. This tokenizer is not the trained checkpoint’s tokenizer or a production vocabulary.',
  ),
  PROMPT: entry(
    'Teaching prompt',
    'Enter text to inspect its UTF-8 bytes, teaching-tokenizer IDs and merge history. Prefill shapes use the resulting token count; Decode step describes a hypothetical one-token query after that prefix.',
    'The architecture study does not run trained weights. For actual local text generation, open Local trained model and explicitly load its bundled checkpoint.',
  ),
  Dimensions: entry(
    'Show tensor dimensions',
    'Show or hide the compact shape annotation on graph components and connection labels. The underlying tensor descriptions and inspector remain available.',
    'Numeric dimensions reflect the current configuration and sequence lengths. Symbolic mode displays named axes such as D, Hq, T and S instead.',
  ),
  'Toggle symbolic dimensions': entry(
    'Symbolic or resolved dimensions',
    'Switch between named axes and their resolved integer values. D is model width, Hq query heads, Hkv KV heads, Dh features per head, T current query positions and S processed extent.',
    'Cache axes R, P and U describe retained positions after a call, the retained prefix before it and working keys. They need not equal S under bounded retention.',
  ),
  'Simulation input packets': entry(
    'Graphical input packets',
    'Set the number of abstract input-token packets in the graphical simulation. Prefill sends their parallel sequence through each layer.',
    'These are opaque teaching packets: this mode does not tokenize text, compute logits or run learned weights. The local trained model is a separate execution path.',
  ),
  'Simulation output budget': entry(
    'Graphical output budget',
    'Set how many confirmed abstract output tokens the scenario should produce. MTP proposals may be accepted or discarded before they count toward this budget.',
    'Proposal choices are scripted. Reaching this budget completes the graphical trace; it is not a measurement of language-model quality.',
  ),
  'Verification scenario': {
    ...entry(
      'Scripted MTP verification',
      'Choose the illustrative agreement pattern between proposed future IDs and target predictions. Verification accepts the matching prefix; a mismatch discards the remaining drafts and restores the committed KV checkpoint.',
    ),
    choices: [
      {
        label: 'Mixed',
        description: 'Show both successful draft acceptance and rejection/rollback.',
      },
      {
        label: 'All drafts match',
        description: 'Illustrate accepting the complete proposed suffix.',
      },
      {
        label: 'First draft mismatches',
        description: 'Illustrate early rejection and cache rollback.',
      },
    ],
  },
  'Simulation step detail': entry(
    'Playback detail',
    'Per stage groups the graphical trace into larger phases. Every operation visits the finer mathematical steps and individual head paths.',
    'This changes playback granularity, not the architecture or the amount of K/V state retained.',
  ),
  'Simulation timeline': entry(
    'Graphical execution timeline',
    'Scrub the abstract packet scenario backward or forward. Each frame identifies active operations, confirmed or proposed packets, and the corresponding logical cache state.',
    'This is an illustrative scripted trace; it does not contain trained activations or computed language-model logits. Use the numerical workbenches for actual tensor values.',
  ),
  'Playback speed': entry(
    'Graphical playback speed',
    'Choose the pause between automatically displayed simulation steps: Slow 1.1 seconds, Normal 0.65 seconds, or Fast 0.25 seconds.',
    'This is animation timing, not inference throughput. It is independent of the configurable tooltip delay.',
  ),
  'Follow the active operation': entry(
    'Follow the active operation',
    'Optionally pan the camera to the current graphical operation during playback or stepping, keeping your exact zoom. Leave this off to watch packets advance across the whole model or any fixed region.',
    'Off by default. Playback, stepping and scrubbing work at every zoom level. You can pan, zoom or Fit graph while the simulation runs; following never forces a close-up or moves graph nodes.',
  ),
  'Numerical parameter seed': entry(
    'Untrained parameter seed',
    'Reproducibly initialize the numerical teaching model’s weights. Changing this seed produces a different untrained reference model and recalculates its captured trace.',
    'These values are actual results of the local reference arithmetic, but the weights are not trained. The trained checkpoint’s learned weights never change with its sampling seed.',
  ),
  'Numerical output budget': entry(
    'Numerical output budget',
    'Choose the number of new vocabulary IDs captured after the input prefix. Each decoder call records actual arrays, attention probabilities and retained K/V state.',
    'The untrained laboratory uses opaque IDs rather than meaningful text. More calls produce a larger trace and more arrays to inspect.',
  ),
  'Numerical head sharing': {
    ...entry(
      'Numerical head sharing',
      'Re-run the small untrained reference model with MHA, GQA or MQA. The graph, K/V projections, parameter arrays and cache-head axis follow the selected architecture.',
      'This experiment changes the reference model; it does not modify the fixed architecture of the bundled trained checkpoint.',
    ),
    choices: [
      { label: 'MHA', description: 'Distinct K/V pairs for every query head.' },
      { label: 'GQA', description: 'Equal query groups share K/V pairs.' },
      { label: 'MQA', description: 'All queries share one K/V pair.' },
    ],
    sources: attention,
  },
  'Numerical token selection': entry(
    'Numerical token selection',
    'Greedy chooses the highest logit. Seeded categorical selection draws an ID from the temperature-adjusted softmax distribution reproducibly.',
    'Selection consumes the computed logits; playback later replays the captured choice rather than sampling again. The untrained reference is not a language-trained model.',
  ),
  'Numerical temperature': entry(
    'Sampling temperature',
    'Before categorical selection, divide logits by a positive temperature and apply softmax. Smaller values concentrate probability on higher logits; larger values flatten the distribution.',
    'Temperature affects sampling, not the learned weights or greedy argmax. It does not make an untrained model language-trained.',
  ),
  'Numerical decoder call': entry(
    'Recorded decoder call',
    'Call 0 is prefill over the input prefix. Later calls consume a previously selected ID and reuse the layer-specific retained K/V state.',
    'Choosing a call changes which captured operations are replayed. A selected output ID does not enter the cache until a subsequent forward pass consumes it.',
  ),
  'Numerical execution timeline': entry(
    'Recorded operation timeline',
    'Scrub through actual captured operations. Tensors become available at the step that computes them; future activations remain unrevealed.',
    'This replays a bounded reference trace. It does not pause an optimized inference kernel at each multiplication.',
  ),
  'Numerical playback speed': entry(
    'Numerical playback speed',
    'Set the pace at which recorded numerical operations are revealed. Stepping and scrubbing remain available independently.',
    'This controls presentation timing, not the model’s generation speed or the tooltip hover delay.',
  ),
  'Local model prompt': entry(
    'Local trained-model prompt',
    'Text is encoded with the checkpoint’s matching scored-BPE vocabulary, including its BOS and dummy-space-prefix behavior. It is not sent to a service.',
    'The inspection path accepts up to 48 prompt IDs including BOS. Stories260K is a small base completion model, so plain story beginnings work better than chat instructions.',
  ),
  'Local model output budget': entry(
    'Trained generation budget',
    'Capture between 1 and 16 newly selected IDs from actual checkpoint inference. Generation can stop earlier on a BOS/EOS special ID.',
    'The checkpoint supports 512 positions, but inspection uses a smaller bounded capture to keep complete intermediate tensors manageable.',
  ),
  'Local model selection': {
    ...entry(
      'Trained token selection',
      'Choose how actual logits become the next token ID. The trained weights remain identical under either choice.',
    ),
    choices: [
      {
        label: 'Greedy argmax',
        description:
          'Always select a highest-scoring ID; temperature and sampling seed do not affect argmax.',
      },
      {
        label: 'Seeded categorical',
        description:
          'Draw reproducibly from temperature-adjusted softmax probabilities using the sampling seed.',
      },
    ],
  },
  'Local model sampling seed': entry(
    'Categorical sampling seed',
    'Reproduce the random draws used for token selection with the same prompt, settings and checkpoint. A different seed can produce a different continuation.',
    'This seed never reinitializes, retrains or edits the checkpoint’s learned weights. It is unused by greedy selection.',
  ),
  'Local model temperature': entry(
    'Trained sampling temperature',
    'For categorical selection, logits are divided by this positive value before softmax. Lower temperatures concentrate choices; higher temperatures spread probability more evenly.',
    'This does not change attention or learned weights within a forward pass. Greedy selection ignores temperature.',
  ),
  'Workspace notes': entry(
    'Workspace notebook',
    'Write your own study notes alongside the architectural document. Notes save locally with the workspace and travel in Export model JSON.',
    'The numerical laboratory and trained checkpoint have separate study notebooks, so experiments do not replace this main notebook.',
  ),
  'Numerical laboratory notes': entry(
    'Laboratory study notes',
    'Record observations alongside this study’s settings, layout and bookmarks. They save in this laboratory’s independent versioned workspace extension.',
    'Export model carries these notes and settings; Export trace separately contains captured numerical arrays. Unsupported future extension fields are preserved.',
  ),
  'Tooltip delay': entry(
    'Tooltip hover delay',
    'The time the pointer must remain over the same control before its explanation appears. The default is 0.7 seconds; set any delay from 0 to 5 seconds.',
    'Moving within one control keeps its timer. Leaving it, moving to another control, clicking, scrolling or pressing Escape cancels the pending explanation. This preference saves on this browser.',
  ),
  'Show explanatory tooltips': entry(
    'Enable explanatory tooltips',
    'Show detailed help after a sustained pointer hover or keyboard focus. Turn this off to keep the interface quiet; the timing/preferences control remains available.',
    'Tooltips do not take keyboard focus. Escape dismisses the visible explanation while leaving its triggering dialog open.',
  ),
}

const actions: [string[], HelpContent][] = [
  [['MHA', 'GQA', 'MQA'], helpCatalog['Attention architecture']],
  [
    ['Edit model configuration'],
    entry(
      'Shape the teaching model',
      'Open the teaching architecture builder to adjust width, heads, layers, feed-forward dimensions, context capacity and cache storage. Linked controls maintain compatible dimensions and valid head sharing.',
      'Changes save automatically and update the architectural studies. The fixed trained checkpoint is configured separately.',
    ),
  ],
  [
    ['Tooltip settings'],
    entry(
      'Tooltip timing & preferences',
      'Choose how long to hover before detailed explanations appear, or turn automatic tooltips off. The default is 0.7 seconds over the same control.',
      'The preference is local to this browser and survives reloads. Moving onto another control always starts a fresh timer.',
    ),
  ],
  [
    ['Auto-layout'],
    entry(
      'Arrange the entire graph',
      'Reposition all components of the current view, including whole-model sections. Large maps pack actual section footprints rather than reserving oversized equal cells.',
      'Connections and labels are routed around obstacles and checked before the result is applied. Saved views are independent; Undo auto-layout restores the previous arrangement.',
    ),
  ],
  [
    ['Undo auto-layout'],
    entry(
      'Restore the previous arrangement',
      'Restore positions, direction, side overrides and viewport from the previous automatic layout in this open graph. Up to ten layout snapshots are kept while the view stays open.',
      'This does not undo model-configuration edits, notebook changes or other views. The control is unavailable until an automatic layout has completed.',
    ),
  ],
  [
    ['Switch layout direction'],
    entry(
      'Change layout direction',
      'Rearrange this graph with horizontal or vertical internal flow. Large maps repack their sections to match the selected direction.',
      'This performs a full automatic layout and fits the result. The previous positions and camera remain available through Undo auto-layout.',
    ),
  ],
  [
    ['Fit graph'],
    entry(
      'Fit the complete diagram',
      'Frame all components, connection labels and exterior routes in the available canvas. Whole-model maps can zoom very far out without omitting layers or heads.',
      'Only the camera changes. Use stage navigation or zoom in to read individual operations; your node positions remain intact.',
    ),
  ],
  [
    ['Actual size'],
    entry(
      'View at actual size',
      'Set the canvas zoom to 100% so component text and tensor shapes are readable at their designed size.',
      'This changes the camera, not the layout. Large models extend beyond the viewport; pan or jump to a stage to explore them.',
    ),
  ],
  [
    ['Zoom in', 'Zoom out'],
    entry(
      'Change canvas magnification',
      'Zoom the camera to inspect an operation or regain context. Components, labels and routed connections scale together.',
      'Node coordinates are unchanged. Fit graph returns to a view of the entire diagram; stage navigation centers a selected section.',
    ),
  ],
  [
    ['Search components', 'Search model components'],
    entry(
      'Search the model',
      'Search names, operations and descriptions across the available graph views. Selecting a result opens its graph, centers the component and displays its explanation.',
      'Press / from the main workbench to open search. This searches architectural content rather than querying an inference service.',
    ),
  ],
  [
    ['Open', 'Open model JSON'],
    entry(
      'Open a saved workspace',
      'Choose a readable model JSON file containing architecture, documentation, notes and saved layouts. The file is validated before replacing your current workspace.',
      'Invalid imports leave the current model intact. Import does not automatically load a trained checkpoint or run inference.',
    ),
  ],
  [
    ['Export model'],
    entry(
      'Export the complete workspace',
      'Download readable model JSON with authored architecture, all saved layouts, notebook content and supported extension settings.',
      'Captured activation arrays and learned weights belong in Export trace, a separate artifact. Browser-specific tooltip preferences are not part of a model document.',
    ),
  ],
  [
    ['Toggle inspector'],
    entry(
      'Show or hide the inspector',
      'Open the side panel for the selected component or connection: descriptions, tensor shapes, equations, sources and the workspace notebook.',
      'This changes available screen space and leaves graph semantics and saved node positions untouched.',
    ),
  ],
  [
    ['Open navigation', 'Close navigation'],
    entry(
      'Model navigation',
      'Show or hide the navigation panel on a narrow screen. Choose a study view, the complete model, simulation or one of the numerical workbenches.',
      'Each view retains its independent layout and viewport. Changing views does not replace your workspace notebook.',
    ),
  ],
  [
    ['Prefill'],
    entry(
      'Prefill shapes',
      'Describe a parallel forward pass over the current tokenized prompt. T is the number of query positions in this call; full causal attention still masks future positions.',
      'This architectural shape view does not execute weights. The numerical and local trained workbenches compute their own recorded prefill traces.',
    ),
  ],
  [
    ['Decode step'],
    entry(
      'Cached decode shapes',
      'Describe a hypothetical next-token forward pass: one current query position reads retained earlier keys and its current key, then writes layer-specific K/V state.',
      'The architecture study changes dimensions without fabricating a generated token. Actual token selection is available in the numerical workbenches.',
    ),
  ],
  [
    ['Autoregressive'],
    entry(
      'Ordinary autoregressive generation',
      'Illustrate one next-token prediction followed by another decoder call that consumes the selected ID and reuses retained K/V state.',
      'This is the abstract graphical scenario. Choose Local trained model to compute a real learned-model continuation.',
    ),
  ],
  [
    ['MTP + verification'],
    entry(
      'Draft multiple tokens, then verify',
      'Illustrate auxiliary future-token proposals followed by causal target verification. Only the matching prefix is committed; rejected drafts cause rollback of tentative cache writes.',
      'Decisions in this graphical mode are scripted. The current numerical reference and Stories260K checkpoint do not execute numerical MTP.',
    ),
  ],
  [
    ['Reset simulation', 'Reset numerical playback'],
    entry(
      'Return playback to the start',
      'Seek to the beginning of this trace and stop automatic playback so you can inspect the initial input and first operations.',
      'This resets the playback position rather than editing the model architecture or deleting saved study notes.',
    ),
  ],
  [
    ['Previous simulation step', 'Previous numerical operation', 'Previous merge'],
    entry(
      'Inspect the previous step',
      'Move backward through the current teaching or recorded execution trace to inspect the previous state. Numerical tensors follow the selected step’s availability.',
      'This is reversible navigation within the trace; it does not undo model settings or change learned parameters.',
    ),
  ],
  [
    ['Next simulation step', 'Next numerical operation', 'Next merge'],
    entry(
      'Inspect the next step',
      'Advance one step in the current trace. The inspector follows the operation and reveals its available tensors, retained state or tokenizer pieces.',
      'Use the timeline to jump farther. In the graphical simulator, the detail selector chooses stage-sized or operation-sized steps.',
    ),
  ],
  [
    ['Play', 'Pause'],
    entry(
      'Play or pause the trace',
      'Automatically advance through the recorded operations or illustrative scenario at the chosen playback speed. Pause to inspect a component, tensor or scalar calculation.',
      'Numerical playback reveals already captured computation; it does not pause a low-level inference kernel at each mathematical operation.',
    ),
  ],
  [
    ['Run to completion', 'Finish numerical playback'],
    entry(
      'Reveal the completed trace',
      'Stop playback and seek to the final captured state, including confirmed output and the final retained cache.',
      'This reveals the recorded result. It does not generate beyond the selected output budget or add another forward pass.',
    ),
  ],
  [
    ['Export trace'],
    entry(
      'Export actual captured tensors',
      'Download separate readable JSON with parameters, activation arrays, IDs, operation recipes and arithmetic provenance for this numerical run.',
      'Trained traces include checkpoint/tokenizer identity and hashes. This can be much larger than a workspace; workspace export saves settings and study notes without duplicating all arrays.',
    ),
  ],
  [
    ['Load trained model', 'Reload trained model'],
    entry(
      'Load the bundled learned checkpoint',
      'Read the pinned Stories260K weights and matching tokenizer from the application’s own assets, verify their SHA-256 digests, and prepare the local worker.',
      'No prompt is sent to an inference service. Loading does not start generation; Generate & inspect explicitly runs the model.',
    ),
  ],
  [
    ['Generate & inspect'],
    entry(
      'Run the trained model locally',
      'Encode the prompt with the matching tokenizer, run the learned weights in a browser worker and capture real activations, per-head attention, logits and retained K/V state.',
      'The bounded trace can then be replayed and inspected in the shared flowchart and tensor microscope. This is a small base completion model, not a chat assistant.',
    ),
  ],
  [
    ['Cancel generation'],
    entry(
      'Stop local generation',
      'Terminate the in-flight local worker run so no more decoder calls are computed for this request.',
      'Existing saved layouts and notes remain. Loading and generation always require explicit controls; nothing is sent to a model service.',
    ),
  ],
  [
    ['Transpose view'],
    entry(
      'Transpose the displayed slice',
      'Swap the displayed row and column axes of this tensor slice. Cell coordinates continue to identify the original tensor entries.',
      'This is a viewing transformation: it does not transpose the stored model weights, rerun the computation or change the mathematical operation.',
    ),
  ],
  [
    ['Δ baseline'],
    entry(
      'Difference from the pinned baseline',
      'Display current minus baseline for compatible cells. Shapes, semantic axes, storage type, absolute positions and visibility masks must match before subtraction is allowed.',
      'Excluded mask entries remain mask markers; subtracting two negative infinities is not meaningful. The cell calculation continues to explain the current run.',
    ),
  ],
  [
    ['Pin baseline', 'Pin this run', 'Clear baseline'],
    entry(
      'Pin or clear a comparison run',
      'Keep an independent reference experiment for compatible cell-by-cell comparisons while changing the current untrained laboratory settings.',
      'Only matching tensor slices are subtracted. Baseline settings save separately from the current run; trained checkpoint parameters remain fixed.',
    ),
  ],
  [
    ['Bookmark operation', 'Bookmark this step'],
    entry(
      'Bookmark this operation',
      'Save the current operation and decoder-call location in this study, with space for a personal observation.',
      'Bookmarks belong to the independent numerical or trained study and travel with workspace settings. They do not embed the entire activation trace.',
    ),
  ],
  [
    ['Copy LaTeX'],
    entry(
      'Copy the mathematical expression',
      'Copy this component or connection’s LaTeX source for use in your notes or another mathematical editor.',
      'The inspector renders the same source with KaTeX and provides the associated tensor descriptions and primary references.',
    ),
  ],
  [
    ['Close dialog', 'Back to the model', 'Close simulator'],
    entry(
      'Return to the model',
      'Close this panel or dialog and return to the surrounding model exploration. Automatically saved settings, notebooks and layouts remain available.',
      'Escape closes a dialog when no tooltip is open; with a tooltip open, the first Escape dismisses that explanation.',
    ),
  ],
]
for (const [names, content] of actions) for (const name of names) helpCatalog[name] = content
Object.assign(helpCatalog, {
  'Follow flow': entry(
    'Follow the recorded flow',
    'Optionally pan the numerical or trained-model canvas to the currently revealed operation, preserving your exact zoom. Leave this off to watch the complete model or a fixed region while computation progresses.',
    'Off by default. Play, step and scrub at any zoom level. Explicit navigation such as a question or chapter can still bring its component into close view. Captured arrays, tensor values and node positions are unchanged.',
  ),
  Notebook: helpCatalog['Workspace notes'],
  'Open numerical lab notes': helpCatalog['Numerical laboratory notes'],
  Inspector: entry(
    'Component and connection inspector',
    'Show the selected component or edge’s explanation, tensor dimensions, equation and primary references. In numerical studies, the microscope additionally exposes available recorded arrays and cell calculations.',
    'Selecting an item changes the explanation being viewed; it does not edit the model’s definition or parameters.',
  ),
  'Getting oriented': entry(
    'Read the learning guide',
    'Read the model document’s overview, axis conventions and distinctions between architectural diagrams, illustrative simulations, untrained numerical references and the local trained checkpoint.',
    'Use this introduction to choose an appropriate study view, then inspect components and connections for their exact mathematics.',
  ),
  'Research library': entry(
    'Primary research library',
    'Browse the original papers and author implementations behind the teaching explanations. References identify the architecture, mathematical operation or implementation convention they support.',
    'An execution kernel, attention architecture and physical cache storage are different concepts; a source about one does not imply the others are used by every model.',
  ),
  'The model format': entry(
    'Portable model document format',
    'Read how authored graph templates, equations, source references, layout positions, notes and extension metadata are described in readable versioned JSON.',
    'Export model saves the architectural workspace. Actual large tensor arrays belong in a separate trace export; opening a document never automatically executes arbitrary weights.',
  ),
  'Open tokenizer lab': helpCatalog['BPE merge step'],
  Done: entry(
    'Keep the tooltip preference',
    'Close tooltip preferences and return to the same underlying view. Valid timing and enable/disable changes are applied immediately and saved separately on this browser.',
    'This does not replace your model workspace, notes or layouts. Unsupported future preference metadata remains preserved.',
  ),
  'Restore tooltip defaults': entry(
    'Restore the 0.7-second default',
    'Enable explanatory tooltips and restore the default sustained-hover delay of 700 milliseconds. The pointer must remain over the same control for that interval.',
    'This resets only tooltip preferences on this browser. Model documents, notebook content, graph geometry and experiment settings remain intact.',
  ),
  K: {
    ...entry(
      'Inspect cached keys',
      'Keys provide the feature vectors compared with queries when computing attention scores. This view selects K state for one layer and KV head, with explicit retained absolute positions and feature axes.',
      'The paired V state supplies the vectors mixed using attention probabilities. Switching this slice does not duplicate, evict or allocate cached state.',
    ),
    sources: cache,
  },
  V: {
    ...entry(
      'Inspect cached values',
      'Values are the feature vectors combined using attention probabilities to form each head result. This view selects V state for one layer and KV head.',
      'Cached keys and values share their logical layer, batch, KV-head, retained-position and feature indexing. The displayed slice does not prescribe GPU paging or physical memory layout.',
    ),
    sources: cache,
  },
  'Back through the calculation': entry(
    'Return through operand history',
    'Return to the tensor cell you inspected before following an operand link. This lets you trace a contribution backward and then recover the result it contributed to.',
    'This is microscope navigation within the current captured run; it does not reverse model arithmetic or alter the playback trace.',
  ),
  'Run to this operation': entry(
    'Reveal this recorded operation',
    'Seek to the point where the selected operation’s outputs have been computed, so its actual tensors and scalar recipes become available for inspection.',
    'Later activations remain unrevealed until their own steps. This is playback over captured inference, rather than an instruction to run arbitrary imported graph equations.',
  ),
  'Remove bookmark': entry(
    'Remove this saved operation',
    'Remove this bookmark and its attached observation from the current study’s saved bookmark list. Other bookmarks and the main notebook remain available.',
    'The operation, captured arrays and model definition are unchanged; you can revisit and bookmark the operation again from its trace.',
  ),
  'Reload checkpoint': helpCatalog['Load trained model'],
  'Generate another trace': helpCatalog['Generate & inspect'],
  'Stop local computation': helpCatalog['Cancel generation'],
})

export function namedHelp(name: string): HelpContent | undefined {
  if (helpCatalog[name]) return helpCatalog[name]
  if (name.startsWith('Load trained model')) return helpCatalog['Load trained model']
  if (name.startsWith('KV cache ·')) return helpCatalog['KV cache retention']
  if (name.startsWith('The field guide')) return helpCatalog['Getting oriented']
  if (/^Tensor .+ slice$/.test(name))
    return entry(
      'Tensor axis slice',
      'Choose a single index of this named tensor axis while displaying the remaining row and column axes. Layer, batch, head and position axes have different meanings; their order is explicit in the tensor header.',
      'Changing the slice selects existing captured values. It does not reshape storage, merge heads or recompute activations.',
    )
  if (/^Numerical input token \d+$/.test(name))
    return entry(
      'Input vocabulary ID',
      'Set one input ID for the untrained numerical reference, or inspect a read-only ID from the trained tokenizer. Each ID gathers a learned or seeded embedding row before entering the decoder.',
      'The trained path keeps IDs tied to its text/tokenizer; edit its prompt and generate again to change that input.',
    )
  if (/^Previous tensor|^Next tensor/.test(name))
    return entry(
      'Page through the tensor slice',
      'Move the visible window by up to eight rows or columns without changing the underlying tensor. Axis labels retain their original position IDs.',
      'Select any visible cell to inspect its exact coordinate, stored value and operation recipe. Paging is a viewing action, not another inference call.',
    )
  if (/^Token \d+:/.test(name))
    return entry(
      'Inspect a token',
      name,
      'Select this token’s byte span and vocabulary ID in the teaching tokenizer. The token count also supplies the architectural sequence dimensions; no trained weights execute in this view.',
    )
  if (/^(Head \d+, query|Query \d+, key)/.test(name))
    return {
      ...entry(
        'Attention query/key cell',
        name,
        'Rows identify query positions and columns identify key positions. A masked entry is excluded before softmax; each visible probability row distributes weight over its permitted keys. The surrounding study identifies whether values are synthetic or from an actual numerical trace.',
      ),
      sources: attention,
    }
  if (/^Cell /.test(name))
    return entry(
      'Inspect the selected tensor entry',
      name,
      'Click to inspect the original tensor coordinate and follow the scalar calculation back to its operands. The compact grid rounds values for display; the stored scalar and recipe remain available in the inspector.',
    )
  if (/^Q\d+$/.test(name))
    return {
      ...entry(
        'Select a query head',
        `Inspect ${name} and its assigned K/V sharing group. Query heads have independent attention paths even when their keys and values are shared.`,
        'The surrounding workbench labels the provenance of its attention values. Selecting a head does not duplicate cached state.',
      ),
      sources: attention,
    }
  if (/^Explore /.test(name))
    return entry(
      'Explore inside this component',
      'Open the detailed study referenced by this component to inspect its internal operations, tensor dimensions, equations and sources.',
      'Each detailed graph has its own saved layout and camera. Opening a component changes the view rather than replacing its authored definition.',
    )
  if (/Switch to (light|dark) mode|Toggle numerical theme/.test(name))
    return entry(
      'Change appearance',
      'Switch between the dark neon palette and a light reading palette. Text, category shapes and patterns supplement colour so the diagrams remain usable with red–green colour blindness.',
      'Theme is saved with the workspace and applies to diagrams, inspectors and explanatory tooltips.',
    )
  if (/^(Find a numerical|Search model)/.test(name))
    return entry(
      'Filter this study',
      'Search the available named operations or parameter tensors. Choosing a match opens its captured tensor or focuses its corresponding graph component.',
      'Filtering changes the visible choices, not the model, arrays or saved notebook.',
    )
  if (/^Note for /.test(name))
    return entry(
      'Bookmark observation',
      'Add a personal explanation or question to this bookmarked operation. The bookmark retains its decoder-call and operation identity.',
      'Notes save with this independent study’s workspace settings, rather than modifying the model’s equations or learned weights.',
    )
  return undefined
}
