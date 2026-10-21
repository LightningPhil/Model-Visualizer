import { namedHelp, type HelpContent } from '../../help/catalog'

export const HELP_TARGETS =
  'button, select, textarea, input:not([type="hidden"]):not([type="file"]), summary, a[href]'

export function controlName(element: HTMLElement) {
  const aria = element.getAttribute('aria-label')
  if (aria) return aria
  if (
    element instanceof HTMLInputElement ||
    element instanceof HTMLSelectElement ||
    element instanceof HTMLTextAreaElement
  ) {
    const label = element.labels?.[0]?.cloneNode(true) as HTMLElement | undefined
    label?.querySelectorAll('input,select,textarea,button').forEach((node) => node.remove())
    if (label?.textContent?.trim()) return label.textContent.trim().replace(/\s+/g, ' ')
  }
  return element.textContent?.trim().replace(/\s+/g, ' ') ?? ''
}

const describe = (title: string, ...paragraphs: string[]): HelpContent => ({ title, paragraphs })

export function resolveHelp(element: HTMLElement): HelpContent {
  const name = element.dataset.help ?? controlName(element)
  const detail =
    element.dataset.tooltipDetail ??
    element.closest<HTMLLabelElement>('label[data-tooltip-detail]')?.dataset.tooltipDetail ??
    element.querySelector<HTMLElement>('[data-tooltip-detail]')?.dataset.tooltipDetail
  let content = namedHelp(name)
  if (!content && element.matches('.model-switcher'))
    content = namedHelp('Edit model configuration')
  if (!content && element.closest('.numeric-tensor-picker'))
    content = describe(
      'Choose an available recorded tensor',
      name,
      'Select this operation’s recorded output or destination operand to inspect its actual shape, semantic axes and stored cells. Only tensors computed by the current playback step are offered.',
    )
  if (!content && element.closest('.numerical-chapters,.numerical-steps,.numerical-questions'))
    content = describe(
      element.dataset.tooltipTitle ?? name,
      detail ??
        'Navigate to this part of the captured numerical execution and focus its corresponding graph operation.',
      'The inspector follows the recorded operation and its available tensors. This changes the playback location without editing learned parameters or the study layout.',
    )
  if (!content && element.closest('.numeric-probabilities'))
    content = describe(
      'Inspect a vocabulary probability',
      name,
      'Select this vocabulary entry’s probability cell in the current recorded decoder call. The top-eight list is a subset of the full computed vocabulary distribution; its visual bars are relative to the largest displayed probability.',
      'This inspects the captured distribution rather than forcing the model to select this ID. Confirmed text follows the actual recorded token choice.',
    )
  if (!content && element.closest('.observatory-query'))
    content = describe(
      'Choose the attention query row',
      `Inspect query position ${name} across the independent head distributions. Columns remain the recorded absolute key positions, including the visibility mask.`,
      'The selection changes the probability slice being displayed and its microscope coordinate. It does not alter attention values or recompute the model.',
    )
  if (!content && element.closest('.numerical-cache-snapshot'))
    content = describe(
      'Inspect this layer’s retained state',
      detail ?? name,
      'Open the recorded K/V state for this layer with its retained absolute-position map. Earlier selected output IDs enter the cache only when a later forward pass consumes them.',
      'These are logical tensor slices; physical paging, allocator overhead and a nested-table representation are not implied.',
    )
  if (!content && element.matches('.numerical-example-button'))
    content = describe(
      'Inspect the numerical example',
      'Open the corresponding operation in the small untrained numerical reference, with actual arrays, semantic axes and scalar calculation recipes.',
      'This reference is independent of edits to the architecture document and is explicitly labelled untrained. The pinned trained model is a separate workbench.',
    )
  if (!content && element.matches('.import-button')) content = namedHelp('Open')
  if (!content && element.matches('.journey-button')) {
    if (name.startsWith('Whole model'))
      content = describe(
        'Explore the complete model',
        'Expand every main decoder layer, query-head result and KV-head slice into a single interactive canvas, using the authored model templates.',
        'Jump between stages or fit the whole map. Dragging, automatic packing and saved layouts let you study a large architecture without hiding components.',
      )
    else if (name.startsWith('Run simulation'))
      content = describe(
        'Follow the graphical execution',
        'Watch abstract packets move from input through attention, cache updates and output, including scripted multi-token proposals and verification.',
        'This graphical mode does not compute trained activations or generate text. The separate numerical and trained workbenches execute actual reference computations.',
      )
    else if (name.startsWith('Local trained'))
      content = describe(
        'Run a real trained model locally',
        'Load the pinned Stories260K checkpoint and its matching tokenizer, then explicitly generate a bounded trace in a browser worker.',
        'Inspect real weights, per-head attention, activations and cache state in the same flowchart and microscope. No prompt is sent to an inference service.',
      )
    else
      content = describe(
        'Open the tensor laboratory',
        'Execute a small local transformer with seeded, untrained parameters and inspect its actual arrays, shapes and individual scalar calculations.',
        'Play a captured prefill/decode trace, compare head sharing and cache policies, and save an independent experiment notebook. This mode does not claim trained-model activations.',
      )
  }
  if (!content && element.matches('.nav-item'))
    content = describe(
      element.dataset.tooltipTitle ?? name.replace(/\s+\d+$/, ''),
      detail ??
        'Open this architectural study to inspect its operations, connections, tensor dimensions and primary sources.',
      'The view remembers its own component positions and camera. Click a component or connection for a deeper explanation; changing views does not replace your model or notebook.',
    )
  if (!content && element.matches('.model-node'))
    content = describe(
      element.dataset.tooltipTitle ?? name,
      detail ?? 'Inspect this component in the current model.',
      'Select the component to inspect its tensor dimensions, mathematical expressions and associated primary references. The surrounding workbench identifies whether displayed values are illustrative or from actual reference inference.',
    )
  if (!content && element.matches('.edge-label'))
    content = describe(
      element.dataset.tooltipTitle ?? name,
      detail ?? 'Inspect the mathematical operation represented by this connection.',
      'Click to see the carried tensor shape, operation equation and explanation in the inspector. A connection can carry an input, an operation result or state access, as described explicitly in that view.',
    )
  if (!content && detail === 'Rotation coefficient')
    content = describe(
      'Fixed RoPE rotation coefficient',
      'This value is a sine or cosine coefficient for the selected absolute position and feature pair. It is determined by the positional-rotation formula, rather than a learned parameter or another input activation.',
      'The adjacent operand shows the feature being rotated. This constant has no separate tensor operand to open; the cell explanation gives its angle and calculation.',
    )
  if (!content && element.closest('.proof-term,.operand-links,.operation-sketch'))
    content = describe(
      'Trace this scalar to its operand',
      detail ?? name,
      'Open the input tensor at the referenced original coordinate. Follow the dependency chain to see which stored values contributed to the selected result; this does not recompute or edit the model.',
    )
  if (!content && element.closest('.parameter-library'))
    content = describe(
      'Inspect a parameter tensor',
      name,
      'Open its stored values, semantic axes and shape in the tensor microscope. The study identifies whether these are seeded untrained parameters or the pinned checkpoint’s learned weights; tied aliases have explicit provenance.',
    )
  if (!content && element.closest('.search-results'))
    content = describe(
      'Open this component',
      name,
      'Navigate to its graph, center it in the viewport and open its explanation. Search results include mathematical operations and descriptions across study views, not just the current canvas.',
    )
  if (!content && element.matches('summary'))
    content = describe(
      name || 'Explore this section',
      'Expand this section to reveal its detailed controls or explanation. Collapse it to make more space while leaving the underlying study and saved settings intact.',
      'Numerical sections retain their surrounding trace provenance: trained, untrained reference or illustrative teaching content is labelled by the workbench.',
    )
  if (!content && element.matches('a[href]'))
    content = describe(
      name || 'Open the reference',
      'Open this linked document or primary source to investigate its original definitions, equations and implementation details.',
      'Sources describe their own architectures and execution conventions. The inspector explains how this application’s tensor axes and row-vector notation relate to them.',
    )
  if (
    !content &&
    /^(Whole model|Split attention heads|Combine attention heads|Batched attention)/.test(name)
  )
    content = describe(
      'Choose the attention diagram',
      'Switch between the compact batched explanation and explicit per-query-head paths. Split paths preserve the model’s KV-sharing groups and expose the separate score, mask, softmax and value operations.',
      'The whole-model variant has its own saved layout. Splitting is a logical view, not an extra computation or a duplicate allocation of the K/V cache.',
    )
  if (!content && /^(Journey|Flow|Inspect)$/.test(name))
    content = describe(
      `${name} pane`,
      'Choose the learning journey, interactive graph, or tensor inspector on a narrow screen. These panes show different parts of the same recorded run.',
      'Switching panes preserves the selected operation, captured arrays and independent study layout.',
    )
  if (
    !content &&
    /^(Start exploring|Update run|Open numerical run setup|Run setup|Back to journey)/.test(name)
  )
    content = describe(
      'Configure or explore the numerical run',
      'Choose the reference experiment’s seed, input IDs, output budget, head sharing, cache policy and token-selection method, then return to its recorded operations.',
      'The small untrained reference runs locally. The trained checkpoint uses separate fixed architecture and explicit loading/generation controls.',
    )
  if (!content && /^(Simulation timeline|Find a numerical operation)/.test(name))
    content = namedHelp('Numerical execution timeline')
  if (!content && /^(Bookmark|Open numerical lab notes|Notebook|Clear baseline)/.test(name))
    content = namedHelp('Numerical laboratory notes')
  if (!content && /^(Add|Remove) numerical input token/.test(name))
    content = describe(
      'Change the reference input sequence',
      'Add or remove an opaque vocabulary ID in the untrained numerical experiment. The reference trace is recalculated with the resulting input sequence and its causal positions.',
      'The trained path instead derives read-only input IDs from its matching text tokenizer. Editing the untrained input does not change trained prompts or weights.',
    )
  if (!content && /^(Dismiss|Close)/.test(name))
    content = describe(
      'Dismiss this panel or message',
      'Remove this temporary notification or close the current panel so you can continue exploring the model.',
      'The surrounding workspace keeps its automatically saved settings, notes and layouts. Dismissing a notification does not undo the action it reported.',
    )
  if (!content && /^(Show all|Show first)/.test(name))
    content = describe(
      'Expand the scalar calculation',
      'Show every recorded contribution to this cell, or return to a shorter list of the first sixteen terms. The final scalar includes the complete computation in either view.',
      'Click an operand to trace its original tensor coordinate; displaying fewer terms does not truncate the model’s arithmetic.',
    )
  if (!content && /^(Inspect this operand|Rotation coefficient)$/.test(detail ?? ''))
    content = describe(
      'Inspect the calculation operand',
      detail!,
      'Follow this scalar’s original tensor and coordinate in the microscope. A fixed rotation coefficient is a mathematical constant for this position rather than another learned parameter.',
    )
  if (!content)
    content = describe(
      element.dataset.tooltipTitle ?? (name || 'Study control'),
      detail ?? 'Inspect or select this item within the current model study.',
      'Use the surrounding inspector for the selected component’s dimensions, equations, numerical provenance and primary references. The workbench labels whether the current content is architectural, illustrative or an actual recorded computation.',
    )
  const selected =
    element instanceof HTMLSelectElement
      ? element.selectedOptions[0]?.textContent?.trim()
      : undefined
  const disabled = element.matches(':disabled')
  return {
    ...content,
    paragraphs: [
      ...content.paragraphs,
      ...(detail &&
      detail !== name &&
      detail !== content.title &&
      !content.paragraphs.includes(detail) &&
      !element.matches('.nav-item,.edge-label')
        ? [`Detail: ${detail}`]
        : []),
      ...(selected ? [`Current choice: ${selected}.`] : []),
      ...(disabled && detail !== 'Rotation coefficient'
        ? [
            'This control is currently unavailable. Its surrounding setup or current playback position determines when it becomes available.',
          ]
        : []),
    ],
  }
}
