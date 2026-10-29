/** Hypothetical state accounting, not a kernel, allocator or model conversion.
 * Sources and assumptions live in the attention-comparison concept guide. */
export type MemoryScenario = {
  layers: number
  batch: number
  positions: number
  queryHeads: number
  kvHeads: number
  headWidth: number
  scalarBytes: number
  window: number
  latentWidth: number
  positionalWidth: number
  recurrentFeatures: number
  fullAttentionLayers: number
  quantBits: number
}
export type MemoryEstimate = {
  id: string
  name: string
  bytes: number
  formula: string
  state: string
  tradeoff: string
  article: string
}
export function compareAttentionMemory(s: MemoryScenario): MemoryEstimate[] {
  for (const value of Object.values(s))
    if (!Number.isSafeInteger(value) || value < 0)
      throw new Error('Use bounded integer dimensions.')
  if (
    Object.entries(s).some(([key, value]) => value === 0 && key !== 'fullAttentionLayers') ||
    s.fullAttentionLayers > s.layers ||
    s.kvHeads > s.queryHeads ||
    s.queryHeads % s.kvHeads ||
    ![2, 4, 8].includes(s.quantBits) ||
    ![2, 4].includes(s.scalarBytes)
  )
    throw new Error('Choose compatible heads, layers and precision.')
  const {
    layers: L,
    batch: B,
    positions: S,
    queryHeads: H,
    kvHeads: G,
    headWidth: D,
    scalarBytes: b,
  } = s
  const dense = (heads: number, positions = S, layers = L) =>
    2 * layers * B * heads * positions * D * b
  const recurrent = (layers = L) => layers * B * H * s.recurrentFeatures * (D + 1) * 4
  // Fictional 64-scalar groups per K/V layer/batch/head plane: pad each final
  // group; one FP16 scale + one FP16 offset (4 bytes). This is NOT KIVI.
  const groups = 2 * L * B * G * Math.ceil((S * D) / 64)
  const rows: MemoryEstimate[] = [
    {
      id: 'mha',
      name: 'Full MHA',
      bytes: dense(H),
      formula: '2 × L × B × Hq × S × Dh × bytes',
      state: 'Separate K and V for every query head and retained position.',
      tradeoff: 'Reference full-context head arrangement. Payload grows with sequence length.',
      article: 'head-sharing',
    },
    {
      id: 'gqa',
      name: 'Full GQA',
      bytes: dense(G),
      formula: '2 × L × B × Hkv × S × Dh × bytes',
      state: 'Groups of queries share explicit K/V heads; all positions remain.',
      tradeoff:
        'Fewer independent K/V representations. Requires compatible trained projections; not a free checkpoint conversion.',
      article: 'head-sharing',
    },
    {
      id: 'mqa',
      name: 'Full MQA',
      bytes: dense(1),
      formula: '2 × L × B × S × Dh × bytes',
      state: 'One K/V head per layer and sequence is shared by all query heads.',
      tradeoff:
        'Maximum head sharing in this family; independent query heads still have separate probability rows.',
      article: 'head-sharing',
    },
    {
      id: 'window',
      name: 'Windowed GQA',
      bytes: dense(G, Math.min(S, s.window)),
      formula: '2 × L × B × Hkv × min(S, W) × Dh × bytes',
      state: 'Only the recent window remains directly available in each layer.',
      tradeoff:
        'Bounds persistent state when W is fixed. Changes direct attention reach; multi-layer propagation is not arbitrary exact retrieval.',
      article: 'attention-patterns',
    },
    {
      id: 'mla',
      name: 'MLA-style latent state',
      bytes: L * B * S * (s.latentWidth + s.positionalWidth) * b,
      formula: 'L × B × S × (latent width + positional width) × bytes',
      state:
        'One joint K/V latent and one decoupled positional key per position, following the cited MLA state structure.',
      tradeoff:
        'Hypothetical widths, not a named checkpoint. Up-projections/absorption and positional design need a compatible architecture. State still grows with S.',
      article: 'mla',
    },
    {
      id: 'linear',
      name: 'Kernel linear attention',
      bytes: recurrent(),
      formula: 'L × B × Hq × feature-map width × (Dh + 1) × 4',
      state:
        'Per head: feature-map-by-value accumulator plus feature normalizer, using FP32 state.',
      tradeoff:
        'Fixed recurrent state for this specific normalized kernel formulation. Changes the attention rule; not exact arbitrary softmax or every recurrent architecture.',
      article: 'linear-attention',
    },
    {
      id: 'hybrid',
      name: 'Hybrid full + recurrent',
      bytes: dense(G, S, s.fullAttentionLayers) + recurrent(L - s.fullAttentionLayers),
      formula: 'full-layer GQA payload + recurrent-layer accumulator payload',
      state: `${s.fullAttentionLayers} full-context GQA layers; ${L - s.fullAttentionLayers} kernel-linear layers.`,
      tradeoff:
        'A hypothetical mixture. Any full-context layers keep a sequence-growing term; trained hybrid architectures have their own state contracts.',
      article: 'memory-budget',
    },
    {
      id: 'quantized',
      name: 'Packed GQA + metadata',
      bytes: groups * ((64 * s.quantBits) / 8 + 4),
      formula: 'groups × (64 × bits / 8 + 4 metadata bytes)',
      state: `Fictional ${s.quantBits}-bit groups of 64 scalars, padded per K/V plane, with scale and offset.`,
      tradeoff:
        'Includes illustrative metadata and padding, but no high-precision residual buffer. Not KIVI, a quality prediction or executed quantization.',
      article: 'cache-quantization',
    },
  ]
  if (rows.some((r) => !Number.isSafeInteger(r.bytes)))
    throw new Error('State exceeds safe accounting range.')
  return rows
}
