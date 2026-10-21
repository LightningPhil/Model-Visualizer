import { parseModelConfig, type ModelConfig } from './schema'

/** The foundations profile partitions D into whole heads with RoPE pairs. */
export function headPartitions(dModel: number, attention?: ModelConfig['attention']) {
  if (!Number.isInteger(dModel) || dModel < 8 || dModel > 65536) return []
  return Array.from({ length: Math.min(256, dModel / 2) }, (_, i) => i + 1)
    .map((nQueryHeads) => ({ nQueryHeads, headDim: dModel / nQueryHeads }))
    .filter(
      (p) =>
        Number.isInteger(p.headDim) &&
        p.headDim >= 2 &&
        p.headDim <= 1024 &&
        p.headDim % 2 === 0 &&
        (!attention || kvHeadChoices(p.nQueryHeads, attention).length > 0),
    )
}

export function kvHeadChoices(heads: number, attention: ModelConfig['attention']): number[] {
  if (attention === 'MHA') return [heads]
  if (attention === 'MQA') return [1]
  return Array.from({ length: Math.max(0, heads - 2) }, (_, i) => i + 2).filter(
    (n) => heads % n === 0,
  )
}

function nearest(values: number[], target: number) {
  return [...values].sort((a, b) => Math.abs(a - target) - Math.abs(b - target) || a - b)[0]
}

/** A single atomic edit: D stays fixed when changing either head partition.
 * Width/architecture changes retain the current partition wherever possible.
 * An unsupported requested value throws before any document state changes. */
export function updateTeachingConfig(
  config: ModelConfig,
  change: Partial<ModelConfig>,
): ModelConfig {
  const next = { ...config, ...change }
  const partitions = headPartitions(next.dModel, next.attention)
  if (change.nQueryHeads !== undefined && change.headDim === undefined)
    next.headDim = next.dModel / next.nQueryHeads
  else if (change.headDim !== undefined && change.nQueryHeads === undefined)
    next.nQueryHeads = next.dModel / next.headDim
  else if (
    (change.dModel !== undefined || change.attention !== undefined) &&
    change.nQueryHeads === undefined &&
    change.headDim === undefined
  ) {
    const heads = nearest(
      partitions.map((p) => p.nQueryHeads),
      config.nQueryHeads,
    )
    if (heads === undefined)
      throw new Error('This width has no valid head partition for that attention type.')
    next.nQueryHeads = heads
    next.headDim = next.dModel / heads
  }
  const kvHeads = kvHeadChoices(next.nQueryHeads, next.attention)
  if (change.nKvHeads === undefined && !kvHeads.includes(next.nKvHeads))
    next.nKvHeads = nearest(kvHeads, config.nKvHeads)
  // Reducing context also reduces its associated retention bounds.
  if (change.contextLength !== undefined && next.kvCache && next.kvCache.mode !== 'full')
    next.kvCache = {
      ...next.kvCache,
      windowSize: Math.min(next.kvCache.windowSize, next.contextLength),
      ...(next.kvCache.mode === 'sparse'
        ? { stride: Math.min(next.kvCache.stride, next.contextLength) }
        : {}),
    }
  return parseModelConfig(next)
}
