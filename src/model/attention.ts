import type { ModelConfig } from './schema'

export type CachePolicy = NonNullable<ModelConfig['kvCache']>
export const cachePolicy = (config: ModelConfig): CachePolicy => config.kvCache ?? { mode: 'full' }
export const cachePolicyLabel = (config: ModelConfig) => {
  const policy = cachePolicy(config)
  return policy.mode === 'full'
    ? 'Full'
    : policy.mode === 'sliding-window'
      ? 'Sliding window'
      : 'Sparse'
}
export const kvHeadFor = (config: ModelConfig, queryHead: number) =>
  Math.floor(queryHead / (config.nQueryHeads / config.nKvHeads))

/** Logical retained positions after processing [0, length). Sparse is the
 * explicitly defined teaching policy: fixed absolute anchors plus recent keys.
 * No kernel layout, learned selection or physical sparse format is implied. */
export function residentCount(config: ModelConfig, length: number) {
  const p = cachePolicy(config)
  if (p.mode === 'full') return length
  const start = Math.max(0, length - p.windowSize)
  return length - start + (p.mode === 'sparse' ? Math.ceil(start / p.stride) : 0)
}
export function residentPositionAt(config: ModelConfig, length: number, index: number) {
  const p = cachePolicy(config)
  if (p.mode === 'full') return index
  const start = Math.max(0, length - p.windowSize)
  const anchors = p.mode === 'sparse' ? Math.ceil(start / p.stride) : 0
  return p.mode === 'sparse' && index < anchors ? index * p.stride : start + index - anchors
}
export function residentPositions(config: ModelConfig, length: number) {
  return Array.from({ length: residentCount(config, length) }, (_, i) =>
    residentPositionAt(config, length, i),
  )
}
export function canAttend(config: ModelConfig, query: number, key: number) {
  if (key < 0 || key > query) return false
  const p = cachePolicy(config)
  return (
    p.mode === 'full' || key > query - p.windowSize || (p.mode === 'sparse' && key % p.stride === 0)
  )
}

/** Synthetic scores are normalized only over architecturally visible keys. */
export function maskedAttentionWeights(config: ModelConfig, count: number, head: number) {
  return Array.from({ length: count }, (_, q) => {
    const scores = Array.from({ length: count }, (_, k) =>
      canAttend(config, q, k) ? Math.exp(Math.sin((head + 1) * (q + 1) * (k + 2)) * 1.4) : 0,
    )
    const sum = scores.reduce((a, b) => a + b, 0)
    return scores.map((v) => v / sum)
  })
}
