import { describe, it, expect } from 'vitest'
import { compareAttentionMemory, type MemoryScenario } from '../src/help/wiki/memory-comparison'

const scenario: MemoryScenario = {
  layers: 1,
  batch: 1,
  positions: 8,
  queryHeads: 4,
  kvHeads: 2,
  headWidth: 2,
  scalarBytes: 2,
  window: 2,
  latentWidth: 3,
  positionalWidth: 1,
  recurrentFeatures: 2,
  fullAttentionLayers: 1,
  quantBits: 4,
}
const bytes = (s = scenario) =>
  Object.fromEntries(compareAttentionMemory(s).map((r) => [r.id, r.bytes]))
describe('hypothetical attention-state comparisons', () => {
  it('counts explicit KV, a joint latent, normalized recurrent state and padded packed groups separately', () => {
    expect(bytes()).toEqual({
      mha: 256,
      gqa: 128,
      mqa: 64,
      window: 32,
      mla: 64,
      linear: 96,
      hybrid: 128,
      quantized: 144,
    })
    // Small padded groups can cost MORE than the unpacked scalar payload.
    expect(bytes().quantized).toBeGreaterThan(bytes().gqa)
  })
  it('distinguishes context growth, saturation and fixed state and uses FP32 recurrent accumulation', () => {
    const a = bytes(),
      b = bytes({ ...scenario, positions: 16 })
    expect(b.mha).toBe(2 * a.mha)
    expect(b.mla).toBe(2 * a.mla)
    expect(b.linear).toBe(a.linear)
    expect(b.window).toBe(a.window)
    const fp32 = bytes({ ...scenario, scalarBytes: 4 })
    expect(fp32.gqa).toBe(a.gqa * 2)
    expect(fp32.linear).toBe(a.linear)
    expect(fp32.quantized).toBe(a.quantized)
  })
  it('sums hybrid layers and rejects invalid dimensions or unsafe totals', () => {
    expect(bytes({ ...scenario, layers: 3, fullAttentionLayers: 1 }).hybrid).toBe(128 + 2 * 96)
    expect(bytes({ ...scenario, layers: 3, fullAttentionLayers: 0 }).hybrid).toBe(3 * 96)
    for (const change of [
      { kvHeads: 3 },
      { positions: 0 },
      { layers: 0 },
      { fullAttentionLayers: 2 },
      { quantBits: 3 },
      { positions: 1.5 },
      { positions: NaN },
      { positions: Number.MAX_SAFE_INTEGER },
    ])
      expect(() => bytes({ ...scenario, ...change })).toThrow()
  })
})
