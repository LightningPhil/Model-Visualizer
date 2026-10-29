import { describe, it, expect } from 'vitest'
import {
  adamSequence,
  muonMatrixExample,
  matrixFromFactors,
  type AdamSettings,
  type Matrix2,
} from '../src/help/wiki/optimizer-experiments'
const settings: AdamSettings = {
  initial: 2,
  rate: 0.1,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-8,
  decay: 0.1,
}
describe('optimizer teaching calculations', () => {
  it('exposes raw moments, bias corrections and an epsilon-outside-root first step', () => {
    const [r] = adamSequence([2], settings, 'adam')
    expect(r.first).toBeCloseTo(0.2)
    expect(r.second).toBeCloseTo(0.004)
    expect(r.correctedFirst).toBeCloseTo(2)
    expect(r.correctedSecond).toBeCloseTo(4)
    expect(r.adaptive).toBeCloseTo(0.2 / (2 + 1e-8))
    expect(r.parameter).toBeCloseTo(1.9)
    for (const row of adamSequence([2, 2, 2, 2], settings, 'adam')) {
      expect(row.correctedFirst).toBeCloseTo(2)
      expect(row.correctedSecond).toBeCloseTo(4)
    }
  })
  it('separates cold-start zero gradients, coupled L2 and decoupled weight decay', () => {
    expect(adamSequence([0], settings, 'adam')[0].parameter).toBe(2)
    const coupled = adamSequence([0], settings, 'l2')[0],
      decoupled = adamSequence([0], settings, 'adamw')[0]
    expect(coupled.effectiveGradient).toBe(0.2)
    expect(coupled.parameter).toBeCloseTo(1.9)
    expect(decoupled.first).toBe(0)
    expect(decoupled.second).toBe(0)
    expect(decoupled.shrinkage).toBeCloseTo(0.02)
    expect(decoupled.parameter).toBe(1.98)
  })
  it('keeps historical motion on an explicit zero step and agrees when decay is zero', () => {
    const rows = adamSequence([2, 0], settings, 'adam')
    expect(rows[1].first).toBeCloseTo(0.18)
    expect(rows[1].adaptive).toBeGreaterThan(0)
    const noDecay = { ...settings, decay: 0 }
    expect(adamSequence([2, -2, 0], noDecay, 'adamw')).toEqual(
      adamSequence([2, -2, 0], noDecay, 'adam'),
    )
  })
  it('uses the declared updated-EMA lookahead and Frobenius normalization', () => {
    const r = muonMatrixExample([2, -1], [1, 2], 0.5, 0, true)
    expect(r.norm).toBeCloseTo(Math.sqrt(1.75 ** 2 + 0.25 ** 2))
    expect(r.initial[0]).toBeCloseTo(1.75 / (r.norm + 1e-7))
    const norm = Math.hypot(...r.candidate.flat())
    expect(norm).toBeCloseTo(r.norm)
    expect(muonMatrixExample([2, -1], [1, 2], 0.5, 0, false).momentum).toEqual(
      matrixFromFactors([1.5, 0.5]),
    )
  })
  it('matches an independent full matrix polynomial and does not claim exact unit singular values', () => {
    const multiply = (a: Matrix2, b: Matrix2) =>
      a.map((row) => [0, 1].map((j) => row.reduce((sum, v, k) => sum + v * b[k][j], 0))) as Matrix2
    const r = muonMatrixExample([4, 0.25], [0, 0], 0.9, 1, true)
    const x = r.candidate.map((row) => row.map((v) => v / (r.norm + 1e-7))) as Matrix2
    const transpose: Matrix2 = [
      [x[0][0], x[1][0]],
      [x[0][1], x[1][1]],
    ]
    const a = multiply(x, transpose),
      ax = multiply(a, x),
      aax = multiply(a, ax)
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        expect(r.output[i][j]).toBeCloseTo(
          3.4445 * x[i][j] - 4.775 * ax[i][j] + 2.0315 * aax[i][j],
          10,
        )
    const five = muonMatrixExample([4, 0.25], [0, 0], 0.9, 5, true)
    expect(five.factors.some((s) => Math.abs(Math.abs(s) - 1) > 1e-3)).toBe(true)
  })
  it('preserves zero and rank-deficient directions without inventing an orthogonal extension', () => {
    expect(muonMatrixExample([0, 0], [0, 0], 0.9, 5, true).factors).toEqual([0, 0])
    const r = muonMatrixExample([-2, 0], [0, 0], 0.9, 5, true)
    expect(r.factors[0]).toBeLessThan(0)
    expect(r.factors[1]).toBe(0)
    expect(r.targetMagnitudes).toEqual([1, 0])
  })
  it('rejects invalid beta, epsilon, nonfinite inputs and iteration contracts', () => {
    expect(() => adamSequence([NaN], settings, 'adam')).toThrow()
    expect(() => adamSequence([1], { ...settings, beta1: 1 }, 'adam')).toThrow()
    expect(() => adamSequence([1], { ...settings, epsilon: 0 }, 'adam')).toThrow()
    expect(() => muonMatrixExample([1, 1], [0, 0], 1, 5, true)).toThrow()
    expect(() => muonMatrixExample([1, 1], [0, 0], 0.9, -1, true)).toThrow()
  })
})
