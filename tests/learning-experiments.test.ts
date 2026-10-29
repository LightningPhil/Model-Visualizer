import { describe, expect, it } from 'vitest'
import {
  regressionStep,
  positionComparison,
  featureMixture,
  interventionExample,
  preferenceExample,
  precisionExample,
  compareNormalizations,
  routedExample,
} from '../src/help/wiki/learning-experiments'

describe('local educational experiments', () => {
  it('updates weight and bias simultaneously and checks the gradient independently', () => {
    const s = regressionStep(2, 3, 0, 0, 0.1)
    expect(s.loss).toBe(4.5)
    expect(s.gradientWeight).toBe(-6)
    expect(s.gradientBias).toBe(-3)
    expect(s.nextWeight).toBeCloseTo(0.6)
    expect(s.nextBias).toBeCloseTo(0.3)
    expect(s.nextLoss).toBeCloseTo(1.125)
    expect(s.finiteDifference).toBeCloseTo(s.gradientWeight, 7)
    expect(regressionStep(2, 3, 0, 0, 1).nextLoss).toBeGreaterThan(s.loss)
  })
  it('preserves relative RoPE/ALiBi scores under common position shifts but masks future keys', () => {
    const a = positionComparison(1, 0, 0.5),
      b = positionComparison(2, 1, 0.5)
    expect(a.rotary).toBeCloseTo(b.rotary)
    expect(a.alibi).toBeCloseTo(b.alibi)
    expect(a.learned).not.toBeCloseTo(b.learned)
    expect(a.sinusoidal).not.toBeCloseTo(b.sinusoidal)
    expect(positionComparison(0, 1, 0.5).visible).toBe(false)
  })
  it('separates feature coefficient, interference and basis-invariant readout', () => {
    const a = featureMixture([1, 0.5, 0], Math.PI / 3, 0),
      b = featureMixture([1, 0.5, 0], Math.PI / 3, Math.PI / 2)
    expect(a.readout).toBeCloseTo(1.25)
    expect(a.interference).toBeCloseTo(0.25)
    expect(b.readout).toBeCloseTo(a.readout)
    expect(b.hidden).not.toEqual(a.hidden)
    expect(featureMixture([1, 0.5, 0], Math.PI / 2, 0).interference).toBeCloseTo(0)
  })
  it('makes restoration and deletion effects explicit under two baselines', () => {
    expect(interventionExample('corrupted', 0).logit).toBe(-2)
    expect(interventionExample('patch', 0)).toMatchObject({
      logit: 1,
      deltaClean: 0,
      deltaCorrupted: 3,
    })
    expect(interventionExample('ablate', 0).logit).toBe(3)
    expect(interventionExample('replace', -1).logit).toBe(-5)
  })
  it('distinguishes demonstration and reference-relative preference losses', () => {
    const a = preferenceExample(0, 0, 0.5),
      b = preferenceExample(0, 2, 0.5)
    expect(a.dpoLoss).toBeCloseTo(Math.log(2))
    expect(a.gradientGap).toBe(-0.25)
    expect(a.preferredProbability).toBe(b.preferredProbability)
    expect(a.demonstrationLoss).toBe(b.demonstrationLoss)
    expect(b.dpoLoss).toBeGreaterThan(a.dpoLoss)
    const h = 1e-5
    expect(
      (preferenceExample(h, 0, 0.5).dpoLoss - preferenceExample(-h, 0, 0.5).dpoLoss) / (2 * h),
    ).toBeCloseTo(a.gradientGap)
  })
  it('exposes binary32 cancellation and nonfinite naive softmax without corrupting stable probabilities', () => {
    const r = precisionExample(1e8, 1000)
    expect(r.sequential32).toBe(0)
    expect(r.cancelFirst32).toBe(1)
    expect(r.binary64).toBe(1)
    expect(r.naive.every(Number.isNaN)).toBe(true)
    expect(r.stable.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    expect(precisionExample(1, 0).stable).toEqual(r.stable)
  })
  it('compares centering invariance and constant rows with explicit epsilon', () => {
    const a = compareNormalizations([3, 4], 0),
      b = compareNormalizations([3, 4], 2)
    expect(a.layer).toEqual(b.layer)
    expect(a.rootMeanSquare).not.toEqual(b.rootMeanSquare)
    expect(compareNormalizations([3, 3], 0).layer).toEqual([0, 0])
    expect(a.mean).toBe(3.5)
    expect(a.variance).toBe(0.25)
  })
  it('executes selected toy experts only, breaks ties by ID and normalizes selected weights', () => {
    const a = routedExample(2, [1, 0, -1], 1)
    expect(a.selected).toEqual([0])
    expect(a.outputs).toEqual([2, undefined, undefined])
    expect(a.output).toBe(2)
    const b = routedExample(2, [0, 0, 0], 2)
    expect(b.selected).toEqual([0, 1])
    expect(b.weights).toEqual([0.5, 0.5, 0])
    expect(b.output).toBe(0.5)
    expect(b.activeExpertParameters).toBe(4)
    expect(b.storedExpertParameters).toBe(6)
  })
  it('rejects invalid toy contracts before calculating', () => {
    expect(() => regressionStep(1, 1, 0, 0, -1)).toThrow()
    expect(() => positionComparison(4, 0, 1)).toThrow()
    expect(() => featureMixture([1], 0, 0)).toThrow()
    expect(() => preferenceExample(0, 0, 0)).toThrow()
    expect(() => precisionExample(NaN, 1)).toThrow()
    expect(() => compareNormalizations([], 1)).toThrow()
    expect(() => routedExample(1, [0, 1, 2], 4)).toThrow()
  })
})
