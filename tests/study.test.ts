import { describe, expect, it } from 'vitest'
import { readFile } from 'node:fs/promises'
import {
  initialParameters,
  networkLoss,
  trainingData,
  trainNetwork,
  defaultTraining,
} from '../src/model/study/training'
import { investigationResult } from '../src/model/study/investigations'
import { newStudyRecord, parseStudyLibrary } from '../src/model/study/records'
import { compareTraces } from '../src/model/study/comparison'
import {
  defaultNumericalSettings,
  executeTeachingModel,
  numericalAdapter,
} from '../src/model/numerics/engine'
import {
  parseStoriesCheckpoint,
  StoriesTokenizer,
  runStories,
  defaultTrainedSettings,
} from '../src/model/checkpoints/stories'
import { explainCell } from '../src/model/numerics/explain'
import { exportNumericalTrace } from '../src/model/numerics/session'

describe('inspectable network training', () => {
  it('matches a central finite difference independently for every parameter', () => {
    const p = initialParameters(7),
      data = trainingData('noise'),
      analytic = networkLoss(p, data).gradient,
      h = 1e-5
    for (let i = 0; i < p.length; i++) {
      const a = [...p],
        b = [...p]
      a[i] += h
      b[i] -= h
      expect(analytic[i]).toBeCloseTo(
        (networkLoss(a, data).loss - networkLoss(b, data).loss) / (2 * h),
        8,
      )
    }
  })
  it('updates simultaneously, corrects Adam bias, and separates AdamW shrinkage', () => {
    const p = initialParameters(7),
      g = networkLoss(p, trainingData('curve')).gradient
    const sgd = trainNetwork({ ...defaultTraining, optimizer: 'sgd', steps: 1 }),
      adam = trainNetwork({ ...defaultTraining, optimizer: 'adam', steps: 1 }),
      aw = trainNetwork({ ...defaultTraining, optimizer: 'adamw', steps: 1 })
    p.forEach((value, i) => {
      expect(sgd.parameters[i]).toBeCloseTo(value - defaultTraining.rate * g[i], 13)
      expect(adam.parameters[i]).toBeCloseTo(
        value - (defaultTraining.rate * g[i]) / (Math.abs(g[i]) + 1e-8),
        12,
      )
      expect(aw.parameters[i] - adam.parameters[i]).toBeCloseTo(
        -defaultTraining.rate * defaultTraining.decay * value,
        12,
      )
    })
  })
  it('reconstructs a deterministic trajectory and fits the visible observations', () => {
    const recipe = { ...defaultTraining, steps: 250 },
      a = trainNetwork(recipe),
      b = trainNetwork(recipe)
    expect(a.parameters).toEqual(b.parameters)
    expect(a.history.at(-1)!.train).toBeLessThan(a.history[0].train / 10)
    expect(a.completed).toBe(250)
    expect(trainNetwork({ ...recipe, task: 'extrapolate' }).parameters).not.toEqual(a.parameters)
    expect(() => trainNetwork({ ...recipe, steps: 1001 })).toThrow()
    expect(() => trainNetwork({ ...recipe, rate: NaN })).toThrow()
  })
})
describe('controlled investigations and portable records', () => {
  it('keeps attention fixed under value changes and computes the output delta', () => {
    const a = investigationResult('values', 0, 1, 1),
      b = investigationResult('values', 3, 1, 1)
    expect(b.probabilities).toEqual(a.probabilities)
    expect(b.current[0] - a.current[0]).toBeCloseTo(3 * b.probabilities[1], 12)
    const shared = investigationResult('sharing', 0, -1, 1)
    expect(shared.probabilities).not.toEqual(shared.baseProbabilities)
    const m = investigationResult('memory', 0, 1, 1024),
      twice = investigationResult('memory', 0, 1, 2048)
    expect(m.baseline[0] / m.current[0]).toBe(4)
    expect(twice.current[0]).toBe(m.current[0] * 2)
  })
  it('round-trips setup, notes, selected evidence and extension metadata', () => {
    const record = newStudyRecord({
      title: 'A test',
      question: 'why',
      prediction: 'same weights',
      observation: 'changed values',
      conclusion: 'conditional',
      recipe: { kind: 'training', settings: { ...defaultTraining, steps: 25 } },
      evidence: { scope: 'synthetic', summary: 'captured', values: [1, '-Infinity'] },
      extensions: { example: 'retained' },
    })
    const parsed = parseStudyLibrary(
      JSON.stringify({ version: 1, records: [record], extra: { retained: true } }),
    )
    expect(parsed.records[0]).toEqual(record)
    expect(parsed.extra).toEqual({ retained: true })
    expect(() => parseStudyLibrary(JSON.stringify({ version: 2, records: [record] }))).toThrow(
      'Unsupported',
    )
    expect(() =>
      parseStudyLibrary(JSON.stringify({ version: 1, records: [record, record] })),
    ).toThrow('Duplicate')
    expect(() =>
      parseStudyLibrary(
        JSON.stringify({
          version: 1,
          records: [
            {
              ...record,
              recipe: { kind: 'training', settings: { ...defaultTraining, steps: 999999 } },
            },
          ],
        }),
      ),
    ).toThrow('Unsupported')
  })
})
describe('aligned run differences and actual head interventions', () => {
  const base = executeTeachingModel({ ...defaultNumericalSettings, generate: 1 })
  it('finds the first changed execution output and refuses changed head parameterization', () => {
    expect(compareTraces(base, base).rows).toHaveLength(0)
    const changed = executeTeachingModel({
      ...base.settings,
      inputIds: [8, ...base.settings.inputIds.slice(1)],
    })
    const result = compareTraces(changed, base)
    expect(result.first?.tensor).toBe('call0/ids')
    expect(result.samePrefix).toBe(false)
    expect(
      compareTraces(executeTeachingModel({ ...base.settings, attention: 'MHA' }), base).issue,
    ).toContain('parameterization')
  })
  it('excludes mask sentinels from RMS rather than treating them as measured zeros', () => {
    const changed = structuredClone(base),
      id = 'call0/l0/masked-0',
      tensor = changed.tensors[id]
    tensor.data[0] += 2
    const row = compareTraces(changed, base).rows.find((r) => r.tensor === id)!,
      finite = Array.from(tensor.data).filter(Number.isFinite).length
    expect(row.total).toBe(finite)
    expect(finite).toBe(10)
    expect(row.rms).toBeCloseTo(
      Math.abs(tensor.data[0] - base.tensors[id].data[0]) / Math.sqrt(finite),
      12,
    )
  })
  it('changes exactly the chosen row before recomputing downstream operations', () => {
    const patch = {
      mode: 'zero' as const,
      layer: 0,
      head: 1,
      position: base.settings.inputIds.length - 1,
      source: 'unit zero ablation',
    }
    const changed = numericalAdapter.run(base.config, {
      ...base.options,
      settings: base.settings,
      intervention: patch,
    })
    const before = base.tensors['call0/l0/head-1'],
      after = changed.tensors[before.id],
      start = (base.settings.inputIds.length - 1) * base.config.headDim
    expect(Array.from(after.data.slice(0, start))).toEqual(Array.from(before.data.slice(0, start)))
    expect(Array.from(after.data.slice(start))).toEqual([0, 0, 0, 0])
    expect(changed.tensors['call0/l0/attention-1'].data).toEqual(
      base.tensors['call0/l0/attention-1'].data,
    )
    expect(compareTraces(changed, base).first?.tensor).toBe(before.id)
    expect(changed.tensors['call0/logits'].data).not.toEqual(base.tensors['call0/logits'].data)
    const proof = explainCell(changed, after, start)
    expect(proof.parents[0].tensor).toContain('before-intervention')
    expect(proof.explanation).toContain('not an original checkpoint activation')
    expect(JSON.parse(exportNumericalTrace(changed)).intervention).toEqual(patch)
    expect(() =>
      numericalAdapter.run(base.config, {
        ...base.options,
        settings: base.settings,
        intervention: { ...patch, head: 9 },
      }),
    ).toThrow('Interventions')
  })
  it('patches a trained checkpoint with its own vector as a no-op, and zero ablation changes the final distribution', async () => {
    const read = async (file: string) => {
      const b = await readFile(`public/models/stories260k/${file}`)
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer
    }
    const weights = parseStoriesCheckpoint(await read('stories260K.bin')),
      tokenizer = new StoriesTokenizer(await read('tok512.bin')),
      settings = { ...defaultTrainedSettings, generate: 1 },
      normal = runStories(weights, tokenizer, settings),
      position = normal.settings.inputIds.length - 1
    const vector = Array.from(
        normal.tensors['call0/l0/head-0'].data.slice(-weights.config.headDim),
      ),
      patch = {
        mode: 'patch' as const,
        layer: 0,
        head: 0,
        position,
        replacement: vector,
        source: 'same-run donor',
      }
    const identical = runStories(weights, tokenizer, settings, undefined, patch),
      zero = runStories(weights, tokenizer, settings, undefined, { ...patch, mode: 'zero' })
    expect(identical.tensors['call0/logits'].data).toEqual(normal.tensors['call0/logits'].data)
    expect(zero.tensors['call0/probabilities'].data).not.toEqual(
      normal.tensors['call0/probabilities'].data,
    )
    expect(zero.checkpoint).toEqual(normal.checkpoint)
    expect(zero.tensors['weights/token embeddings'].data).toEqual(
      normal.tensors['weights/token embeddings'].data,
    )
  })
})
