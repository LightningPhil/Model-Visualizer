import { describe, expect, it } from 'vitest'
import katex from 'katex'
import {
  executeTeachingModel,
  defaultNumericalSettings,
  numericalAdapter,
  teachingConfig,
  type NumericalSettings,
} from '../src/model/numerics/engine'
import { makeTensor, matmul, rmsnorm, rope, softmax } from '../src/model/numerics/tensor'
import { explainCell, valueLabel } from '../src/model/numerics/explain'
import { canAttend } from '../src/model/attention'
import { numericalExampleFor, resolveNumericalExample } from '../src/model/numerics/examples'
import { inspectConnection } from '../src/model/numerics/inspection'
import { expandWholeModel } from '../src/model/whole-model'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import {
  loadNumericalSession,
  numericalGraph,
  exportNumericalTrace,
  NUMERICAL_SESSION_KEY,
  sessionSchema,
} from '../src/model/numerics/session'

const tensor = (id: string, shape: number[], values: number[]) =>
  makeTensor(
    id,
    id,
    shape,
    shape.map((_, i) => `axis${i}`),
    values,
    { op: 'input' },
  )
describe('small reference tensor operations', () => {
  it('high-precision labels round-trip stored Float32 values, including tiny magnitudes', () => {
    for (const n of [1.2345678e-6, 0.000123456789, -0.01963182, 3.14159265, 1.234567e10]) {
      const value = Math.fround(n)
      expect(Math.fround(Number(valueLabel(value, 9)))).toBe(value)
    }
    expect(valueLabel(2147483647, 8)).toBe('2147483647')
  })
  it('contracts a row and column, rather than multiplying matching cells', () => {
    const a = tensor('a', [2, 3], [1, 2, 3, 4, 5, 6]),
      b = tensor('b', [3, 2], [7, 8, 9, 10, 11, 12]),
      c = matmul('c', 'result', a, b)
    expect(Array.from(c.data)).toEqual([58, 64, 139, 154])
    const proof = explainCell({ tensors: { a, b, c } }, c, 3)
    expect(proof.terms.map((t) => t.result)).toEqual([32, 50, 72])
    expect(proof.result).toBe(154)
  })
  it('normalizes by RMS without subtracting the mean', () => {
    const x = tensor('x', [1, 2], [3, 4]),
      gain = tensor('g', [2], [1, 2]),
      out = rmsnorm('n', x, gain, 1e-5)
    expect(out.data[0]).toBeCloseTo(3 / Math.sqrt(12.5 + 1e-5), 6)
    expect(out.data[1]).toBeCloseTo(8 / Math.sqrt(12.5 + 1e-5), 6)
  })
  it('rotates pairs at absolute positions and preserves their norm to FP32 precision', () => {
    const x = tensor('x', [1, 1, 2, 4], [1, 2, 3, 4, 1, 2, 3, 4]),
      r = rope('r', x, [0, 17])
    expect(Array.from(r.data.slice(0, 4))).toEqual([1, 2, 3, 4])
    expect(r.data[4]).toBeCloseTo(Math.cos(17) - 2 * Math.sin(17), 6)
    expect(r.data[4] ** 2 + r.data[5] ** 2).toBeCloseTo(5, 5)
  })
  it('handles large logits and excluded keys without overflow', () => {
    const x = tensor('x', [1, 3], [1000, -Infinity, 1001]),
      p = softmax('p', x)
    expect(p.data[1]).toBe(0)
    expect(p.data[0]).toBeCloseTo(1 / (1 + Math.E), 6)
    expect(Array.from(p.data).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  })
})
describe('numerical transformer traces', () => {
  it('inspects actual shared connection operands without mixing decoder calls or future tensors', () => {
    const trace = executeTeachingModel(defaultNumericalSettings),
      prediction = trace.frames.find((f) => f.call === 0 && f.nodeIds.includes('output.sample'))!
    const selected = inspectConnection(
      trace,
      { source: 'output.sample', target: 'generation.next' },
      prediction.id,
    )
    expect(selected.preferred).toBe('call0/choice')
    const projected = trace.frames.find(
      (f) => f.call === 0 && f.nodeIds.includes('layer-0.attn.q'),
    )!
    expect(
      inspectConnection(trace, { source: 'layer-0.attn.x', target: 'layer-0.attn.q' }, projected.id)
        .preferred,
    ).toBe('call0/l0/norm1')
    const next = trace.frames.find((f) => f.call === 1 && f.nodeIds.includes('tokens.lookup'))!
    const mixed = inspectConnection(
      trace,
      { source: 'tokens.lookup', target: 'layer-0.norm1' },
      next.id,
    )
    expect(mixed.available).toEqual(['call1/embedding'])
    for (const id of selected.available)
      expect(trace.tensors[id].createdAt).toBeLessThanOrEqual(prediction.id)
  })
  it('owns its input snapshots and rejects invalid profile state', () => {
    const settings = structuredClone(defaultNumericalSettings),
      config = teachingConfig(settings),
      options = {
        settings,
        promptTokens: 4,
        newTokens: 3,
        mtp: false,
        scenario: 'accept' as const,
        splitHeads: true,
      },
      trace = numericalAdapter.run(config, options)
    settings.inputIds[0] = 99
    config.nLayers = 4
    expect(trace.settings.inputIds[0]).toBe(7)
    expect(trace.options.promptTokens).toBe(4)
    expect(trace.config.nLayers).toBe(2)
    expect(() => executeTeachingModel({ ...defaultNumericalSettings, temperature: NaN })).toThrow()
    expect(() =>
      executeTeachingModel({
        ...defaultNumericalSettings,
        kvCache: { mode: 'sliding-window', windowSize: 128 },
      }),
    ).toThrow(/capacity/)
    expect(() =>
      numericalAdapter.run({ ...teachingConfig(defaultNumericalSettings), nKvHeads: 3 }, options),
    ).toThrow(/divisible/)
    expect(() =>
      numericalAdapter.run(teachingConfig(defaultNumericalSettings), { ...options, mtp: true }),
    ).toThrow(/autoregressive/)
    expect(() =>
      numericalAdapter.run(
        {
          ...teachingConfig(defaultNumericalSettings),
          kvCache: { mode: 'sliding-window', windowSize: 2 },
        },
        options,
      ),
    ).toThrow(/cache policy/)
  })
  it('runs actual reproducible math with seeded weights, preserving all snapshots', () => {
    const a = executeTeachingModel(defaultNumericalSettings),
      b = executeTeachingModel(defaultNumericalSettings)
    expect(a.provenance.kind).toBe('numerical')
    expect(a.outputIds).toEqual(b.outputIds)
    expect(a.outputIds).toHaveLength(3)
    expect(a.frames[0].cachePositions).toEqual([[], []])
    expect(a.frames.at(-1)!.generated).toBe(3)
    expect(Array.from(a.tensors['call0/logits'].data)).toEqual(
      Array.from(b.tensors['call0/logits'].data),
    )
    const changed = executeTeachingModel({ ...defaultNumericalSettings, seed: 38 })
    expect(Array.from(changed.tensors['call0/logits'].data)).not.toEqual(
      Array.from(a.tensors['call0/logits'].data),
    )
    for (const t of Object.values(a.tensors)) {
      expect(t.data.length).toBe(t.shape.reduce((s, n) => s * n, 1))
      if (t.recipe.op !== 'mask') expect(Array.from(t.data).every(Number.isFinite)).toBe(true)
    }
  })
  for (const attention of ['MHA', 'GQA', 'MQA'] as const)
    for (const kvCache of [
      { mode: 'full' },
      { mode: 'sliding-window', windowSize: 2 },
      { mode: 'sparse', windowSize: 2, stride: 3 },
    ] as const)
      it(`${attention} / ${kvCache.mode}: cached decode agrees with a full causal forward pass`, () => {
        const settings: NumericalSettings = {
          ...defaultNumericalSettings,
          generate: 2,
          attention,
          kvCache,
        }
        const incremental = executeTeachingModel(settings)
        const full = executeTeachingModel({
          ...settings,
          inputIds: [...settings.inputIds, incremental.outputIds[0]],
          generate: 1,
        })
        const actual = incremental.tensors['call1/logits'].data,
          expected = full.tensors['call0/logits'].data.slice(-256)
        actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 5))
        const weights = incremental.tensors['call0/l0/attention-0']
        for (let q = 0; q < settings.inputIds.length; q++)
          for (let k = 0; k < settings.inputIds.length; k++)
            if (!canAttend(incremental.config, q, k))
              expect(weights.data[q * settings.inputIds.length + k]).toBe(0)
      })
  it('does not allow future input tokens to alter earlier causal states', () => {
    const a = executeTeachingModel({ ...defaultNumericalSettings, generate: 1 }),
      b = executeTeachingModel({
        ...defaultNumericalSettings,
        generate: 1,
        inputIds: [7, 19, 31, 99],
      })
    expect(Array.from(a.tensors['call0/l1/residual2'].data.slice(0, 48))).toEqual(
      Array.from(b.tensors['call0/l1/residual2'].data.slice(0, 48)),
    )
  })
  it('every scalar proof renders and recomputes its recorded result', () => {
    const trace = executeTeachingModel({ ...defaultNumericalSettings, generate: 1 })
    for (const t of Object.values(trace.tensors).filter((t) => t.data.length)) {
      for (const index of [...new Set([0, Math.floor(t.data.length / 2), t.data.length - 1])]) {
        const proof = explainCell(trace, t, index)
        expect(proof.result).toBe(t.data[index])
        for (const parent of [
          ...proof.parents,
          ...proof.terms.flatMap((term) => [term.source, ...(term.second ? [term.second] : [])]),
        ]) {
          const source = trace.tensors[parent.tensor]
          expect(source, `${t.id}: missing source ${parent.tensor}`).toBeDefined()
          expect(parent.index).toBeGreaterThanOrEqual(0)
          expect(parent.index).toBeLessThan(source.data.length)
          expect(source.createdAt).toBeLessThanOrEqual(t.createdAt)
        }
        expect(() =>
          katex.renderToString(proof.tex, { throwOnError: true, trust: false }),
        ).not.toThrow()
        if (['matmul', 'rope'].includes(t.recipe.op))
          expect(Math.fround(proof.terms.reduce((sum, p) => sum + p.result, 0))).toBe(t.data[index])
      }
    }
  })
  it('retains absolute positions while earlier prefill queries use transient keys', () => {
    const trace = executeTeachingModel({
      ...defaultNumericalSettings,
      kvCache: { mode: 'sliding-window', windowSize: 2 },
    })
    expect(trace.tensors['call0/l0/state-k'].positions).toEqual({ R: [2, 3] })
    expect(trace.tensors['call0/l0/working-k'].positions).toEqual({ U: [0, 1, 2, 3] })
    expect(trace.tensors['call0/l0/attention-0'].data[0]).toBe(1)
    expect(trace.tensors['call1/l0/state-k'].positions).toEqual({ R: [3, 4] })
    expect(trace.tensors['call1/l1/residual2'].positions).toEqual({ T: [4] })
    expect(trace.frames.at(-1)!.confirmed).toHaveLength(7)
    expect(trace.frames.at(-1)!.cachePositions).toEqual([
      [4, 5],
      [4, 5],
    ])
    expect(trace.frames.at(-1)!.cacheLengths).toEqual([6, 6])
  })
  it('samples deterministically from the recorded FP32 categorical distribution', () => {
    const settings = { ...defaultNumericalSettings, sampling: 'sample' as const, generate: 8 }
    const a = executeTeachingModel(settings),
      b = executeTeachingModel(settings)
    expect(a.outputIds).toEqual(b.outputIds)
    for (let call = 0; call < 8; call++) {
      const choice = a.tensors[`call${call}/choice`],
        r = choice.recipe
      if (r.op !== 'choice') throw Error('Expected choice')
      expect(r.method).toBe('sample')
      const p = a.tensors[r.input],
        width = p.shape.at(-1)!,
        row = Array.from(p.data.slice(r.row * width, (r.row + 1) * width)),
        sum = row.reduce((s, v) => s + v, 0)
      const lower = row.slice(0, r.token).reduce((s, v) => s + v, 0) / sum,
        upper = lower + row[r.token] / sum
      expect(r.draw!).toBeGreaterThanOrEqual(lower)
      expect(r.draw!).toBeLessThanOrEqual(upper)
      expect(explainCell(a, choice, 0).tex).toContain('Categorical')
    }
  })
  it('keeps a full trace serializable without erasing excluded-score semantics', () => {
    const trace = executeTeachingModel(defaultNumericalSettings),
      exported = JSON.parse(exportNumericalTrace(trace))
    expect(exported.tensors['call0/l0/masked-0'].data[1]).toBe('-Infinity')
    expect(exported.tensors['call0/l0/scores-0'].data).toEqual(
      Array.from(trace.tensors['call0/l0/scores-0'].data),
    )
    expect(exported.provenance.kind).toBe('numerical')
    expect(exported.config.dtypeBytes).toBe(4)
  })
})

describe('numerical laboratory extension and authored graph reuse', () => {
  const doc = parseDocument(starter)
  it('links ordinary and expanded components to valid numerical teaching examples', () => {
    const config = teachingConfig(defaultNumericalSettings),
      available = new Set(
        numericalGraph(doc, defaultNumericalSettings, {
          direction: 'RIGHT',
          positions: {},
        }).nodes.map((n) => n.id),
      )
    for (const source of doc.graphs) {
      const graph = expandWholeModel(doc, source)
      for (const node of graph.nodes) {
        const example = numericalExampleFor(graph.id, node, 5)
        if (example)
          expect(
            available.has(resolveNumericalExample(example, config)),
            `${node.id} → ${example}`,
          ).toBe(true)
      }
    }
    expect(
      numericalExampleFor('memory', doc.graphs.find((g) => g.id === 'memory')!.nodes[0]),
    ).toBeUndefined()
    const whole = expandWholeModel(
      doc,
      doc.graphs.find((g) => g.id === 'whole-model')!,
    )
    expect(
      numericalExampleFor(
        whole.id,
        whole.nodes.find((n) => n.id === 'layer-5.attn.head-7'),
      ),
    ).toBe('layer-5.attn.head-7')
    expect(
      numericalExampleFor(
        whole.id,
        whole.nodes.find((n) => n.id === 'layer-5.attn.cachek-1'),
      ),
    ).toBe('layer-5.attn.cachek-1')
    expect(
      resolveNumericalExample('layer-5.attn.cachek-7', { nLayers: 2, nQueryHeads: 4, nKvHeads: 1 }),
    ).toBe('layer-1.attn.cachek-0')
  })
  it('compiles the authored templates without mutating model content or layouts', () => {
    const before = JSON.stringify(doc),
      graph = numericalGraph(doc, defaultNumericalSettings, { direction: 'RIGHT', positions: {} }),
      trace = executeTeachingModel(defaultNumericalSettings),
      ids = new Set(graph.nodes.map((n) => n.id))
    expect(JSON.stringify(doc)).toBe(before)
    expect(ids.size).toBe(graph.nodes.length)
    for (const frame of trace.frames) {
      for (const id of frame.nodeIds) expect(ids.has(id), id).toBe(true)
      for (const id of [...frame.inputs, ...frame.outputs, ...Object.values(frame.bindings).flat()])
        expect(trace.tensors[id].createdAt).toBeLessThanOrEqual(frame.id)
    }
    expect(graph.nodes.some((n) => n.id.includes('mtp') || n.id.includes('memory.'))).toBe(false)
    expect(graph.nodes.find((n) => n.id === 'generation.commit')!.subtitle).toBe(
      'Append one output token',
    )
  })
  it('saves only settings, geometry and notes while preserving other metadata', () => {
    const session = loadNumericalSession(doc).session
    session.notes = 'Observe query head 3.'
    session.layout.positions['layer-0.attn.scores-3'] = { x: 120, y: 240 }
    session.bookmarks.push({
      id: '0/score3',
      title: 'Head 3',
      node: 'layer-0.attn.scores-3',
      call: 0,
      note: 'Compare group 1.',
    })
    const saved = parseDocument({
      ...doc,
      extensions: {
        ...doc.extensions,
        'someone-else.example': { valuable: 'keep' },
        [NUMERICAL_SESSION_KEY]: session,
      },
    })
    expect(sessionSchema.parse(loadNumericalSession(saved).session)).toEqual(session)
    expect(saved.extensions!['someone-else.example']).toEqual({ valuable: 'keep' })
    expect(saved.config).toEqual(doc.config)
    expect(JSON.stringify(session)).not.toContain('weights/')
  })
  it('preserves unknown laboratory versions and unsupported metadata without overwriting them', () => {
    for (const raw of [
      { version: 2, notes: 'future schema', settings: { x: 1 } },
      { ...loadNumericalSession(doc).session, unknownField: 'must survive' },
    ]) {
      const saved = { ...doc, extensions: { ...doc.extensions, [NUMERICAL_SESSION_KEY]: raw } },
        before = JSON.stringify(saved)
      expect(loadNumericalSession(saved).writable).toBe(false)
      expect(JSON.stringify(saved)).toBe(before)
    }
  })
  it('opens an out-of-profile saved experiment temporarily instead of crashing or erasing it', () => {
    const raw = loadNumericalSession(doc).session
    raw.settings.kvCache = { mode: 'sparse', windowSize: 128, stride: 3 }
    const saved = { ...doc, extensions: { ...doc.extensions, [NUMERICAL_SESSION_KEY]: raw } }
    expect(loadNumericalSession(saved).writable).toBe(false)
    expect(() => executeTeachingModel(loadNumericalSession(saved).session.settings)).not.toThrow()
    expect(saved.extensions[NUMERICAL_SESSION_KEY]).toBe(raw)
  })
})
