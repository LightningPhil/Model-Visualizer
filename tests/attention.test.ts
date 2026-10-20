import { describe, expect, it } from 'vitest'
import starter from '../examples/transformer-foundations.model.json'
import { cacheBytes, dimensions, parseDocument, type ModelConfig } from '../src/model/schema'
import {
  canAttend,
  kvHeadFor,
  maskedAttentionWeights,
  residentCount,
  residentPositions,
  type CachePolicy,
} from '../src/model/attention'
import { expandWholeModel } from '../src/model/whole-model'
import { graphicalAdapter } from '../src/model/simulation'
import { configurationLayouts } from '../src/model/configuration-layouts'
import { upgradeCachePolicies } from '../src/model/policy-upgrade'

const doc = parseDocument(starter)
const config = doc.config
const policies: CachePolicy[] = [
  { mode: 'full' },
  { mode: 'sliding-window', windowSize: 1 },
  { mode: 'sliding-window', windowSize: 4 },
  { mode: 'sparse', windowSize: 2, stride: 4 },
]
describe('position-aware attention and cache retention', () => {
  it('upgrades earlier foundations shapes and paths while preserving user geometry, notes and metadata', () => {
    const old = structuredClone(doc)
    delete old.extensions!['model-visualizer.cache-policy-version']
    delete old.config.kvCache
    old.ui.notes = 'Do not replace this notebook.'
    const g = old.graphs.find((g) => g.id === 'cache')!
    g.layout.positions.keys = { x: 123, y: 456 }
    g.nodes.find((n) => n.id === 'keys')!.extensions = { 'org.example': { annotation: 'saved' } }
    for (const n of g.nodes)
      for (const t of n.tensors) t.shape = t.shape.map((a) => (a === 'R' ? 'S' : a))
    g.edges = g.edges.filter((e) => !e.id.endsWith('-current'))
    const next = upgradeCachePolicies(old, doc)
    expect(next.ui).toEqual(old.ui)
    const updated = next.graphs.find((x) => x.id === g.id)!
    expect(updated.layout).toEqual(g.layout)
    expect(updated.nodes.find((n) => n.id === 'keys')!.extensions).toEqual(
      g.nodes.find((n) => n.id === 'keys')!.extensions,
    )
    expect(updated.nodes.find((n) => n.id === 'keys')!.tensors[0].shape).toContain('R')
    expect(updated.edges.filter((e) => e.id.endsWith('-current'))).toHaveLength(2)
    expect(upgradeCachePolicies(next, doc)).toBe(next)
    expect(parseDocument(next)).toEqual(next)
  })
  for (const policy of policies)
    it(`${JSON.stringify(policy)}: resident positions, visibility and byte counts agree`, () => {
      const c = { ...config, kvCache: policy }
      for (let length = 0; length <= 64; length++) {
        const expected = Array.from({ length }, (_, i) => i).filter((k) =>
          canAttend(c, length - 1, k),
        )
        expect(residentPositions(c, length)).toEqual(expected)
        expect(residentCount(c, length)).toBe(expected.length)
        expect(cacheBytes(c, length)).toBe(
          2 * c.nLayers * c.nKvHeads * expected.length * c.headDim * c.dtypeBytes,
        )
        // Future eligible keys must be recoverable from retained prefix + new key.
        const working = new Set([...expected, length])
        for (let k = 0; k <= length; k++)
          if (canAttend(c, length, k)) expect(working.has(k)).toBe(true)
      }
      const weights = maskedAttentionWeights(c, 8, 7)
      weights.forEach((row, q) => {
        expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
        row.forEach((w, k) => expect(w > 0).toBe(canAttend(c, q, k)))
      })
    })
  it('uses absolute positions rather than compact slots and distinguishes working from resident state', () => {
    const c: ModelConfig = { ...config, kvCache: { mode: 'sliding-window', windowSize: 4 } }
    expect(residentPositions(c, 10)).toEqual([6, 7, 8, 9])
    expect(canAttend(c, 9, 0)).toBe(false)
    expect(canAttend(c, 9, 6)).toBe(true)
    expect(canAttend(c, 9, 10)).toBe(false)
    expect(dimensions(c, 12)).toMatchObject({ S: 12, R: 4, P: 0, U: 12, T: 12 })
    expect(dimensions(c, 12, true)).toMatchObject({ S: 13, R: 4, P: 4, U: 5, T: 1 })
  })
  it('accepts legacy documents as full and rejects invalid policy contracts', () => {
    const legacy = structuredClone(doc)
    delete legacy.config.kvCache
    expect(cacheBytes(parseDocument(legacy).config, 12)).toBe(cacheBytes(config, 12))
    for (const kvCache of [
      { mode: 'sliding-window', windowSize: 0 },
      { mode: 'sliding-window', windowSize: config.contextLength + 1 },
      { mode: 'sparse', windowSize: 4, stride: 1 },
      { mode: 'sparse', windowSize: 4, stride: 4, unknown: true },
    ])
      expect(() => parseDocument({ ...doc, config: { ...config, kvCache } })).toThrow()
  })
})

describe('explicit independent attention heads', () => {
  it('preserves independent configuration layouts, notes and unrelated extension metadata', () => {
    const saved = structuredClone(doc)
    const graph = saved.graphs.find((g) => g.id === 'attention-heads')!
    graph.layout.positions['query-7'] = { x: 480, y: 210 }
    saved.ui.notes = 'Keep this notebook.'
    saved.extensions = { 'org.example': { keep: [1, 2] } }
    const mqa = configurationLayouts(saved, {
      ...saved,
      config: { ...saved.config, attention: 'MQA', nKvHeads: 1 },
    })
    expect(mqa.graphs.find((g) => g.id === graph.id)!.layout.positions).toEqual({})
    const restored = parseDocument(configurationLayouts(mqa, { ...mqa, config: saved.config }))
    expect(restored.graphs.find((g) => g.id === graph.id)!.layout).toEqual(graph.layout)
    expect(restored.ui.notes).toBe(saved.ui.notes)
    expect(restored.extensions!['org.example']).toEqual(saved.extensions!['org.example'])
    const future = {
      ...saved,
      extensions: { 'model-visualizer.configuration-layouts': { version: 999, custom: true } },
    }
    const next = configurationLayouts(future, {
      ...future,
      config: { ...future.config, attention: 'MQA', nKvHeads: 1 },
    })
    expect(next.extensions).toEqual(future.extensions)
  })
  for (const [attention, nKvHeads] of [
    ['MHA', 8],
    ['GQA', 2],
    ['MQA', 1],
  ] as const)
    it(`${attention}: all head operations use exactly the assigned K/V group`, () => {
      const d = parseDocument({ ...doc, config: { ...config, attention, nKvHeads } })
      const graph = expandWholeModel(
        d,
        d.graphs.find((g) => g.id === 'attention-heads')!,
      )
      for (let h = 0; h < config.nQueryHeads; h++) {
        const kv = kvHeadFor(d.config, h)
        expect(
          graph.edges
            .filter((e) => e.target === `scores-${h}` && e.source.startsWith('cachek-'))
            .map((e) => e.source),
        ).toEqual([`cachek-${kv}`])
        expect(
          graph.edges
            .filter((e) => e.target === `head-${h}` && e.source.startsWith('cachev-'))
            .map((e) => e.source),
        ).toEqual([`cachev-${kv}`])
        expect(
          graph.edges.some((e) => e.source === `scale-${h}` && e.target === `softmax-${h}`),
        ).toBe(true)
        expect(graph.edges.some((e) => e.source === `head-${h}` && e.target === 'concat')).toBe(
          true,
        )
        expect(graph.nodes.find((n) => n.id === `scores-${h}`)!.tensors[0].shape).toEqual([
          'B',
          'T',
          'U',
        ])
      }
      expect(graph.nodes.filter((n) => n.kind === 'attention-head')).toHaveLength(
        config.nQueryHeads,
      )
      expect(graph.nodes.filter((n) => n.id.startsWith('cachek-'))).toHaveLength(nKvHeads)
    })
})

describe('retention through speculative verification', () => {
  for (const policy of policies)
    for (const mtp of [false, true])
      it(`${policy.mode} / ${mtp ? 'MTP' : 'AR'}: checkpoints survive trials and commit the correct retained positions`, () => {
        const c = { ...config, kvCache: policy }
        const frames = graphicalAdapter.run(c, {
          promptTokens: 6,
          newTokens: 8,
          mtp,
          scenario: 'reject',
          splitHeads: true,
        }).frames
        const graph = expandWholeModel(
          { ...doc, config: c },
          doc.graphs.find((g) => g.id === 'whole-model-heads')!,
        )
        const ids = new Set(graph.nodes.map((n) => n.id))
        expect(frames.every((f) => f.nodeIds.every((id) => ids.has(id)))).toBe(true)
        for (const f of frames)
          for (let l = 0; l < c.nLayers; l++) {
            if (!f.tentative || f.phase === 'rollback')
              expect(f.cachePositions[l]).toEqual(residentPositions(c, f.cacheLengths[l]))
            expect(new Set(f.cachePositions[l]).size).toBe(f.cachePositions[l].length)
          }
        for (const f of frames.filter((f) => f.title === 'Write tentative K/V')) {
          const layer = f.layer!
          const checkpoint = residentPositions(c, f.confirmed.length)
          expect(checkpoint.every((p) => f.cachePositions[layer].includes(p))).toBe(true)
        }
        expect(frames[0].cachePositions.every((p) => p.length === 0)).toBe(true)
        expect(frames.at(-1)!.generated).toBe(8)
        if (policy.mode !== 'full')
          expect(frames.some((f) => f.evictedPositions.some((p) => p.length))).toBe(true)
      })
})
