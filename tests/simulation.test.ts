import { describe, expect, it } from 'vitest'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import { graphicalAdapter, nextFrame } from '../src/model/simulation'
import { expandWholeModel, isWholeModel } from '../src/model/whole-model'

const doc = parseDocument(starter)
const graph = expandWholeModel(doc, doc.graphs.find(isWholeModel)!)
const ids = new Set(graph.nodes.map((n) => n.id))
const options = { promptTokens: 6, newTokens: 8, mtp: true, scenario: 'mixed' as const }

describe('graphical execution trace', () => {
  for (const mtp of [true, false])
    for (const scenario of ['mixed', 'accept', 'reject'] as const)
      it(`${mtp ? 'MTP' : 'autoregressive'} / ${scenario}: visits every layer and commits exactly the reference sequence`, () => {
        const trace = graphicalAdapter.run(doc.config, { ...options, mtp, scenario })
        expect(trace.provenance.kind).toBe('illustrative')
        const last = trace.frames.at(-1)!
        expect(last.phase).toBe('done')
        expect(last.generated).toBe(8)
        expect(last.confirmed.slice(6).map((p) => p.id)).toEqual(
          Array.from({ length: 8 }, (_, i) => `G${i}`),
        )
        expect(trace.frames.every((f) => f.nodeIds.every((id) => ids.has(id)))).toBe(true)
        for (let layer = 0; layer < doc.config.nLayers; layer++)
          expect(trace.frames.some((f) => f.nodeIds.includes(`layer-${layer}.add2`))).toBe(true)
        for (const frame of trace.frames) {
          expect(frame.confirmed.every((p) => !p.id.startsWith('D'))).toBe(true)
          expect(frame.generated).toBe(frame.confirmed.length - 6)
          expect(frame.cacheLengths).toHaveLength(doc.config.nLayers)
          expect(frame.cacheLengths.every((s) => s >= 0 && s <= frame.contextLength + 3)).toBe(true)
        }
      })
  it('shows causal batched verification, rejection, suffix discard and rollback before committing', () => {
    const frames = graphicalAdapter.run(doc.config, options).frames
    const firstReject = frames.findIndex((f) => f.proposals.some((p) => p.status === 'rejected'))
    expect(firstReject).toBeGreaterThan(0)
    expect(frames[firstReject].tentative).toBe(true)
    const rollback = frames.slice(firstReject).find((f) => f.phase === 'rollback')!
    expect(rollback.cacheLengths).toEqual(Array(6).fill(8)) // prefix 6 + two matching proposals
    const commit = frames.slice(firstReject).find((f) => f.phase === 'commit')!
    expect(commit.confirmed.at(-1)).toMatchObject({ id: 'G2', status: 'correction' })
    expect(commit.generated).toBe(3)
    expect(commit.tentative).toBe(false)
    expect(frames.some((f) => f.phase === 'verify' && f.queryLength === 3 && f.layer === 5)).toBe(
      true,
    )
  })
  it('discards every proposal after the first mismatching MTP draft', () => {
    const frames = graphicalAdapter.run(doc.config, { ...options, scenario: 'reject' }).frames
    const rejected = frames.find((f) => f.proposals.some((p) => p.status === 'rejected'))!
    expect(rejected.proposals.map((p) => p.status)).toEqual(['accepted', 'rejected', 'discarded'])
    expect(rejected.accepted).toBe(1)
    expect(rejected.rejected).toBe(2)
  })
  it('appends a bonus after a fully matching draft and reuses the accepted KV prefix', () => {
    const frames = graphicalAdapter.run(doc.config, { ...options, scenario: 'accept' }).frames
    const commit = frames.find((f) => f.phase === 'commit')!
    expect(commit.generated).toBe(4)
    expect(commit.confirmed.at(-1)).toMatchObject({ id: 'G3', status: 'bonus' })
    expect(commit.cacheLengths).toEqual(Array(6).fill(9))
    const nextLayerWrite = frames
      .slice(commit.id + 1)
      .find((f) => f.nodeIds.includes('layer-0.attn.cachek'))!
    expect(nextLayerWrite.cacheLengths[0]).toBe(10)
    expect(nextLayerWrite.queryLength).toBe(1)
  })
  it('preserves history snapshots and stops exactly at short budgets', () => {
    for (const newTokens of [1, 2, 3, 4, 5]) {
      const trace = graphicalAdapter.run(doc.config, { ...options, newTokens })
      expect(trace.frames[0].generated).toBe(0)
      expect(trace.frames[0].cacheLengths).toEqual(Array(6).fill(0))
      expect(trace.frames.at(-1)!.generated).toBe(newTokens)
    }
  })
  it('supports reversible operation and milestone navigation without recomputing decisions', () => {
    const trace = graphicalAdapter.run(doc.config, options)
    expect(nextFrame(trace, 0, 'operation')).toBe(1)
    const next = nextFrame(trace, 0, 'stage')
    expect(next).toBeGreaterThan(1)
    expect(trace.frames[next].milestone).toBe(true)
    expect(nextFrame(trace, next, 'stage', -1)).toBe(0)
    expect(nextFrame(trace, trace.frames.length - 1, 'stage')).toBe(trace.frames.length - 1)
  })
  it('rejects impossible packet and context budgets explicitly', () => {
    expect(() => graphicalAdapter.run(doc.config, { ...options, promptTokens: 0 })).toThrow()
    expect(() => graphicalAdapter.run({ ...doc.config, contextLength: 8 }, options)).toThrow(
      /context capacity/,
    )
  })
})
