import { describe, expect, it } from 'vitest'
import starter from '../examples/transformer-foundations.model.json'
import { headPartitions, kvHeadChoices, updateTeachingConfig } from '../src/model/teaching-config'
import { parseModelConfig, type ModelConfig } from '../src/model/schema'

const config = parseModelConfig(starter.config)

describe('coupled teaching-model dimensions', () => {
  it('repartitions a fixed model width from either head control', () => {
    const moreHeads = updateTeachingConfig(config, { nQueryHeads: 16 })
    expect(moreHeads).toMatchObject({ dModel: 512, nQueryHeads: 16, headDim: 32 })
    expect(updateTeachingConfig(moreHeads, { headDim: 128 })).toMatchObject({
      dModel: 512,
      nQueryHeads: 4,
      headDim: 128,
      nKvHeads: 2,
    })
  })
  it('updates model width atomically, keeping the head count when possible', () => {
    expect(updateTeachingConfig(config, { dModel: 768 })).toMatchObject({
      dModel: 768,
      nQueryHeads: 8,
      headDim: 96,
    })
  })
  it('handles non-power-of-two head counts without fractional KV heads', () => {
    const six = updateTeachingConfig(config, { dModel: 768, nQueryHeads: 6, headDim: 128 })
    expect(six.nKvHeads).toBe(2)
    expect(kvHeadChoices(6, 'GQA')).toEqual([2, 3])
    expect(updateTeachingConfig(six, { nKvHeads: 3 }).nKvHeads).toBe(3)
    expect(updateTeachingConfig(six, { attention: 'MHA' }).nKvHeads).toBe(6)
    expect(updateTeachingConfig(six, { attention: 'MQA' }).nKvHeads).toBe(1)
  })
  it('changes a prime head count to a supported partition when requesting GQA', () => {
    const prime = parseModelConfig({
      ...config,
      dModel: 384,
      nQueryHeads: 3,
      headDim: 128,
      attention: 'MQA',
      nKvHeads: 1,
    })
    expect(updateTeachingConfig(prime, { attention: 'GQA' })).toMatchObject({
      nQueryHeads: 4,
      nKvHeads: 2,
      headDim: 96,
    })
    expect(() =>
      updateTeachingConfig(
        { ...prime, dModel: 10, nQueryHeads: 5, headDim: 2 },
        { attention: 'GQA' },
      ),
    ).toThrow()
  })
  for (const attention of ['MHA', 'GQA', 'MQA'] as const)
    it(`${attention}: every offered head partition and KV grouping is a valid profile`, () => {
      for (const dModel of [8, 10, 64, 128, 384, 512, 768, 1024, 1536, 4096, 65536])
        for (const partition of headPartitions(dModel, attention))
          for (const nKvHeads of kvHeadChoices(partition.nQueryHeads, attention)) {
            const candidate: ModelConfig = { ...config, dModel, ...partition, nKvHeads, attention }
            expect(parseModelConfig(candidate)).toEqual(candidate)
          }
    })
  it('rejects incompatible axes and does not mutate the original config', () => {
    const before = structuredClone(config)
    for (const change of [{ nQueryHeads: 3 }, { headDim: 63 }, { nKvHeads: 3 }, { dModel: 513 }])
      expect(() => updateTeachingConfig(config, change)).toThrow()
    expect(config).toEqual(before)
  })
  it('keeps retention within a reduced context capacity', () => {
    expect(
      updateTeachingConfig(
        { ...config, kvCache: { mode: 'sparse', windowSize: 8192, stride: 4096 } },
        { contextLength: 2048 },
      ).kvCache,
    ).toEqual({ mode: 'sparse', windowSize: 2048, stride: 2048 })
  })
})
