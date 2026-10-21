import { describe, expect, it } from 'vitest'
import { helpCatalog, namedHelp } from '../src/help/catalog'
import {
  defaultHelpPreferences,
  positionHelp,
  readHelpPreferences,
  serializeHelpPreferences,
} from '../src/help/preferences'

describe('help preferences', () => {
  it('defaults to 700 ms without a preference and rejects invalid stored timings', () => {
    for (const raw of [
      null,
      'broken',
      '{}',
      '{"version":1,"delayMs":-1}',
      '{"version":1,"delayMs":9000}',
      '{"version":1,"delayMs":"0.7"}',
    ])
      expect(readHelpPreferences(raw)).toEqual(defaultHelpPreferences)
    expect(readHelpPreferences('{"version":1,"delayMs":0,"enabled":false}')).toEqual({
      delayMs: 0,
      enabled: false,
    })
  })
  it('preserves extra device metadata and refuses to overwrite future preferences', () => {
    const raw = '{"version":1,"org.example":{"notes":[1,2]}}'
    expect(JSON.parse(serializeHelpPreferences(raw, { delayMs: 1200, enabled: true })!)).toEqual({
      version: 1,
      delayMs: 1200,
      enabled: true,
      'org.example': { notes: [1, 2] },
    })
    expect(
      serializeHelpPreferences('{"version":2,"other":"future"}', defaultHelpPreferences),
    ).toBeUndefined()
  })
})

describe('detailed control explanations', () => {
  it('covers every teaching configuration option and its enum choices', () => {
    for (const name of [
      'Model width D',
      'Decoder layers',
      'Query heads',
      'Features per head',
      'Attention architecture',
      'KV heads',
      'Feed-forward width',
      'Context capacity',
      'Cache scalar storage',
      'KV cache retention',
      'KV window size',
      'Sparse anchor stride',
    ])
      expect(namedHelp(name)?.paragraphs.join(' ').length, name).toBeGreaterThan(140)
    expect(helpCatalog['Attention architecture'].choices?.map((c) => c.label)).toEqual([
      'MHA',
      'GQA',
      'MQA',
    ])
    expect(helpCatalog['KV cache retention'].choices).toHaveLength(3)
  })
  it('keeps graphical, untrained and learned-model provenance distinct', () => {
    expect(helpCatalog['Simulation input packets'].paragraphs.join(' ')).toContain(
      'does not tokenize text',
    )
    expect(helpCatalog['Numerical parameter seed'].paragraphs.join(' ')).toContain('not trained')
    expect(helpCatalog['Local model sampling seed'].paragraphs.join(' ')).toContain(
      'never reinitializes',
    )
    expect(namedHelp('Tensor Hq slice')?.title).toBe('Tensor axis slice')
    expect(namedHelp('Cell 0, 1: 0.003')?.paragraphs.join(' ')).toContain(
      'original tensor coordinate',
    )
  })
})

describe('tooltip placement', () => {
  it('keeps help inside a narrow viewport and flips above a low control', () => {
    const position = positionHelp(
      { x: 260, y: 700, width: 100, height: 36 },
      { width: 366, height: 300 },
      { width: 390, height: 844 },
    )
    expect(position.x).toBe(12)
    expect(position.y + 300).toBeLessThanOrEqual(700)
  })
  it('uses space beside a tall control when neither vertical side fits', () => {
    const position = positionHelp(
      { x: 200, y: 100, width: 100, height: 600 },
      { width: 300, height: 450 },
      { width: 1440, height: 900 },
    )
    expect(position.x).toBe(310)
    expect(position.y).toBe(100)
  })
})
