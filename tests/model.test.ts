import { describe, expect, it } from 'vitest'
import katex from 'katex'
import starter from '../examples/transformer-foundations.model.json'
import { cacheBytes, dimensions, parseDocument, resolveAxis, shapeLabel } from '../src/model/schema'
import { attentionWeights, demoTokenizer } from '../src/model/tokenizer'
import { autoLayout, NODE_HEIGHT, NODE_WIDTH } from '../src/model/layout'

describe('portable model contract', () => {
  it('loads and round-trips the complete authored document', () => {
    const doc = parseDocument(starter)
    expect(parseDocument(JSON.parse(JSON.stringify(doc)))).toEqual(doc)
    expect(doc.graphs).toHaveLength(13)
    expect(doc.graphs.reduce((sum, g) => sum + g.nodes.length, 0)).toBe(79)
    expect(doc.config.vocabSize).toBe(demoTokenizer.vocabularySize)
  })
  it('preserves nested extension metadata', () => {
    const doc = {
      ...structuredClone(starter),
      extensions: { 'org.example': { routing: [2, 4], annotation: 'custom' } },
    }
    expect(parseDocument(doc).extensions).toEqual(doc.extensions)
  })
  it('rejects unknown standard fields instead of losing content', () => {
    expect(() => parseDocument({ ...starter, accidentalMetadata: 'preserve me' })).toThrow()
  })
  it.each([
    'dangling edge',
    'duplicate node',
    'unknown child',
    'unknown source',
    'bad viewport',
    'future version',
    'unsafe URL',
    'non-divisible heads',
    'wrong MHA sharing',
    'wrong MQA sharing',
    'unknown dimension',
  ])('rejects %s before replacing a workspace', (fault) => {
    const doc = structuredClone(starter)
    const graph = doc.graphs[0]
    if (fault === 'dangling edge') graph.edges[0].target = 'missing'
    if (fault === 'duplicate node') graph.nodes.push(graph.nodes[0])
    if (fault === 'unknown child') graph.nodes[0].childGraph = 'missing'
    if (fault === 'unknown source') graph.nodes[0].sourceIds.push('missing')
    if (fault === 'bad viewport')
      Object.assign(graph.layout, { viewport: { x: 0, y: 0, zoom: -2 } })
    if (fault === 'future version') doc.schemaVersion = '99.0'
    if (fault === 'unsafe URL') doc.sources[0].url = 'javascript:alert(1)'
    if (fault === 'non-divisible heads') doc.config.nKvHeads = 3
    if (fault === 'wrong MHA sharing') doc.config.attention = 'MHA'
    if (fault === 'wrong MQA sharing') doc.config.attention = 'MQA'
    if (fault === 'unknown dimension') graph.nodes[0].tensors[0].shape = ['Missing']
    expect(() => parseDocument(doc)).toThrow()
  })
  it('renders every authored LaTeX equation without parse errors', () => {
    const doc = parseDocument(starter)
    const equations = doc.graphs
      .flatMap((g) => [...g.nodes, ...g.edges])
      .flatMap((item) => (item.equation ? [item.equation] : []))
    expect(equations.length).toBeGreaterThan(20)
    for (const tex of equations)
      expect(() => katex.renderToString(tex, { throwOnError: true, trust: false })).not.toThrow()
  })
})

describe('tensor dimensions and storage', () => {
  const config = parseDocument(starter).config
  it('distinguishes prefill queries from decode prefix length', () => {
    expect(dimensions(config, 7)).toMatchObject({ T: 7, S: 7 })
    expect(dimensions(config, 7, true)).toMatchObject({ T: 1, S: 8 })
    expect(shapeLabel(['B', 'Hq', 'T', 'S'], dimensions(config, 7, true))).toBe('[1 × 8 × 1 × 8]')
  })
  it('calculates cache bytes from KV heads, never query heads', () => {
    expect(cacheBytes(config, 6)).toBe(18432)
    expect(cacheBytes({ ...config, attention: 'MHA', nKvHeads: 8 }, 6)).toBe(
      4 * cacheBytes(config, 6),
    )
    expect(cacheBytes({ ...config, attention: 'MQA', nKvHeads: 1 }, 6)).toBe(
      cacheBytes(config, 6) / 2,
    )
    expect(cacheBytes(config, config.contextLength)).toBe(6291456)
  })
  it('parses precedence and parentheses without evaluating code', () => {
    expect(resolveAxis('2*(D + F) / Hq', { D: 512, F: 1536, Hq: 8 })).toBe(512)
    expect(shapeLabel(['T-1'], { T: 1 })).toBe('[?]')
  })
  it.each(['D.foo', 'alert(1)', 'D/0', 'D/3', '(D', 'D)', '0', 'D-999', 'D;1', 'constructor'])(
    'rejects invalid dimension %s',
    (expression) => {
      expect(() => resolveAxis(expression, { D: 8 })).toThrow()
    },
  )
})

describe('executable teaching tokenizer', () => {
  it.each([
    '',
    'The future is beautifully complex.',
    'Hello world!',
    '你好世界',
    '🧠🚀',
    'café\n\t naïve',
    'a\u0000b',
    'e\u0301',
  ])('losslessly encodes and decodes %j', (input) => {
    const encoded = demoTokenizer.encode(input)
    expect(demoTokenizer.decode(encoded.tokens.map((t) => t.id))).toBe(input)
    expect(encoded.tokens.flatMap((t) => t.bytes)).toEqual([...new TextEncoder().encode(input)])
  })
  it('records every merge and ends at the emitted IDs', () => {
    const encoded = demoTokenizer.encode('The future is beautifully complex.')
    expect(encoded.tokens).toHaveLength(6)
    expect(encoded.steps.at(-1)?.pieces).toEqual(encoded.tokens.map((t) => t.id))
    expect(encoded.steps.length).toBe(encoded.byteLength - encoded.tokens.length)
    for (const step of encoded.steps)
      expect(demoTokenizer.decode([step.result])).toBe(demoTokenizer.decode(step.pair))
  })
  it('is deterministic and does not normalize Unicode', () => {
    expect(demoTokenizer.encode('café')).toEqual(demoTokenizer.encode('café'))
    expect(demoTokenizer.encode('é').tokens).not.toEqual(demoTokenizer.encode('e\u0301').tokens)
  })
  it('rejects token IDs outside its vocabulary', () =>
    expect(() => demoTokenizer.decode([99999])).toThrow())
})

describe('attention teaching probabilities', () => {
  it('applies a causal mask and normalizes each query row', () => {
    for (const row of attentionWeights(12, 2))
      expect(row.reduce((sum, v) => sum + v, 0)).toBeCloseTo(1, 12)
    attentionWeights(12, 2).forEach((row, r) =>
      row.forEach((v, c) => {
        if (c > r) expect(v).toBe(0)
      }),
    )
  })
  it('shows distinct deterministic patterns per head', () =>
    expect(attentionWeights(6, 0)).not.toEqual(attentionWeights(6, 1)))
})

describe('automatic layout', () => {
  it('places every attention component without overlap in both directions', async () => {
    const graph = parseDocument(starter).graphs.find((g) => g.id === 'attention')!
    for (const direction of ['RIGHT', 'DOWN'] as const) {
      const positions = await autoLayout(graph, direction)
      expect(Object.keys(positions)).toHaveLength(graph.nodes.length)
      const points = Object.values(positions)
      points.forEach((point, i) =>
        points.slice(i + 1).forEach((other) => {
          const overlaps =
            Math.abs(point.x - other.x) < NODE_WIDTH && Math.abs(point.y - other.y) < NODE_HEIGHT
          expect(overlaps).toBe(false)
        }),
      )
    }
  }, 20000)
})
