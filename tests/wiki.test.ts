import { describe, expect, it } from 'vitest'
import katex from 'katex'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import {
  buildWikiEntries,
  conceptsFor,
  entryId,
  learningPaths,
  searchWiki,
  wikiArticles,
  modelExamplesFor,
  suggestWikiQueries,
  type WikiContext,
} from '../src/help/wiki/catalog'
import { wikiSources } from '../src/help/wiki/types'
import {
  attentionExample,
  cacheExample,
  gateExample,
  normalizationExample,
  probabilityExample,
  rotationExample,
  projectionExample,
  residualExample,
  attentionKeys,
  attentionValues,
} from '../src/help/wiki/explorations'
import { perspectives } from '../src/help/wiki/perspectives'
import { glossary } from '../src/help/wiki/glossary'
import { readLibrary, readingKey, serializeLibrary } from '../src/help/wiki/reading'

const doc = parseDocument(starter)
const context: WikiContext = { doc, graph: doc.graphs[0], mode: 'architecture' }

describe('the educational wiki', () => {
  it('provides a separately authored introduction and valid technical derivation for every guide', () => {
    expect(Object.keys(perspectives).sort()).toEqual(wikiArticles.map((a) => a.id).sort())
    for (const article of wikiArticles) {
      const { introduction, technical } = perspectives[article.id]
      expect(introduction).toHaveLength(4)
      expect(technical.length).toBeGreaterThan(0)
      for (const section of introduction) {
        expect(section.equation).toBeUndefined()
        expect(
          [...section.paragraphs, ...(section.bullets ?? [])].join(' ').length,
        ).toBeGreaterThan(70)
      }
      expect(new Set([...article.sections, ...technical].map((s) => s.id)).size).toBe(
        article.sections.length + technical.length,
      )
      for (const section of technical)
        if (section.equation)
          expect(
            () =>
              katex.renderToString(section.equation!, {
                throwOnError: true,
                strict: 'error',
                trust: false,
              }),
            article.id,
          ).not.toThrow()
    }
  })
  it('has complete, uniquely addressable guides with resolving sources, prerequisites and paths', () => {
    const ids = new Set(wikiArticles.map((article) => article.id))
    expect(ids.size).toBe(wikiArticles.length)
    expect(wikiArticles.length).toBeGreaterThanOrEqual(30)
    for (const article of wikiArticles) {
      expect(new Set(article.sections.map((s) => s.id)).size, article.id).toBe(
        article.sections.length,
      )
      expect(article.sections.length, article.id).toBeGreaterThanOrEqual(3)
      for (const id of [...article.related, ...article.prerequisites])
        expect(ids.has(id), `${article.id} → ${id}`).toBe(true)
      for (const source of article.sources) expect(wikiSources[source], source).toBeDefined()
      for (const section of article.sections)
        if (section.equation)
          expect(
            () =>
              katex.renderToString(section.equation!, {
                throwOnError: true,
                trust: false,
                strict: 'error',
              }),
            article.id,
          ).not.toThrow()
    }
    for (const path of learningPaths) for (const id of path.articles) expect(ids.has(id)).toBe(true)
  })

  it('finds symbols, natural questions, aliases and full explanation text with collection filters', () => {
    const entries = buildWikiEntries(context)
    expect(
      searchWiki(entries, 'Why is the future invisible?').some((e) => e.id === 'causal-mask'),
    ).toBe(true)
    expect(searchWiki(entries, 'Hkv').some((e) => e.id === 'head-sharing')).toBe(true)
    expect(searchWiki(entries, 'Hadamard').some((e) => e.id === 'swiglu')).toBe(true)
    expect(searchWiki(entries, 'denominator').some((e) => e.id === 'rmsnorm')).toBe(true)
    expect(
      searchWiki(entries, 'RoPE', 'Model document').every((e) => e.origin === 'document'),
    ).toBe(true)
    expect(searchWiki(entries, 'doesnotexist123')).toEqual([])
    expect(searchWiki(entries, ' ', 'Attention').length).toBeGreaterThan(0)
  })

  it('reads imported components and connections directly without changing the document', () => {
    const imported = structuredClone(doc)
    const graph = imported.graphs[0]
    graph.nodes[0].title = 'Custom tokenizer with preserved terminology'
    graph.nodes[0].description = 'Document-specific customword123 explanation.'
    graph.nodes[0].notes.push('A notebook-independent authored note.')
    graph.edges[0].description = 'A special edgeword123 operation.'
    imported.documentation = 'My modeldocword123 documentation.'
    imported.extensions = { 'org.example': { content: 'Keep this' } }
    const before = JSON.stringify(imported)
    const entries = buildWikiEntries({ ...context, doc: imported, graph })
    expect(searchWiki(entries, 'customword123')[0].title).toBe(graph.nodes[0].title)
    expect(searchWiki(entries, 'edgeword123')[0].target?.type).toBe('edge')
    expect(searchWiki(entries, 'modeldocword123')[0].id).toBe('document:overview')
    expect(new Set(entries.map((e) => e.id)).size).toBe(entries.length)
    expect(JSON.stringify(imported)).toBe(before)
  })

  it('uses actual current runtime IDs and scopes numerical navigation to the runtime graph', () => {
    const runtime = {
      ...doc.graphs[0],
      id: 'runtime',
      nodes: [{ ...doc.graphs[0].nodes[0], id: 'layer-0.attn.scores-0', kind: 'scores' }],
    }
    const entries = buildWikiEntries({ ...context, graph: runtime, currentGraphOnly: true })
    const selected = entries.find(
      (e) => e.id === entryId({ graphId: runtime.id, type: 'node', id: runtime.nodes[0].id }),
    )!
    expect(selected.related).toContain('attention-scores')
    expect(entries.filter((e) => e.target).every((e) => e.target?.graphId === 'runtime')).toBe(true)
    expect(conceptsFor({ kind: 'cachek' })).toContain('kv-cache')
    expect(conceptsFor({ kind: 'unknown-imported-kind' }, 'unknown-graph')).toEqual([])
  })
})

describe('worked explanations and learning navigation', () => {
  it('isolates value interventions from attention probabilities and exposes projection and residual arithmetic', () => {
    const before = attentionExample([1, 2], 1, true)
    const changed = attentionExample([1, 2], 1, true, attentionKeys, [
      [6, 0],
      ...attentionValues.slice(1),
    ])
    expect(changed.probabilities).toEqual(before.probabilities)
    expect(changed.output[0] - before.output[0]).toBeCloseTo(4 * before.probabilities[0])
    expect(
      projectionExample(
        [2, -1],
        [
          [3, 4],
          [5, -2],
        ],
      ),
    ).toEqual({
      products: [
        [6, -5],
        [8, 2],
      ],
      output: [1, 10],
    })
    expect(residualExample(2, -0.5, 0.2, true).output).toBeCloseTo(1.7)
    expect(residualExample(2, -0.5, 0.2, false)).toEqual({ afterAttention: -0.5, output: 0.2 })
  })
  it('calculates masked attention and excludes future inputs before row normalization', () => {
    const row = attentionExample([1, 2], 0, true)
    expect(row.probabilities).toEqual([1, 0, 0])
    expect(row.output).toEqual([2, 0])
    const noncausal = attentionExample([1, 2], 0, false)
    expect(noncausal.probabilities.reduce((a, b) => a + b, 0)).toBeCloseTo(1)
    expect(noncausal.output).not.toEqual(row.output)
    expect(noncausal.output.every((v) => v >= 0 && v <= 4)).toBe(true)
    expect(probabilityExample([1, 2]).probabilities).toEqual(
      probabilityExample([101, 102]).probabilities,
    )
    expect(() => probabilityExample([1], 0)).toThrow()
    expect(() => probabilityExample([1], 1, [false])).toThrow()
  })
  it('checks the rotation, normalization and gate examples against hand calculations', () => {
    expect(rotationExample([1, 0.5], 0, 0).output).toEqual([1, 0.5])
    const rotated = rotationExample([1, 0.5], Math.PI / 2, 0)
    expect(rotated.output[0]).toBeCloseTo(-0.5)
    expect(rotated.output[1]).toBeCloseTo(1)
    expect(Math.hypot(...rotated.output)).toBeCloseTo(Math.hypot(1, 0.5))
    expect(rotationExample([1, 0], 1, 1).frequency).toBe(0.01)
    const rms = normalizationExample([3, 4])
    expect(rms.meanSquare).toBe(12.5)
    expect(rms.output[0]).toBeCloseTo(3 / Math.sqrt(12.50001))
    expect(normalizationExample([0, 0]).output).toEqual([0, 0])
    expect(gateExample(2, -3).product).toBeCloseTo(-5.2847824679)
    expect(gateExample(-1, 1).silu).toBeLessThan(0)
  })
  it('uses retained position counts and the actual profile for working-state accounting', () => {
    const sliding = { ...doc.config, kvCache: { mode: 'sliding-window' as const, windowSize: 3 } }
    const example = cacheExample(sliding, 10, 4)
    expect(example.prefix).toBe(3)
    expect(example.working).toBe(7)
    expect(example.retained).toBe(3)
    expect(example.bytes).toBe(
      2 * doc.config.nLayers * doc.config.nKvHeads * 3 * doc.config.headDim * doc.config.dtypeBytes,
    )
    const prefill = cacheExample(sliding, 10, 10)
    expect(prefill.prefix).toBe(0)
    expect(prefill.working).toBe(10)
    expect(prefill.retained).toBe(3)
  })
  it('resolves glossary definitions, typo suggestions and exact model mechanisms', () => {
    const entries = buildWikiEntries(context)
    for (const term of glossary)
      expect(
        wikiArticles.some((a) => a.id === term.article),
        term.id,
      ).toBe(true)
    expect(searchWiki(entries, 'why do we need masking').some((e) => e.id === 'causal-mask')).toBe(
      true,
    )
    expect(searchWiki(entries, 'normalisation').some((e) => e.id === 'rmsnorm')).toBe(true)
    expect(suggestWikiQueries('softmx')).toContain('softmax')
    expect(suggestWikiQueries('rotarry')).toContain('rotary')
    const rope = wikiArticles.find((a) => a.id === 'rope')!
    const matches = modelExamplesFor(rope, entries, context)
    expect(matches.map((e) => e.target?.id)).toContain('ropeq')
    expect(matches.every((e) => ['ropeq', 'ropek', 'newk'].includes(e.semanticKind ?? ''))).toBe(
      true,
    )
  })
})

describe('a separate reading library', () => {
  it('keeps bookmarks and places scoped to their model and preserves future metadata', () => {
    expect(readingKey('rope', 'model-a')).toBe(readingKey('rope', 'model-b'))
    expect(readingKey('document:custom:node:q', 'model-a')).not.toBe(
      readingKey('document:custom:node:q', 'model-b'),
    )
    const place = {
      key: 'rope',
      entryId: 'rope',
      title: 'RoPE',
      at: 1,
      scroll: 500,
      custom: { keep: true },
    }
    const encoded = serializeLibrary('{"version":1,"org.example":{"keep":true}}', {
      bookmarks: [place],
      recent: [place],
    })!
    expect(JSON.parse(encoded)['org.example']).toEqual({ keep: true })
    expect(readLibrary(encoded).bookmarks[0].custom).toEqual({ keep: true })
    expect(readLibrary('broken')).toEqual({ bookmarks: [], recent: [] })
    expect(
      serializeLibrary('{"version":2,"keep":"future"}', { bookmarks: [], recent: [] }),
    ).toBeUndefined()
    expect(
      readLibrary(
        JSON.stringify({
          version: 1,
          bookmarks: [{ ...place, scroll: -1 }],
          recent: [place, place],
        }),
      ),
    ).toEqual({ bookmarks: [], recent: [place] })
  })
})
