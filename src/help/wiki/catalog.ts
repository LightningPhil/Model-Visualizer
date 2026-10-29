import type { Component, ModelConfig, ModelDocument, ModelGraph, Tensor } from '../../model/schema'
import { foundations } from './foundations'
import { attentionArticles } from './attention'
import { practiceArticles } from './practice'
import type { WikiArticle, WikiSource } from './types'
import { wikiSources } from './types'
import { glossary } from './glossary'
import { perspectives } from './perspectives'
import { memoryArticles } from './memory'
import { learningArticles } from './learning'
import { optimizerArticles } from './optimizers'
import type { StudyRecipe } from '../../model/study/records'

export const wikiArticles: WikiArticle[] = [
  ...foundations,
  ...attentionArticles,
  ...practiceArticles,
  ...memoryArticles,
  ...learningArticles,
  ...optimizerArticles,
]
export const wikiCategories = [...new Set(wikiArticles.map((article) => article.category))]

export type WikiTarget = { graphId: string; type: 'node' | 'edge' | 'graph'; id: string }
export type WikiContext = {
  doc: ModelDocument
  graph: ModelGraph
  selection?: { type: 'node' | 'edge'; id: string } | null
  mode: 'architecture' | 'simulation' | 'untrained' | 'trained'
  symbols?: Record<string, number>
  inspection?: {
    name: string
    index: number
    coordinate: number[]
    axes: string[]
    value: number | string
    explanation: string
    tex: string
  }
  config?: ModelConfig
  initialArticle?: string
  initialStudy?: StudyRecipe
  onRestoreStudy?: (recipe: StudyRecipe) => void
  onNavigate?: (target: WikiTarget) => void
  onExperiment?: (entry: string) => void
  /** Numerical graphs reuse a reference document but only this runtime graph is navigable. */
  currentGraphOnly?: boolean
}

export type WikiEntry = WikiArticle & {
  origin: 'concept' | 'document' | 'glossary'
  semanticKind?: string
  target?: WikiTarget
  tensors?: Tensor[]
  references: WikiSource[]
}

export const entryId = (target: WikiTarget) =>
  `document:${encodeURIComponent(target.graphId)}:${target.type}:${encodeURIComponent(target.id)}`

export function conceptsFor(component: Pick<Component, 'kind'> | undefined, graphId?: string) {
  const exact = component
    ? wikiArticles.filter((article) => article.kinds.includes(component.kind)).map((a) => a.id)
    : []
  return exact.length
    ? exact
    : wikiArticles.filter((article) => article.graphs.includes(graphId ?? '')).map((a) => a.id)
}

export function buildWikiEntries(context: WikiContext): WikiEntry[] {
  const entries: WikiEntry[] = wikiArticles.map((article) => ({
    ...article,
    origin: 'concept',
    references: article.sources.map((id) => wikiSources[id]),
  }))
  entries.push(
    ...glossary.map((term): WikiEntry => ({
      id: `glossary:${term.id}`,
      title: term.term,
      summary: term.definition,
      category: 'Glossary',
      aliases: [...(term.aliases ?? []), ...(term.symbol ? [term.symbol] : [])],
      sections: [
        { id: 'definition', title: 'Meaning in this workbench', paragraphs: [term.definition] },
      ],
      prerequisites: [],
      related: [term.article],
      sources: [],
      kinds: [],
      graphs: [],
      origin: 'glossary',
      references: (wikiArticles.find((a) => a.id === term.article)?.sources ?? []).map(
        (id) => wikiSources[id],
      ),
    })),
  )
  const graphs = context.currentGraphOnly
    ? [context.graph]
    : [...context.doc.graphs.filter((g) => g.id !== context.graph.id), context.graph]
  const sourceMap = new Map(context.doc.sources.map((source) => [source.id, source]))
  const referencesFor = (ids: string[]) =>
    ids.flatMap((id) => {
      const source = sourceMap.get(id)
      return source ? [{ title: source.title, url: source.url, note: source.note }] : []
    })
  entries.push({
    id: 'document:overview',
    title: context.doc.name,
    summary: context.doc.description,
    category: 'Model document',
    aliases: ['model documentation', 'research library', 'sources', 'overview'],
    origin: 'document',
    sections: [
      {
        id: 'documentation',
        title: 'Model documentation',
        paragraphs: [context.doc.documentation],
      },
    ],
    prerequisites: [],
    related: ['reading-the-model', 'workspace', 'evidence'],
    sources: [],
    kinds: [],
    graphs: [],
    references: referencesFor(context.doc.sources.map((source) => source.id)),
  })
  for (const graph of graphs) {
    const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]))
    const target = { graphId: graph.id, type: 'graph', id: graph.id } as const
    entries.push({
      id: entryId(target),
      title: graph.title,
      summary: graph.subtitle,
      category: 'Model document',
      aliases: [graph.id, graph.title],
      origin: 'document',
      sections: [{ id: 'description', title: 'About this view', paragraphs: [graph.description] }],
      prerequisites: [],
      related: conceptsFor(undefined, graph.id),
      sources: [],
      kinds: [],
      graphs: [],
      references: [],
      target,
    })
    for (const node of graph.nodes) {
      const target = { graphId: graph.id, type: 'node', id: node.id } as const
      const related = conceptsFor(node, graph.id)
      entries.push({
        id: entryId(target),
        title: node.title,
        summary: node.subtitle || node.description,
        category: 'Model document',
        aliases: [node.id, node.kind, graph.title, ...node.tensors.map((t) => t.name)],
        origin: 'document',
        semanticKind: node.kind,
        target,
        tensors: node.tensors,
        sections: [
          {
            id: 'description',
            title: 'The authored explanation',
            paragraphs: [node.description],
            equation: node.equation,
          },
          ...(node.notes.length
            ? [{ id: 'notes', title: 'Worth understanding', paragraphs: node.notes }]
            : []),
        ],
        prerequisites: [],
        related,
        sources: [],
        kinds: [],
        graphs: [],
        references: referencesFor(node.sourceIds),
      })
    }
    for (const edge of graph.edges) {
      const target = { graphId: graph.id, type: 'edge', id: edge.id } as const
      const source = nodeMap.get(edge.source)
      const destination = nodeMap.get(edge.target)
      entries.push({
        id: entryId(target),
        title: edge.label,
        summary: `${source?.title ?? edge.source} → ${destination?.title ?? edge.target}`,
        category: 'Model document',
        aliases: [edge.id, edge.operation, graph.title],
        origin: 'document',
        target,
        sections: [
          {
            id: 'operation',
            title: edge.operation,
            paragraphs: [edge.description],
            equation: edge.equation,
          },
        ],
        tensors: edge.shape.length
          ? [{ name: 'Carried / result tensor', role: 'output', shape: edge.shape }]
          : [],
        prerequisites: [],
        related: [
          ...new Set([...conceptsFor(destination, graph.id), ...conceptsFor(source, graph.id)]),
        ],
        sources: [],
        kinds: [],
        graphs: [],
        references: referencesFor([
          ...new Set([...(source?.sourceIds ?? []), ...(destination?.sourceIds ?? [])]),
        ]),
      })
    }
  }
  return entries
}

export function modelExamplesFor(article: WikiArticle, entries: WikiEntry[], context: WikiContext) {
  return entries
    .filter(
      (entry) =>
        entry.origin === 'document' &&
        entry.target &&
        ((entry.target.type === 'node' && article.kinds.includes(entry.semanticKind ?? '')) ||
          (entry.target.type === 'graph' && article.graphs.includes(entry.target.graphId))),
    )
    .map((entry, index) => ({
      entry,
      index,
      score:
        (entry.target!.graphId === context.graph.id ? 100 : 0) +
        (context.selection?.id === entry.target!.id ? 100 : 0) +
        (entry.target!.type === 'node' ? 20 : 0),
    }))
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(({ entry }) => entry)
}

const stopWords = new Set([
  'a',
  'an',
  'the',
  'is',
  'are',
  'does',
  'do',
  'what',
  'why',
  'how',
  'can',
  'i',
  'my',
  'it',
  'in',
  'of',
  'to',
  'for',
  'and',
  'with',
  'this',
  'that',
  'about',
  'from',
  'why',
  'when',
  'where',
  'happen',
  'happens',
  'work',
  'works',
  'need',
  'needed',
  'mean',
  'means',
  'please',
  'explain',
  'would',
  'should',
  'use',
  'using',
])
export const normalizeWikiText = (value: string) =>
  value
    .toLowerCase()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[−–—-]/g, ' ')

export function buildWikiSearchIndex(entries: WikiEntry[]) {
  return entries.map((entry, index) => ({
    entry,
    index,
    title: normalizeWikiText(entry.title),
    aliases: normalizeWikiText(entry.aliases.join(' ')),
    body: normalizeWikiText(
      [
        entry.summary,
        ...(perspectives[entry.id]?.introduction ?? []).flatMap((s) => [
          s.title,
          ...s.paragraphs,
          ...(s.bullets ?? []),
        ]),
        ...(perspectives[entry.id]?.technical ?? []).flatMap((s) => [
          s.title,
          ...s.paragraphs,
          s.equation ?? '',
        ]),
        ...entry.sections.flatMap((s) => [
          s.title,
          ...s.paragraphs,
          ...(s.bullets ?? []),
          s.equation ?? '',
        ]),
        entry.experiment?.title ?? '',
        ...(entry.experiment?.steps ?? []),
        entry.experiment?.observation ?? '',
        ...entry.references.flatMap((source) => [source.title, source.note]),
        ...(entry.tensors?.flatMap((t) => [t.name, t.description ?? '', ...t.shape]) ?? []),
      ].join(' '),
    ),
  }))
}

export function searchWiki(
  entries: WikiEntry[],
  query: string,
  category = 'All topics',
  prepared = buildWikiSearchIndex(entries),
) {
  const tokens = normalizeWikiText(query)
    .split(/[^\p{L}\p{N}]+/u)
    .filter((term) => term && !stopWords.has(term))
  const synonyms: Record<string, string[]> = {
    memory: ['cache', 'state', 'memory'],
    normalisation: ['normalization', 'rmsnorm'],
    normalization: ['normalization', 'rmsnorm'],
    future: ['future', 'causal'],
    masking: ['mask', 'masking'],
    tokens: ['token'],
    weights: ['weight', 'parameter'],
    activations: ['activation'],
    position: ['position', 'rotary'],
    temperature: ['temperature', 'sampling'],
    probabilities: ['probabilit', 'softmax'],
    probability: ['probabilit', 'softmax'],
    cached: ['cache', 'cached'],
  }
  return prepared
    .flatMap(({ entry, index, title, aliases, body }) => {
      if (category !== 'All topics' && entry.category !== category) return []
      if (
        !tokens.every((token) =>
          (synonyms[token] ?? [token]).some((term) => `${title} ${aliases} ${body}`.includes(term)),
        )
      )
        return []
      const score =
        tokens.reduce(
          (sum, token) =>
            sum +
            (title.includes(token) ? 12 : 0) +
            (aliases.includes(token) ? 8 : 0) +
            (body.includes(token) ? 1 : 0),
          0,
        ) + (entry.origin === 'concept' ? 2 : 0)
      return [{ entry, score, index }]
    })
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .map(({ entry }) => entry)
}

function editDistance(a: string, b: string) {
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 0; i < a.length; i++) {
    const current = [i + 1]
    for (let j = 0; j < b.length; j++)
      current.push(
        Math.min(current[j] + 1, previous[j + 1] + 1, previous[j] + Number(a[i] !== b[j])),
      )
    previous = current
  }
  return previous[b.length]
}

/** Conservative typo suggestions for editorial terminology, never a guessed
 * answer or a silent change to the user's search. */
export function suggestWikiQueries(query: string) {
  if (query.length > 100) return []
  const dictionary = [
    ...new Set(
      [
        ...wikiArticles.flatMap((a) => [a.title, ...a.aliases]),
        ...glossary.flatMap((t) => [t.term, ...(t.aliases ?? [])]),
      ]
        .join(' ')
        .toLowerCase()
        .match(/[a-z]{4,}/g) ?? [],
    ),
  ]
  const words = query.toLowerCase().split(/\s+/)
  const suggestions: string[] = []
  for (let i = 0; i < words.length; i++) {
    const word = words[i]
    if (word.length < 4 || dictionary.includes(word)) continue
    const candidate = dictionary
      .filter((term) => Math.abs(term.length - word.length) <= 2)
      .map((term) => ({ term, distance: editDistance(word, term) }))
      .filter((r) => r.distance <= (word.length > 6 ? 2 : 1))
      .sort((a, b) => a.distance - b.distance || a.term.localeCompare(b.term))[0]
    if (candidate)
      suggestions.push(
        words.map((value, index) => (index === i ? candidate.term : value)).join(' '),
      )
  }
  return suggestions.slice(0, 3)
}

export const learningPaths = [
  {
    title: 'How learning changes behaviour',
    description:
      'Follow gradients, training objectives, representation hypotheses and controlled tests.',
    articles: [
      'learning-loop',
      'adam',
      'adamw',
      'muon',
      'training-objective',
      'post-training',
      'numerical-precision',
      'feature-representations',
      'causal-interventions',
    ],
  },
  {
    title: 'Why attention uses memory',
    description: 'Compare head sharing, latent state, recurrence, precision and physical sharing.',
    articles: [
      'attention-comparison',
      'memory-budget',
      'head-sharing',
      'attention-patterns',
      'mla',
      'linear-attention',
      'cache-quantization',
      'prefix-sharing',
      'storage-kernels',
    ],
  },
  {
    title: 'Follow a token',
    description: 'From bytes to a next-token choice.',
    articles: [
      'tokenization',
      'embeddings',
      'parameters-activations-state',
      'residual',
      'attention-overview',
      'swiglu',
      'logits',
      'sampling',
    ],
  },
  {
    title: 'Understand attention',
    description: 'Reconstruct the calculation one operation at a time.',
    articles: [
      'tensors',
      'linear-maps',
      'attention-scores',
      'causal-mask',
      'softmax',
      'rope',
      'position-methods',
      'head-sharing',
    ],
  },
  {
    title: 'Understand generation state',
    description: 'Positions, cached values, memory and verification.',
    articles: [
      'prefill-decode',
      'kv-cache',
      'cache-retention',
      'cache-bytes',
      'speculative-decoding',
      'storage-kernels',
    ],
  },
  {
    title: 'Investigate actual values',
    description: 'Choose evidence, follow operands and record findings.',
    articles: [
      'evidence',
      'tensor-laboratory',
      'trained-model',
      'attention-interpretation',
      'study-method',
    ],
  },
]

export const modeLabels: Record<WikiContext['mode'], string> = {
  architecture: 'Architecture study · teaching shapes and labelled illustrations',
  simulation: 'Graphical simulation · scripted choices, no computed activations',
  untrained: 'Numerical reference · actual arrays, untrained parameters',
  trained: 'Local trained checkpoint · actual learned weights and activations',
}
