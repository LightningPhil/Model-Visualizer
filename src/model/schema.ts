import { z } from 'zod'
import { expandWholeModel } from './whole-model.ts'
import { residentCount } from './attention.ts'

const text = z.string().max(24000)
const id = z
  .string()
  .min(1)
  .max(100)
  .regex(/^[a-zA-Z0-9_.-]+$/)
const axis = z
  .string()
  .min(1)
  .max(60)
  .regex(/^[A-Za-z0-9+*() /-]+$/)
const point = z.strictObject({ x: z.number().finite(), y: z.number().finite() })

export const configSchema = z.strictObject({
  dModel: z.number().int().min(8).max(65536),
  nLayers: z.number().int().min(1).max(512),
  nQueryHeads: z.number().int().min(1).max(256),
  nKvHeads: z.number().int().min(1).max(256),
  headDim: z.number().int().min(2).max(1024),
  ffnDim: z.number().int().min(8).max(262144),
  vocabSize: z.number().int().min(256).max(1000000),
  contextLength: z.number().int().min(32).max(1048576),
  dtypeBytes: z.union([z.literal(1), z.literal(2), z.literal(4)]),
  attention: z.enum(['MHA', 'GQA', 'MQA']),
  kvCache: z
    .discriminatedUnion('mode', [
      z.strictObject({ mode: z.literal('full') }),
      z.strictObject({
        mode: z.literal('sliding-window'),
        windowSize: z.number().int().min(1).max(1048576),
      }),
      z.strictObject({
        mode: z.literal('sparse'),
        windowSize: z.number().int().min(1).max(1048576),
        stride: z.number().int().min(2).max(1048576),
      }),
    ])
    .optional(),
})

const tensorSchema = z.strictObject({
  name: text,
  shape: z.array(axis).max(10),
  role: z.enum(['input', 'output', 'weight', 'state']),
  description: text.optional(),
})

export const componentSchema = z.strictObject({
  id,
  kind: id,
  title: z.string().max(80),
  subtitle: z.string().max(100),
  category: z.enum(['data', 'compute', 'memory', 'output']),
  description: text,
  equation: text.optional(),
  tensors: z.array(tensorSchema).max(30),
  notes: z.array(text).max(20).default([]),
  sourceIds: z.array(id).max(20).default([]),
  childGraph: id.optional(),
  repeat: axis.optional(),
  extensions: z.record(z.string(), z.unknown()).optional(),
})

const connectionSchema = z.strictObject({
  id,
  source: id,
  target: id,
  label: z.string().max(80),
  operation: text,
  equation: text.optional(),
  shape: z.array(axis).max(10),
  description: text,
  kind: z.enum(['flow', 'residual', 'state', 'auxiliary']).default('flow'),
  sourceSide: z.enum(['top', 'right', 'bottom', 'left']).optional(),
  targetSide: z.enum(['top', 'right', 'bottom', 'left']).optional(),
})

const graphSchema = z.strictObject({
  id,
  title: text,
  subtitle: text,
  description: text,
  nodes: z.array(componentSchema).min(1).max(250),
  edges: z.array(connectionSchema).max(800),
  layout: z.strictObject({
    direction: z.enum(['RIGHT', 'DOWN']),
    positions: z.record(id, point),
    viewport: z
      .strictObject({
        x: z.number().finite(),
        y: z.number().finite(),
        zoom: z.number().min(0.001).max(4),
      })
      .optional(),
  }),
})

export const documentSchema = z.strictObject({
  $schema: z.string().optional(),
  schemaVersion: z.literal('1.0'),
  id,
  name: z.string().min(1).max(120),
  description: text,
  documentation: text,
  config: configSchema,
  tokenizer: z.strictObject({ adapter: z.string().min(1).max(100), description: text }),
  sources: z
    .array(
      z.strictObject({
        id,
        title: text,
        url: z.url().refine((v) => /^https?:\/\//.test(v), 'Use an HTTP(S) source URL'),
        note: text,
      }),
    )
    .max(100),
  graphs: z.array(graphSchema).min(1).max(30),
  ui: z.strictObject({
    activeGraph: id,
    theme: z.enum(['dark', 'light']),
    prompt: z.string().max(600),
    notes: text,
  }),
  extensions: z.record(z.string(), z.unknown()).optional(),
})

export type ModelDocument = z.infer<typeof documentSchema>
export type ModelConfig = z.infer<typeof configSchema>
export type Component = z.infer<typeof componentSchema>
export type Connection = z.infer<typeof connectionSchema>
export type ModelGraph = z.infer<typeof graphSchema>
export type Tensor = z.infer<typeof tensorSchema>

/** Only +, -, *, /, integer literals, parentheses and known symbols. Never eval. */
export function resolveAxis(expression: string, symbols: Record<string, number>): number {
  const tokens = expression.match(/[A-Za-z][A-Za-z0-9]*|\d+|[()+*/-]/g) ?? []
  if (tokens.join('') !== expression.replace(/\s/g, ''))
    throw new Error(`Invalid dimension: ${expression}`)
  let cursor = 0
  function primary(): number {
    const token = tokens[cursor++]
    if (token === '(') {
      const value = sum()
      if (tokens[cursor++] !== ')') throw new Error('Unclosed dimension expression')
      return value
    }
    if (token && /^\d+$/.test(token)) return Number(token)
    if (token && Object.hasOwn(symbols, token)) return symbols[token]
    throw new Error(`Unknown dimension: ${token ?? expression}`)
  }
  function product(): number {
    let value = primary()
    while (tokens[cursor] === '*' || tokens[cursor] === '/') {
      const op = tokens[cursor++]
      const right = primary()
      value = op === '*' ? value * right : value / right
    }
    return value
  }
  function sum(): number {
    let value = product()
    while (tokens[cursor] === '+' || tokens[cursor] === '-') {
      const op = tokens[cursor++]
      const right = product()
      value = op === '+' ? value + right : value - right
    }
    return value
  }
  const value = sum()
  if (
    cursor !== tokens.length ||
    !Number.isSafeInteger(value) ||
    (value < 1 && !(value === 0 && ['P', 'R'].includes(expression.trim())))
  )
    throw new Error(`Dimension must be a positive integer: ${expression}`)
  return value
}

export function dimensions(config: ModelConfig, tokenCount = 12, decode = false) {
  return {
    B: 1,
    T: decode ? 1 : tokenCount,
    S: tokenCount + (decode ? 1 : 0),
    R: residentCount(config, tokenCount + (decode ? 1 : 0)),
    P: residentCount(config, decode ? tokenCount : 0),
    U: residentCount(config, decode ? tokenCount : 0) + (decode ? 1 : tokenCount),
    D: config.dModel,
    L: config.nLayers,
    Hq: config.nQueryHeads,
    Hkv: config.nKvHeads,
    Dh: config.headDim,
    F: config.ffnDim,
    V: config.vocabSize,
    C: config.contextLength,
    N: 4,
    M: 4096,
    E: 128,
  }
}

export function parseDocument(input: unknown): ModelDocument {
  const doc = documentSchema.parse(input)
  const c = doc.config
  if (
    c.kvCache &&
    c.kvCache.mode !== 'full' &&
    (c.kvCache.windowSize > c.contextLength ||
      (c.kvCache.mode === 'sparse' && c.kvCache.stride > c.contextLength))
  )
    throw new Error('Cache window and sparse stride must not exceed the context capacity.')
  if (c.nQueryHeads % c.nKvHeads !== 0)
    throw new Error('Query heads must be divisible by KV heads.')
  if (c.dModel !== c.nQueryHeads * c.headDim)
    throw new Error('This teaching architecture requires D = Hq × Dh.')
  if (c.headDim % 2 !== 0)
    throw new Error('RoPE requires an even head dimension in this architecture.')
  if (c.attention === 'MHA' && c.nKvHeads !== c.nQueryHeads)
    throw new Error('MHA requires one KV head per query head.')
  if (c.attention === 'MQA' && c.nKvHeads !== 1)
    throw new Error('MQA requires exactly one KV head.')
  if (c.attention === 'GQA' && (c.nKvHeads <= 1 || c.nKvHeads >= c.nQueryHeads))
    throw new Error(
      'GQA requires more than one and fewer than Hq KV heads. Use MHA or MQA at the endpoints.',
    )
  const graphIds = new Set(doc.graphs.map((g) => g.id))
  if (graphIds.size !== doc.graphs.length) throw new Error('Graph IDs must be unique.')
  if (!graphIds.has(doc.ui.activeGraph)) throw new Error('The active graph does not exist.')
  const sourceIds = new Set(doc.sources.map((s) => s.id))
  if (sourceIds.size !== doc.sources.length) throw new Error('Source IDs must be unique.')
  const symbols = dimensions(c)
  for (const authored of doc.graphs) {
    const graph = expandWholeModel(doc, authored)
    const ids = new Set(graph.nodes.map((n) => n.id))
    if (ids.size !== graph.nodes.length) throw new Error(`Duplicate node ID in ${graph.id}.`)
    if (new Set(graph.edges.map((e) => e.id)).size !== graph.edges.length)
      throw new Error(`Duplicate edge ID in ${graph.id}.`)
    for (const node of graph.nodes) {
      if (node.childGraph && !graphIds.has(node.childGraph))
        throw new Error(`Unknown child graph: ${node.childGraph}`)
      for (const source of node.sourceIds)
        if (!sourceIds.has(source)) throw new Error(`Unknown source: ${source}`)
      for (const tensor of node.tensors) for (const axis of tensor.shape) resolveAxis(axis, symbols)
      if (node.repeat) resolveAxis(node.repeat, symbols)
    }
    for (const edge of graph.edges) {
      if (!ids.has(edge.source) || !ids.has(edge.target))
        throw new Error(`Dangling connection: ${edge.id}`)
      for (const axis of edge.shape) resolveAxis(axis, symbols)
    }
    for (const key of Object.keys(graph.layout.positions))
      if (!ids.has(key)) throw new Error(`Position references unknown node: ${key}`)
  }
  return doc
}

export function shapeLabel(shape: string[], symbols: Record<string, number>, symbolic = false) {
  return `[${shape
    .map((axis) => {
      if (symbolic) return axis
      try {
        return resolveAxis(axis, symbols).toLocaleString('en-GB')
      } catch {
        return '?'
      }
    })
    .join(' × ')}]`
}

export function cacheBytes(config: ModelConfig, sequenceLength: number) {
  return (
    2 *
    config.nLayers *
    config.nKvHeads *
    residentCount(config, sequenceLength) *
    config.headDim *
    config.dtypeBytes
  )
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  const unit = bytes >= 1024 ** 3 ? 'GiB' : bytes >= 1024 ** 2 ? 'MiB' : 'KiB'
  const divisor = unit === 'GiB' ? 1024 ** 3 : unit === 'MiB' ? 1024 ** 2 : 1024
  return `${(bytes / divisor).toLocaleString('en-GB', { maximumFractionDigits: 2 })} ${unit}`
}
