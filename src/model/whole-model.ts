import type { Component, Connection, ModelDocument, ModelGraph } from './schema'

export const EXPANSION_KEY = 'model-visualizer.expansion'
export const INSTANCE_KEY = 'model-visualizer.instance'
type Ref = { graph: string; entry: string; exit: string }
type Recipe = {
  version: 1
  input: Ref
  layers: Ref & { expand: { node: string; graph: string; entry: string; exit: string }[] }
  output: Ref
  memory: Ref
  generation: Ref
}
export type Instance = {
  stage: string
  graph: string
  node: string
  layer?: number
  head?: number
  optional?: boolean
}

export function isWholeModel(graph: ModelGraph) {
  return (
    graph.nodes.length === 1 &&
    graph.nodes[0].kind === 'model-expansion' &&
    Object.hasOwn(graph.nodes[0].extensions ?? {}, EXPANSION_KEY)
  )
}
export function instanceOf(node: Component) {
  return node.extensions?.[INSTANCE_KEY] as Instance | undefined
}

/** Expand a portable recipe from the authored document. No generated content
 * is stored alongside its source templates. IDs remain stable across layouts. */
export function expandWholeModel(
  doc: Pick<ModelDocument, 'graphs' | 'config'>,
  recipeGraph: ModelGraph,
): ModelGraph {
  if (!isWholeModel(recipeGraph)) return recipeGraph
  const recipe = recipeGraph.nodes[0].extensions?.[EXPANSION_KEY] as Recipe | undefined
  if (
    !recipe ||
    recipe.version !== 1 ||
    !recipe.input ||
    !recipe.layers ||
    !recipe.output ||
    !recipe.generation ||
    !recipe.memory ||
    !Array.isArray(recipe.layers.expand)
  )
    throw new Error('Unsupported whole-model expansion recipe.')
  const nodes: Component[] = [],
    edges: Connection[] = []
  const refGraph = (id: string) => {
    const g = doc.graphs.find((g) => g.id === id)
    if (!g || isWholeModel(g)) throw new Error(`Expansion template is missing or recursive: ${id}`)
    return g
  }
  const layerSize =
    refGraph(recipe.layers.graph).nodes.length -
    recipe.layers.expand.length +
    recipe.layers.expand.reduce((sum, e) => sum + refGraph(e.graph).nodes.length, 0) +
    doc.config.nQueryHeads +
    2 * doc.config.nKvHeads
  if (layerSize * doc.config.nLayers + 50 > 4096)
    throw new Error(
      'The expanded model exceeds the 4,096-component view budget. Use a smaller teaching configuration.',
    )
  function instantiate(ref: Ref, prefix: string, stage: string, layer?: number, optional = false) {
    const g = refGraph(ref.graph)
    if (!g.nodes.some((n) => n.id === ref.entry) || !g.nodes.some((n) => n.id === ref.exit))
      throw new Error(`Invalid expansion boundary in ${ref.graph}.`)
    nodes.push(
      ...g.nodes.map((n) => ({
        ...n,
        id: `${prefix}.${n.id}`,
        repeat: undefined,
        title: layer === undefined ? n.title : `L${layer} · ${n.title}`,
        extensions: {
          ...n.extensions,
          [INSTANCE_KEY]: { stage, graph: g.id, node: n.id, layer, optional } satisfies Instance,
        },
      })),
    )
    edges.push(
      ...g.edges.map((e) => ({
        ...e,
        id: `${prefix}.${e.id}`,
        source: `${prefix}.${e.source}`,
        target: `${prefix}.${e.target}`,
        sourceSide: undefined,
        targetSide: undefined,
      })),
    )
    return { entry: `${prefix}.${ref.entry}`, exit: `${prefix}.${ref.exit}` }
  }
  function link(
    source: string,
    target: string,
    label: string,
    kind: Connection['kind'] = 'flow',
    shape?: string[],
  ) {
    edges.push({
      id: `${source}--${target}`,
      source,
      target,
      label,
      kind,
      operation: label,
      description: `${label}. This connects instantiated components in the complete inference graph.`,
      shape: shape ??
        nodes
          .find((n) => n.id === source)
          ?.tensors.find((t) => t.role === 'output' || t.role === 'state')?.shape ?? [
          'B',
          'T',
          'D',
        ],
    })
  }
  const input = instantiate(recipe.input, 'tokens', 'Input')
  let previous = input.exit
  for (let layer = 0; layer < doc.config.nLayers; layer++) {
    const prefix = `layer-${layer}`,
      stage = `Layer ${String(layer).padStart(2, '0')}`
    const block = instantiate(recipe.layers, prefix, stage, layer)
    link(previous, block.entry, layer ? 'next decoder layer' : 'embedded sequence')
    for (const expansion of recipe.layers.expand) {
      const replaced = `${prefix}.${expansion.node}`
      if (!nodes.some((n) => n.id === replaced))
        throw new Error(`Unknown expansion node: ${replaced}`)
      const inner = instantiate(expansion, `${prefix}.${expansion.node}`, stage, layer)
      for (const edge of edges) {
        if (edge.source === replaced) edge.source = inner.exit
        if (edge.target === replaced) edge.target = inner.entry
      }
      nodes.splice(
        nodes.findIndex((n) => n.id === replaced),
        1,
      )
    }
    const attentionPrefix = `${prefix}.attn`
    const weighted = nodes.find((n) => n.id === `${attentionPrefix}.weighted`)
    const concatenate = nodes.find((n) => n.id === `${attentionPrefix}.concat`)
    if (weighted && concatenate) {
      const original = edges.findIndex(
        (e) => e.source === weighted.id && e.target === concatenate.id,
      )
      if (original >= 0) edges.splice(original, 1)
      for (let head = 0; head < doc.config.nQueryHeads; head++) {
        const kv = Math.floor(head / (doc.config.nQueryHeads / doc.config.nKvHeads)),
          id = `${attentionPrefix}.head-${head}`
        nodes.push({
          ...weighted,
          id,
          kind: 'attention-head',
          title: `L${layer} · Query head ${head}`,
          subtitle: `Q${head} → K${kv} / V${kv} · head result`,
          description: `Head ${head} uses KV head ${kv}. This node exposes the per-head slice of the batched attention result; it does not apply attention a second time. Each layer has its own weights and state.`,
          equation: `O_{${head}}=\\operatorname{softmax}\\!\\left(\\frac{Q_{${head}}K_{${kv}}^\\top}{\\sqrt{d_h}}+M\\right)V_{${kv}}`,
          tensors: [{ name: `Head ${head} result`, role: 'output', shape: ['B', 'T', 'Dh'] }],
          extensions: {
            [INSTANCE_KEY]: {
              stage,
              graph: 'attention',
              node: 'weighted',
              layer,
              head,
            } satisfies Instance,
          },
        })
        link(weighted.id, id, `slice Q${head}`, 'flow', ['B', 'T', 'Dh'])
        link(id, concatenate.id, `join Q${head}`)
      }
    }
    for (const [cacheNode, destination, plane] of [
      ['cachek', 'scores', 'K'],
      ['cachev', 'weighted', 'V'],
    ] as const) {
      const cache = nodes.find((n) => n.id === `${attentionPrefix}.${cacheNode}`)
      const target = `${attentionPrefix}.${destination}`
      if (!cache || !nodes.some((n) => n.id === target)) continue
      const original = edges.findIndex((e) => e.source === cache.id && e.target === target)
      if (original >= 0) edges.splice(original, 1)
      for (let head = 0; head < doc.config.nKvHeads; head++) {
        const id = `${attentionPrefix}.${cacheNode}-${head}`
        nodes.push({
          ...cache,
          id,
          title: `L${layer} · ${plane} head ${head}`,
          subtitle: `Layer ${layer} · KV head ${head} · [B, S, Dh]`,
          description: `A logical head slice of the ${plane} cache at decoder layer ${layer}. The complete logical state has axes [L, B, Hkv, S, Dh]. A slice is a view, not a second allocation. Physical storage and paging are implementation choices.`,
          tensors: [{ name: `${plane} slice`, role: 'state', shape: ['B', 'S', 'Dh'] }],
          extensions: {
            [INSTANCE_KEY]: {
              stage,
              graph: 'attention',
              node: cacheNode,
              layer,
              head,
            } satisfies Instance,
          },
        })
        link(cache.id, id, `view ${plane}${head}`, 'state', ['B', 'S', 'Dh'])
        link(id, target, `read ${plane}${head}`, 'state')
      }
    }
    previous = block.exit
  }
  const output = instantiate(recipe.output, 'output', 'Output')
  link(previous, output.entry, 'final hidden states')
  const memory = instantiate(recipe.memory, 'memory', 'Optional n-gram memory', undefined, true)
  link('tokens.ids', memory.entry, 'optional n-grams', 'auxiliary')
  link(input.exit, 'memory.context', 'context for optional gate', 'auxiliary')
  link(memory.exit, 'layer-0.input', 'optional memory contribution', 'auxiliary')
  const generation = instantiate(recipe.generation, 'generation', 'Generation & verification')
  link('output.sample', generation.entry, 'base next-token prediction')
  link(previous, 'generation.hidden', 'last hidden state', 'auxiliary')
  link('generation.verify', 'tokens.lookup', 'embed proposal IDs', 'state', ['B', 'T'])
  link('output.lmhead', 'generation.compare', 'shifted target logits', 'auxiliary', ['B', 'T', 'V'])
  link(generation.exit, 'tokens.ids', 'next iteration · token IDs', 'state')
  if (nodes.length > 4096 || edges.length > 12000)
    throw new Error(
      'The expanded model exceeds the 4,096-component view budget. Use a smaller teaching configuration.',
    )
  const ids = new Set(nodes.map((n) => n.id))
  if (ids.size !== nodes.length || edges.some((e) => !ids.has(e.source) || !ids.has(e.target)))
    throw new Error('Invalid expanded-model references.')
  return { ...recipeGraph, nodes, edges }
}

/** Opt existing compatible workspaces into the new authored recipe without
 * replacing their content, notes, metadata or any saved study layout. */
export function addWholeModelView(doc: ModelDocument, reference: ModelDocument): ModelDocument {
  const recipe = reference.graphs.find(isWholeModel)
  if (!recipe || doc.graphs.some(isWholeModel) || doc.graphs.some((g) => g.id === recipe.id))
    return doc
  const generation = reference.graphs.find((g) => g.id === 'generation')
  const graphs = [
    ...doc.graphs,
    ...(generation && !doc.graphs.some((g) => g.id === generation.id)
      ? [structuredClone(generation)]
      : []),
    structuredClone(recipe),
  ]
  const candidate = {
    ...doc,
    graphs,
    sources: [
      ...doc.sources,
      ...reference.sources.filter((s) => !doc.sources.some((old) => old.id === s.id)),
    ],
  }
  if (candidate.graphs.length > 30 || candidate.sources.length > 100) return doc
  try {
    expandWholeModel(candidate, recipe)
    return candidate
  } catch {
    return doc
  }
}
