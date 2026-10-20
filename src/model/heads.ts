import type { Component, ModelDocument, ModelGraph } from './schema'
import { kvHeadFor } from './attention.ts'

export const HEADS_KEY = 'model-visualizer.heads'
export function isHeadsGraph(graph: ModelGraph) {
  return (
    graph.nodes.length === 1 &&
    graph.nodes[0].kind === 'attention-expansion' &&
    Object.hasOwn(graph.nodes[0].extensions ?? {}, HEADS_KEY)
  )
}

/** Explicit mathematical head paths. Projections remain batched; splitting is
 * a view/reshape, not a second projection or a physical duplication of K/V. */
export function expandHeads(
  doc: Pick<ModelDocument, 'graphs' | 'config'>,
  graph: ModelGraph,
): ModelGraph {
  if (!isHeadsGraph(graph)) return graph
  const recipe = graph.nodes[0].extensions![HEADS_KEY] as { version: number; graph: string }
  const source = doc.graphs.find((g) => g.id === recipe.graph)
  if (
    recipe.version !== 1 ||
    !source ||
    isHeadsGraph(source) ||
    source.nodes.some((n) => n.kind === 'model-expansion')
  )
    throw new Error('Invalid or recursive attention-head recipe.')
  const original = (id: string) => {
    const node = source.nodes.find((n) => n.id === id)
    if (!node) throw new Error(`Missing attention template: ${id}`)
    return node
  }
  const baseIds = new Set([
    'x',
    'q',
    'k',
    'v',
    'ropeq',
    'ropek',
    'cachek',
    'cachev',
    'concat',
    'project',
  ])
  const nodes = source.nodes.filter((n) => baseIds.has(n.id)).map((n) => structuredClone(n))
  for (const node of nodes)
    if (['cachek', 'cachev'].includes(node.id)) {
      node.tensors = node.tensors.map((t) => ({
        ...t,
        shape: t.shape.map((a) => (a === 'S' ? 'R' : a)),
      }))
    }
  const edges = source.edges
    .filter((e) => baseIds.has(e.source) && baseIds.has(e.target))
    .map((e) => structuredClone(e))
  const link = (
    a: string,
    b: string,
    label: string,
    shape: string[],
    description: string,
    state = false,
  ) => {
    edges.push({
      id: `${a}--${b}`,
      source: a,
      target: b,
      label,
      shape,
      description,
      operation: description,
      kind: state ? 'state' : 'flow',
    })
  }
  const add = (
    template: string,
    id: string,
    title: string,
    description: string,
    shape: string[],
    equation?: string,
    memory = false,
  ) => {
    const node: Component = {
      ...original(template),
      id,
      title,
      subtitle: shape.join(' × '),
      description,
      childGraph: undefined,
      equation,
      tensors: [{ name: title, role: memory ? 'state' : 'output', shape }],
      sourceIds: [...new Set(['attention', 'gqa', ...original(template).sourceIds])],
    }
    nodes.push(node)
  }
  for (let h = 0; h < doc.config.nKvHeads; h++) {
    for (const [id, plane, projected] of [
      ['cachek', 'K', 'ropek'],
      ['cachev', 'V', 'v'],
    ] as const) {
      const slice = `${id}-${h}`
      add(
        id,
        slice,
        `${plane} head ${h} · working ${plane === 'K' ? 'keys' : 'values'}`,
        `KV group ${h}: gather the retained prefix by absolute position and concatenate this call’s new ${plane} vectors. U = P + T working positions include transient keys needed by earlier queries in a prefill or verification batch. Cache reads use the pre-update snapshot; retention is committed after the call. No KV vectors are copied for each query head.`,
        ['B', 'U', 'Dh'],
      )
      link(
        id,
        slice,
        `prefix ${plane}${h}`,
        ['B', 'P', 'Dh'],
        `Read KV head ${h} from the pre-update cache snapshot, preserving its absolute position map.`,
        true,
      )
      link(
        projected,
        slice,
        `new ${plane}${h}`,
        ['B', 'T', 'Dh'],
        `View new head ${h}; concatenate with retained prefix keys in absolute-position order.`,
      )
    }
  }
  for (let h = 0; h < doc.config.nQueryHeads; h++) {
    const kv = kvHeadFor(doc.config, h)
    add(
      'q',
      `query-${h}`,
      `Query head ${h}`,
      `View Q[:, ${h}, :, :] after projection and positional rotation. The slice has Dh features. It reads K${kv} and V${kv}; each query head still has its own queries, scores and softmax.`,
      ['B', 'T', 'Dh'],
    )
    add(
      'scores',
      `scores-${h}`,
      `Q${h} · K${kv}ᵀ`,
      `Contract Dh independently for query head ${h}, using shared KV group ${kv}. The U working key slots have an explicit absolute-position map.`,
      ['B', 'T', 'U'],
      `Z^{(${h})}_{b,t,u}=\\sum_j Q_{b,${h},t,j}K_{b,${kv},u,j}`,
    )
    add(
      'scale',
      `scale-${h}`,
      `Head ${h} · scale + mask`,
      'Divide scores by √Dh. Set disallowed absolute query/key pairs to −∞ according to the full, sliding-window or fixed-anchor sparse causal policy. Future positions are always excluded.',
      ['B', 'T', 'U'],
      `\\widetilde Z^{(${h})}=Z^{(${h})}/\\sqrt{d_h}+M`,
    )
    add(
      'softmax',
      `softmax-${h}`,
      `Head ${h} · softmax`,
      `Normalize only visible key slots for head ${h}. Other query heads have independent attention distributions even when they share this KV group.`,
      ['B', 'T', 'U'],
      `A^{(${h})}_{t,u}=\\operatorname{softmax}_u(\\widetilde Z^{(${h})}_{t,u})`,
    )
    add(
      'weighted',
      `head-${h}`,
      `Query head ${h} · result`,
      `Contract the key-slot axis with V${kv}, returning Dh features. This is the actual head operation, not another attention pass over a batched result.`,
      ['B', 'T', 'Dh'],
      `O^{(${h})}_{b,t,j}=\\sum_u A^{(${h})}_{b,t,u}V_{b,${kv},u,j}`,
    )
    nodes.at(-1)!.kind = 'attention-head'
    link(
      'ropeq',
      `query-${h}`,
      `view Q${h}`,
      ['B', 'T', 'Dh'],
      'Select a query-head slice; projection is performed once in the batched Wq operation.',
    )
    link(
      `query-${h}`,
      `scores-${h}`,
      'dot product · Dh',
      ['B', 'T', 'Dh'],
      'Use this query head as the left operand; contract feature axis Dh with the transposed key operand.',
    )
    link(
      `cachek-${kv}`,
      `scores-${h}`,
      `K${kv} transpose`,
      ['B', 'U', 'Dh'],
      'Shared key operand; transpose only its last two axes. Sharing does not average query heads.',
      true,
    )
    link(
      `scores-${h}`,
      `scale-${h}`,
      'scale scores',
      ['B', 'T', 'U'],
      'Scale dot products by the inverse square root of head width.',
    )
    link(
      `scale-${h}`,
      `softmax-${h}`,
      'masked scores',
      ['B', 'T', 'U'],
      'Apply the absolute-position visibility mask, then normalize over keys.',
    )
    link(
      `softmax-${h}`,
      `head-${h}`,
      'weights',
      ['B', 'T', 'U'],
      'Attention weights are the left operand of the value-weighted sum.',
    )
    link(
      `cachev-${kv}`,
      `head-${h}`,
      `read V${kv}`,
      ['B', 'U', 'Dh'],
      'Values use the same KV group and absolute-position mapping as keys.',
      true,
    )
    link(
      `head-${h}`,
      'concat',
      `join Q${h}`,
      ['B', 'T', 'Dh'],
      'Concatenate head features in query-head order; do not sum heads.',
    )
  }
  if (nodes.length > 4096 || edges.length > 12000)
    throw new Error('Attention head expansion exceeds the view budget.')
  return { ...graph, nodes, edges }
}
