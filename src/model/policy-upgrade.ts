import { parseDocument, type ModelDocument } from './schema'

const KEY = 'model-visualizer.cache-policy-version'
/** Add policy-aware axes/working-key paths to earlier foundations workspaces.
 * Existing IDs, prose, notes, extension data and geometry remain available. */
export function upgradeCachePolicies(doc: ModelDocument, reference: ModelDocument): ModelDocument {
  if (doc.id !== reference.id || Object.hasOwn(doc.extensions ?? {}, KEY)) return doc
  const note =
    'Policy-aware state: R counts resident positions, P the retained pre-call prefix, and U = P + T working keys. Absolute position IDs are separate. Current-call K/V remain transiently available before final retention; descriptions of a cached prefix mean the retained prefix under the selected policy.'
  const graphs = doc.graphs.map((graph) => {
    if (!['attention', 'cache'].includes(graph.id)) return graph
    const current = reference.graphs.find((g) => g.id === graph.id)!
    const nodes = graph.nodes.map((node) => {
      const template = current.nodes.find((n) => n.id === node.id && n.kind === node.kind)
      if (!template) return node
      let changed = false
      const tensors = node.tensors.map((t) => {
        const canonical = template.tensors.find((x) => x.name === t.name && x.role === t.role)
        if (!canonical || canonical.shape.length !== t.shape.length) return t
        const shape = t.shape.map((axis, i) => {
          const next = canonical.shape[i]
          if (axis === 'S' && ['R', 'P', 'U'].includes(next)) {
            changed = true
            return next
          }
          return axis
        })
        return { ...t, shape }
      })
      if (!changed && node.id !== 'scale') return node
      return {
        ...node,
        tensors,
        description: node.description.replaceAll('Hkv, S, Dh', 'Hkv, R, Dh'),
        equation: node.id === 'scale' ? template.equation : node.equation,
        notes: [...node.notes, ...(node.notes.includes(note) ? [] : [note])],
        sourceIds: [...new Set([...node.sourceIds, ...template.sourceIds])],
      }
    })
    const edges = graph.edges.map((edge) => {
      const template = current.edges.find(
        (e) => e.id === edge.id && e.source === edge.source && e.target === edge.target,
      )
      if (!template || template.shape.length !== edge.shape.length) return edge
      return {
        ...edge,
        shape: edge.shape.map((axis, i) =>
          axis === 'S' && ['P', 'R', 'U'].includes(template.shape[i]) ? template.shape[i] : axis,
        ),
      }
    })
    const ids = new Set(nodes.map((n) => n.id))
    const additions = current.edges.filter(
      (e) =>
        e.id.endsWith('-current') &&
        ids.has(e.source) &&
        ids.has(e.target) &&
        !edges.some((old) => old.id === e.id),
    )
    return { ...graph, nodes, edges: [...edges, ...structuredClone(additions)] }
  })
  const candidate = {
    ...doc,
    graphs,
    extensions: { ...doc.extensions, [KEY]: 1 },
    sources: [
      ...doc.sources,
      ...reference.sources.filter((s) => !doc.sources.some((old) => old.id === s.id)),
    ],
  }
  try {
    parseDocument(candidate)
    return candidate
  } catch {
    return doc
  }
}
