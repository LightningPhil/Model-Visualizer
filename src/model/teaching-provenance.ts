import { parseDocument, type ModelDocument } from './schema'

export const TEACHING_PROVENANCE_KEY = 'model-visualizer.teaching-provenance-version'
/** Historical stock text is only an exact-match migration key. Current teaching
 * content comes from the authored JSON; custom explanations are never replaced. */
export const LEGACY_DOCUMENTATION =
  'Start at the architecture overview, then open a block or inspect an edge. Symbolic axes are B (batch), T (current query positions), S (visible key positions), D (hidden width), Hq (query heads), Hkv (KV heads), Dh (features per head), F (FFN width), V (vocabulary) and L (layers). The workbench computes exact dimensions and a real demo BPE trace. Attention probabilities are synthetic examples; no trained model weights are loaded. Optional memory and MTP views illustrate separate architecture extensions. Model definitions, equations, citations, notes, positions and viewports travel together in this versioned JSON document.'
export const LEGACY_SIMULATION_NOTE =
  'Inference teaching path. The runnable simulator uses explicitly disclosed tiny, untrained weights.'

export function upgradeTeachingProvenance(
  doc: ModelDocument,
  reference: ModelDocument,
): ModelDocument {
  if (doc.id !== reference.id || Object.hasOwn(doc.extensions ?? {}, TEACHING_PROVENANCE_KEY))
    return doc
  const graph = reference.graphs.find((g) => g.id === 'generation')
  const candidate = {
    ...doc,
    documentation:
      doc.documentation === LEGACY_DOCUMENTATION ? reference.documentation : doc.documentation,
    graphs: doc.graphs.map((g) =>
      g.id !== 'generation'
        ? g
        : {
            ...g,
            nodes: g.nodes.map((n) => {
              const template = graph?.nodes.find((t) => t.id === n.id && t.kind === n.kind),
                note = template?.notes.find((s) => s.startsWith('Graphical simulation uses'))
              return note
                ? { ...n, notes: n.notes.map((s) => (s === LEGACY_SIMULATION_NOTE ? note : s)) }
                : n
            }),
          },
    ),
    extensions: { ...doc.extensions, [TEACHING_PROVENANCE_KEY]: 1 },
  }
  try {
    parseDocument(candidate)
    return candidate
  } catch {
    return doc
  }
}
