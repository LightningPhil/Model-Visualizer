import { z } from 'zod'
import { documentSchema, type ModelDocument } from './schema'
import { expandWholeModel, isWholeModel } from './whole-model'
import { isHeadsGraph } from './heads'

const KEY = 'model-visualizer.configuration-layouts'
const archiveSchema = z.strictObject({
  version: z.literal(1),
  layouts: z.record(
    z.string(),
    z.record(z.string(), documentSchema.shape.graphs.element.shape.layout),
  ),
})
const signature = (doc: ModelDocument) =>
  `${doc.config.nLayers}/${doc.config.nQueryHeads}/${doc.config.nKvHeads}`

/** Topology edits need new routes and often a new arrangement. Preserve the
 * previous authored geometry by configuration, restoring it when revisited. */
export function configurationLayouts(
  previous: ModelDocument,
  candidate: ModelDocument,
): ModelDocument {
  const changed = signature(previous) !== signature(candidate)
  const stored = previous.extensions?.[KEY]
  const parsed = archiveSchema.safeParse(stored ?? { version: 1, layouts: {} })
  const archive = parsed.success ? parsed.data : undefined
  const oldKey = signature(previous),
    newKey = signature(candidate)
  if (changed && archive)
    archive.layouts[oldKey] = {
      ...archive.layouts[oldKey],
      ...Object.fromEntries(
        previous.graphs
          .filter((g) => isWholeModel(g) || isHeadsGraph(g))
          .map((g) => [g.id, structuredClone(g.layout)]),
      ),
    }
  return {
    ...candidate,
    extensions:
      changed && archive ? { ...candidate.extensions, [KEY]: archive } : candidate.extensions,
    graphs: candidate.graphs.map((g) => {
      if (!isWholeModel(g) && !isHeadsGraph(g)) return g
      const ids = new Set(expandWholeModel(candidate, g).nodes.map((n) => n.id))
      const layout =
        changed && archive
          ? (archive.layouts[newKey]?.[g.id] ?? { direction: g.layout.direction, positions: {} })
          : g.layout
      return {
        ...g,
        layout: {
          ...layout,
          positions: Object.fromEntries(
            Object.entries(layout.positions).filter(([id]) => ids.has(id)),
          ),
        },
      }
    }),
  }
}
