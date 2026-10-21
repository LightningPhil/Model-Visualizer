import type { ModelDocument, ModelGraph } from './schema'
import { expandWholeModel } from './whole-model'

/** Runtime-only compilation cache. Layout, camera and notebook edits never
 * regenerate semantic instances. Imported content/configuration invalidates it.
 * The portable authored document remains the only source of model content. */
export function createGraphCompiler() {
  let config: ModelDocument['config'] | undefined
  let templates: ModelGraph[] = []
  const compiled = new Map<string, ModelGraph>()
  return (doc: Pick<ModelDocument, 'graphs' | 'config'>, graph: ModelGraph): ModelGraph => {
    if (
      config !== doc.config ||
      templates.length !== doc.graphs.length ||
      doc.graphs.some((g, i) => {
        const before = templates[i]
        return g.id !== before.id || g.nodes !== before.nodes || g.edges !== before.edges
      })
    ) {
      config = doc.config
      templates = doc.graphs
      compiled.clear()
    }
    let result = compiled.get(graph.id)
    if (!result) {
      result = expandWholeModel(doc, graph)
      compiled.set(graph.id, result)
    }
    return result === graph ? graph : { ...graph, nodes: result.nodes, edges: result.edges }
  }
}
