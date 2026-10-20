import type { ModelGraph } from './schema'

export const NODE_WIDTH = 246
export const NODE_HEIGHT = 112

export async function autoLayout(graph: ModelGraph, direction = graph.layout.direction) {
  const { default: ELK } = await import('elkjs/lib/elk.bundled.js')
  const elk = new ELK()
  const result = await elk.layout({
    id: graph.id,
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': direction,
      'elk.spacing.nodeNode': '50',
      'elk.layered.spacing.nodeNodeBetweenLayers': direction === 'RIGHT' ? '92' : '64',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.padding': '[top=40,left=40,bottom=40,right=40]',
      'elk.randomSeed': '1',
    },
    children: graph.nodes.map((node) => ({ id: node.id, width: NODE_WIDTH, height: NODE_HEIGHT })),
    edges: graph.edges
      .filter((edge) => edge.kind !== 'auxiliary')
      .map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] })),
  })
  return Object.fromEntries(
    (result.children ?? []).map((node) => [node.id, { x: node.x ?? 0, y: node.y ?? 0 }]),
  )
}
