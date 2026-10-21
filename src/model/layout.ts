import type { ModelGraph } from './schema'
import type { AvoidEngine } from './avoid'
import type { ELK } from 'elkjs/lib/elk-api'
import { labelSize, NODE_HEIGHT, NODE_WIDTH } from './geometry'
import { routeGraph, routingProblems, type Positions, type RoutingResult } from './routing'
import { planGrouped, usesStageLayout } from './group-layout'

export { NODE_WIDTH, NODE_HEIGHT } from './geometry'
export type LayoutPlan = { positions: Positions; routing: RoutingResult }

let layoutEngine: ELK | undefined
/** The ELK bundle installs self.onmessage when loaded inside a worker. Give it
 * its own worker instead of allowing it to replace our diagram protocol. */
export async function configureLayoutWorker(url: string) {
  const { default: ELK } = await import('elkjs/lib/elk-api.js')
  layoutEngine = new ELK({ workerUrl: url })
}

async function placeNodes(graph: ModelGraph, direction: 'RIGHT' | 'DOWN', attempt: number) {
  if (!layoutEngine) {
    const { default: ELK } = await import('elkjs/lib/elk.bundled.js')
    layoutEngine = new ELK()
  }
  const elk = layoutEngine
  const vertical = direction === 'DOWN'
  const result = await elk.layout({
    id: graph.id,
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': direction,
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.edgeLabels.inline': 'true',
      'elk.edgeLabels.placement': 'CENTER',
      'elk.spacing.nodeNode': String(74 + attempt * 22),
      'elk.spacing.edgeNode': '24',
      'elk.spacing.edgeEdge': '18',
      'elk.spacing.labelNode': '24',
      'elk.spacing.labelLabel': '18',
      // Center labels occupy their own ELK layer. This spacing applies on
      // BOTH sides of that layer; a large value needlessly doubles the gaps.
      'elk.layered.spacing.nodeNodeBetweenLayers': String((vertical ? 26 : 36) + attempt * 24),
      'elk.layered.spacing.edgeNodeBetweenLayers': '18',
      'elk.layered.spacing.edgeEdgeBetweenLayers': '18',
      'elk.layered.nodePlacement.strategy': attempt === 1 ? 'NETWORK_SIMPLEX' : 'BRANDES_KOEPF',
      'elk.layered.nodePlacement.bk.fixedAlignment': 'BALANCED',
      'elk.layered.nodePlacement.favorStraightEdges': 'true',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
      'elk.layered.thoroughness': '20',
      'elk.padding': '[top=56,left=56,bottom=56,right=56]',
      'elk.randomSeed': '1',
    },
    children: graph.nodes.map((node) => ({
      id: node.id,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      layoutOptions: { 'elk.portConstraints': 'FIXED_SIDE' },
      ports: graph.edges.flatMap((edge) => [
        ...(edge.source === node.id
          ? [
              {
                id: `s-${edge.id}`,
                width: 0,
                height: 0,
                layoutOptions: { 'elk.port.side': vertical ? 'SOUTH' : 'EAST' },
              },
            ]
          : []),
        ...(edge.target === node.id
          ? [
              {
                id: `t-${edge.id}`,
                width: 0,
                height: 0,
                layoutOptions: { 'elk.port.side': vertical ? 'NORTH' : 'WEST' },
              },
            ]
          : []),
      ]),
    })),
    // State, residual and auxiliary links all need room in the same diagram.
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      sources: [`s-${edge.id}`],
      targets: [`t-${edge.id}`],
      labels: [{ id: `label-${edge.id}`, text: edge.label, ...labelSize(edge.label) }],
      layoutOptions: { 'elk.layered.priority.direction': edge.kind === 'flow' ? '10' : '1' },
    })),
  })
  return Object.fromEntries(
    (result.children ?? []).map((node) => [node.id, { x: node.x ?? 0, y: node.y ?? 0 }]),
  )
}

export async function autoLayout(graph: ModelGraph, direction = graph.layout.direction) {
  return placeNodes(graph, direction, 0)
}

/** Compare complete diagrams rather than scoring node positions alone. */
export async function planLayout(
  A: AvoidEngine,
  graph: ModelGraph,
  direction = graph.layout.direction,
): Promise<LayoutPlan> {
  const automatic: ModelGraph = {
    ...graph,
    edges: graph.edges.map((edge) => ({ ...edge, sourceSide: undefined, targetSide: undefined })),
    layout: { ...graph.layout, direction },
  }
  if (usesStageLayout(automatic)) return planGrouped(A, automatic, direction)
  let best: (LayoutPlan & { score: number }) | undefined
  for (let attempt = 0; attempt < 3; attempt++) {
    const positions = await placeNodes(automatic, direction, attempt)
    let routing: RoutingResult
    try {
      routing = routeGraph(A, automatic, positions, direction)
    } catch {
      continue
    } // A cramped candidate must not prevent trying wider arrangements.
    if (routingProblems(automatic, positions, routing).length) continue
    const length = Object.values(routing.edges).reduce(
      (sum, edge) =>
        sum +
        edge.points
          .slice(1)
          .reduce(
            (s, p, i) => s + Math.abs(p.x - edge.points[i].x) + Math.abs(p.y - edge.points[i].y),
            0,
          ),
      0,
    )
    const score =
      routing.crossings * 1200 +
      routing.bends * 100 +
      length * 0.015 +
      routing.bounds.width * routing.bounds.height * 0.00001
    if (!best || score < best.score) best = { positions, routing, score }
  }
  if (!best)
    throw new Error('A clear layout could not be found. Your previous layout is preserved.')
  return { positions: best.positions, routing: best.routing }
}
