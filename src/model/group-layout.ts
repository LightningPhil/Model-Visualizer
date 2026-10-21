import type { ModelGraph } from './schema'
import type { AvoidEngine } from './avoid'
import { instanceOf } from './whole-model'
import { planLayout, type LayoutPlan } from './layout'
import {
  allocatePorts,
  routeGraph,
  routingProblems,
  type Positions,
  type RoutedEdge,
  type RoutingResult,
} from './routing'
import { diagramBounds, NODE_HEIGHT, NODE_WIDTH, segments, type Rect, type Point } from './geometry'

/** Pack complete routed footprints, not equal cells sized for the largest
 * stage. Adjacent sections share a routing gutter; short rows and columns
 * never reserve the unused extent of an unrelated decoder layer. */
export function packStageFootprints(
  bounds: Rect[],
  direction: 'RIGHT' | 'DOWN',
  gap = 180,
): Point[] {
  if (!bounds.length) return []
  const vertical = direction === 'DOWN'
  const major = (r: Rect) => (vertical ? r.height : r.width)
  const minor = (r: Rect) => (vertical ? r.width : r.height)
  const area = bounds.reduce((sum, b) => sum + (b.width + gap) * (b.height + gap), 0)
  const target = Math.max(...bounds.map(major), Math.sqrt(area * 1.8))
  const bands: number[][] = [[]]
  let length = 0
  bounds.forEach((b, i) => {
    if (length && length + gap + major(b) > target) {
      bands.push([])
      length = 0
    }
    bands.at(-1)!.push(i)
    length += (length ? gap : 0) + major(b)
  })
  const offsets: Point[] = []
  let across = 0
  bands.forEach((band, row) => {
    let along = 0
    // Keep consecutive sections beside each other at each wrap. Each stage's
    // internal direction remains intact; the router owns the connecting turn.
    for (const i of row % 2 ? band.toReversed() : band) {
      offsets[i] = vertical ? { x: across, y: along } : { x: along, y: across }
      along += major(bounds[i]) + gap
    }
    across += Math.max(...band.map((i) => minor(bounds[i]))) + gap
  })
  return offsets
}

export function stageGraphs(graph: ModelGraph): ModelGraph[] {
  const stages = new Map<string, string[]>()
  for (const node of graph.nodes) {
    const stage = instanceOf(node)?.stage ?? 'Other'
    stages.set(stage, [...(stages.get(stage) ?? []), node.id])
  }
  return [...stages].map(([stage, ids]) => {
    const own = new Set(ids)
    return {
      ...graph,
      id: stage,
      nodes: graph.nodes.filter((n) => own.has(n.id)),
      edges: graph.edges.filter((e) => own.has(e.source) && own.has(e.target)),
      layout: {
        ...graph.layout,
        positions: Object.fromEntries(
          ids
            .filter((id) => graph.layout.positions[id])
            .map((id) => [id, graph.layout.positions[id]]),
        ),
      },
    }
  })
}

export function usesStageLayout(graph: ModelGraph) {
  return new Set(graph.nodes.flatMap((n) => (instanceOf(n) ? [instanceOf(n)!.stage] : []))).size > 1
}

/** Fixed local geometry is an obstacle for the inter-stage router. This keeps
 * complexity bounded without dropping any component, edge or label. */
export function routeGrouped(
  A: AvoidEngine,
  graph: ModelGraph,
  positions: Positions,
  direction = graph.layout.direction,
): RoutingResult {
  const ports = allocatePorts(graph, positions, direction)
  const edges: Record<string, RoutedEdge> = {}
  for (const stage of stageGraphs(graph)) {
    const localPositions = Object.fromEntries(stage.nodes.map((n) => [n.id, positions[n.id]]))
    const localPorts = Object.fromEntries(stage.nodes.map((n) => [n.id, ports[n.id]]))
    let routed: RoutingResult
    try {
      routed = routeGraph(A, stage, localPositions, direction, { ports: localPorts })
    } catch (error) {
      throw new Error(`${stage.id}: ${error instanceof Error ? error.message : 'Routing failed.'}`)
    }
    Object.assign(edges, routed.edges)
  }
  const internal = Object.values(edges)
  const external = { ...graph, edges: graph.edges.filter((e) => !edges[e.id]) }
  const leaders = internal.flatMap((e) => (e.labelLeader ? [e.labelLeader] : []))
  const leaderObstacles = leaders.flatMap(segments).map(({ a, b }) => ({
    x: Math.min(a.x, b.x) - 0.5,
    y: Math.min(a.y, b.y) - 0.5,
    width: Math.max(1, Math.abs(b.x - a.x)),
    height: Math.max(1, Math.abs(b.y - a.y)),
  }))
  const routed = routeGraph(A, external, positions, direction, {
    ports,
    obstacles: [...internal.map((e) => e.label), ...leaderObstacles],
    paths: [...internal.map((e) => e.points), ...leaders],
  })
  Object.assign(edges, routed.edges)
  const result = {
    edges,
    ports,
    bounds: diagramBounds(
      Object.values(positions).map((p) => ({ ...p, width: NODE_WIDTH, height: NODE_HEIGHT })),
      [
        ...Object.values(edges).map((e) => e.label),
        ...Object.values(edges).flatMap((e) =>
          e.points.map((p) => ({ ...p, width: 0, height: 0 })),
        ),
      ],
    ),
    crossings: routed.crossings,
    bends: Object.values(edges).reduce((sum, e) => sum + Math.max(0, e.points.length - 2), 0),
  }
  return result
}

export async function planGrouped(
  A: AvoidEngine,
  graph: ModelGraph,
  direction: 'RIGHT' | 'DOWN',
): Promise<LayoutPlan> {
  const stages = stageGraphs(graph)
  const plans: { stage: ModelGraph; positions: Positions; bounds: Rect }[] = []
  const templates = new Map<string, { positions: { x: number; y: number }[]; bounds: Rect }>()
  for (const stage of stages) {
    const indices = new Map(stage.nodes.map((n, i) => [n.id, i]))
    // Equal fixed-size topologies can share placement work across layers. Routes
    // and ports are still recomputed for every instance in the merged graph.
    const key = JSON.stringify([
      stage.nodes.length,
      stage.edges.map((e) => [indices.get(e.source), indices.get(e.target), e.label, e.kind]),
    ])
    let template = templates.get(key)
    if (!template) {
      const plan = await planLayout(A, stage, direction)
      template = {
        positions: stage.nodes.map((n) => plan.positions[n.id]),
        bounds: plan.routing.bounds,
      }
      templates.set(key, template)
    }
    plans.push({
      stage,
      positions: Object.fromEntries(stage.nodes.map((n, i) => [n.id, template!.positions[i]])),
      bounds: template.bounds,
    })
  }
  let problem = ''
  for (const gap of [180, 260, 380]) {
    const offsets = packStageFootprints(
      plans.map((p) => p.bounds),
      direction,
      gap,
    )
    const positions: Positions = {}
    plans.forEach(({ stage, positions: local, bounds }, i) => {
      const dx = Math.round(offsets[i].x + 56 - bounds.x),
        dy = Math.round(offsets[i].y + 56 - bounds.y)
      for (const node of stage.nodes)
        positions[node.id] = { x: local[node.id].x + dx, y: local[node.id].y + dy }
    })
    try {
      const routing = routeGrouped(A, graph, positions, direction)
      const problems = routingProblems(graph, positions, routing)
      if (!problems.length) return { positions, routing }
      problem = problems.slice(0, 2).join('; ')
    } catch (error) {
      problem = error instanceof Error ? error.message : 'Routing failed.'
    }
  }
  throw new Error(`Stage layout needs more room: ${problem}`)
}

export function stageBounds(graph: ModelGraph, positions: Positions): { id: string; rect: Rect }[] {
  return stageGraphs(graph).map((stage) => ({
    id: stage.id,
    rect: diagramBounds(
      stage.nodes.map((n) => ({ ...positions[n.id], width: NODE_WIDTH, height: NODE_HEIGHT })),
    ),
  }))
}
