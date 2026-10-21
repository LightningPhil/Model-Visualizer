import type { ModelGraph, Connection } from './schema'
import type { AvoidEngine, NativeRouter, NativeShape, NativeConnector } from './avoid'
import {
  crossing,
  diagramBounds,
  EPSILON,
  labelSize,
  LABEL_CLEARANCE,
  NODE_CLEARANCE,
  NODE_HEIGHT,
  NODE_WIDTH,
  rectanglesOverlap,
  segmentHitsRect,
  segmentLength,
  segments,
  simplifyPoints,
  type Point,
  type Rect,
  type Side,
} from './geometry'

export type Positions = ModelGraph['layout']['positions']
export type Port = {
  id: string
  nodeId: string
  edgeId: string
  type: 'source' | 'target'
  side: Side
  point: Point
  offset: number
  classId: number
}
export type RoutedEdge = {
  id: string
  points: Point[]
  label: Rect
  labelLeader?: Point[]
  source: Port
  target: Port
}
export type RoutingResult = {
  edges: Record<string, RoutedEdge>
  ports: Record<string, Port[]>
  bounds: Rect
  crossings: number
  bends: number
}
const directionFlags: Record<Side, number> = { top: 1, bottom: 2, left: 4, right: 8 }
const opposite: Record<Side, Side> = { top: 'bottom', bottom: 'top', left: 'right', right: 'left' }

export function nodeRect(position: Point): Rect {
  return { ...position, width: NODE_WIDTH, height: NODE_HEIGHT }
}

function chooseSides(
  edge: Connection,
  a: Rect,
  b: Rect,
  direction: 'RIGHT' | 'DOWN',
): [Side, Side] {
  let side: Side
  if (edge.source === edge.target) return [edge.sourceSide ?? 'right', edge.targetSide ?? 'bottom']
  const horizontal = b.x >= a.x + a.width + NODE_CLEARANCE || a.x >= b.x + b.width + NODE_CLEARANCE
  const vertical = b.y >= a.y + a.height + NODE_CLEARANCE || a.y >= b.y + b.height + NODE_CLEARANCE
  if (horizontal && (!vertical || direction === 'RIGHT')) side = b.x > a.x ? 'right' : 'left'
  else if (vertical) side = b.y > a.y ? 'bottom' : 'top'
  else
    side =
      Math.abs(b.x - a.x) > Math.abs(b.y - a.y)
        ? b.x > a.x
          ? 'right'
          : 'left'
        : b.y > a.y
          ? 'bottom'
          : 'top'
  return [edge.sourceSide ?? side, edge.targetSide ?? opposite[side]]
}

/** Distinct ports for every incident connection, ordered by their destination. */
export function allocatePorts(
  graph: ModelGraph,
  positions: Positions,
  direction = graph.layout.direction,
) {
  const ports: Record<string, Port[]> = Object.fromEntries(graph.nodes.map((n) => [n.id, []]))
  for (const edge of graph.edges) {
    const a = positions[edge.source],
      b = positions[edge.target]
    if (!a || !b) throw new Error(`Missing layout position for ${edge.id}.`)
    const [sourceSide, targetSide] = chooseSides(edge, nodeRect(a), nodeRect(b), direction)
    ports[edge.source].push({
      id: `s-${edge.id}`,
      nodeId: edge.source,
      edgeId: edge.id,
      type: 'source',
      side: sourceSide,
      point: { x: 0, y: 0 },
      offset: 0,
      classId: 0,
    })
    ports[edge.target].push({
      id: `t-${edge.id}`,
      nodeId: edge.target,
      edgeId: edge.id,
      type: 'target',
      side: targetSide,
      point: { x: 0, y: 0 },
      offset: 0,
      classId: 0,
    })
  }
  for (const [nodeId, attached] of Object.entries(ports)) {
    const rect = nodeRect(positions[nodeId])
    let classId = 1
    for (const side of ['top', 'right', 'bottom', 'left'] as const) {
      const vertical = side === 'left' || side === 'right'
      const group = attached
        .filter((p) => p.side === side)
        .sort((a, b) => {
          const edgeA = graph.edges.find((e) => e.id === a.edgeId)!,
            edgeB = graph.edges.find((e) => e.id === b.edgeId)!
          const otherA = positions[a.type === 'source' ? edgeA.target : edgeA.source],
            otherB = positions[b.type === 'source' ? edgeB.target : edgeB.source]
          return (vertical ? otherA.y - otherB.y : otherA.x - otherB.x) || a.id.localeCompare(b.id)
        })
      group.forEach((port, i) => {
        const length = vertical ? rect.height : rect.width
        const offset =
          group.length === 1 ? length / 2 : 20 + ((length - 40) * (i + 0.5)) / group.length
        port.classId = classId++
        port.offset = offset
        port.point = {
          x: rect.x + (side === 'right' ? rect.width : side === 'left' ? 0 : offset),
          y: rect.y + (side === 'bottom' ? rect.height : side === 'top' ? 0 : offset),
        }
      })
    }
  }
  return ports
}

function makeShape(A: AvoidEngine, router: NativeRouter, rect: Rect) {
  const a = new A.Point(rect.x, rect.y),
    b = new A.Point(rect.x + rect.width, rect.y + rect.height)
  const polygon = new A.Rectangle(a, b)
  const shape = new A.ShapeRef(router, polygon)
  polygon.delete()
  a.delete()
  b.delete()
  return shape
}

function addPin(
  A: AvoidEngine,
  shape: NativeShape,
  rect: Rect,
  point: Point,
  side: Side,
  id: number,
) {
  const pin = new A.ShapeConnectionPin(
    shape,
    id,
    (point.x - rect.x) / rect.width,
    (point.y - rect.y) / rect.height,
    true,
    0,
    directionFlags[side],
  )
  pin.setExclusive(false)
  // Pins and shape/connector native instances are owned by the Router. Deleting
  // the router releases them; deleting each wrapper too would double-free them.
  return pin
}

function connect(
  A: AvoidEngine,
  router: NativeRouter,
  a: NativeShape,
  pinA: number,
  b: NativeShape,
  pinB: number,
) {
  const start = new A.ConnEnd(a, pinA),
    end = new A.ConnEnd(b, pinB)
  const conn = new A.ConnRef(router, start, end)
  conn.setRoutingType(A.ConnType.ConnType_Orthogonal)
  start.delete()
  end.delete()
  return conn
}

function readRoute(connection: NativeConnector, edgeId: string): Point[] {
  const route = connection.displayRoute()
  const points: Point[] = []
  try {
    for (let i = 0; i < route.size(); i++) {
      const point = route.at(i)
      points.push({ x: point.x, y: point.y })
      point.delete()
    }
  } finally {
    route.delete()
  }
  if (points.length < 2)
    throw new Error('A connection could not be routed. Move overlapping components apart.')
  for (const { a, b } of segments(points))
    if (Math.abs(a.x - b.x) >= EPSILON && Math.abs(a.y - b.y) >= EPSILON)
      throw new Error(`Connection ${edgeId} could not find an orthogonal route.`)
  return simplifyPoints(points)
}

type LabelPlacement = { rect: Rect; horizontal: boolean; forward: boolean; offset?: boolean }

function placeLabels(
  graph: ModelGraph,
  positions: Positions,
  paths: Record<string, Point[]>,
  obstacles: Rect[] = [],
  reservedPaths: Point[][] = [],
): Record<string, LabelPlacement> {
  const nodes = [...Object.values(positions).map(nodeRect), ...obstacles]
  const reservedSegments = reservedPaths.flatMap(segments)
  const allSegments = Object.fromEntries([
    ...Object.entries(paths).map(([id, p]) => [id, segments(p)]),
    ['@reserved', reservedSegments],
  ]) as Record<string, ReturnType<typeof segments>>
  const labels: Record<string, LabelPlacement> = {}
  const candidates = graph.edges
    .map((edge) => {
      const size = labelSize(edge.label)
      const pathSegments = allSegments[edge.id]
      const total = pathSegments.reduce((sum, s) => sum + segmentLength(s), 0)
      let traversed = 0
      const choices: (LabelPlacement & { score: number; clashes: number })[] = []
      for (const segment of pathSegments) {
        const horizontal = Math.abs(segment.a.y - segment.b.y) < EPSILON
        const length = segmentLength(segment),
          span = horizontal ? size.width : size.height
        const low =
          (horizontal ? Math.min(segment.a.x, segment.b.x) : Math.min(segment.a.y, segment.b.y)) +
          span / 2 +
          NODE_CLEARANCE
        const high =
          (horizontal ? Math.max(segment.a.x, segment.b.x) : Math.max(segment.a.y, segment.b.y)) -
          span / 2 -
          NODE_CLEARANCE
        if (high >= low) {
          const mid = (low + high) / 2
          const samples = new Set([mid, low, high, low + (high - low) / 4, high - (high - low) / 4])
          // Exterior lanes in whole-model maps can span tens of thousands of
          // pixels. Bound uniform sampling, then include obstacle boundaries
          // so a short clear interval is still considered at any map scale.
          const step = Math.max(24, (high - low) / 48)
          for (let p = low; p <= high; p += step) samples.add(p)
          if (step > 24) {
            const boundary = (rect: Rect, clearance: number) => {
              const cross = horizontal ? segment.a.y : segment.a.x
              const halfCross = (horizontal ? size.height : size.width) / 2
              const startCross = horizontal ? rect.y : rect.x
              const endCross = startCross + (horizontal ? rect.height : rect.width)
              if (
                endCross + clearance < cross - halfCross ||
                startCross - clearance > cross + halfCross
              )
                return
              const start = horizontal ? rect.x : rect.y
              const end = start + (horizontal ? rect.width : rect.height)
              for (const p of [start - clearance - span / 2 - 1, end + clearance + span / 2 + 1])
                if (p >= low && p <= high) samples.add(p)
            }
            nodes.forEach((n) => boundary(n, NODE_CLEARANCE))
            for (const s of [...reservedSegments, ...pathSegments.filter((s) => s !== segment)])
              boundary(
                {
                  x: Math.min(s.a.x, s.b.x),
                  y: Math.min(s.a.y, s.b.y),
                  width: Math.abs(s.b.x - s.a.x),
                  height: Math.abs(s.b.y - s.a.y),
                },
                LABEL_CLEARANCE,
              )
          }
          for (const coordinate of samples) {
            const center = horizontal
              ? { x: coordinate, y: segment.a.y }
              : { x: segment.a.x, y: coordinate }
            const rect = { x: center.x - size.width / 2, y: center.y - size.height / 2, ...size }
            if (
              nodes.some((n) => rectanglesOverlap(rect, n, NODE_CLEARANCE)) ||
              reservedSegments.some((s) => segmentHitsRect(s, rect, LABEL_CLEARANCE))
            )
              continue
            // A label must sit on a straight run, never over its own bend/other leg.
            if (
              pathSegments.some((s) => s !== segment && segmentHitsRect(s, rect, LABEL_CLEARANCE))
            )
              continue
            const clashes = Object.entries(allSegments)
              .filter(([id]) => id !== edge.id)
              .reduce(
                (sum, [, ss]) =>
                  sum + ss.filter((s) => segmentHitsRect(s, rect, NODE_CLEARANCE + 2)).length,
                0,
              )
            const along =
              traversed + Math.abs(horizontal ? coordinate - segment.a.x : coordinate - segment.a.y)
            choices.push({
              rect,
              horizontal,
              forward: horizontal ? segment.b.x > segment.a.x : segment.b.y > segment.a.y,
              clashes,
              score: clashes * 100000 + Math.abs(along - total / 2) + (horizontal ? 0 : 24),
            })
          }
        }
        traversed += length
      }
      return { edge, choices: choices.sort((a, b) => a.score - b.score) }
    })
    .sort((a, b) => a.choices.length - b.choices.length || a.edge.id.localeCompare(b.edge.id))

  for (const { edge, choices } of candidates) {
    const clear = choices.find(
      (choice) =>
        !Object.values(labels).some((other) =>
          rectanglesOverlap(choice.rect, other.rect, NODE_CLEARANCE * 2),
        ),
    )
    if (clear) {
      labels[edge.id] = clear
      continue
    }
    // Keep short connections straight in existing/manual layouts. Move only
    // their annotation into a nearby free gutter, with a single straight leader.
    // Never send the connection on a detour just to make room for its wording.
    const size = labelSize(edge.label)
    let offset: LabelPlacement | undefined
    for (const distance of [22, 38, 54, 70, 86, 102, 134, 166]) {
      if (offset) break
      for (const segment of [...allSegments[edge.id]].sort(
        (a, b) => segmentLength(b) - segmentLength(a),
      )) {
        const horizontal = Math.abs(segment.a.y - segment.b.y) < EPSILON
        const center = { x: (segment.a.x + segment.b.x) / 2, y: (segment.a.y + segment.b.y) / 2 }
        for (const sign of [-1, 1]) {
          const rect = {
            x: center.x - size.width / 2 + (horizontal ? 0 : sign * (size.width / 2 + distance)),
            y: center.y - size.height / 2 + (horizontal ? sign * (size.height / 2 + distance) : 0),
            ...size,
          }
          const boundary = horizontal
            ? { x: center.x, y: sign < 0 ? rect.y + rect.height : rect.y }
            : { x: sign < 0 ? rect.x + rect.width : rect.x, y: center.y }
          const leader = { a: boundary, b: center }
          if (
            nodes.some(
              (node) =>
                rectanglesOverlap(rect, node, NODE_CLEARANCE) || segmentHitsRect(leader, node, 4),
            )
          )
            continue
          if (
            Object.values(labels).some(
              (label) =>
                rectanglesOverlap(rect, label.rect, NODE_CLEARANCE * 2) ||
                segmentHitsRect(leader, label.rect, 4),
            )
          )
            continue
          if (
            Object.entries(allSegments).some(([id, ss]) =>
              ss.some(
                (s) =>
                  segmentHitsRect(s, rect, NODE_CLEARANCE) ||
                  (id !== edge.id && crossing(s, leader)),
              ),
            )
          )
            continue
          offset = { rect, horizontal, forward: true, offset: true }
          break
        }
        if (offset) break
      }
    }
    if (offset) {
      labels[edge.id] = offset
      continue
    }
    // A cramped manually composed view can lack a straight run wide enough for
    // its label. Reserve a clean exterior lane; never hide a tag inside a node.
    const bounds = diagramBounds(
      nodes,
      Object.values(labels).map((l) => l.rect),
    )
    const source = positions[edge.source],
      target = positions[edge.target]
    const horizontal = graph.layout.direction === 'RIGHT'
    labels[edge.id] = {
      rect: horizontal
        ? {
            x: (source.x + target.x + NODE_WIDTH) / 2 - size.width / 2,
            y: bounds.y - 50 - size.height,
            ...size,
          }
        : {
            x: bounds.x - 50 - size.width,
            y: (source.y + target.y + NODE_HEIGHT) / 2 - size.height / 2,
            ...size,
          },
      horizontal,
      forward: horizontal ? target.x >= source.x : target.y >= source.y,
    }
  }
  return labels
}

/** One joint routing transaction accounts for every node, port and label. */
export function routeGraph(
  A: AvoidEngine,
  graph: ModelGraph,
  positions: Positions,
  direction = graph.layout.direction,
  fixed: { ports?: Record<string, Port[]>; obstacles?: Rect[]; paths?: Point[][] } = {},
): RoutingResult {
  const router = new A.Router(A.RouterFlag.OrthogonalRouting.value)
  const portMap = fixed.ports ?? allocatePorts(graph, positions, direction)
  const portList = Object.values(portMap).flat()
  const shapes: Record<string, NativeShape> = {}
  const connections: Record<string, NativeConnector> = {}
  try {
    const parameters = {
      segmentPenalty: 100,
      crossingPenalty: 0,
      fixedSharedPathPenalty: 100,
      portDirectionPenalty: 100,
      shapeBufferDistance: NODE_CLEARANCE / 2,
      idealNudgingDistance: 14,
      reverseDirectionPenalty: 120,
    } as const
    for (const [key, value] of Object.entries(parameters))
      router.setRoutingParameter(A.RoutingParameter[key as keyof typeof parameters], value)
    router.setRoutingOption(A.RoutingOption.nudgeOrthogonalSegmentsConnectedToShapes, false)
    router.setRoutingOption(A.RoutingOption.nudgeOrthogonalTouchingColinearSegments, true)
    router.setRoutingOption(A.RoutingOption.nudgeSharedPathsWithCommonEndPoint, false)
    for (const obstacle of fixed.obstacles ?? []) makeShape(A, router, obstacle)
    for (const node of graph.nodes) {
      const rect = nodeRect(positions[node.id])
      const shape = (shapes[node.id] = makeShape(A, router, rect))
      for (const port of portMap[node.id])
        addPin(A, shape, rect, port.point, port.side, port.classId)
    }
    const endpoints = (edge: Connection) => ({
      source: portList.find((p) => p.edgeId === edge.id && p.type === 'source')!,
      target: portList.find((p) => p.edgeId === edge.id && p.type === 'target')!,
    })
    for (const edge of graph.edges) {
      const { source, target } = endpoints(edge)
      connections[edge.id] = connect(
        A,
        router,
        shapes[edge.source],
        source.classId,
        shapes[edge.target],
        target.classId,
      )
    }
    router.processTransaction()
    const initial = Object.fromEntries(
      graph.edges.map((e) => [e.id, readRoute(connections[e.id], e.id)]),
    )
    const labels = placeLabels(
      { ...graph, layout: { ...graph.layout, direction } },
      positions,
      initial,
      fixed.obstacles,
      fixed.paths,
    )
    // Labels become actual routing obstacles with an entrance and exit on
    // opposite faces. Thus a label cannot obscure any unrelated connection,
    // and its own connection passes straight through it without a dogleg.
    const halves: Record<string, NativeConnector[]> = {}
    for (const edge of graph.edges) {
      router.deleteConnector(connections[edge.id])
      const { rect, horizontal, forward, offset } = labels[edge.id]
      const shape = makeShape(A, router, rect)
      const { source, target } = endpoints(edge)
      if (offset) {
        halves[edge.id] = [
          connect(
            A,
            router,
            shapes[edge.source],
            source.classId,
            shapes[edge.target],
            target.classId,
          ),
        ]
        continue
      }
      const entrySide: Side = horizontal ? (forward ? 'left' : 'right') : forward ? 'top' : 'bottom'
      const exitSide = opposite[entrySide]
      const entry = {
        x:
          rect.x + (entrySide === 'left' ? 0 : entrySide === 'right' ? rect.width : rect.width / 2),
        y:
          rect.y +
          (entrySide === 'top' ? 0 : entrySide === 'bottom' ? rect.height : rect.height / 2),
      }
      const exit = {
        x: rect.x + (exitSide === 'left' ? 0 : exitSide === 'right' ? rect.width : rect.width / 2),
        y:
          rect.y + (exitSide === 'top' ? 0 : exitSide === 'bottom' ? rect.height : rect.height / 2),
      }
      addPin(A, shape, rect, entry, entrySide, 1)
      addPin(A, shape, rect, exit, exitSide, 2)
      halves[edge.id] = [
        connect(A, router, shapes[edge.source], source.classId, shape, 1),
        connect(A, router, shape, 2, shapes[edge.target], target.classId),
      ]
    }
    router.processTransaction()
    const edges: Record<string, RoutedEdge> = {}
    for (const edge of graph.edges) {
      edges[edge.id] = {
        id: edge.id,
        points: simplifyPoints(halves[edge.id].flatMap((half) => readRoute(half, edge.id))),
        label: labels[edge.id].rect,
        ...endpoints(edge),
      }
    }
    const routes = Object.values(edges)
    for (const route of routes)
      if (labels[route.id].offset) {
        const rect = route.label
        const cx = rect.x + rect.width / 2,
          cy = rect.y + rect.height / 2
        const leaders = segments(route.points)
          .flatMap((segment) => {
            if (
              Math.abs(segment.a.y - segment.b.y) < EPSILON &&
              cx >= Math.min(segment.a.x, segment.b.x) &&
              cx <= Math.max(segment.a.x, segment.b.x)
            )
              return [
                [
                  { x: cx, y: segment.a.y < cy ? rect.y : rect.y + rect.height },
                  { x: cx, y: segment.a.y },
                ],
              ]
            if (
              Math.abs(segment.a.x - segment.b.x) < EPSILON &&
              cy >= Math.min(segment.a.y, segment.b.y) &&
              cy <= Math.max(segment.a.y, segment.b.y)
            )
              return [
                [
                  { x: segment.a.x < cx ? rect.x : rect.x + rect.width, y: cy },
                  { x: segment.a.x, y: cy },
                ],
              ]
            return []
          })
          .filter((points) => {
            const line = { a: points[0], b: points[1] }
            return (
              !(fixed.obstacles ?? []).some((rect) => segmentHitsRect(line, rect, 2)) &&
              !(fixed.paths ?? []).flatMap(segments).some((s) => crossing(s, line)) &&
              !Object.values(positions).some((p) => segmentHitsRect(line, nodeRect(p), 2)) &&
              !routes.some(
                (other) =>
                  other.id !== route.id &&
                  (segmentHitsRect(line, other.label, 2) ||
                    segments(other.points).some((s) => crossing(s, line))),
              )
            )
          })
          .sort((a, b) => segmentLength({ a: a[0], b: a[1] }) - segmentLength({ a: b[0], b: b[1] }))
        if (!leaders[0])
          throw new Error('A connection annotation needs more room. Use Auto-layout.')
        route.labelLeader = leaders[0]
      }
    let crossings = 0
    routes.forEach((route, i) =>
      routes.slice(i + 1).forEach((other) =>
        segments(route.points).forEach((a) =>
          segments(other.points).forEach((b) => {
            if (crossing(a, b)) crossings++
          }),
        ),
      ),
    )
    return {
      edges,
      ports: portMap,
      bounds: diagramBounds(Object.values(positions).map(nodeRect), [
        ...routes.map((r) => r.label),
        ...routes.flatMap((r) => r.points.map((p) => ({ ...p, width: 0, height: 0 }))),
      ]),
      crossings,
      bends: routes.reduce((sum, route) => sum + Math.max(0, route.points.length - 2), 0),
    }
  } finally {
    router.delete()
  }
}

export function routingProblems(
  graph: ModelGraph,
  positions: Positions,
  result: RoutingResult,
): string[] {
  const problems: string[] = []
  const routes = Object.values(result.edges)
  for (const route of routes) {
    const first = route.points[0],
      last = route.points.at(-1)
    if (
      !first ||
      !last ||
      Math.hypot(first.x - route.source.point.x, first.y - route.source.point.y) > EPSILON ||
      Math.hypot(last.x - route.target.point.x, last.y - route.target.point.y) > EPSILON
    )
      problems.push(`${route.id}: disconnected port`)
    for (let i = 1; i < route.points.length - 1; i++) {
      const a = route.points[i - 1],
        b = route.points[i],
        c = route.points[i + 1]
      if ((b.x - a.x) * (c.x - b.x) + (b.y - a.y) * (c.y - b.y) < -EPSILON)
        problems.push(`${route.id}: doubles back`)
      if (
        !route.labelLeader &&
        b.x > route.label.x - 5 &&
        b.x < route.label.x + route.label.width + 5 &&
        b.y > route.label.y - 5 &&
        b.y < route.label.y + route.label.height + 5
      )
        problems.push(`${route.id}: bend at label`)
    }
    for (const { a, b } of segments(route.points)) {
      if (Math.abs(a.x - b.x) > EPSILON && Math.abs(a.y - b.y) > EPSILON) {
        problems.push(`${route.id}: diagonal segment`)
        continue
      }
      for (const node of graph.nodes)
        if (segmentHitsRect({ a, b }, nodeRect(positions[node.id]), 0))
          problems.push(`${route.id}: crosses node ${node.id}`)
      for (const other of routes)
        if (other.id !== route.id && segmentHitsRect({ a, b }, other.label, 2))
          problems.push(`${route.id}: crosses label ${other.id}`)
    }
    for (const node of graph.nodes)
      if (rectanglesOverlap(route.label, nodeRect(positions[node.id]), 2))
        problems.push(`${route.id}: label overlaps node ${node.id}`)
    for (const other of routes)
      if (other.id > route.id && rectanglesOverlap(route.label, other.label, 2))
        problems.push(`${route.id}: label overlaps ${other.id}`)
    if (route.labelLeader)
      for (const line of segments(route.labelLeader)) {
        for (const node of graph.nodes)
          if (segmentHitsRect(line, nodeRect(positions[node.id]), 2))
            problems.push(`${route.id}: annotation crosses node ${node.id}`)
        for (const other of routes)
          if (
            other.id !== route.id &&
            (segmentHitsRect(line, other.label, 2) ||
              segments(other.points).some((s) => crossing(s, line)))
          )
            problems.push(`${route.id}: annotation crosses ${other.id}`)
      }
  }
  return [...new Set(problems)]
}
