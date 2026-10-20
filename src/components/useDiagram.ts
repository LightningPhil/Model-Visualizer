import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  applyNodeChanges,
  getViewportForBounds,
  MarkerType,
  useReactFlow,
  useStore,
} from '@xyflow/react'
import type { NodeChange } from '@xyflow/react'
import type { DiagramProps, LayoutSnapshot, VisualNode, VisualEdge } from './GraphCanvas'
import { DiagramWorker } from '../model/layout-client'
import {
  NODE_HEIGHT,
  NODE_WIDTH,
  rectanglesOverlap,
  diagramBounds,
  type Point,
  type Rect,
} from '../model/geometry'
import type { RoutingResult, Positions } from '../model/routing'
import type { ModelGraph } from '../model/schema'

function version(graph: ModelGraph, positions: Positions) {
  return JSON.stringify([
    graph.layout.direction,
    graph.edges.map((e) => [e.id, e.source, e.target, e.label, e.sourceSide, e.targetSide]),
    graph.nodes.map((n) => [n.id, positions[n.id]]),
  ])
}

/** Nodes cannot be dragged over one another and swallow connection endpoints. */
export function clearDrop(
  id: string,
  position: Point,
  nodes: Pick<VisualNode, 'id' | 'position'>[],
): Point {
  const others = nodes
    .filter((n) => n.id !== id)
    .map((n) => ({ ...n.position, width: NODE_WIDTH, height: NODE_HEIGHT }))
  const clear = (p: Point) =>
    !others.some((rect) =>
      rectanglesOverlap({ ...p, width: NODE_WIDTH, height: NODE_HEIGHT }, rect, 28),
    )
  if (clear(position)) return position
  const candidates = others.flatMap((rect) => [
    { x: rect.x - NODE_WIDTH - 28, y: position.y },
    { x: rect.x + rect.width + 28, y: position.y },
    { x: position.x, y: rect.y - NODE_HEIGHT - 28 },
    { x: position.x, y: rect.y + rect.height + 28 },
  ])
  const bounds = diagramBounds(others)
  candidates.push(
    { x: bounds.x - NODE_WIDTH - 28, y: position.y },
    { x: bounds.x + bounds.width + 28, y: position.y },
    { x: position.x, y: bounds.y - NODE_HEIGHT - 28 },
    { x: position.x, y: bounds.y + bounds.height + 28 },
  )
  return (
    candidates
      .filter(clear)
      .sort(
        (a, b) =>
          Math.hypot(a.x - position.x, a.y - position.y) -
          Math.hypot(b.x - position.x, b.y - position.y),
      )[0] ?? position
  )
}

export function useDiagram(props: DiagramProps) {
  const { graph, symbols, selection, showShapes, symbolic, onOpen, notify } = props
  const flow = useReactFlow<VisualNode, VisualEdge>()
  const canvasRef = useRef<HTMLDivElement>(null)
  const propsRef = useRef(props)
  propsRef.current = props
  const [client, setClient] = useState<DiagramWorker | null>(null)
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(
    Object.keys(graph.layout.positions).length === graph.nodes.length,
  )
  const [routed, setRouted] = useState<{ key: string; result: RoutingResult } | null>(null)
  const [routingError, setRoutingError] = useState('')
  const [history, setHistory] = useState<LayoutSnapshot[]>([])
  const generation = useRef(0),
    request = useRef(0)
  const initialized = useRef(false),
    fitted = useRef(Boolean(graph.layout.viewport))
  const resultRef = useRef<RoutingResult | null>(null)

  const buildNodes = useCallback(
    (positions: Positions): VisualNode[] =>
      graph.nodes.map((component, i) => ({
        id: component.id,
        type: 'component',
        position: positions[component.id] ?? { x: i * 400, y: 100 },
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        ariaLabel: `${component.title}. ${component.subtitle}.`,
        data: { component, symbols, showShapes, symbolic, onOpen, ports: [] },
        selected: selection?.type === 'node' && selection.id === component.id,
      })),
    [graph.nodes, symbols, showShapes, symbolic, onOpen, selection],
  )
  const [nodeState, setNodes] = useState<VisualNode[]>(() => buildNodes(graph.layout.positions))
  const positions = useMemo(
    () => Object.fromEntries(nodeState.map((n) => [n.id, n.position])),
    [nodeState],
  )
  const routeVersion = version(graph, positions)
  const result = routed?.key === routeVersion ? routed.result : null
  resultRef.current = result

  const fitGraph = useCallback(
    (bounds?: Rect, duration = 350) => {
      const rect = bounds ?? resultRef.current?.bounds
      // Read the current container size: the Flow store can be a frame behind
      // when fitting immediately after opening/closing a side panel.
      const width = canvasRef.current?.clientWidth ?? 0
      const height = canvasRef.current?.clientHeight ?? 0
      if (!rect || !width || !height) return
      const viewport = getViewportForBounds(rect, width, height, 0.035, 1, {
        top: '64px',
        bottom: '82px',
        left: '30px',
        right: '34px',
      })
      void flow.setViewport(viewport, { duration })
    },
    [flow],
  )

  useEffect(() => {
    const worker = new DiagramWorker()
    setClient(worker)
    return () => worker.dispose()
  }, [])

  useEffect(() => {
    if (!client) return
    const current = ++generation.current
    if (Object.keys(graph.layout.positions).length === graph.nodes.length)
      return () => {
        generation.current++
      }
    setBusy(true)
    void client
      .layout(graph)
      .then((plan) => {
        if (generation.current !== current) return
        const automatic = {
          ...graph,
          edges: graph.edges.map((e) => ({ ...e, sourceSide: undefined, targetSide: undefined })),
        }
        setNodes(buildNodes(plan.positions))
        setRouted({ key: version(automatic, plan.positions), result: plan.routing })
        setReady(true)
        fitted.current = true
        initialized.current = true
        propsRef.current.onLayout(plan.positions, graph.layout.direction, true)
        requestAnimationFrame(() => fitGraph(plan.routing.bounds))
      })
      .catch((error) => {
        if (generation.current === current && error.name !== 'AbortError') notify(error.message)
      })
      .finally(() => {
        if (generation.current === current) setBusy(false)
      })
    return () => {
      generation.current++
    }
    // This initializes one graph; later selection/configuration edits preserve it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client])

  useEffect(() => {
    if (!client || !ready || busy) return
    const current = ++request.current
    const frame = requestAnimationFrame(() => {
      void client
        .route(graph, positions)
        .then((next) => {
          if (request.current !== current) return
          setRouted({ key: routeVersion, result: next })
          setRoutingError('')
          initialized.current = true
          if (!fitted.current) {
            fitted.current = true
            requestAnimationFrame(() => fitGraph(next.bounds, 0))
          }
        })
        .catch((error) => {
          if (request.current === current && error.name !== 'AbortError')
            setRoutingError(error.message)
        })
    })
    return () => {
      cancelAnimationFrame(frame)
      request.current++
    }
    // The serialized geometry is the dependency, not selection or viewport state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [client, ready, busy, routeVersion])

  // Retain React Flow's measured dimensions and drag state. Rebuilding only
  // authored node data must not trigger an endless ResizeObserver measurement.
  const nodes = useMemo(
    () =>
      buildNodes(positions).map((node) => ({
        ...nodeState.find((current) => current.id === node.id),
        ...node,
        data: { ...node.data, ports: result?.ports[node.id] ?? [] },
      })),
    [buildNodes, positions, result, nodeState],
  )
  const portsReady = useStore((state) =>
    Boolean(
      result &&
      Object.entries(result.ports).every(([id, ports]) => {
        const bounds = state.nodeLookup.get(id)?.internals.handleBounds
        return ports.every((port) => bounds?.[port.type]?.some((handle) => handle.id === port.id))
      }),
    ),
  )
  const edges = useMemo(
    (): VisualEdge[] =>
      !result || !portsReady
        ? []
        : graph.edges.map((connection) => {
            const route = result.edges[connection.id]
            const selected = selection?.type === 'edge' && selection.id === connection.id
            const color = selected
              ? 'var(--cyan)'
              : connection.kind === 'state'
                ? 'var(--violet)'
                : connection.kind === 'auxiliary'
                  ? 'var(--amber)'
                  : connection.kind === 'residual'
                    ? 'var(--cyan)'
                    : 'var(--edge)'
            return {
              id: connection.id,
              source: connection.source,
              target: connection.target,
              sourceHandle: route.source.id,
              targetHandle: route.target.id,
              type: 'operation',
              markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
              selected,
              data: {
                connection,
                route,
                onSelect: (id) => propsRef.current.onSelect({ type: 'edge', id }),
              },
              style: {
                stroke: color,
                strokeWidth: selected ? 2 : 1.4,
                strokeDasharray: connection.kind === 'flow' ? undefined : '5 5',
              },
            }
          }),
    [result, portsReady, graph.edges, selection],
  )

  async function arrange(direction = graph.layout.direction) {
    if (!client || busy) return
    const current = generation.current
    request.current++
    setBusy(true)
    setRoutingError('')
    try {
      const plan = await client.layout(graph, direction)
      if (generation.current !== current) return
      setHistory((h) => [
        ...h.slice(-9),
        {
          layout: { ...graph.layout, positions, viewport: flow.getViewport() },
          edges: graph.edges,
        },
      ])
      setNodes(buildNodes(plan.positions))
      setRouted({
        key: version(
          {
            ...graph,
            layout: { ...graph.layout, direction },
            edges: graph.edges.map((e) => ({ ...e, sourceSide: undefined, targetSide: undefined })),
          },
          plan.positions,
        ),
        result: plan.routing,
      })
      propsRef.current.onLayout(plan.positions, direction, true)
      fitted.current = true
      requestAnimationFrame(() => fitGraph(plan.routing.bounds, 450))
      notify('Layout arranged. Connections and labels have clear routes.')
    } catch (error) {
      if (generation.current === current && error instanceof Error && error.name !== 'AbortError')
        notify(error.message)
    } finally {
      if (generation.current === current) setBusy(false)
    }
  }

  function undoLayout() {
    const snapshot = history.at(-1)
    if (!snapshot || busy) return
    request.current++
    setNodes(buildNodes(snapshot.layout.positions))
    setRouted(null)
    setRoutingError('')
    propsRef.current.onRestore(snapshot)
    setHistory((h) => h.slice(0, -1))
    if (snapshot.layout.viewport) void flow.setViewport(snapshot.layout.viewport, { duration: 300 })
    notify('Previous layout restored.')
  }

  function onNodesChange(changes: NodeChange<VisualNode>[]) {
    setNodes((current) => {
      const next = applyNodeChanges(changes, current)
      for (const change of changes)
        if (change.type === 'position' && change.position) {
          const node = next.find((n) => n.id === change.id)
          if (node) node.position = clearDrop(node.id, node.position, next)
        }
      return next
    })
  }

  function onDragStop(node: VisualNode) {
    const position = clearDrop(node.id, node.position, nodeState)
    setNodes((current) => current.map((n) => (n.id === node.id ? { ...n, position } : n)))
    propsRef.current.onLayout({ ...positions, [node.id]: position })
  }

  return {
    canvasRef,
    nodes,
    edges,
    busy,
    initialized,
    arrange,
    onNodesChange,
    onDragStop,
    fitGraph,
    undoLayout,
    canUndo: history.length > 0,
    routingError,
    routing: Boolean(client && ready && !busy && !result && !routingError),
  }
}
