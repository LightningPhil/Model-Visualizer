import { memo, useCallback, useEffect, useMemo, useRef } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  MiniMap,
  Handle,
  Position,
  BaseEdge,
  EdgeLabelRenderer,
  useReactFlow,
  useStore,
  useUpdateNodeInternals,
  ViewportPortal,
} from '@xyflow/react'
import type { Node, Edge, NodeProps, EdgeProps, Viewport } from '@xyflow/react'
import {
  ArrowDown,
  ArrowRight,
  Expand,
  Maximize2,
  Minus,
  Plus,
  Scan,
  WandSparkles,
  Undo2,
} from 'lucide-react'
import { shapeLabel, type Component, type Connection, type ModelGraph } from '../model/schema'
import { roundedPath, NODE_HEIGHT, NODE_WIDTH } from '../model/geometry'
import { instanceOf } from '../model/whole-model'
import type { Port, RoutedEdge } from '../model/routing'
import { useDiagram } from './useDiagram'
import { ComponentIcon } from './Icons'
import { OverviewCanvas } from './overview/OverviewCanvas'

export type Selection = { type: 'node' | 'edge'; id: string } | null
type VisualData = {
  component: Component
  symbols: Record<string, number>
  showShapes: boolean
  symbolic: boolean
  onOpen: (id: string) => void
  ports: Port[]
  execution?: 'active' | 'done' | 'pending'
}
export type VisualNode = Node<VisualData, 'component'>
export type VisualEdge = Edge<
  {
    connection: Connection
    route: RoutedEdge
    onSelect: (id: string) => void
    active?: boolean
    playing?: boolean
  },
  'operation'
>
const sidePositions = {
  top: Position.Top,
  right: Position.Right,
  bottom: Position.Bottom,
  left: Position.Left,
}

const ModelNode = memo(function ModelNode({ data, selected }: NodeProps<VisualNode>) {
  const { component, symbols, symbolic } = data
  const output =
    component.tensors.find((t) => t.role === 'output') ??
    component.tensors.find((t) => t.role === 'state')
  return (
    <div
      className={`model-node category-${component.category} ${selected ? 'is-selected' : ''} ${data.execution ? `execution-${data.execution}` : ''} ${instanceOf(component)?.optional ? 'optional-component' : ''}`}
      data-execution={data.execution}
      data-tooltip-title={component.title}
      data-tooltip-detail={component.description}
    >
      {data.ports.map((port) => (
        <Handle
          key={port.id}
          id={port.id}
          type={port.type}
          position={sidePositions[port.side]}
          style={
            port.side === 'left' || port.side === 'right'
              ? { top: port.offset }
              : { left: port.offset }
          }
          isConnectable={false}
        />
      ))}
      <div className="node-heading">
        <span className="node-icon">
          <ComponentIcon component={component} />
        </span>
        <span className="node-category">{component.category}</span>
        {data.execution === 'active' && <span className="node-live-label">ACTIVE</span>}
        {component.repeat && (
          <span className="repeat-badge">× {symbols[component.repeat] ?? component.repeat}</span>
        )}
        {component.childGraph && (
          <button
            className="node-open nodrag"
            aria-label={`Explore ${component.title}`}
            data-tooltip-detail="Explore inside"
            onClick={(event) => {
              event.stopPropagation()
              data.onOpen(component.childGraph!)
            }}
          >
            <Expand size={13} />
          </button>
        )}
      </div>
      <strong>{component.title}</strong>
      <span className="node-subtitle">{component.subtitle}</span>
      {data.showShapes && output && (
        <code className="node-shape">{shapeLabel(output.shape, symbols, symbolic)}</code>
      )}
    </div>
  )
})

const OperationEdge = memo(function OperationEdge(props: EdgeProps<VisualEdge>) {
  const route = props.data!.route
  const path = useMemo(() => roundedPath(route.points, 5), [route.points])
  const points = useMemo(() => JSON.stringify(route.points), [route.points])
  const connection = props.data!.connection
  return (
    <>
      {route.labelLeader && (
        <path className="edge-annotation-leader" d={roundedPath(route.labelLeader, 0)} />
      )}
      <g
        className={`routed-connection ${props.data!.active ? 'connection-active' : ''}`}
        data-edge-id={props.id}
        data-route-points={points}
      >
        <BaseEdge
          id={props.id}
          path={path}
          markerEnd={props.markerEnd}
          style={props.style}
          interactionWidth={24}
        />
        {props.data!.active && (
          <circle
            className="execution-packet"
            r="4"
            fill="var(--cyan)"
            stroke="var(--panel)"
            strokeWidth="1.5"
          >
            <animateMotion
              dur={props.data!.playing ? '1s' : '2s'}
              repeatCount="indefinite"
              path={path}
            />
          </circle>
        )}
      </g>
      <EdgeLabelRenderer>
        <button
          className={`edge-label nodrag nopan edge-${connection.kind} ${props.selected ? 'is-selected' : ''}`}
          style={{
            transform: `translate(${route.label.x}px,${route.label.y}px)`,
            width: route.label.width,
            height: route.label.height,
          }}
          data-label-edge={props.id}
          onClick={() => props.data!.onSelect(props.id)}
          data-tooltip-title={connection.operation}
          data-tooltip-detail={connection.description}
        >
          {connection.label}
        </button>
      </EdgeLabelRenderer>
    </>
  )
})

const nodeTypes = { component: ModelNode }
const edgeTypes = { operation: OperationEdge }
const minimapColors = { data: '#46b6cc', compute: '#747fc4', memory: '#a88de8', output: '#d3ae65' }
const minimapColor = (node: VisualNode) => minimapColors[node.data.component.category]

function AdaptiveBackground() {
  // A fixed world-space grid becomes hundreds of thousands of subpixel dots
  // at whole-model zoom. Keep its screen spacing in a useful, bounded range.
  const level = useStore((state) => Math.max(0, Math.ceil(Math.log2(0.65 / state.transform[2]))))
  const scale = 2 ** level
  return (
    <Background
      variant={BackgroundVariant.Dots}
      gap={24 * scale}
      size={scale}
      color="var(--grid)"
    />
  )
}

// Subscribe only this small control to zoom. Panning must never render the
// diagram, rebuild node data, or traverse hundreds of port/stage definitions.
function ZoomControls({ onFit }: { onFit: () => void }) {
  const flow = useReactFlow()
  const percent = useStore((state) => Math.round(state.transform[2] * 100))
  return (
    <div className="zoom-controls">
      <button
        aria-label="Zoom out"
        data-tooltip-detail="Zoom out"
        onClick={() => void flow.zoomOut({ duration: 200 })}
      >
        <Minus size={15} />
      </button>
      <span>{percent}%</span>
      <small
        className="overview-renderer-label"
        data-tooltip-title="Fast overview rendering"
        data-tooltip-detail="This distant view draws every component and validated route on a bounded canvas in a background worker. Zoom in for full text, tensor shapes and operation labels. The same nodes and links remain clickable and keyboard accessible."
      >
        Overview
      </small>
      <button
        aria-label="Zoom in"
        data-tooltip-detail="Zoom in"
        onClick={() => void flow.zoomIn({ duration: 200 })}
      >
        <Plus size={15} />
      </button>
      <i />
      <button aria-label="Fit graph" data-tooltip-detail="Fit graph" onClick={onFit}>
        <Scan size={17} />
      </button>
      <button
        aria-label="Actual size"
        data-tooltip-detail="Actual size"
        onClick={() => void flow.zoomTo(1, { duration: 300 })}
      >
        <Maximize2 size={14} />
      </button>
    </div>
  )
}

export type LayoutSnapshot = { layout: ModelGraph['layout']; edges: ModelGraph['edges'] }
export type CameraFocus = { id: string; revision: number; preserveZoom?: boolean }

export type DiagramProps = {
  graph: ModelGraph
  symbols: Record<string, number>
  selection: Selection
  showShapes: boolean
  symbolic: boolean
  theme: 'dark' | 'light'
  onSelect: (selection: Selection) => void
  onOpen: (id: string) => void
  onLayout: (
    positions: ModelGraph['layout']['positions'],
    direction?: 'RIGHT' | 'DOWN',
    resetViewport?: boolean,
  ) => void
  onRestore: (snapshot: LayoutSnapshot) => void
  onViewport: (viewport: Viewport) => void
  notify: (message: string) => void
  execution?: { active: string[]; completed: Set<string>; revision: number; playing: boolean }
  focus?: CameraFocus
  onReady?: (ready: boolean) => void
}

function Canvas(props: DiagramProps) {
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])
  const { graph } = props
  const flow = useReactFlow<VisualNode, VisualEdge>()
  const currentProps = useRef(props)
  currentProps.current = props
  const nodeClick = useCallback(
    (_: unknown, node: VisualNode) => currentProps.current.onSelect({ type: 'node', id: node.id }),
    [],
  )
  const nodeDoubleClick = useCallback((_: unknown, node: VisualNode) => {
    if (node.data.component.childGraph) currentProps.current.onOpen(node.data.component.childGraph)
  }, [])
  const edgeClick = useCallback(
    (_: unknown, edge: VisualEdge) => currentProps.current.onSelect({ type: 'edge', id: edge.id }),
    [],
  )
  const paneClick = useCallback(() => currentProps.current.onSelect(null), [])
  const {
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
    canUndo,
    routingError,
    routing,
  } = useDiagram(props)
  const updateNodeInternals = useUpdateNodeInternals()
  const portSignature = useMemo(
    () =>
      nodes
        .map((n) => `${n.id}:${n.data.ports.map((p) => `${p.id}:${p.side}:${p.offset}`).join(',')}`)
        .join('|'),
    [nodes],
  )
  useEffect(() => {
    if (nodes.some((n) => n.data.ports.length)) updateNodeInternals(nodes.map((n) => n.id))
    // One measurement transaction for the entire graph, instead of a forced
    // synchronous render for every individual node in a large model.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [portSignature, updateNodeInternals])
  const stageMap = useMemo(() => {
    const stages = new Map<string, { x: number; y: number; right: number; bottom: number }>()
    for (const node of nodes) {
      const stage = instanceOf(node.data.component)?.stage
      if (!stage) continue
      const current = stages.get(stage)
      stages.set(stage, {
        x: Math.min(current?.x ?? Infinity, node.position.x),
        y: Math.min(current?.y ?? Infinity, node.position.y),
        right: Math.max(current?.right ?? -Infinity, node.position.x + NODE_WIDTH),
        bottom: Math.max(current?.bottom ?? -Infinity, node.position.y + NODE_HEIGHT),
      })
    }
    return stages
  }, [nodes])
  const stageBoxes = useMemo(
    () =>
      [...stageMap].map(
        ([name, box]) =>
          [
            name,
            {
              x: box.x - 46,
              y: box.y - 80,
              width: box.right - box.x + 92,
              height: box.bottom - box.y + 120,
            },
          ] as [string, { x: number; y: number; width: number; height: number }],
      ),
    [stageMap],
  )
  const dragStop = useCallback((_: unknown, node: VisualNode) => onDragStop(node), [onDragStop])
  const moveEnd = useCallback(
    (_: unknown, next: Viewport) => {
      if (initialized.current && mounted.current) currentProps.current.onViewport(next)
    },
    [initialized],
  )
  const init = useCallback(() => {
    initialized.current = true
  }, [initialized])
  const fit = useCallback(() => fitGraph(), [fitGraph])
  useEffect(() => {
    props.onReady?.(!busy && edges.length === graph.edges.length && Boolean(nodes.length))
  }, [busy, edges.length, graph.edges.length, nodes.length, props.onReady])
  const consumedFocus = useRef<CameraFocus | undefined>(undefined)
  useEffect(() => {
    if (
      !props.focus ||
      busy ||
      !edges.length ||
      (consumedFocus.current?.id === props.focus.id &&
        consumedFocus.current.revision === props.focus.revision &&
        consumedFocus.current.preserveZoom === props.focus.preserveZoom)
    )
      return
    const node = nodes.find((n) => n.id === props.focus!.id)
    if (node) {
      consumedFocus.current = props.focus
      void flow.setCenter(node.position.x + NODE_WIDTH / 2, node.position.y + NODE_HEIGHT / 2, {
        zoom: props.focus.preserveZoom
          ? flow.getZoom()
          : Math.max(0.55, Math.min(0.9, flow.getZoom())),
        duration: props.execution?.playing ? 180 : 350,
      })
    }
    // Routing and focus revisions matter; ordinary selection/viewport changes do not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.focus?.id, props.focus?.revision, props.focus?.preserveZoom, busy, edges.length])

  return (
    <div ref={canvasRef} className="graph-canvas" data-testid="graph-canvas">
      <ReactFlow<VisualNode, VisualEdge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onNodeClick={nodeClick}
        onNodeDoubleClick={nodeDoubleClick}
        onEdgeClick={edgeClick}
        onPaneClick={paneClick}
        onNodeDragStop={dragStop}
        onMoveEnd={moveEnd}
        onInit={init}
        defaultViewport={graph.layout.viewport}
        minZoom={0.001}
        maxZoom={1.8}
        nodesDraggable={!busy}
        nodesConnectable={false}
        deleteKeyCode={null}
        colorMode={props.theme}
        panOnScroll
        zoomOnDoubleClick={false}
        selectionOnDrag={false}
      >
        <AdaptiveBackground />
        <OverviewCanvas
          nodes={nodes}
          edges={edges}
          stages={stageBoxes}
          parent={canvasRef}
          theme={props.theme}
        />
        {stageMap.size > 1 && (
          <ViewportPortal>
            {[...stageMap].map(([stage, box]) => (
              <div
                key={stage}
                className="model-stage-outline"
                style={{
                  transform: `translate(${box.x - 46}px,${box.y - 80}px)`,
                  width: box.right - box.x + 92,
                  height: box.bottom - box.y + 120,
                }}
              >
                <span>{stage}</span>
              </div>
            ))}
          </ViewportPortal>
        )}
        <MiniMap
          pannable
          zoomable
          position="bottom-right"
          nodeColor={minimapColor}
          maskColor="var(--minimap-mask)"
        />
      </ReactFlow>
      <div className="canvas-caption">
        <span className="live-dot" />
        {graph.id === 'overview'
          ? 'ARCHITECTURE MAP'
          : graph.id === 'attention'
            ? 'ATTENTION PATHWAY'
            : 'COMPONENT GRAPH'}
        <span className="caption-divider" />
        {graph.nodes.length} components
      </div>
      <div className="layout-tools">
        <button
          className="icon-button undo-layout"
          aria-label="Undo auto-layout"
          data-tooltip-detail="Restore previous layout"
          disabled={!canUndo || busy}
          onClick={undoLayout}
        >
          <Undo2 size={15} />
        </button>
        <button className="button button-layout" disabled={busy} onClick={() => void arrange()}>
          <WandSparkles size={14} className={busy ? 'spin' : ''} />
          {busy ? 'Arranging…' : 'Auto-layout'}
        </button>
        <button
          className="icon-button direction-button"
          aria-label="Switch layout direction"
          data-tooltip-detail="Switch layout direction"
          disabled={busy}
          onClick={() => void arrange(graph.layout.direction === 'RIGHT' ? 'DOWN' : 'RIGHT')}
        >
          {graph.layout.direction === 'RIGHT' ? <ArrowRight size={15} /> : <ArrowDown size={15} />}
        </button>
      </div>
      <ZoomControls onFit={fit} />
      {routingError && (
        <div className="routing-message" role="status">
          {routingError}
        </div>
      )}
      {routing && <div className="routing-pending">Routing connections…</div>}
      <div className="canvas-hint">
        Drag to arrange<span>·</span>Scroll to pan<span>·</span>Pinch or Ctrl + scroll to zoom
      </div>
    </div>
  )
}

export function GraphCanvas(props: DiagramProps) {
  return (
    <ReactFlowProvider key={`${props.graph.id}:${props.graph.nodes.map((n) => n.id).join('|')}`}>
      <Canvas {...props} />
    </ReactFlowProvider>
  )
}
