import { memo, useEffect } from 'react'
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
  useViewport,
  useUpdateNodeInternals,
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
import { roundedPath } from '../model/geometry'
import type { Port, RoutedEdge } from '../model/routing'
import { useDiagram } from './useDiagram'
import { ComponentIcon } from './Icons'

export type Selection = { type: 'node' | 'edge'; id: string } | null
type VisualData = {
  component: Component
  symbols: Record<string, number>
  showShapes: boolean
  symbolic: boolean
  onOpen: (id: string) => void
  ports: Port[]
}
export type VisualNode = Node<VisualData, 'component'>
export type VisualEdge = Edge<
  { connection: Connection; route: RoutedEdge; onSelect: (id: string) => void },
  'operation'
>
const sidePositions = {
  top: Position.Top,
  right: Position.Right,
  bottom: Position.Bottom,
  left: Position.Left,
}

const ModelNode = memo(function ModelNode({ data, selected, id }: NodeProps<VisualNode>) {
  const updateNodeInternals = useUpdateNodeInternals()
  const portSignature = data.ports.map((port) => `${port.id}:${port.side}:${port.offset}`).join('|')
  useEffect(() => {
    updateNodeInternals(id)
  }, [id, portSignature, updateNodeInternals])
  const { component, symbols, symbolic } = data
  const output =
    component.tensors.find((t) => t.role === 'output') ??
    component.tensors.find((t) => t.role === 'state')
  return (
    <div className={`model-node category-${component.category} ${selected ? 'is-selected' : ''}`}>
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
        {component.repeat && (
          <span className="repeat-badge">× {symbols[component.repeat] ?? component.repeat}</span>
        )}
        {component.childGraph && (
          <button
            className="node-open nodrag"
            aria-label={`Explore ${component.title}`}
            title="Explore inside"
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

function OperationEdge(props: EdgeProps<VisualEdge>) {
  const route = props.data!.route
  const path = roundedPath(route.points, 5)
  const connection = props.data!.connection
  return (
    <>
      {route.labelLeader && (
        <path className="edge-annotation-leader" d={roundedPath(route.labelLeader, 0)} />
      )}
      <g
        className="routed-connection"
        data-edge-id={props.id}
        data-route-points={JSON.stringify(route.points)}
      >
        <BaseEdge
          id={props.id}
          path={path}
          markerEnd={props.markerEnd}
          style={props.style}
          interactionWidth={24}
        />
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
          title={connection.operation}
        >
          {connection.label}
        </button>
      </EdgeLabelRenderer>
    </>
  )
}

const nodeTypes = { component: ModelNode }
const edgeTypes = { operation: OperationEdge }

export type LayoutSnapshot = { layout: ModelGraph['layout']; edges: ModelGraph['edges'] }

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
}

function Canvas(props: DiagramProps) {
  const { graph, onSelect, onOpen } = props
  const flow = useReactFlow<VisualNode, VisualEdge>()
  const viewport = useViewport()
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

  return (
    <div ref={canvasRef} className="graph-canvas" data-testid="graph-canvas">
      <ReactFlow<VisualNode, VisualEdge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onNodeClick={(_, node) => onSelect({ type: 'node', id: node.id })}
        onNodeDoubleClick={(_, node) =>
          node.data.component.childGraph && onOpen(node.data.component.childGraph)
        }
        onEdgeClick={(_, edge) => onSelect({ type: 'edge', id: edge.id })}
        onPaneClick={() => onSelect(null)}
        onNodeDragStop={(_, node) => onDragStop(node)}
        onMoveEnd={(_, next) => {
          if (initialized.current) props.onViewport(next)
        }}
        onInit={() => {
          initialized.current = true
        }}
        defaultViewport={graph.layout.viewport}
        minZoom={0.035}
        maxZoom={1.8}
        nodesDraggable={!busy}
        nodesConnectable={false}
        deleteKeyCode={null}
        colorMode={props.theme}
        panOnScroll
        zoomOnDoubleClick={false}
        selectionOnDrag={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="var(--grid)" />
        <MiniMap
          pannable
          zoomable
          position="bottom-right"
          nodeColor={(node) =>
            ({ data: '#46b6cc', compute: '#747fc4', memory: '#a88de8', output: '#d3ae65' })[
              (node.data as VisualData).component.category
            ]
          }
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
          title="Restore previous layout"
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
          title="Switch layout direction"
          disabled={busy}
          onClick={() => void arrange(graph.layout.direction === 'RIGHT' ? 'DOWN' : 'RIGHT')}
        >
          {graph.layout.direction === 'RIGHT' ? <ArrowRight size={15} /> : <ArrowDown size={15} />}
        </button>
      </div>
      <div className="zoom-controls">
        <button
          aria-label="Zoom out"
          title="Zoom out"
          onClick={() => void flow.zoomOut({ duration: 200 })}
        >
          <Minus size={15} />
        </button>
        <span>{Math.round(viewport.zoom * 100)}%</span>
        <button
          aria-label="Zoom in"
          title="Zoom in"
          onClick={() => void flow.zoomIn({ duration: 200 })}
        >
          <Plus size={15} />
        </button>
        <i />
        <button aria-label="Fit graph" title="Fit graph" onClick={() => fitGraph()}>
          <Scan size={17} />
        </button>
        <button
          aria-label="Actual size"
          title="Actual size"
          onClick={() => void flow.zoomTo(1, { duration: 300 })}
        >
          <Maximize2 size={14} />
        </button>
      </div>
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
    <ReactFlowProvider key={props.graph.id}>
      <Canvas {...props} />
    </ReactFlowProvider>
  )
}
