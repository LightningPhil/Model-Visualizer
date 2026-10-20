import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  BackgroundVariant,
  MiniMap,
  MarkerType,
  Handle,
  Position,
  BaseEdge,
  EdgeLabelRenderer,
  getSmoothStepPath,
  applyNodeChanges,
  useReactFlow,
  useViewport,
} from '@xyflow/react'
import type { Node, Edge, NodeProps, EdgeProps, NodeChange, Viewport } from '@xyflow/react'
import {
  ArrowDown,
  ArrowRight,
  Expand,
  Maximize2,
  Minus,
  Plus,
  Scan,
  WandSparkles,
} from 'lucide-react'
import { shapeLabel, type Component, type Connection, type ModelGraph } from '../model/schema'
import { autoLayout, NODE_HEIGHT, NODE_WIDTH } from '../model/layout'
import { ComponentIcon } from './Icons'

export type Selection = { type: 'node' | 'edge'; id: string } | null
type VisualData = {
  component: Component
  symbols: Record<string, number>
  showShapes: boolean
  symbolic: boolean
  onOpen: (id: string) => void
}
type VisualNode = Node<VisualData, 'component'>
type VisualEdge = Edge<{ connection: Connection; onSelect: (id: string) => void }, 'operation'>
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
    <div className={`model-node category-${component.category} ${selected ? 'is-selected' : ''}`}>
      {Object.entries(sidePositions).flatMap(([side, position]) => [
        <Handle
          key={`t-${side}`}
          id={`t-${side}`}
          type="target"
          position={position}
          isConnectable={false}
        />,
        <Handle
          key={`s-${side}`}
          id={`s-${side}`}
          type="source"
          position={position}
          isConnectable={false}
        />,
      ])}
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
  const [path, x, y] = getSmoothStepPath({ ...props, borderRadius: 18, offset: 32 })
  const connection = props.data!.connection
  return (
    <>
      <BaseEdge
        id={props.id}
        path={path}
        markerEnd={props.markerEnd}
        style={props.style}
        interactionWidth={24}
      />
      <EdgeLabelRenderer>
        <button
          className={`edge-label nodrag nopan edge-${connection.kind} ${props.selected ? 'is-selected' : ''}`}
          style={{ transform: `translate(-50%, -50%) translate(${x}px,${y}px)` }}
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

type Props = {
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
  onViewport: (viewport: Viewport) => void
  notify: (message: string) => void
}

function Canvas(props: Props) {
  const { graph, symbols, selection, showShapes, symbolic, onSelect, onOpen, onLayout, notify } =
    props
  const flow = useReactFlow<VisualNode, VisualEdge>()
  const viewport = useViewport()
  const [nodes, setNodes] = useState<VisualNode[]>([])
  const [busy, setBusy] = useState(false)
  const initialized = useRef(false)
  const generation = useRef(0)
  const onOpenRef = useRef(onOpen)
  onOpenRef.current = onOpen

  const buildNodes = useCallback(
    (positions: ModelGraph['layout']['positions']) =>
      graph.nodes.map((component, i): VisualNode => ({
        id: component.id,
        type: 'component',
        position: positions[component.id] ?? { x: i * 340, y: 100 },
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
        ariaLabel: `${component.title}. ${component.subtitle}.`,
        data: { component, symbols, showShapes, symbolic, onOpen: (id) => onOpenRef.current(id) },
        selected: selection?.type === 'node' && selection.id === component.id,
      })),
    [graph.nodes, symbols, showShapes, symbolic, selection],
  )

  useEffect(() => {
    setNodes((previous) => {
      const current = new Map(previous.map((n) => [n.id, n.position]))
      return buildNodes(graph.layout.positions).map((node) => ({
        ...node,
        position: current.get(node.id) ?? node.position,
      }))
    })
  }, [buildNodes, graph.layout.positions])

  // A Canvas is keyed by graph ID: each view restores its own positions and viewport.
  useEffect(() => {
    const current = ++generation.current
    if (Object.keys(graph.layout.positions).length === graph.nodes.length) {
      setNodes(buildNodes(graph.layout.positions))
      initialized.current = true
      return
    }
    setBusy(true)
    autoLayout(graph)
      .then((positions) => {
        if (generation.current !== current) return
        setNodes(buildNodes(positions))
        onLayout(positions)
        initialized.current = true
        requestAnimationFrame(() => void flow.fitView({ padding: 0.16, duration: 300, maxZoom: 1 }))
      })
      .catch(() => notify('Layout could not be computed. You can still arrange nodes manually.'))
      .finally(() => {
        if (generation.current === current) setBusy(false)
      })
    return () => {
      generation.current++
    }
    // Graph remounts on navigation. Avoid resetting manual layouts on selection changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const edges = useMemo(
    (): VisualEdge[] =>
      graph.edges.map((connection) => {
        const defaultSource = graph.layout.direction === 'RIGHT' ? 'right' : 'bottom'
        const defaultTarget = graph.layout.direction === 'RIGHT' ? 'left' : 'top'
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
          sourceHandle: `s-${connection.sourceSide ?? defaultSource}`,
          targetHandle: `t-${connection.targetSide ?? defaultTarget}`,
          type: 'operation',
          markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
          selected,
          data: { connection, onSelect: (id) => onSelect({ type: 'edge', id }) },
          style: {
            stroke: color,
            strokeWidth: selected ? 2 : 1.4,
            strokeDasharray: connection.kind === 'flow' ? undefined : '5 5',
          },
        }
      }),
    [graph.edges, graph.layout.direction, selection, onSelect],
  )

  async function arrange(direction = graph.layout.direction) {
    if (busy) return
    const current = generation.current
    setBusy(true)
    try {
      const positions = await autoLayout(graph, direction)
      if (generation.current !== current) return
      setNodes(buildNodes(positions))
      onLayout(positions, direction, true)
      requestAnimationFrame(() => void flow.fitView({ padding: 0.16, duration: 500, maxZoom: 1 }))
      notify('Layout arranged. Drag any component to make it yours.')
    } catch {
      notify('Auto-layout failed. Your previous layout is preserved.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="graph-canvas" data-testid="graph-canvas">
      <ReactFlow<VisualNode, VisualEdge>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={(changes: NodeChange<VisualNode>[]) =>
          setNodes((current) => applyNodeChanges(changes, current))
        }
        onNodeClick={(_, node) => onSelect({ type: 'node', id: node.id })}
        onNodeDoubleClick={(_, node) =>
          node.data.component.childGraph && onOpen(node.data.component.childGraph)
        }
        onEdgeClick={(_, edge) => onSelect({ type: 'edge', id: edge.id })}
        onPaneClick={() => onSelect(null)}
        onNodeDragStop={(_, node) =>
          onLayout({ ...graph.layout.positions, [node.id]: node.position })
        }
        onMoveEnd={(_, next) => {
          if (initialized.current) props.onViewport(next)
        }}
        onInit={() => {
          initialized.current = true
        }}
        fitView={!graph.layout.viewport}
        defaultViewport={graph.layout.viewport}
        fitViewOptions={{ padding: 0.16, maxZoom: 1 }}
        minZoom={0.12}
        maxZoom={1.8}
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
        <button
          aria-label="Fit graph"
          title="Fit graph"
          onClick={() => void flow.fitView({ padding: 0.16, duration: 400, maxZoom: 1 })}
        >
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
      <div className="canvas-hint">
        Drag to arrange<span>·</span>Scroll to pan<span>·</span>Pinch or Ctrl + scroll to zoom
      </div>
    </div>
  )
}

export function GraphCanvas(props: Props) {
  return (
    <ReactFlowProvider key={props.graph.id}>
      <Canvas {...props} />
    </ReactFlowProvider>
  )
}
