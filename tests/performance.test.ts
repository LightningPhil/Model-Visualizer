import { afterEach, describe, expect, it, vi } from 'vitest'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import { createGraphCompiler } from '../src/model/graph-compiler'
import { reconcileNodes } from '../src/components/diagram-view'
import type { VisualNode, VisualEdge } from '../src/components/GraphCanvas'
import { finiteMagnitude } from '../src/model/numerics/tensor'
import { DiagramWorker } from '../src/model/layout-client'
import type { RoutingResult } from '../src/model/routing'
import { overviewScene, staleOverviewRoutes } from '../src/components/overview/scene'

describe('large-map work stays proportional to changes', () => {
  it('retains semantic instances across camera, layout and notebook edits; recompiles authored/config changes', () => {
    const doc = parseDocument(starter),
      compile = createGraphCompiler()
    const graph = doc.graphs.find((g) => g.id === 'whole-model-heads')!
    const first = compile(doc, graph)
    const layout = { ...graph.layout, viewport: { x: 17, y: 29, zoom: 0.08 } }
    const moved = { ...graph, layout, title: 'My study title' }
    const nextDoc = {
      ...doc,
      graphs: doc.graphs.map((g) => (g.id === graph.id ? moved : g)),
      ui: { ...doc.ui, notes: 'My notes' },
    }
    const next = compile(nextDoc, moved)
    expect(next.nodes).toBe(first.nodes)
    expect(next.edges).toBe(first.edges)
    expect(next.layout).toBe(layout)
    expect(next.title).toBe('My study title')
    const changed = { ...nextDoc, config: { ...doc.config, nLayers: doc.config.nLayers + 1 } }
    expect(compile(changed, moved).nodes.length).toBeGreaterThan(next.nodes.length)
    const edited = {
      ...nextDoc,
      graphs: nextDoc.graphs.map((g) =>
        g.id !== 'attention'
          ? g
          : {
              ...g,
              nodes: g.nodes.map((n) => (n.id !== 'q' ? n : { ...n, title: 'My Q projection' })),
            },
      ),
    }
    expect(compile(edited, moved).nodes.some((n) => n.title.includes('My Q projection'))).toBe(true)
  })

  it('only replaces the dragged node and retains measured handles while new paths are pending', () => {
    const doc = parseDocument(starter),
      symbols = { D: 64 },
      open = () => {}
    const authored: VisualNode[] = doc.graphs[0].nodes.map((component, i) => ({
      id: component.id,
      type: 'component',
      position: { x: i * 400, y: 0 },
      measured: { width: 246, height: 112 },
      data: { component, symbols, showShapes: true, symbolic: false, onOpen: open, ports: [] },
    }))
    const ports = [
      {
        id: 's-test',
        nodeId: authored[0].id,
        edgeId: 'test',
        type: 'source' as const,
        side: 'right' as const,
        offset: 56,
        point: { x: 246, y: 56 },
        classId: 1,
      },
    ]
    const routes: RoutingResult = {
      ports: { [authored[0].id]: ports },
      edges: {},
      bounds: { x: 0, y: 0, width: 3000, height: 112 },
      bends: 0,
      crossings: 0,
    }
    const first = reconcileNodes(authored, authored, [], routes)
    const moved = authored.map((node, i) =>
      i ? node : { ...node, position: { x: 23, y: 31 }, dragging: true },
    )
    const second = reconcileNodes(
      moved.map((node) => ({ ...node, data: { ...node.data, ports: [] } })),
      moved,
      first,
      null,
    )
    expect(second[0]).not.toBe(first[0])
    expect(second[0].data).toBe(first[0].data)
    expect(second[0].data.ports).toBe(ports)
    expect(second[0].measured).toBe(first[0].measured)
    expect(second[0].dragging).toBe(true)
    for (let i = 1; i < first.length; i++) expect(second[i]).toBe(first[i])
    const scene = overviewScene(second, [], [])
    expect(scene.nodes.map((n) => n.id)).toEqual(second.map((n) => n.id))
    expect(scene.nodes[0]).toMatchObject({ x: 23, y: 31, width: 246, height: 112 })
    const connection = doc.graphs[0].edges[0]
    const edge: VisualEdge = {
      id: connection.id,
      source: connection.source,
      target: connection.target,
      data: {
        connection,
        onSelect: () => {},
        route: {
          id: connection.id,
          points: [
            { x: 246, y: 56 },
            { x: 400, y: 56 },
          ],
          label: { x: 290, y: 44, width: 60, height: 24 },
          source: ports[0],
          target: { ...ports[0], type: 'target' },
        },
      },
    }
    const before = overviewScene(first, [edge], [])
    const sameGeometry = overviewScene(
      first.map((node) => ({ ...node, selected: true })),
      [edge],
      [],
    )
    expect(staleOverviewRoutes(before, sameGeometry)).toBe(false)
    expect(staleOverviewRoutes(before, overviewScene(second, [edge], []))).toBe(true)
    expect(staleOverviewRoutes(before, scene)).toBe(true)
    expect(staleOverviewRoutes(scene, overviewScene(second, [edge], []))).toBe(false)
  })

  it('scales a million-value tensor without boxed copies, spread limits or mask contamination', () => {
    const values = new Float32Array(1_000_000)
    values[0] = NaN
    values[1] = -Infinity
    values[2] = Infinity
    values[999_999] = -987
    expect(finiteMagnitude(values)).toBe(987)
    expect(finiteMagnitude(new Float32Array())).toBe(1e-12)
    expect(finiteMagnitude([-Infinity, NaN])).toBe(1e-12)
  })
})

class FakeWorker {
  static instances: FakeWorker[] = []
  messages: { id: number; graph?: unknown; type: string }[] = []
  onmessage?: (event: { data: unknown }) => void
  onerror?: () => void
  stopped = false
  constructor() {
    FakeWorker.instances.push(this)
  }
  postMessage(message: { id: number; graph?: unknown; type: string }) {
    this.messages.push(message)
  }
  terminate() {
    this.stopped = true
  }
  reply(index: number) {
    this.onmessage?.({ data: { id: this.messages[index].id, result: { edges: {}, ports: {} } } })
  }
}
afterEach(() => {
  vi.unstubAllGlobals()
  FakeWorker.instances = []
})

describe('diagram worker traffic', () => {
  it('sends authored content once, then only positions until semantics change', async () => {
    vi.stubGlobal('Worker', FakeWorker)
    const client = new DiagramWorker(),
      worker = FakeWorker.instances[0]
    const graph = parseDocument(starter).graphs[0]
    const first = client.route(graph, graph.layout.positions)
    worker.reply(0)
    await first
    await new Promise((resolve) => setTimeout(resolve, 0))
    const second = client.route(
      { ...graph, layout: { ...graph.layout, viewport: { x: 4, y: 7, zoom: 1 } } },
      graph.layout.positions,
    )
    expect(worker.messages[0].graph).toBe(graph)
    expect(worker.messages[1].graph).toBeUndefined()
    worker.reply(1)
    await second
    await new Promise((resolve) => setTimeout(resolve, 0))
    const updated = {
      ...graph,
      edges: graph.edges.map((edge) => ({ ...edge, label: `${edge.label}!` })),
    }
    const third = client.route(updated, graph.layout.positions)
    expect(worker.messages[2].graph).toBe(updated)
    worker.reply(2)
    await third
    client.dispose()
    expect(worker.stopped).toBe(true)
  })

  it('coalesces queued drags and rejects both active and queued jobs on disposal', async () => {
    vi.stubGlobal('Worker', FakeWorker)
    const client = new DiagramWorker(),
      worker = FakeWorker.instances[0],
      graph = parseDocument(starter).graphs[0]
    const first = client.route(graph, {}).catch((e: Error) => e.name)
    const superseded = client.route(graph, {}).catch((e: Error) => e.name)
    const last = client.route(graph, {}).catch((e: Error) => e.name)
    expect(await superseded).toBe('AbortError')
    expect(worker.messages).toHaveLength(1)
    client.dispose()
    expect(await first).toBe('AbortError')
    expect(await last).toBe('AbortError')
    expect(worker.messages).toHaveLength(1)
  })
})
