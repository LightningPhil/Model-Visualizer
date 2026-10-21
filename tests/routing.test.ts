import { beforeAll, describe, expect, it } from 'vitest'
import { AvoidLib } from 'libavoid-js'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import { planLayout } from '../src/model/layout'
import { routeGraph, routingProblems } from '../src/model/routing'
import {
  rectanglesOverlap,
  NODE_HEIGHT,
  NODE_WIDTH,
  segments,
  segmentLength,
} from '../src/model/geometry'
import type { AvoidEngine } from '../src/model/avoid'
import type { ModelGraph } from '../src/model/schema'
import { clearDrop } from '../src/components/useDiagram'
import { isWholeModel, expandWholeModel } from '../src/model/whole-model'
import { numericalGraph } from '../src/model/numerics/session'
import { defaultNumericalSettings } from '../src/model/numerics/engine'
import { storiesManifest } from '../src/model/checkpoints/stories'
import { parseModelConfig } from '../src/model/schema'
import { packStageFootprints, stageBounds } from '../src/model/group-layout'

let engine: AvoidEngine
const document = parseDocument(starter)
beforeAll(async () => {
  await AvoidLib.load()
  engine = AvoidLib.getInstance()
})

describe('complete diagram layout', () => {
  for (const id of ['whole-model', 'whole-model-heads'])
    for (const direction of ['RIGHT', 'DOWN'] as const)
      it(`${id} ${direction}: packs differently sized stages and keeps every route clear`, async () => {
        const graph = expandWholeModel(
          document,
          document.graphs.find((g) => g.id === id)!,
        )
        const plan = await planLayout(engine, graph, direction)
        expect(routingProblems(graph, plan.positions, plan.routing)).toEqual([])
        expect(Object.keys(plan.routing.edges)).toHaveLength(graph.edges.length)
        const bounds = stageBounds(graph, plan.positions).map((s) => s.rect)
        const maxWidth = Math.max(...bounds.map((b) => b.width)) + 440
        const maxHeight = Math.max(...bounds.map((b) => b.height)) + 440
        const columns = direction === 'RIGHT' ? 3 : 2
        const legacyArea = maxWidth * columns * maxHeight * Math.ceil(bounds.length / columns)
        expect(plan.routing.bounds.width * plan.routing.bounds.height).toBeLessThan(
          legacyArea * 0.8,
        )
      }, 90000)
  for (const direction of ['RIGHT', 'DOWN'] as const)
    it(`trained Stories260K ${direction}: all 354 components and links remain inspectable`, async () => {
      const graph = numericalGraph(
        document,
        defaultNumericalSettings,
        { direction, positions: {} },
        parseModelConfig(storiesManifest.config),
      )
      const plan = await planLayout(engine, graph, direction)
      expect(graph.nodes).toHaveLength(354)
      expect(routingProblems(graph, plan.positions, plan.routing)).toEqual([])
      expect(Object.keys(plan.routing.edges)).toHaveLength(graph.edges.length)
    }, 90000)
  for (const attention of ['MHA', 'GQA', 'MQA'] as const)
    for (const direction of ['RIGHT', 'DOWN'] as const)
      it(`numerical ${attention} ${direction}: every link and annotation stays clear`, async () => {
        const graph = numericalGraph(
            document,
            { ...defaultNumericalSettings, attention },
            { direction, positions: {} },
          ),
          plan = await planLayout(engine, graph, direction)
        expect(routingProblems(graph, plan.positions, plan.routing)).toEqual([])
        expect(Object.keys(plan.routing.edges)).toHaveLength(graph.edges.length)
      }, 30000)
  for (const graph of document.graphs
    .filter((g) => !isWholeModel(g))
    .map((g) => expandWholeModel(document, g)))
    for (const direction of ['RIGHT', 'DOWN'] as const) {
      it(`${graph.id} ${direction}: nodes, labels and all connections remain clear`, async () => {
        const plan = await planLayout(engine, graph, direction)
        const automatic = {
          ...graph,
          edges: graph.edges.map((e) => ({ ...e, sourceSide: undefined, targetSide: undefined })),
        }
        expect(routingProblems(automatic, plan.positions, plan.routing)).toEqual([])
        expect(Object.keys(plan.routing.edges)).toHaveLength(graph.edges.length)
        const rects = Object.values(plan.positions).map((p) => ({
          ...p,
          width: NODE_WIDTH,
          height: NODE_HEIGHT,
        }))
        rects.forEach((rect, i) =>
          rects
            .slice(i + 1)
            .forEach((other) => expect(rectanglesOverlap(rect, other, 24)).toBe(false)),
        )
        for (const route of Object.values(plan.routing.edges)) {
          expect(route.points[0].x).toBeCloseTo(route.source.point.x, 2)
          expect(route.points[0].y).toBeCloseTo(route.source.point.y, 2)
          expect(route.points.at(-1)!.x).toBeCloseTo(route.target.point.x, 2)
          expect(route.points.at(-1)!.y).toBeCloseTo(route.target.point.y, 2)
          expect(segments(route.points).every((s) => segmentLength(s) > 0)).toBe(true)
        }
      }, 30000)
    }
  it('routes the hand-composed overview without moving any nodes', () => {
    const graph = document.graphs[0]
    const positions = structuredClone(graph.layout.positions)
    const routes = routeGraph(engine, graph, positions)
    expect(routingProblems(graph, positions, routes)).toEqual([])
    expect(positions).toEqual(graph.layout.positions)
  })
  for (const graph of document.graphs.slice(1))
    if (Object.keys(graph.layout.positions).length === graph.nodes.length)
      it(`${graph.id}: preserves and routes the authored layout`, () => {
        const positions = structuredClone(graph.layout.positions)
        const routes = routeGraph(engine, graph, positions)
        expect(routingProblems(graph, positions, routes)).toEqual([])
        expect(positions).toEqual(graph.layout.positions)
      })
})

describe('stage packing', () => {
  for (const direction of ['RIGHT', 'DOWN'] as const)
    it(`${direction}: uses the actual footprint of each unequal section`, () => {
      const sizes = [
        { x: -20, y: 10, width: 1000, height: 200 },
        { x: 40, y: -80, width: 5000, height: 1800 },
        { x: 0, y: 0, width: 800, height: 300 },
        { x: 0, y: 0, width: 1600, height: 600 },
      ]
      const offsets = packStageFootprints(sizes, direction)
      expect(packStageFootprints(sizes, direction)).toEqual(offsets)
      const packed = offsets.map((p, i) => ({ ...sizes[i], ...p }))
      for (const [i, a] of packed.entries())
        for (const b of packed.slice(i + 1)) expect(rectanglesOverlap(a, b, 179)).toBe(false)
      const area =
        Math.max(...packed.map((r) => r.x + r.width)) *
        Math.max(...packed.map((r) => r.y + r.height))
      expect(area).toBeLessThan(4 * 5440 * 2240 * 0.65)
    })
})

function fixture(
  positions: ModelGraph['layout']['positions'],
  pairs: [string, string][],
  label = 'dot product',
): ModelGraph {
  const base = document.graphs[0]
  return {
    ...base,
    id: 'routing-fixture',
    nodes: Object.keys(positions).map((id) => ({ ...base.nodes[0], id })),
    edges: pairs.map(([source, target], i) => ({
      ...base.edges[0],
      id: `e${i}`,
      source,
      target,
      sourceSide: undefined,
      targetSide: undefined,
      label,
    })),
    layout: { direction: 'RIGHT', positions },
  }
}

describe('routing difficult geometry', () => {
  it('keeps a short connection straight when its long annotation cannot fit in the gap', () => {
    const graph = fixture(
      { a: { x: 0, y: 0 }, b: { x: 274, y: 0 } },
      [['a', 'b']],
      'a long explanation of a dot product',
    )
    const result = routeGraph(engine, graph, graph.layout.positions)
    expect(routingProblems(graph, graph.layout.positions, result)).toEqual([])
    expect(result.edges.e0.points).toHaveLength(2)
    expect(result.edges.e0.labelLeader).toHaveLength(2)
  })
  it('uses exactly four bends around a blocking component without label doglegs', () => {
    const graph = fixture({ a: { x: 0, y: 0 }, obstacle: { x: 420, y: 0 }, b: { x: 840, y: 0 } }, [
      ['a', 'b'],
    ])
    const result = routeGraph(engine, graph, graph.layout.positions)
    expect(routingProblems(graph, graph.layout.positions, result)).toEqual([])
    expect(result.edges.e0.points.length - 2).toBe(4)
  })
  it('assigns separate ports and lanes to fan-out, fan-in and long residual paths', async () => {
    const graph = fixture(
      {
        a: { x: 0, y: 300 },
        b: { x: 700, y: 0 },
        c: { x: 700, y: 300 },
        d: { x: 700, y: 600 },
        e: { x: 1400, y: 300 },
      },
      [
        ['a', 'b'],
        ['a', 'c'],
        ['a', 'd'],
        ['b', 'e'],
        ['c', 'e'],
        ['d', 'e'],
        ['a', 'e'],
      ],
    )
    for (const direction of ['RIGHT', 'DOWN'] as const) {
      const plan = await planLayout(engine, graph, direction)
      expect(routingProblems(graph, plan.positions, plan.routing)).toEqual([])
      const sourcePorts = plan.routing.ports.a.map((p) => JSON.stringify(p.point))
      expect(new Set(sourcePorts).size).toBe(sourcePorts.length)
    }
  })
  it('routes cycles and a self-loop without hiding a connection inside its own node', async () => {
    const graph = fixture({ a: { x: 0, y: 0 }, b: { x: 700, y: 0 }, c: { x: 1400, y: 0 } }, [
      ['a', 'b'],
      ['b', 'c'],
      ['c', 'a'],
      ['b', 'b'],
    ])
    const plan = await planLayout(engine, graph)
    expect(routingProblems(graph, plan.positions, plan.routing)).toEqual([])
  })
  it('is deterministic across repeated automatic layout and JSON restoration', async () => {
    const graph = document.graphs.find((g) => g.id === 'attention')!
    const first = await planLayout(engine, graph)
    const second = await planLayout(engine, JSON.parse(JSON.stringify(graph)))
    expect(second).toEqual(first)
    const restored = {
      ...graph,
      layout: { ...graph.layout, positions: first.positions },
      edges: graph.edges.map((e) => ({ ...e, sourceSide: undefined, targetSide: undefined })),
    }
    expect(routeGraph(engine, restored, restored.layout.positions)).toEqual(first.routing)
  })
  it('moves a dragged component to the nearest free space when dropped over another', () => {
    const nodes = [
      { id: 'a', position: { x: 0, y: 0 } },
      { id: 'b', position: { x: 600, y: 0 } },
    ]
    const drop = clearDrop('b', { x: 20, y: 30 }, nodes)
    expect(
      rectanglesOverlap(
        { ...drop, width: NODE_WIDTH, height: NODE_HEIGHT },
        { ...nodes[0].position, width: NODE_WIDTH, height: NODE_HEIGHT },
        28,
      ),
    ).toBe(false)
    expect(clearDrop('b', { x: 600, y: 0 }, nodes)).toEqual({ x: 600, y: 0 })
  })
})
