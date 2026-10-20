import { describe, expect, it } from 'vitest'
import { AvoidLib } from 'libavoid-js'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import {
  addWholeModelView,
  expandWholeModel,
  instanceOf,
  isWholeModel,
} from '../src/model/whole-model'
import { planLayout } from '../src/model/layout'
import { routingProblems } from '../src/model/routing'

const doc = parseDocument(starter)
const recipe = doc.graphs.find(isWholeModel)!
describe('whole-model composition', () => {
  it('upgrades compatible saved workspaces without replacing notes, layouts or custom metadata', () => {
    const old = {
      ...structuredClone(doc),
      graphs: structuredClone(doc.graphs.slice(0, 9)),
      sources: doc.sources.filter((s) => s.id !== 'speculative'),
      extensions: { 'org.example': { keep: true } },
    }
    old.ui.notes = 'Preserve my research.'
    old.graphs[0].layout.positions.decoder = { x: 900, y: 450 }
    const upgraded = addWholeModelView(old, doc)
    expect(upgraded.graphs).toHaveLength(11)
    expect(upgraded.graphs.slice(0, 9)).toEqual(old.graphs)
    expect(upgraded.ui.notes).toBe(old.ui.notes)
    expect(upgraded.extensions).toEqual(old.extensions)
    expect(parseDocument(upgraded)).toEqual(upgraded)
    expect(addWholeModelView(upgraded, doc)).toBe(upgraded)
  })
  it('validates expanded saved positions and very small whole-map viewports on round trip', () => {
    const saved = structuredClone(doc)
    const recipe = saved.graphs.find(isWholeModel)!
    recipe.layout = {
      direction: 'RIGHT',
      positions: { 'layer-5.attn.head-7': { x: 100, y: 200 } },
      viewport: { x: 0, y: 0, zoom: 0.012 },
    }
    expect(parseDocument(JSON.parse(JSON.stringify(saved)))).toEqual(saved)
    recipe.layout.positions.missing = { x: 1, y: 2 }
    expect(() => parseDocument(saved)).toThrow(/unknown node/)
  })
  it('rejects recursive recipes and oversized expansions before rendering', () => {
    const recursive = structuredClone(doc)
    const recipe = recursive.graphs.find(isWholeModel)!
    const extension = recipe.nodes[0].extensions!['model-visualizer.expansion'] as {
      input: { graph: string }
    }
    extension.input.graph = recipe.id
    expect(() => parseDocument(recursive)).toThrow(/recursive/)
    expect(() => parseDocument({ ...doc, config: { ...doc.config, nLayers: 512 } })).toThrow(
      /view budget/,
    )
  })
  it('expands every layer, query head and KV-head state view on one canvas', () => {
    const graph = expandWholeModel(doc, recipe)
    expect(graph.nodes.length).toBeGreaterThan(240)
    for (let layer = 0; layer < doc.config.nLayers; layer++) {
      expect(graph.nodes.some((n) => n.id === `layer-${layer}.swiglu.down`)).toBe(true)
      expect(
        graph.nodes.filter((n) => instanceOf(n)?.layer === layer && n.kind === 'attention-head'),
      ).toHaveLength(doc.config.nQueryHeads)
    }
    expect(new Set(graph.nodes.map((n) => n.id)).size).toBe(graph.nodes.length)
    expect(
      graph.edges.some((e) => e.source === 'generation.feedback' && e.target === 'tokens.ids'),
    ).toBe(true)
  })
  for (const direction of ['RIGHT', 'DOWN'] as const)
    it(`arranges and routes the complete expanded model ${direction}`, async () => {
      await AvoidLib.load()
      const graph = expandWholeModel(doc, recipe)
      const plan = await planLayout(AvoidLib.getInstance(), graph, direction)
      expect(routingProblems(graph, plan.positions, plan.routing)).toEqual([])
    }, 120000)
})
