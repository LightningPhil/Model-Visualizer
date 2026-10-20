import { describe, expect, it } from 'vitest'
import starter from '../examples/transformer-foundations.model.json'
import { parseDocument } from '../src/model/schema'
import {
  LEGACY_DOCUMENTATION,
  LEGACY_SIMULATION_NOTE,
  TEACHING_PROVENANCE_KEY,
  upgradeTeachingProvenance,
} from '../src/model/teaching-provenance'

const base = parseDocument(starter)
// Distinct canonical wording proves the migration reads the authored reference.
const reference = {
  ...base,
  documentation: 'Canonical authored documentation.',
  graphs: base.graphs.map((g) =>
    g.id !== 'generation'
      ? g
      : {
          ...g,
          nodes: g.nodes.map((n) => ({
            ...n,
            notes: ['Graphical simulation uses abstract packets.'],
          })),
        },
  ),
}
function older() {
  const doc = structuredClone(base)
  if (doc.extensions) delete doc.extensions[TEACHING_PROVENANCE_KEY]
  doc.documentation = LEGACY_DOCUMENTATION
  const node = doc.graphs.find((g) => g.id === 'generation')!.nodes[0]
  node.notes = [LEGACY_SIMULATION_NOTE, 'My extra annotation.']
  doc.ui.notes = 'My notebook must survive.'
  return doc
}
describe('stock teaching-provenance clarification', () => {
  it('the authored glossary distinguishes processed, retained and working extents', () => {
    expect(base.documentation).toContain('S (absolute processed extent)')
    expect(base.documentation).toContain('R (retained positions after a call)')
    expect(base.documentation).toContain('U (working keys P + T)')
    expect(base.extensions![TEACHING_PROVENANCE_KEY]).toBe(1)
    for (const node of base.graphs.find((g) => g.id === 'generation')!.nodes)
      expect(node.notes).not.toContain(LEGACY_SIMULATION_NOTE)
  })
  it('updates only exact old stock explanations while retaining IDs, geometry, notes and metadata', () => {
    const original = older()
    original.extensions = { ...original.extensions, 'someone-else.data': { keep: ['all of it'] } }
    const next = upgradeTeachingProvenance(original, reference)
    expect(next.documentation).toBe(reference.documentation)
    expect(next.graphs.find((g) => g.id === 'generation')!.nodes[0].notes).toEqual([
      'Graphical simulation uses abstract packets.',
      'My extra annotation.',
    ])
    expect(next.ui).toEqual(original.ui)
    expect(next.graphs.map((g) => g.layout)).toEqual(original.graphs.map((g) => g.layout))
    expect(next.graphs.map((g) => g.nodes.map((n) => n.id))).toEqual(
      original.graphs.map((g) => g.nodes.map((n) => n.id)),
    )
    expect(next.extensions!['someone-else.data']).toEqual({ keep: ['all of it'] })
    expect(next.extensions![TEACHING_PROVENANCE_KEY]).toBe(1)
    expect(upgradeTeachingProvenance(next, reference)).toBe(next)
  })
  it('keeps custom documentation and edited prose intact', () => {
    const doc = older()
    doc.documentation = 'My own explanation of this model.'
    doc.graphs.find((g) => g.id === 'generation')!.nodes[0].notes[0] += ' My own addition.'
    const next = upgradeTeachingProvenance(doc, reference)
    expect(next.documentation).toBe(doc.documentation)
    expect(next.graphs.find((g) => g.id === 'generation')!.nodes[0].notes).toEqual(
      doc.graphs.find((g) => g.id === 'generation')!.nodes[0].notes,
    )
  })
  it('preserves unknown future migration metadata and unrelated documents', () => {
    const future = { ...older(), extensions: { [TEACHING_PROVENANCE_KEY]: 2, unknown: 'keep' } }
    expect(upgradeTeachingProvenance(future, reference)).toBe(future)
    const other = { ...older(), id: 'another-model' }
    expect(upgradeTeachingProvenance(other, reference)).toBe(other)
  })
})
