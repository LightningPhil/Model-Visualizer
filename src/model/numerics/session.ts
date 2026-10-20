import { z } from 'zod'
import { documentSchema, type ModelDocument, type ModelGraph } from '../schema'
import { expandWholeModel, INSTANCE_KEY } from '../whole-model'
import {
  defaultNumericalSettings,
  teachingConfig,
  type NumericalSettings,
  type NumericalTrace,
} from './engine'

import { NUMERICAL_SESSION_KEY } from './identity'
export { NUMERICAL_SESSION_KEY } from './identity'
const policy = documentSchema.shape.config.shape.kvCache
export const numericalSettingsSchema = z
  .strictObject({
    seed: z.number().int().min(0).max(2147483647),
    inputIds: z.array(z.number().int().min(0).max(255)).min(1).max(8),
    generate: z.number().int().min(1).max(8),
    attention: z.enum(['MHA', 'GQA', 'MQA']),
    kvCache: policy.unwrap(),
    sampling: z.enum(['greedy', 'sample']),
    temperature: z.number().min(0.1).max(3),
  })
  .refine(
    (settings) =>
      settings.kvCache.mode === 'full' ||
      (settings.kvCache.windowSize <= 64 &&
        (settings.kvCache.mode !== 'sparse' || settings.kvCache.stride <= 64)),
    'Cache settings exceed the numerical profile capacity.',
  )
export const sessionSchema = z.strictObject({
  version: z.literal(1),
  settings: numericalSettingsSchema,
  layout: documentSchema.shape.graphs.element.shape.layout,
  layouts: z.record(z.string(), documentSchema.shape.graphs.element.shape.layout).default({}),
  bookmarks: z
    .array(
      z.strictObject({
        id: z.string().max(160),
        title: z.string().max(120),
        node: z.string().max(100),
        call: z.number().int().min(0).max(8),
        note: z.string().max(2000),
      }),
    )
    .max(24),
  notes: z.string().max(24000),
  pinned: numericalSettingsSchema.optional(),
})
export type NumericalSession = z.infer<typeof sessionSchema>
export function loadNumericalSession(doc: ModelDocument): {
  session: NumericalSession
  writable: boolean
} {
  const raw = doc.extensions?.[NUMERICAL_SESSION_KEY],
    parsed = sessionSchema.safeParse(raw)
  if (parsed.success) return { session: parsed.data, writable: true }
  return {
    session: {
      version: 1,
      settings: structuredClone(defaultNumericalSettings),
      layout: { direction: 'RIGHT', positions: {} },
      layouts: {},
      bookmarks: [],
      notes: '',
    },
    writable: raw === undefined,
  }
}

/** Reuse authored semantic templates. The lab owns only a separate execution
 * profile and layout, never a second generated model document. */
export function numericalGraph(
  doc: ModelDocument,
  settings: NumericalSettings,
  layout: ModelGraph['layout'],
): ModelGraph {
  const recipe = doc.graphs.find((g) => g.id === 'whole-model-heads')
  if (!recipe) throw new Error('The numerical lab requires the foundations split-head recipe.')
  const full = expandWholeModel(
    { ...doc, config: teachingConfig(settings) },
    { ...recipe, id: 'numerical-lab', layout },
  )
  const keep = new Set([
    'tokens.ids',
    'tokens.lookup',
    'output.finalnorm',
    'output.lmhead',
    'output.sample',
    'generation.next',
    'generation.commit',
    'generation.decode',
    'generation.feedback',
  ])
  const nodes = full.nodes
    .filter((n) => n.id.startsWith('layer-') || keep.has(n.id))
    .map((n) => ({
      ...n,
      childGraph: undefined,
      ...(n.id === 'tokens.ids'
        ? {
            subtitle: 'Opaque vocabulary indices',
            description:
              'Input IDs for the local numerical teaching model. No text tokenizer is used.',
          }
        : {}),
      ...(n.id === 'tokens.lookup'
        ? {
            subtitle: 'Rows of seeded E[V,D]',
            description:
              'Each input vocabulary index selects its corresponding embedding row. These local parameters are seeded and untrained.',
          }
        : {}),
      ...(n.id === 'generation.next'
        ? {
            title: 'Chosen next ID',
            subtitle: 'Argmax or seeded categorical draw',
            description:
              'The final query row selects one vocabulary ID. This profile uses autoregressive decoding, without numerical MTP heads.',
            tensors: [{ name: 'Selected ID', role: 'output' as const, shape: ['B', '1'] }],
            equation: undefined,
          }
        : {}),
      ...(n.id === 'generation.commit'
        ? {
            title: 'Commit chosen ID',
            subtitle: 'Append one output token',
            description:
              'The chosen ID joins the confirmed sequence. Its K/V is computed only when that ID is consumed by the next decoder call.',
            tensors: [{ name: 'Confirmed new ID', role: 'output' as const, shape: ['B', '1'] }],
            equation: undefined,
          }
        : {}),
      ...(n.id === 'generation.decode'
        ? {
            title: 'Output token IDs',
            subtitle: 'Opaque IDs · no text decoder',
            description:
              'The numerical lab shows actual vocabulary indices from an untrained model; no text tokenizer or language decoder is used.',
            tensors: [{ name: 'Latest output ID', role: 'output' as const, shape: ['B', '1'] }],
            equation: undefined,
          }
        : {}),
      ...(n.id === 'generation.feedback'
        ? {
            subtitle: 'One ID + retained K/V',
            description:
              'Reuse retained state and feed the single latest vocabulary ID to the next decoder call. No text is re-tokenized.',
            tensors: [{ name: 'Next input ID', role: 'output' as const, shape: ['B', '1'] }],
            equation: undefined,
            sourceIds: ['attention', 'gqa'],
          }
        : {}),
      extensions: {
        ...n.extensions,
        [INSTANCE_KEY]: {
          ...(n.extensions?.[INSTANCE_KEY] as object),
          optional: false,
          ...(n.id.startsWith('generation.') ? { stage: 'AUTOREGRESSIVE GENERATION' } : {}),
        },
      },
    }))
  const ids = new Set(nodes.map((n) => n.id))
  const edges = full.edges
    .filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((e) =>
      e.source.startsWith('generation.') || e.target.startsWith('generation.')
        ? {
            ...e,
            shape: ['B', '1'],
            ...(e.source === 'generation.next' && e.target === 'generation.commit'
              ? {
                  kind: 'flow' as const,
                  label: 'selected ID',
                  operation: 'Commit the selected vocabulary ID',
                  description:
                    'Append the actual chosen ID to the confirmed sequence. Its K/V will be computed by the next decoder call.',
                }
              : {}),
          }
        : e,
    )
  if (!edges.some((e) => e.source === 'generation.next' && e.target === 'generation.commit'))
    edges.push({
      id: 'numerical-next-commit',
      source: 'generation.next',
      target: 'generation.commit',
      label: 'selected ID',
      operation: 'Commit the actual selected vocabulary ID.',
      description:
        'Greedy argmax or seeded categorical sampling selects from the final query row. This teaching model has untrained parameters.',
      shape: ['B', '1'],
      kind: 'flow',
    })
  if (!edges.some((e) => e.source === 'generation.decode' && e.target === 'generation.feedback'))
    edges.push({
      id: 'numerical-output-feedback',
      source: 'generation.decode',
      target: 'generation.feedback',
      label: 'next decoder call',
      operation: 'Feed the selected ID into the next call.',
      description:
        'The latest output ID is uncached until the next decoder call. Retained prefix K/V is reused.',
      shape: ['B', '1'],
      kind: 'state',
    })
  return {
    ...full,
    title: 'The mathematics, alive.',
    subtitle: 'A tiny transformer. Actual numbers. Every operation.',
    nodes,
    edges,
  }
}
export function exportNumericalTrace(trace: NumericalTrace) {
  // JSON cannot express infinities; represent masked scores explicitly.
  return (
    JSON.stringify(
      {
        ...trace,
        tensors: Object.fromEntries(
          Object.entries(trace.tensors).map(([id, t]) => [
            id,
            {
              ...t,
              data: Array.from(t.data, (v) =>
                Number.isFinite(v)
                  ? v
                  : v === -Infinity
                    ? '-Infinity'
                    : v === Infinity
                      ? 'Infinity'
                      : 'NaN',
              ),
            },
          ]),
        ),
      },
      null,
      2,
    ) + '\n'
  )
}
