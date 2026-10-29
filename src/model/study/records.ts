import { z } from 'zod'
import { numericalSettingsSchema } from '../numerics/session'
import { trainedSettingsSchema } from '../checkpoints/session'

export const trainingRecipeSchema = z
  .object({
    seed: z.number().int().min(0).max(2147483647),
    task: z.enum(['curve', 'extrapolate', 'noise']),
    optimizer: z.enum(['sgd', 'adam', 'adamw']),
    rate: z.number().positive().max(2),
    decay: z.number().min(0).max(1),
    steps: z.number().int().min(0).max(1000),
  })
  .passthrough()
export const headInterventionSchema = z
  .object({
    mode: z.enum(['zero', 'patch']),
    layer: z.number().int().min(0).max(4),
    head: z.number().int().min(0).max(7),
    donorPrompt: z
      .string()
      .max(600)
      .refine((s) => !s.includes('\0'))
      .optional(),
  })
  .passthrough()
export type HeadIntervention = z.infer<typeof headInterventionSchema>
export const trainedComparisonSchema = z
  .object({ settings: trainedSettingsSchema, intervention: headInterventionSchema.optional() })
  .passthrough()
const recipeSchema = z.discriminatedUnion('kind', [
  z
    .object({
      kind: z.literal('investigation'),
      investigation: z.enum(['values', 'sharing', 'memory']),
      value: z.number().min(-3).max(3),
      query: z.number().min(-3).max(3),
      length: z.number().int().min(1).max(8192),
    })
    .passthrough(),
  z.object({ kind: z.literal('training'), settings: trainingRecipeSchema }).passthrough(),
  z
    .object({
      kind: z.literal('numerical'),
      settings: numericalSettingsSchema,
      baseline: numericalSettingsSchema.optional(),
      node: z.string().max(160),
      call: z.number().int().min(0).max(8),
      tensor: z.string().max(160).optional(),
      index: z.number().int().min(0).max(1000000).optional(),
    })
    .passthrough(),
  z
    .object({
      kind: z.literal('trained'),
      settings: trainedSettingsSchema,
      checkpoint: z.string().max(160),
      revision: z.string().max(160),
      node: z.string().max(160).optional(),
      call: z.number().int().min(0).max(15).optional(),
      tensor: z.string().max(160).optional(),
      index: z.number().int().min(0).max(1000000).optional(),
      intervention: headInterventionSchema.optional(),
      comparison: trainedComparisonSchema.optional(),
    })
    .passthrough(),
])
export type StudyRecipe = z.infer<typeof recipeSchema>
export const studyRecordSchema = z
  .object({
    id: z.string().min(1).max(100),
    created: z.string().max(50),
    title: z.string().min(1).max(200),
    question: z.string().max(4000),
    prediction: z.string().max(4000),
    observation: z.string().max(8000),
    conclusion: z.string().max(8000),
    recipe: recipeSchema,
    evidence: z
      .object({
        scope: z.string().max(200),
        tensor: z.string().max(160).optional(),
        shape: z.array(z.number().int().min(0).max(1000000)).max(8).optional(),
        axes: z.array(z.string().max(80)).max(8).optional(),
        start: z.number().int().min(0).optional(),
        values: z
          .array(z.union([z.number().finite(), z.string().max(40)]))
          .max(64)
          .optional(),
        baseline: z
          .array(z.union([z.number().finite(), z.string().max(40)]))
          .max(64)
          .optional(),
        summary: z.string().max(8000),
      })
      .passthrough(),
  })
  .passthrough()
export type StudyRecord = z.infer<typeof studyRecordSchema>
export const studyLibrarySchema = z
  .object({ version: z.literal(1), records: z.array(studyRecordSchema).max(100) })
  .passthrough()
export const STUDY_KEY = 'model-visualizer.investigations.v1'
export function parseStudyLibrary(text: string) {
  if (text.length > 2_000_000) throw new Error('Investigation files must be under 2 MB.')
  const result = studyLibrarySchema.safeParse(JSON.parse(text))
  if (!result.success)
    throw new Error('Unsupported investigation file. Existing records are unchanged.')
  if (new Set(result.data.records.map((r) => r.id)).size !== result.data.records.length)
    throw new Error('Duplicate record IDs. Existing records are unchanged.')
  return result.data
}
export function newStudyRecord(input: Omit<StudyRecord, 'id' | 'created'>): StudyRecord {
  return studyRecordSchema.parse({
    ...input,
    id: crypto.randomUUID(),
    created: new Date().toISOString(),
  })
}
export function downloadStudyLibrary(records: StudyRecord[]) {
  const blob = new Blob([JSON.stringify({ version: 1, records }, null, 2)], {
      type: 'application/json',
    }),
    url = URL.createObjectURL(blob),
    a = document.createElement('a')
  a.href = url
  a.download = 'model-investigations.json'
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
