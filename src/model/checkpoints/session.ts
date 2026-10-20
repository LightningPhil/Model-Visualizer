import { z } from 'zod'
import { sessionSchema } from '../numerics/session'
import { defaultNumericalSettings } from '../numerics/engine'
import { CHECKPOINT_SESSION_KEY, defaultTrainedSettings, storiesManifest } from './stories'
import type { ModelDocument } from '../schema'

export const trainedSettingsSchema = z.strictObject({
  prompt: z
    .string()
    .max(600)
    .refine((s) => !s.includes('\0')),
  generate: z.number().int().min(1).max(16),
  seed: z.number().int().min(0).max(2147483647),
  sampling: z.enum(['greedy', 'sample']),
  temperature: z.number().min(0.1).max(3),
})
export const checkpointSessionSchema = z.strictObject({
  version: z.literal(1),
  checkpointId: z.literal('karpathy-stories260k'),
  revision: z.literal(storiesManifest.revision),
  settings: trainedSettingsSchema,
  study: z.strictObject({
    layout: sessionSchema.shape.layout,
    bookmarks: z
      .array(
        sessionSchema.shape.bookmarks.element.extend({ call: z.number().int().min(0).max(15) }),
      )
      .max(24),
    notes: sessionSchema.shape.notes,
  }),
})
export type CheckpointSession = z.infer<typeof checkpointSessionSchema>
export function loadCheckpointSession(doc: ModelDocument) {
  const raw = doc.extensions?.[CHECKPOINT_SESSION_KEY],
    parsed = checkpointSessionSchema.safeParse(raw)
  return {
    writable: raw === undefined || parsed.success,
    session: parsed.success
      ? parsed.data
      : {
          version: 1 as const,
          checkpointId: 'karpathy-stories260k' as const,
          revision: storiesManifest.revision,
          settings: structuredClone(defaultTrainedSettings),
          study: {
            layout: { direction: 'RIGHT' as const, positions: {} },
            bookmarks: [],
            notes: '',
          },
        },
  }
}
export function checkpointStudySession(session: CheckpointSession) {
  return {
    version: 1 as const,
    settings: structuredClone(defaultNumericalSettings),
    layouts: {},
    ...session.study,
  }
}
