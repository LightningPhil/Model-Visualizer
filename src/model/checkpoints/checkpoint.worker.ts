import {
  parseStoriesCheckpoint,
  StoriesTokenizer,
  verifyAsset,
  storiesManifest,
  runStoriesExperiment,
  type TrainedRunSettings,
} from './stories'
import type { CheckpointWeights } from '../numerics/engine'
import {
  headInterventionSchema,
  trainedComparisonSchema,
  type HeadIntervention,
} from '../study/records'
import { trainedSettingsSchema } from './session'

let weights: CheckpointWeights | undefined, tokenizer: StoriesTokenizer | undefined
self.onmessage = async (
  event: MessageEvent<{
    id: number
    weights?: ArrayBuffer
    tokenizer?: ArrayBuffer
    settings?: TrainedRunSettings
    intervention?: HeadIntervention
    comparison?: unknown
  }>,
) => {
  const { id } = event.data
  try {
    if (event.data.weights && event.data.tokenizer) {
      await verifyAsset(event.data.weights, storiesManifest.weights)
      await verifyAsset(event.data.tokenizer, storiesManifest.tokenizer)
      weights = parseStoriesCheckpoint(event.data.weights)
      tokenizer = new StoriesTokenizer(event.data.tokenizer)
      self.postMessage({ id, ready: true })
    } else if (event.data.settings && weights && tokenizer) {
      const experiment = runStoriesExperiment(
        weights,
        tokenizer,
        trainedSettingsSchema.parse(event.data.settings),
        event.data.intervention ? headInterventionSchema.parse(event.data.intervention) : undefined,
        (ids, text) => self.postMessage({ id, progress: { count: ids.length, text } }),
      )
      const trace = experiment.trace
      let baseline = experiment.baseline
      if (event.data.comparison && !event.data.intervention) {
        const comparison = trainedComparisonSchema.parse(event.data.comparison)
        baseline = runStoriesExperiment(
          weights,
          tokenizer,
          comparison.settings,
          comparison.intervention,
        ).trace
      }
      const buffers = [
        ...new Set(
          [trace, ...(baseline ? [baseline] : [])].flatMap((run) =>
            Object.values(run.tensors).map((t) => t.data.buffer),
          ),
        ),
      ]
      self.postMessage({ id, trace, baseline }, { transfer: buffers })
    } else throw new Error('Load and verify the checkpoint before running it.')
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : 'Local model failed.' })
  }
}
