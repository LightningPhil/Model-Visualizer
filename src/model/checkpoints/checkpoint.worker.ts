import {
  parseStoriesCheckpoint,
  StoriesTokenizer,
  verifyAsset,
  storiesManifest,
  runStories,
  type TrainedRunSettings,
} from './stories'
import type { CheckpointWeights } from '../numerics/engine'

let weights: CheckpointWeights | undefined, tokenizer: StoriesTokenizer | undefined
self.onmessage = async (
  event: MessageEvent<{
    id: number
    weights?: ArrayBuffer
    tokenizer?: ArrayBuffer
    settings?: TrainedRunSettings
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
      const trace = runStories(weights, tokenizer, event.data.settings, (ids, text) =>
        self.postMessage({ id, progress: { count: ids.length, text } }),
      )
      const buffers = [...new Set(Object.values(trace.tensors).map((t) => t.data.buffer))]
      self.postMessage({ id, trace }, { transfer: buffers })
    } else throw new Error('Load and verify the checkpoint before running it.')
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : 'Local model failed.' })
  }
}
