import { foundationPerspectives } from './perspectives-foundations'
import { attentionPerspectives } from './perspectives-attention'
import { practicePerspectives } from './perspectives-practice'
import type { TopicPerspectives } from './perspective-types'
import { memoryPerspectives } from './perspectives-memory'
import { learningPerspectives } from './perspectives-learning'
import { optimizerPerspectives } from './optimizers'

export const perspectives: Record<string, TopicPerspectives> = {
  ...foundationPerspectives,
  ...attentionPerspectives,
  ...practicePerspectives,
  ...memoryPerspectives,
  ...learningPerspectives,
  ...optimizerPerspectives,
}
