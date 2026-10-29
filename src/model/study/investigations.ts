import {
  attentionExample,
  attentionKeys,
  attentionValues,
  type Pair,
} from '../../help/wiki/explorations'
export const investigationQuestions = [
  {
    id: 'values',
    title: 'Can the output change while attention stays the same?',
    guide: 'attention-interpretation',
    instruction: 'Hold Q and K fixed. Change only feature 0 of the second value vector.',
    hint: 'The probability row is calculated before V enters the operation.',
    explanation:
      'The scores and probabilities stay fixed because Q and K did not change. Only the value-weighted sum changes: Δoutput₀ = attention₁ × Δvalue₁,₀. A heatmap alone cannot describe the output contribution.',
  },
  {
    id: 'sharing',
    title: 'Do shared keys and values make query heads identical?',
    guide: 'head-sharing',
    instruction:
      'Two query heads read the same three K/V rows. Hold head A at [1,2] and vary feature 0 of head B.',
    hint: 'Each query supplies a different operand to the dot products.',
    explanation:
      'K/V sharing does not tie the query vectors. The heads share stored content but independently score it. Equal queries give equal rows in this controlled example; different queries can give different probabilities and outputs.',
  },
  {
    id: 'memory',
    title: 'Where does attention memory go as context grows?',
    guide: 'attention-comparison',
    instruction:
      'Vary the retained length with layer count, head width, batch and precision held fixed. Compare eight KV heads with two.',
    hint: 'Count every layer × K and V × retained position × KV head × feature × scalar byte.',
    explanation:
      'For this synthetic configuration, MHA payload = 2 × 4 layers × 1 batch × 8 KV heads × S × 16 features × 2 bytes. GQA replaces 8 with 2, giving one quarter of this payload. This excludes weights, temporary workspace, metadata and padding; it does not establish equivalent quality or speed.',
  },
] as const
export function investigationResult(
  kind: 'values' | 'sharing' | 'memory',
  value: number,
  query: number,
  length: number,
) {
  const base = attentionExample([1, 2], 2, true)
  if (kind === 'memory')
    return {
      baseline: [2 * 4 * 8 * length * 16 * 2],
      current: [2 * 4 * 2 * length * 16 * 2],
      probabilities: [],
      label: 'Logical K/V payload bytes (MHA → GQA)',
    }
  const values = attentionValues.map((row) => [...row] as Pair)
  values[1][0] = value
  const other =
    kind === 'values'
      ? attentionExample([1, 2], 2, true, attentionKeys, values)
      : attentionExample([query, 2], 2, true)
  return {
    baseline: base.output,
    current: other.output,
    probabilities: other.probabilities,
    baseProbabilities: base.probabilities,
    label: 'Output feature vector (baseline → edited)',
  }
}
