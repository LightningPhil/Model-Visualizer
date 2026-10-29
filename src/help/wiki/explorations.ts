import { cacheBytes, type ModelConfig } from '../../model/schema'
import { residentCount } from '../../model/attention'

/** Small, explicit teaching calculations. These never consume a trace, mutate a
 * model, or claim to reproduce learned activations or an execution kernel. */
function finite(values: number[]) {
  if (!values.length || values.some((value) => !Number.isFinite(value)))
    throw new Error('The teaching calculation needs finite values.')
}

export function probabilityExample(
  scores: number[],
  temperature = 1,
  visible = scores.map(() => true),
) {
  finite(scores)
  if (
    !Number.isFinite(temperature) ||
    temperature <= 0 ||
    visible.length !== scores.length ||
    !visible.some(Boolean)
  )
    throw new Error('Softmax needs a positive temperature and at least one visible score.')
  const scaled = scores.map((score, i) => (visible[i] ? score / temperature : -Infinity))
  const maximum = Math.max(...scaled)
  const exponentials = scaled.map((score) => Math.exp(score - maximum))
  const denominator = exponentials.reduce((sum, value) => sum + value, 0)
  return {
    scaled,
    maximum,
    exponentials,
    denominator,
    probabilities: exponentials.map((value) => value / denominator),
  }
}

export type Pair = [number, number]
export const attentionKeys: Pair[] = [
  [1, 0],
  [0, 1],
  [1, 1],
]
export const attentionValues: Pair[] = [
  [2, 0],
  [0, 4],
  [2, 2],
]
export function attentionExample(
  query: Pair,
  position: number,
  causal: boolean,
  keys: Pair[] = attentionKeys,
  values: Pair[] = attentionValues,
) {
  finite(query)
  if (!Number.isInteger(position) || position < 0 || position > 2)
    throw new Error('Choose query position 0–2.')
  if (
    keys.length !== 3 ||
    values.length !== 3 ||
    [...keys, ...values].some((row) => row.length !== 2)
  )
    throw new Error('The teaching row needs three two-feature keys and values.')
  finite([...keys.flat(), ...values.flat()])
  const dots = keys.map((key) => key.reduce((sum, value, d) => sum + query[d] * value, 0))
  const visible = keys.map((_, i) => !causal || i <= position)
  const distribution = probabilityExample(
    dots.map((dot) => dot / Math.sqrt(2)),
    1,
    visible,
  )
  const contributions = values.map((value, i) =>
    value.map((v) => v * distribution.probabilities[i]),
  )
  const output = [0, 1].map((d) => contributions.reduce((sum, row) => sum + row[d], 0))
  return { keys, values, dots, visible, ...distribution, contributions, output }
}

export function projectionExample(input: Pair, weights: [Pair, Pair]) {
  finite([...input, ...weights.flat()])
  const products = [0, 1].map((column) => input.map((value, row) => value * weights[row][column]))
  return { products, output: products.map((terms) => terms.reduce((sum, term) => sum + term, 0)) }
}
export function residualExample(
  input: number,
  attention: number,
  feedForward: number,
  skip: boolean,
) {
  finite([input, attention, feedForward])
  const afterAttention = (skip ? input : 0) + attention
  return { afterAttention, output: (skip ? afterAttention : 0) + feedForward }
}

export function rotationExample(
  input: [number, number],
  position: number,
  pair: number,
  width = 4,
) {
  finite([...input, position])
  if (
    !Number.isInteger(width) ||
    width < 2 ||
    width % 2 ||
    !Number.isInteger(pair) ||
    pair < 0 ||
    pair >= width / 2
  )
    throw new Error('RoPE needs a valid feature pair in an even head width.')
  const frequency = 10000 ** ((-2 * pair) / width)
  const angle = position * frequency
  const cosine = Math.cos(angle),
    sine = Math.sin(angle)
  return {
    frequency,
    angle,
    cosine,
    sine,
    output: [input[0] * cosine - input[1] * sine, input[0] * sine + input[1] * cosine],
    norm: Math.hypot(...input),
  }
}

export function normalizationExample(input: number[], gain = 1, epsilon = 1e-5) {
  finite([...input, gain, epsilon])
  if (!input.length || epsilon <= 0) throw new Error('RMSNorm needs a row and positive epsilon.')
  const squares = input.map((value) => value * value)
  const meanSquare = squares.reduce((sum, value) => sum + value, 0) / input.length
  const denominator = Math.sqrt(meanSquare + epsilon)
  return {
    squares,
    meanSquare,
    denominator,
    output: input.map((value) => (gain * value) / denominator),
  }
}

export function gateExample(gate: number, up: number) {
  finite([gate, up])
  const sigmoid = 1 / (1 + Math.exp(-gate)),
    silu = gate * sigmoid
  return { sigmoid, silu, product: silu * up }
}

export function cacheExample(config: ModelConfig, extent: number, queries = 1) {
  if (
    !Number.isSafeInteger(extent) ||
    extent < 1 ||
    !Number.isSafeInteger(queries) ||
    queries < 1 ||
    queries > extent
  )
    throw new Error('Choose a valid processed extent and query count.')
  const prefix = residentCount(config, extent - queries)
  return {
    extent,
    queries,
    prefix,
    working: prefix + queries,
    retained: residentCount(config, extent),
    bytes: cacheBytes(config, extent),
    mhaBytes: cacheBytes({ ...config, nKvHeads: config.nQueryHeads }, extent),
    mqaBytes: cacheBytes({ ...config, nKvHeads: 1 }, extent),
  }
}

export const explorationFor: Record<
  string,
  | 'attention'
  | 'rotation'
  | 'normalization'
  | 'gate'
  | 'probabilities'
  | 'cache'
  | 'heads'
  | 'projection'
  | 'residual'
  | 'embedding'
  | 'memory-comparison'
  | 'learning'
  | 'positions'
  | 'features'
  | 'interventions'
  | 'post-training'
  | 'precision'
  | 'experts'
  | 'adam-lab'
  | 'muon-lab'
> = {
  adam: 'adam-lab',
  adamw: 'adam-lab',
  muon: 'muon-lab',
  'learning-loop': 'learning',
  'position-methods': 'positions',
  'feature-representations': 'features',
  'causal-interventions': 'interventions',
  'post-training': 'post-training',
  'numerical-precision': 'precision',
  'mixture-of-experts': 'experts',
  'attention-comparison': 'memory-comparison',
  'memory-budget': 'memory-comparison',
  'attention-overview': 'attention',
  'attention-scores': 'attention',
  'causal-mask': 'attention',
  softmax: 'attention',
  'attention-interpretation': 'attention',
  rope: 'rotation',
  rmsnorm: 'normalization',
  swiglu: 'gate',
  logits: 'probabilities',
  sampling: 'probabilities',
  'kv-cache': 'cache',
  'cache-retention': 'cache',
  'cache-bytes': 'cache',
  'head-sharing': 'heads',
  'linear-maps': 'projection',
  residual: 'residual',
  embeddings: 'embedding',
}

/** Explicit destinations in the existing foundations numerical profile. */
export const experimentEntries: Record<string, string> = {
  embeddings: 'tokens.lookup',
  'linear-maps': 'layer-0.attn.q',
  residual: 'layer-0.add1',
  rmsnorm: 'layer-0.norm1',
  'attention-overview': 'layer-0.attn.scores-0',
  'attention-scores': 'layer-0.attn.scores-0',
  'causal-mask': 'layer-0.attn.scale-0',
  softmax: 'layer-0.attn.softmax-0',
  rope: 'layer-0.attn.ropeq',
  'head-sharing': 'layer-0.attn.head-0',
  swiglu: 'layer-0.swiglu.multiply',
  'kv-cache': 'layer-0.attn.cachek',
  'cache-retention': 'layer-0.attn.cachek',
  'cache-bytes': 'layer-0.attn.cachek',
  logits: 'output.lmhead',
  sampling: 'output.sample',
  'prefill-decode': 'generation.commit',
}
