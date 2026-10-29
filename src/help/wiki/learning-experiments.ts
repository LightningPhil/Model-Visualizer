/** Explicit synthetic teaching calculations, independent of graphs and checkpoints. */
export const sigmoid = (z: number) =>
  z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z))
const softplus = (z: number) => Math.max(z, 0) + Math.log1p(Math.exp(-Math.abs(z)))
function valid(values: number[]) {
  if (values.some((v) => !Number.isFinite(v))) throw new Error('Use finite teaching operands.')
}
export function regressionStep(
  x: number,
  target: number,
  weight: number,
  bias: number,
  rate: number,
) {
  valid([x, target, weight, bias, rate])
  if (rate < 0) throw new Error('Learning rate must be nonnegative.')
  const prediction = weight * x + bias,
    error = prediction - target
  const gradientWeight = error * x,
    gradientBias = error
  const nextWeight = weight - rate * gradientWeight,
    nextBias = bias - rate * gradientBias
  const after = nextWeight * x + nextBias - target
  const h = 1e-4
  const lossAt = (w: number) => 0.5 * (w * x + bias - target) ** 2
  return {
    prediction,
    error,
    loss: 0.5 * error ** 2,
    gradientWeight,
    gradientBias,
    nextWeight,
    nextBias,
    nextLoss: 0.5 * after ** 2,
    finiteDifference: (lossAt(weight + h) - lossAt(weight - h)) / (2 * h),
  }
}
const dot = (a: number[], b: number[]) => a.reduce((sum, v, i) => sum + v * b[i], 0)
export const rotatePair = (v: number[], angle: number) => [
  v[0] * Math.cos(angle) - v[1] * Math.sin(angle),
  v[0] * Math.sin(angle) + v[1] * Math.cos(angle),
]
export function positionComparison(queryPosition: number, keyPosition: number, slope: number) {
  valid([queryPosition, keyPosition, slope])
  if (
    ![queryPosition, keyPosition].every((p) => Number.isInteger(p) && p >= 0 && p < 4) ||
    slope < 0
  )
    throw new Error('Choose positions in the fictional table.')
  const query = [1, 0.5],
    key = [0.5, 1]
  const table = [
    [0.2, -0.1],
    [0.4, 0.3],
    [-0.3, 0.6],
    [0.1, -0.5],
  ]
  const add = (a: number[], b: number[]) => a.map((v, i) => v + b[i])
  const sine = (p: number) => [Math.sin(p), Math.cos(p)]
  const score = (q: number[], k: number[]) => dot(q, k) / Math.sqrt(2)
  return {
    visible: keyPosition <= queryPosition,
    content: score(query, key),
    learned: score(add(query, table[queryPosition]), add(key, table[keyPosition])),
    sinusoidal: score(add(query, sine(queryPosition)), add(key, sine(keyPosition))),
    rotary: score(rotatePair(query, queryPosition), rotatePair(key, keyPosition)),
    alibi: score(query, key) - slope * (queryPosition - keyPosition),
    query,
    key,
    queryEncoding: table[queryPosition],
    keyEncoding: table[keyPosition],
  }
}
export function featureMixture(strengths: number[], angle: number, view: number) {
  valid([...strengths, angle, view])
  if (strengths.length !== 3) throw new Error('Three fictional features are required.')
  const directions = [
    [1, 0],
    [Math.cos(angle), Math.sin(angle)],
    [Math.cos((2 * Math.PI) / 3), Math.sin((2 * Math.PI) / 3)],
  ]
  const hidden = [0, 1].map((d) => strengths.reduce((sum, a, i) => sum + a * directions[i][d], 0))
  const displayed = rotatePair(hidden, view),
    rotatedDirections = directions.map((d) => rotatePair(d, view))
  return {
    directions: rotatedDirections,
    hidden: displayed,
    readout: dot(displayed, rotatedDirections[0]),
    targetStrength: strengths[0],
    interference: dot(hidden, directions[0]) - strengths[0],
  }
}
export type Intervention = 'clean' | 'corrupted' | 'ablate' | 'patch' | 'replace'
export function interventionExample(mode: Intervention, replacement: number) {
  valid([replacement])
  const hidden =
    mode === 'corrupted'
      ? [0, 1]
      : mode === 'ablate'
        ? [1, 0]
        : mode === 'replace'
          ? [replacement, 1]
          : [1, 1]
  const logit = 3 * hidden[0] - 2 * hidden[1]
  return {
    hidden,
    contributions: [3 * hidden[0], -2 * hidden[1]],
    logit,
    probability: sigmoid(logit),
    deltaClean: logit - 1,
    deltaCorrupted: logit + 2,
  }
}
export function preferenceExample(gap: number, referenceGap: number, beta: number) {
  valid([gap, referenceGap, beta])
  if (beta <= 0) throw new Error('Beta must be positive.')
  const margin = beta * (gap - referenceGap)
  return {
    preferredProbability: sigmoid(gap),
    demonstrationLoss: softplus(-gap),
    dpoLoss: softplus(-margin),
    margin,
    gradientGap: -beta * sigmoid(-margin),
  }
}
export function precisionExample(magnitude: number, logit: number) {
  valid([magnitude, logit])
  const a = Math.fround(magnitude)
  const sequential32 = Math.fround(Math.fround(a + Math.fround(1)) + Math.fround(-a))
  const cancelFirst32 = Math.fround(Math.fround(a + Math.fround(-a)) + Math.fround(1))
  const exponentials = [Math.exp(logit), Math.exp(logit - 1)]
  const sum = exponentials[0] + exponentials[1]
  return {
    stored: a,
    sequential32,
    cancelFirst32,
    binary64: magnitude + 1 - magnitude,
    naive: exponentials.map((v) => v / sum),
    stable: [1 / (1 + Math.exp(-1)), Math.exp(-1) / (1 + Math.exp(-1))],
    exponentials,
  }
}
export function compareNormalizations(input: number[], offset: number, gain = 1, epsilon = 1e-5) {
  valid([...input, offset, gain, epsilon])
  if (!input.length || epsilon <= 0) throw new Error('Use a nonempty row and positive epsilon.')
  const row = input.map((x) => x + offset),
    mean = row.reduce((a, b) => a + b, 0) / row.length
  const variance = row.reduce((sum, x) => sum + (x - mean) ** 2, 0) / row.length
  const rms = Math.sqrt(row.reduce((sum, x) => sum + x * x, 0) / row.length + epsilon)
  return {
    row,
    mean,
    variance,
    rms,
    layer: row.map((x) => (gain * (x - mean)) / Math.sqrt(variance + epsilon)),
    rootMeanSquare: row.map((x) => (gain * x) / rms),
  }
}
export function routedExample(input: number, scores: number[], count: number) {
  valid([input, ...scores])
  if (scores.length !== 3 || !Number.isInteger(count) || count < 1 || count > 3)
    throw new Error('Choose one to three toy experts.')
  const selected = [0, 1, 2].sort((a, b) => scores[b] - scores[a] || a - b).slice(0, count)
  const max = Math.max(...selected.map((i) => scores[i])),
    denominator = selected.reduce((sum, i) => sum + Math.exp(scores[i] - max), 0)
  const weights = scores.map((s, i) => (selected.includes(i) ? Math.exp(s - max) / denominator : 0))
  const outputs = scores.map((_, i) =>
    selected.includes(i) ? (i === 0 ? input : i === 1 ? -input + 1 : 2 * input - 1) : undefined,
  )
  const contributions = outputs.map((v, i) => (v === undefined ? 0 : v * weights[i]))
  return {
    selected,
    weights,
    outputs,
    contributions,
    output: contributions.reduce((sum, v) => sum + v, 0),
    denseOutput: 2 * input - 1,
    storedExpertParameters: 6,
    activeExpertParameters: 2 * count,
  }
}
