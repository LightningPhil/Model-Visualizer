/** A synthetic 1 → 4 tanh → 1 regressor. Full-batch analytic backprop;
 * JavaScript Number arithmetic, deliberately independent of checkpoint inference. */
export type TrainingRecipe = {
  seed: number
  task: 'curve' | 'extrapolate' | 'noise'
  optimizer: 'sgd' | 'adam' | 'adamw'
  rate: number
  decay: number
  steps: number
}
export const defaultTraining: TrainingRecipe = {
  seed: 7,
  task: 'curve',
  optimizer: 'adam',
  rate: 0.03,
  decay: 0.01,
  steps: 0,
}
export type Sample = { x: number; y: number }
export const targetFunction = (x: number) => Math.sin(Math.PI * x)
export function trainingData(task: TrainingRecipe['task']) {
  const xs = task === 'extrapolate' ? [-0.5, -0.25, 0, 0.25, 0.5] : [-1, -0.5, 0, 0.5, 1]
  return xs.map((x, i) => ({
    x,
    y: targetFunction(x) + (task === 'noise' ? [0.4, -0.4, 0.3, -0.4, 0.4][i] : 0),
  }))
}
export const holdoutData: Sample[] = [-0.9, -0.7, -0.3, -0.1, 0.1, 0.3, 0.7, 0.9].map((x) => ({
  x,
  y: targetFunction(x),
}))
export const parameterNames = [
  ...Array.from({ length: 4 }, (_, i) => `W1[${i}]`),
  ...Array.from({ length: 4 }, (_, i) => `b1[${i}]`),
  ...Array.from({ length: 4 }, (_, i) => `W2[${i}]`),
  'b2',
]
export function initialParameters(seed: number) {
  let state = seed >>> 0
  return parameterNames.map(() => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0
    return (state / 4294967296 - 0.5) * 1.5
  })
}
export function networkForward(parameters: number[], x: number) {
  const z = Array.from({ length: 4 }, (_, j) => x * parameters[j] + parameters[4 + j])
  const h = z.map(Math.tanh)
  return { z, h, output: h.reduce((sum, v, j) => sum + v * parameters[8 + j], parameters[12]) }
}
/** L = mean((prediction-target)^2) / 2. All gradients use the same pre-update parameters. */
export function networkLoss(parameters: number[], data: Sample[]) {
  const gradient = Array(13).fill(0) as number[]
  let loss = 0
  for (const { x, y } of data) {
    const { h, output } = networkForward(parameters, x),
      error = output - y
    loss += (error * error) / (2 * data.length)
    for (let j = 0; j < 4; j++) {
      const dz = (error * parameters[8 + j] * (1 - h[j] * h[j])) / data.length
      gradient[j] += dz * x
      gradient[4 + j] += dz
      gradient[8 + j] += (error * h[j]) / data.length
    }
    gradient[12] += error / data.length
  }
  return { loss, gradient }
}
export function trainNetwork(recipe: TrainingRecipe) {
  if (
    !Number.isInteger(recipe.seed) ||
    recipe.seed < 0 ||
    recipe.seed > 2147483647 ||
    !Number.isInteger(recipe.steps) ||
    recipe.steps < 0 ||
    recipe.steps > 1000 ||
    !Number.isFinite(recipe.rate) ||
    recipe.rate <= 0 ||
    recipe.rate > 2 ||
    !Number.isFinite(recipe.decay) ||
    recipe.decay < 0 ||
    recipe.decay > 1 ||
    !['curve', 'extrapolate', 'noise'].includes(recipe.task) ||
    !['sgd', 'adam', 'adamw'].includes(recipe.optimizer)
  )
    throw new Error('Invalid training recipe.')
  let parameters = initialParameters(recipe.seed)
  const m = Array(13).fill(0) as number[],
    v = Array(13).fill(0) as number[]
  const history = [
    {
      step: 0,
      train: networkLoss(parameters, trainingData(recipe.task)).loss,
      holdout: networkLoss(parameters, holdoutData).loss,
    },
  ]
  let lastGradient = Array(13).fill(0) as number[],
    lastDelta = [...lastGradient],
    completed = 0,
    stopped = false
  for (let t = 1; t <= recipe.steps; t++) {
    const oldM = [...m],
      oldV = [...v]
    const { gradient } = networkLoss(parameters, trainingData(recipe.task))
    const delta = gradient.map((g, i) => {
      m[i] = 0.9 * m[i] + 0.1 * g
      v[i] = 0.999 * v[i] + 0.001 * g * g
      const direction =
        recipe.optimizer === 'sgd'
          ? g
          : m[i] / (1 - 0.9 ** t) / (Math.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
      return (
        -recipe.rate *
        (direction + (recipe.optimizer === 'adamw' ? recipe.decay * parameters[i] : 0))
      )
    })
    const next = parameters.map((p, i) => p + delta[i])
    const train = networkLoss(next, trainingData(recipe.task)).loss,
      holdout = networkLoss(next, holdoutData).loss
    if (![...next, train, holdout].every(Number.isFinite)) {
      m.splice(0, m.length, ...oldM)
      v.splice(0, v.length, ...oldV)
      stopped = true
      break
    }
    parameters = next
    completed = t
    lastGradient = gradient
    lastDelta = delta
    history.push({ step: t, train, holdout })
  }
  return {
    parameters,
    m,
    v,
    history,
    lastGradient,
    lastDelta,
    completed,
    stopped,
    current: networkLoss(parameters, trainingData(recipe.task)),
  }
}
