/** Synthetic optimizer arithmetic. No graph, model state or trained parameters. */
export type AdamMode = 'adam' | 'l2' | 'adamw'
export type AdamSettings = {
  initial: number
  rate: number
  beta1: number
  beta2: number
  epsilon: number
  decay: number
}
export function adamSequence(gradients: number[], settings: AdamSettings, mode: AdamMode) {
  if (
    [...gradients, ...Object.values(settings)].some((v) => !Number.isFinite(v)) ||
    gradients.length > 100 ||
    settings.rate < 0 ||
    settings.decay < 0 ||
    settings.epsilon <= 0 ||
    settings.beta1 < 0 ||
    settings.beta1 >= 1 ||
    settings.beta2 < 0 ||
    settings.beta2 >= 1
  )
    throw new Error('Choose finite gradients, valid moment betas and a positive stabilizer.')
  let parameter = settings.initial,
    first = 0,
    second = 0
  return gradients.map((lossGradient, index) => {
    const before = parameter,
      t = index + 1
    const effectiveGradient = lossGradient + (mode === 'l2' ? settings.decay * before : 0)
    first = settings.beta1 * first + (1 - settings.beta1) * effectiveGradient
    second = settings.beta2 * second + (1 - settings.beta2) * effectiveGradient ** 2
    const correctedFirst = first / (1 - settings.beta1 ** t),
      correctedSecond = second / (1 - settings.beta2 ** t)
    const denominator = Math.sqrt(correctedSecond) + settings.epsilon
    const adaptive = (settings.rate * correctedFirst) / denominator
    const shrinkage = mode === 'adamw' ? settings.rate * settings.decay * before : 0
    parameter = before - adaptive - shrinkage
    return {
      t,
      lossGradient,
      effectiveGradient,
      before,
      first,
      second,
      correctedFirst,
      correctedSecond,
      denominator,
      adaptive,
      shrinkage,
      parameter,
    }
  })
}
export type Matrix2 = [[number, number], [number, number]]
/** U diag(s0,s1) V^T for fixed rotations U=R(30°), V=R(-20°).
 * Signed factors allow flips; actual singular magnitudes are abs(s). */
export function matrixFromFactors(factors: number[]): Matrix2 {
  const a = Math.PI / 6,
    b = -Math.PI / 9,
    ca = Math.cos(a),
    sa = Math.sin(a),
    cb = Math.cos(b),
    sb = Math.sin(b)
  const [x, y] = factors
  return [
    [ca * x * cb + sa * y * sb, ca * x * sb - sa * y * cb],
    [sa * x * cb - ca * y * sb, sa * x * sb + ca * y * cb],
  ]
}
export function muonMatrixExample(
  gradient: [number, number],
  previous: [number, number],
  beta: number,
  steps: number,
  nesterov: boolean,
) {
  if (
    [...gradient, ...previous, beta].some((v) => !Number.isFinite(v)) ||
    beta < 0 ||
    beta >= 1 ||
    !Number.isInteger(steps) ||
    steps < 0 ||
    steps > 10
  )
    throw new Error(
      'Choose finite matrix factors, a valid momentum beta and zero to ten iterations.',
    )
  const momentum = gradient.map((g, i) => beta * previous[i] + (1 - beta) * g)
  const candidate = gradient.map((g, i) =>
    nesterov ? (1 - beta) * g + beta * momentum[i] : momentum[i],
  )
  const norm = Math.hypot(...candidate),
    initial = candidate.map((s) => s / (norm + 1e-7))
  let factors = initial
  const iterations = [{ iteration: 0, factors, magnitudes: factors.map(Math.abs) }]
  for (let k = 1; k <= steps; k++) {
    factors = factors.map((s) => 3.4445 * s - 4.775 * s ** 3 + 2.0315 * s ** 5)
    iterations.push({ iteration: k, factors, magnitudes: factors.map(Math.abs) })
  }
  const ideal = candidate.map(Math.sign)
  return {
    gradient: matrixFromFactors(gradient),
    momentum: matrixFromFactors(momentum),
    candidate: matrixFromFactors(candidate),
    norm,
    initial,
    iterations,
    factors,
    output: matrixFromFactors(factors),
    ideal: matrixFromFactors(ideal),
    targetMagnitudes: ideal.map(Math.abs),
  }
}
