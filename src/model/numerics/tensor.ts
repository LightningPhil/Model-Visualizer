/** Small, explicit FP32 tensor operations for the numerical teaching model.
 * Storage is Float32Array. JS Number accumulates sums; outputs round to FP32.
 * This is a reference implementation, not a reproduction of a GPU kernel. */
export type Recipe =
  | { op: 'parameter'; seed: number }
  | {
      op: 'checkpoint'
      id: string
      name: string
      revision: string
      sha256: string
      tokenizerSha256: string
      source: string
      tiedTo?: string
    }
  | { op: 'input' }
  | { op: 'embedding'; table: string; tokens: number[] }
  | { op: 'matmul'; a: string; b: string }
  | { op: 'rmsnorm'; input: string; gain: string; epsilon: number }
  | { op: 'split'; input: string; heads: number }
  | { op: 'head'; input: string; head: number }
  | { op: 'transpose'; input: string }
  | { op: 'rope'; input: string; positions: number[]; base: number }
  | { op: 'concat'; inputs: string[]; axis: number }
  | { op: 'gather'; input: string; indices: number[]; axis: number }
  | {
      op: 'mask'
      input: string
      queries: number[]
      keys: number[]
      headDim: number
      allowed: boolean[]
    }
  | { op: 'softmax'; input: string; temperature: number }
  | { op: 'add' | 'multiply'; a: string; b: string }
  | { op: 'silu'; input: string }
  | {
      op: 'choice'
      input: string
      token: number
      row: number
      method: 'greedy' | 'sample'
      draw?: number
    }

export type NumericTensor = {
  id: string
  name: string
  shape: number[]
  axes: string[]
  data: Float32Array | Int32Array
  dtype: 'float32' | 'int32'
  role: 'activation' | 'parameter' | 'state' | 'input'
  recipe: Recipe
  positions?: Record<string, number[]>
  createdAt: number
}
export const elements = (shape: number[]) => shape.reduce((a, b) => a * b, 1)
/** Subtract only semantically aligned values. Infinities encode exclusion, so
 * changed visibility must be compared through finite probabilities instead. */
export function tensorComparisonIssue(current: NumericTensor, baseline: NumericTensor) {
  if (current.shape.join() !== baseline.shape.join()) return 'shape'
  if (current.axes.join() !== baseline.axes.join() || current.dtype !== baseline.dtype)
    return 'axes'
  if (
    current.axes.some(
      (axis) =>
        JSON.stringify(current.positions?.[axis]) !== JSON.stringify(baseline.positions?.[axis]),
    )
  )
    return 'positions'
  if (current.recipe.op === 'mask' || baseline.recipe.op === 'mask') {
    if (
      current.recipe.op !== 'mask' ||
      baseline.recipe.op !== 'mask' ||
      current.recipe.allowed.join() !== baseline.recipe.allowed.join()
    )
      return 'visibility'
  }
  return undefined
}
/** Position IDs follow semantic axes, never a physical slot offset. */
export function inheritPositions(result: NumericTensor, ...inputs: NumericTensor[]) {
  const positions: Record<string, number[]> = {}
  for (const input of inputs)
    for (const [axis, ids] of Object.entries(input.positions ?? {})) {
      const at = result.axes.indexOf(axis)
      if (at >= 0 && result.shape[at] === ids.length) positions[axis] = [...ids]
    }
  if (Object.keys(positions).length) result.positions = positions
  return result
}
export function unravel(flat: number, shape: number[]) {
  const indices = Array(shape.length).fill(0) as number[]
  if (shape.some((n) => n === 0)) return indices
  for (let a = shape.length - 1; a >= 0; a--) {
    indices[a] = flat % shape[a]
    flat = Math.floor(flat / shape[a])
  }
  return indices
}
export function offset(indices: number[], shape: number[]) {
  return indices.reduce((value, index, axis) => value * shape[axis] + index, 0)
}
export function makeTensor(
  id: string,
  name: string,
  shape: number[],
  axes: string[],
  values: ArrayLike<number>,
  recipe: Recipe,
  role: NumericTensor['role'] = 'activation',
): NumericTensor {
  if (
    shape.length !== axes.length ||
    elements(shape) !== values.length ||
    shape.some((n) => !Number.isInteger(n) || n < 0) ||
    values.length > 1000000
  )
    throw new Error('Invalid teaching tensor shape.')
  const integer = (recipe.op === 'input' && role === 'input') || recipe.op === 'choice'
  return {
    id,
    name,
    shape: [...shape],
    axes: [...axes],
    data: integer ? Int32Array.from(values) : Float32Array.from(values),
    dtype: integer ? 'int32' : 'float32',
    recipe,
    role,
    createdAt: -1,
  }
}
export function matmul(
  id: string,
  name: string,
  a: NumericTensor,
  b: NumericTensor,
): NumericTensor {
  const [m, k] = a.shape.slice(-2),
    [k2, n] = b.shape.slice(-2)
  if (
    k !== k2 ||
    a.shape.length < 2 ||
    b.shape.length < 2 ||
    (b.shape.length !== 2 && a.shape.slice(0, -2).join() !== b.shape.slice(0, -2).join())
  )
    throw new Error('Incompatible matrix multiplication.')
  const batch = elements(a.shape.slice(0, -2)),
    data = new Float32Array(batch * m * n)
  for (let z = 0; z < batch; z++)
    for (let r = 0; r < m; r++)
      for (let c = 0; c < n; c++) {
        let sum = 0
        for (let j = 0; j < k; j++)
          sum +=
            a.data[z * m * k + r * k + j] *
            b.data[(b.shape.length === 2 ? 0 : z) * k * n + j * n + c]
        data[z * m * n + r * n + c] = sum
      }
  return inheritPositions(
    makeTensor(
      id,
      name,
      [...a.shape.slice(0, -2), m, n],
      [...a.axes.slice(0, -1), b.axes.at(-1)!],
      data,
      { op: 'matmul', a: a.id, b: b.id },
    ),
    a,
    b,
  )
}
export function rmsnorm(id: string, input: NumericTensor, gain: NumericTensor, epsilon = 1e-5) {
  const width = input.shape.at(-1)!,
    data = new Float32Array(input.data.length)
  for (let r = 0; r < input.data.length / width; r++) {
    let square = 0
    for (let j = 0; j < width; j++) square += input.data[r * width + j] ** 2
    const scale = 1 / Math.sqrt(square / width + epsilon)
    for (let j = 0; j < width; j++)
      data[r * width + j] = input.data[r * width + j] * scale * gain.data[j]
  }
  return inheritPositions(
    makeTensor(id, 'RMS-normalized features', input.shape, input.axes, data, {
      op: 'rmsnorm',
      input: input.id,
      gain: gain.id,
      epsilon,
    }),
    input,
  )
}
export function splitHeads(id: string, input: NumericTensor, heads: number) {
  const [b, t, width] = input.shape,
    dh = width / heads,
    data = new Float32Array(input.data.length)
  if (!Number.isInteger(dh)) throw new Error('Head width must be integral.')
  for (let z = 0; z < b; z++)
    for (let h = 0; h < heads; h++)
      for (let q = 0; q < t; q++)
        for (let j = 0; j < dh; j++)
          data[((z * heads + h) * t + q) * dh + j] = input.data[(z * t + q) * width + h * dh + j]
  return inheritPositions(
    makeTensor(id, 'Split head axis', [b, heads, t, dh], ['B', 'H', 'T', 'Dh'], data, {
      op: 'split',
      input: input.id,
      heads,
    }),
    input,
  )
}
export function headSlice(id: string, input: NumericTensor, head: number) {
  const [b, h, t, d] = input.shape,
    data = new Float32Array(b * t * d)
  for (let z = 0; z < b; z++)
    for (let q = 0; q < t; q++)
      for (let j = 0; j < d; j++)
        data[(z * t + q) * d + j] = input.data[((z * h + head) * t + q) * d + j]
  const result = makeTensor(
    id,
    `Head ${head} slice`,
    [b, t, d],
    ['B', input.axes[2], input.axes[3]],
    data,
    { op: 'head', input: input.id, head },
  )
  result.positions = input.positions
  return result
}
export function transpose(id: string, input: NumericTensor) {
  const [m, n] = input.shape.slice(-2),
    batch = elements(input.shape.slice(0, -2)),
    data = new Float32Array(input.data.length)
  for (let z = 0; z < batch; z++)
    for (let r = 0; r < m; r++)
      for (let c = 0; c < n; c++) data[z * m * n + c * m + r] = input.data[z * m * n + r * n + c]
  const result = makeTensor(
    id,
    `${input.name} · transpose`,
    [...input.shape.slice(0, -2), n, m],
    [...input.axes.slice(0, -2), input.axes.at(-1)!, input.axes.at(-2)!],
    data,
    { op: 'transpose', input: input.id },
  )
  result.positions = input.positions
  return result
}
export function rope(id: string, input: NumericTensor, positions: number[], base = 10000) {
  const [b, heads, t, d] = input.shape,
    data = new Float32Array(input.data.length)
  if (d % 2 || positions.length !== t) throw new Error('Invalid RoPE axes.')
  for (let z = 0; z < b; z++)
    for (let h = 0; h < heads; h++)
      for (let q = 0; q < t; q++)
        for (let j = 0; j < d; j += 2) {
          const i = ((z * heads + h) * t + q) * d + j,
            angle = positions[q] * base ** (-j / d),
            c = Math.cos(angle),
            s = Math.sin(angle)
          data[i] = input.data[i] * c - input.data[i + 1] * s
          data[i + 1] = input.data[i] * s + input.data[i + 1] * c
        }
  const result = makeTensor(id, 'RoPE-rotated features', input.shape, input.axes, data, {
    op: 'rope',
    input: input.id,
    positions: [...positions],
    base,
  })
  result.positions = { T: [...positions] }
  return result
}
export function concatenate(
  id: string,
  name: string,
  inputs: NumericTensor[],
  axis: number,
  axes = inputs[0].axes,
) {
  const shape = [...inputs[0].shape]
  shape[axis] = inputs.reduce((s, t) => s + t.shape[axis], 0)
  const data = new Float32Array(elements(shape)),
    outer = elements(shape.slice(0, axis)),
    inner = elements(shape.slice(axis + 1))
  for (let z = 0; z < outer; z++) {
    let at = z * shape[axis] * inner
    for (const t of inputs) {
      const size = t.shape[axis] * inner
      data.set(t.data.subarray(z * size, (z + 1) * size), at)
      at += size
    }
  }
  return inheritPositions(
    makeTensor(id, name, shape, axes, data, {
      op: 'concat',
      inputs: inputs.map((t) => t.id),
      axis,
    }),
    ...inputs,
  )
}
export function gather(
  id: string,
  name: string,
  input: NumericTensor,
  indices: number[],
  axis: number,
  axisName = input.axes[axis],
) {
  const shape = [...input.shape]
  shape[axis] = indices.length
  const axes = [...input.axes]
  axes[axis] = axisName
  const data = new Float32Array(elements(shape)),
    outer = elements(shape.slice(0, axis)),
    inner = elements(shape.slice(axis + 1))
  for (let z = 0; z < outer; z++)
    for (let r = 0; r < indices.length; r++)
      for (let j = 0; j < inner; j++)
        data[(z * indices.length + r) * inner + j] =
          input.data[(z * input.shape[axis] + indices[r]) * inner + j]
  return makeTensor(id, name, shape, axes, data, {
    op: 'gather',
    input: input.id,
    indices: [...indices],
    axis,
  })
}
export function elementwise(
  id: string,
  name: string,
  a: NumericTensor,
  b: NumericTensor,
  op: 'add' | 'multiply',
) {
  if (a.shape.join() !== b.shape.join())
    throw new Error('Element-wise operands must have equal shapes.')
  return inheritPositions(
    makeTensor(
      id,
      name,
      a.shape,
      a.axes,
      a.data.map((v, i) => (op === 'add' ? v + b.data[i] : v * b.data[i])),
      { op, a: a.id, b: b.id },
    ),
    a,
    b,
  )
}
export const sigmoid = (x: number) =>
  x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))
export function silu(id: string, input: NumericTensor) {
  return inheritPositions(
    makeTensor(
      id,
      'SiLU gate',
      input.shape,
      input.axes,
      input.data.map((x) => x * sigmoid(x)),
      { op: 'silu', input: input.id },
    ),
    input,
  )
}
export function softmax(id: string, input: NumericTensor, temperature = 1) {
  if (!(temperature > 0)) throw new Error('Temperature must be positive.')
  const width = input.shape.at(-1)!,
    data = new Float32Array(input.data.length)
  for (let r = 0; r < input.data.length / width; r++) {
    const row = Array.from(input.data.subarray(r * width, (r + 1) * width), (v) => v / temperature),
      max = Math.max(...row),
      exp = row.map((v) => Math.exp(v - max)),
      sum = exp.reduce((a, b) => a + b, 0)
    if (!Number.isFinite(sum) || sum === 0) throw new Error('Softmax needs a visible finite score.')
    for (let j = 0; j < width; j++) data[r * width + j] = exp[j] / sum
  }
  return inheritPositions(
    makeTensor(id, 'Row-normalized probabilities', input.shape, input.axes, data, {
      op: 'softmax',
      input: input.id,
      temperature,
    }),
    input,
  )
}
