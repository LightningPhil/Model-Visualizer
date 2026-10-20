import { offset, sigmoid, unravel, type NumericTensor } from './tensor'
import type { NumericalTrace } from './engine'

export type OperandCell = { tensor: string; index: number }
export type ProofTerm = {
  label: string
  a: number
  b?: number
  result: number
  source: OperandCell
  second?: OperandCell
}
export type CellProof = {
  title: string
  tex: string
  explanation: string
  result: number
  terms: ProofTerm[]
  details: { label: string; value: number | string }[]
  parents: OperandCell[]
  sourceIds: string[]
}
export function valueLabel(value: number, precision = 4) {
  if (value === Infinity) return '∞'
  if (value === -Infinity) return '−∞'
  if (Number.isNaN(value)) return 'NaN'
  if (value === 0) return '0'
  if (precision >= 7)
    return Number.isInteger(value) ? String(value) : Number(value.toPrecision(9)).toString()
  if (Math.abs(value) < 0.0001 || Math.abs(value) >= 10000)
    return value.toExponential(Math.max(2, precision - 1))
  return Number(value.toFixed(precision)).toString()
}
export function explainCell(
  trace: Pick<NumericalTrace, 'tensors'>,
  tensor: NumericTensor,
  index: number,
): CellProof {
  const data = trace.tensors,
    r = tensor.recipe,
    indices = unravel(index, tensor.shape)
  const proof: CellProof = {
    title: 'Follow this value',
    tex: 'y=x',
    explanation: 'Select an operand to follow its provenance.',
    result: tensor.data[index],
    terms: [],
    details: [],
    parents: [],
    sourceIds:
      r.op === 'rmsnorm'
        ? ['rms']
        : r.op === 'rope'
          ? ['rope']
          : r.op === 'silu' || r.op === 'multiply'
            ? ['swiglu']
            : ['matmul', 'softmax', 'mask', 'split', 'head', 'concat', 'transpose', 'add'].includes(
                  r.op,
                )
              ? ['attention']
              : [],
  }
  const cell = (t: string, i: number) => ({ tensor: t, index: i })
  if (r.op === 'parameter')
    return {
      ...proof,
      title: 'A seeded, untrained parameter',
      tex: tensor.name.includes('gain')
        ? String.raw`g_i\sim\mathcal U(0.9,1.1)`
        : String.raw`w_i\sim\mathcal U(-a,a),\quad a=0.7\sqrt{3/n_{\mathrm{in}}}`,
      explanation:
        'This value is generated deterministically from the run seed and parameter name. It was not learned from data. RMS gains instead use U(0.9,1.1).',
      details: [
        { label: 'Seed', value: r.seed },
        { label: 'Storage', value: 'Float32' },
      ],
    }
  if (r.op === 'input')
    return {
      ...proof,
      title: 'An input or initial state',
      tex:
        tensor.role === 'input'
          ? String.raw`i\in\{0,\ldots,V-1\}`
          : String.raw`K,V\in\mathbb R^{B\times H_{kv}\times 0\times d_h}`,
      explanation:
        tensor.role === 'input'
          ? 'Token indices select embedding rows. They are opaque IDs here, without a language tokenizer.'
          : 'The initial per-layer K/V prefix has zero positions. No features are stored before the first forward pass.',
    }
  if (r.op === 'matmul') {
    const a = data[r.a],
      b = data[r.b],
      [m, k] = a.shape.slice(-2),
      n = b.shape.at(-1)!,
      col = indices.at(-1)!,
      row = indices.at(-2)!,
      batch = Math.floor(index / (m * n))
    for (let j = 0; j < k; j++) {
      const ai = batch * m * k + row * k + j,
        bi = (b.shape.length === 2 ? 0 : batch) * k * n + j * n + col
      proof.terms.push({
        label: `j = ${j}`,
        a: a.data[ai],
        b: b.data[bi],
        result: a.data[ai] * b.data[bi],
        source: cell(a.id, ai),
        second: cell(b.id, bi),
      })
    }
    return {
      ...proof,
      title: 'One row × one column',
      tex: String.raw`C_{r,c}=\sum_{j=0}^{K-1}A_{r,j}B_{j,c}`,
      explanation: `${k} products contract the shared axis. Number arithmetic accumulates the sum; the stored result rounds once to Float32.`,
      details: [
        { label: 'Row', value: row },
        { label: 'Column', value: col },
        { label: 'Contracted size K', value: k },
        { label: 'Sum before FP32 rounding', value: proof.terms.reduce((s, t) => s + t.result, 0) },
      ],
      parents: [
        cell(a.id, batch * m * k + row * k),
        cell(b.id, (b.shape.length === 2 ? 0 : batch) * k * n + col),
      ],
    }
  }
  if (r.op === 'rmsnorm') {
    const a = data[r.input],
      gain = data[r.gain],
      d = a.shape.at(-1)!,
      row = Math.floor(index / d),
      j = index % d
    for (let k = 0; k < d; k++) {
      const i = row * d + k
      proof.terms.push({
        label: `feature ${k}`,
        a: a.data[i],
        result: a.data[i] ** 2,
        source: cell(a.id, i),
      })
    }
    const mean = proof.terms.reduce((s, t) => s + t.result, 0) / d,
      denominator = Math.sqrt(mean + r.epsilon)
    return {
      ...proof,
      title: 'Normalize one token’s feature scale',
      tex: String.raw`y_j=\gamma_j\,x_j/\sqrt{\frac1D\sum_kx_k^2+\epsilon}`,
      explanation:
        'RMSNorm uses the root mean square, with no mean subtraction. Gain is a learned per-feature parameter (seeded, untrained here).',
      details: [
        { label: 'Mean of squares', value: mean },
        { label: 'Epsilon', value: r.epsilon },
        { label: 'Denominator', value: denominator },
        { label: 'Gain γ', value: gain.data[j] },
        { label: 'Input x', value: a.data[index] },
      ],
      parents: [cell(a.id, index), cell(gain.id, j)],
    }
  }
  if (r.op === 'rope') {
    const a = data[r.input],
      d = a.shape.at(-1)!,
      feature = indices.at(-1)!,
      pair = Math.floor(feature / 2),
      q = indices.at(-2)!,
      start = index - feature + pair * 2,
      x = a.data[start],
      y = a.data[start + 1],
      angle = r.positions[q] * r.base ** ((-2 * pair) / d),
      c = Math.cos(angle),
      s = Math.sin(angle)
    return {
      ...proof,
      title: 'A two-feature rotation',
      tex: String.raw`\begin{bmatrix}x^\prime\\y^\prime\end{bmatrix}=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix}`,
      explanation:
        'The angle uses the absolute token position, not the cache slot. This profile rotates adjacent feature pairs.',
      terms: [
        {
          label: 'x contribution',
          a: x,
          b: feature % 2 ? s : c,
          result: x * (feature % 2 ? s : c),
          source: cell(a.id, start),
        },
        {
          label: 'y contribution',
          a: y,
          b: feature % 2 ? c : -s,
          result: y * (feature % 2 ? c : -s),
          source: cell(a.id, start + 1),
        },
      ],
      details: [
        { label: 'Absolute position', value: r.positions[q] },
        { label: 'Pair', value: pair },
        { label: 'Angle (radians)', value: angle },
        { label: 'cos θ', value: c },
        { label: 'sin θ', value: s },
        { label: 'x', value: x },
        { label: 'y', value: y },
      ],
      parents: [cell(a.id, start), cell(a.id, start + 1)],
    }
  }
  if (r.op === 'mask') {
    const width = tensor.shape.at(-1)!,
      q = Math.floor(index / width) % r.queries.length,
      k = index % width,
      allowed = r.allowed[q * width + k]
    return {
      ...proof,
      title: allowed
        ? 'Visible key: scale the dot product'
        : 'Masked key: probability will be zero',
      tex: String.raw`\widetilde Z_{q,k}=Z_{q,k}/\sqrt{d_h}+M_{q,k}`,
      explanation: allowed
        ? 'The policy permits this absolute query/key pair. Divide its score by √Dh.'
        : 'The key is future or excluded by the selected retention policy. Its score becomes −∞ before softmax.',
      details: [
        { label: 'Query absolute position', value: r.queries[q] },
        { label: 'Key absolute position', value: r.keys[k] },
        { label: 'Raw score', value: data[r.input].data[index] },
        { label: 'Scale divisor √Dh', value: Math.sqrt(r.headDim) },
        { label: 'Mask', value: allowed ? '0' : '−∞' },
      ],
      parents: [cell(r.input, index)],
    }
  }
  if (r.op === 'softmax') {
    const input = data[r.input],
      width = input.shape.at(-1)!,
      row = Math.floor(index / width),
      values = Array.from(input.data.subarray(row * width, (row + 1) * width)),
      max = Math.max(...values.map((v) => v / r.temperature)),
      exp = values.map((v) => Math.exp(v / r.temperature - max)),
      sum = exp.reduce((a, b) => a + b, 0)
    proof.terms = values.map((v, j) => ({
      label: `${input.axes.at(-1) === 'V' ? 'token' : 'key'} ${input.positions?.[input.axes.at(-1)!]?.[j] ?? j}`,
      a: v,
      result: exp[j],
      source: cell(input.id, row * width + j),
    }))
    return {
      ...proof,
      title: 'Exponentiate, then normalize this row',
      tex: String.raw`p_k=\frac{\exp(z_k/\tau-m)}{\sum_j\exp(z_j/\tau-m)}`,
      explanation:
        'Subtracting the maximum keeps exponentials stable. Masked −∞ contributes exp(−∞) = 0; it never receives probability mass.',
      details: [
        { label: 'Temperature τ', value: r.temperature },
        { label: 'Row maximum m', value: max },
        { label: 'Selected exponential', value: exp[index % width] },
        { label: 'Sum of exponentials', value: sum },
        {
          label: 'Stored row sum',
          value: Array.from(tensor.data.subarray(row * width, (row + 1) * width)).reduce(
            (a, b) => a + b,
            0,
          ),
        },
      ],
      parents: [cell(input.id, index)],
    }
  }
  if (r.op === 'add' || r.op === 'multiply') {
    const a = data[r.a],
      b = data[r.b]
    return {
      ...proof,
      title: r.op === 'add' ? 'Two residual features meet' : 'Element-wise gating',
      tex: r.op === 'add' ? 'y_i=a_i+b_i' : String.raw`y_i=a_i\,b_i`,
      explanation:
        r.op === 'add'
          ? 'Matching features add. No sequence positions or head axes are contracted.'
          : 'Matching features multiply. This is a Hadamard product; it is not a dot product.',
      terms: [
        {
          label: 'matching feature',
          a: a.data[index],
          b: b.data[index],
          result: r.op === 'add' ? a.data[index] + b.data[index] : a.data[index] * b.data[index],
          source: cell(a.id, index),
          second: cell(b.id, index),
        },
      ],
      parents: [cell(a.id, index), cell(b.id, index)],
    }
  }
  if (r.op === 'silu')
    return {
      ...proof,
      title: 'The SiLU gate',
      tex: String.raw`\operatorname{SiLU}(x)=x\,\sigma(x)=\frac{x}{1+e^{-x}}`,
      explanation:
        'A smooth gate multiplies x by its sigmoid. Negative inputs are not clipped to zero.',
      details: [
        { label: 'x', value: data[r.input].data[index] },
        { label: 'sigmoid(x)', value: sigmoid(data[r.input].data[index]) },
      ],
      parents: [cell(r.input, index)],
    }
  if (r.op === 'embedding') {
    const d = tensor.shape.at(-1)!,
      row = Math.floor(index / d),
      j = index % d,
      token = r.tokens[row]
    return {
      ...proof,
      title: `Token #${token} selects embedding row ${token}`,
      tex: String.raw`X_{t,j}=E_{\mathrm{id}_t,j}`,
      explanation:
        'Embedding lookup gathers a row. It is not a vocabulary-length one-hot matrix multiplication in this implementation.',
      details: [
        { label: 'Token ID', value: token },
        { label: 'Feature', value: j },
      ],
      parents: [cell(r.table, token * d + j)],
    }
  }
  if (r.op === 'choice')
    return {
      ...proof,
      title: `Selected token #${r.token}`,
      tex:
        r.method === 'sample'
          ? String.raw`t\sim\operatorname{Categorical}(p)`
          : String.raw`t^\star=\arg\max_t p_t\quad\text{(greedy)}`,
      explanation:
        'The selected ID comes from the last query row. Greedy chooses its maximum; seeded sampling draws from the distribution when that mode is selected.',
      details: [
        {
          label: 'Selection',
          value: r.method === 'sample' ? 'Seeded categorical sampling' : 'Greedy argmax',
        },
        ...(r.draw === undefined ? [] : [{ label: 'Uniform draw [0,1)', value: r.draw }]),
        {
          label: 'Selected probability',
          value: data[r.input].data[r.row * data[r.input].shape.at(-1)! + r.token],
        },
      ],
      parents: [cell(r.input, r.row * data[r.input].shape.at(-1)! + r.token)],
    }
  if (r.op === 'head') {
    const input = data[r.input],
      mapped = [indices[0], r.head, ...indices.slice(1)]
    return {
      ...proof,
      title: `View head ${r.head}`,
      tex: 'Y_{b,t,j}=X_{b,h,t,j}',
      explanation:
        'Select the head axis. The numerical value is unchanged; a fused implementation can keep this as a strided view.',
      parents: [cell(input.id, offset(mapped, input.shape))],
    }
  }
  if (r.op === 'transpose') {
    const mapped = [...indices]
    ;[mapped[mapped.length - 1], mapped[mapped.length - 2]] = [
      mapped[mapped.length - 2],
      mapped[mapped.length - 1],
    ]
    return {
      ...proof,
      title: 'Transpose the final two axes',
      tex: String.raw`Y_{\ldots,j,i}=X_{\ldots,i,j}`,
      explanation:
        'Transpose changes indexing, not the scalar value. Batch and head dimensions remain in place.',
      parents: [cell(r.input, offset(mapped, data[r.input].shape))],
    }
  }
  if (r.op === 'split') {
    const [b, h, t, j] = indices,
      d = tensor.shape.at(-1)!,
      mapped = [b, t, h * d + j]
    return {
      ...proof,
      title: 'Expose a query or KV head',
      tex: 'Y_{b,h,t,j}=X_{b,t,h d_h+j}',
      explanation:
        'Reshape the projected feature axis and transpose to make heads explicit. No new learned projection occurs.',
      parents: [cell(r.input, offset(mapped, data[r.input].shape))],
    }
  }
  if (r.op === 'gather') {
    const mapped = [...indices]
    mapped[r.axis] = r.indices[indices[r.axis]]
    return {
      ...proof,
      title: 'Retain this absolute position',
      tex: String.raw`Y_{\ldots,r,\ldots}=X_{\ldots,\pi(r),\ldots}`,
      explanation:
        'The position map selects retained slots. A compact slot index is separate from its absolute sequence position.',
      parents: [cell(r.input, offset(mapped, data[r.input].shape))],
    }
  }
  if (r.op === 'concat') {
    let selected = indices[r.axis],
      input = data[r.inputs[0]]
    for (const id of r.inputs) {
      input = data[id]
      if (selected < input.shape[r.axis]) break
      selected -= input.shape[r.axis]
    }
    const mapped = [...indices]
    mapped[r.axis] = selected
    return {
      ...proof,
      title: 'Concatenate without summing',
      tex: String.raw`Y=\operatorname{Concat}(X_0,\ldots,X_n)`,
      explanation:
        'This output scalar comes directly from one operand. Head concatenation preserves head order; working-cache concatenation joins the prefix and new positions.',
      parents: [cell(input.id, offset(mapped, input.shape))],
    }
  }
  return proof
}
