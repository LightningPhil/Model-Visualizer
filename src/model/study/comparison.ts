import type { NumericalTrace } from '../numerics/engine'
import { tensorComparisonIssue } from '../numerics/tensor'

/** Execution-order differences, not an attribution of final causal importance. */
export function compareTraces(current: NumericalTrace, baseline: NumericalTrace, tolerance = 1e-6) {
  const architecture = (trace: NumericalTrace) => {
    const { kvCache: _policy, ...config } = trace.config
    return JSON.stringify(config)
  }
  const incompatible =
    current.checkpoint?.sha256 !== baseline.checkpoint?.sha256 ||
    current.checkpoint?.tokenizerSha256 !== baseline.checkpoint?.tokenizerSha256 ||
    architecture(current) !== architecture(baseline)
  if (incompatible)
    return {
      issue:
        'Compare runs with the same architecture, checkpoint and tokenizer. Head-sharing changes alter the parameterization.',
      rows: [],
    }
  const rows: {
    tensor: string
    title: string
    frame: number
    call: number
    index: number
    changed: number
    total: number
    max: number
    rms: number
  }[] = []
  let skipped = 0
  const seen = new Set<string>()
  current.frames.forEach((frame, index) =>
    frame.outputs.forEach((id) => {
      if (seen.has(id)) return
      seen.add(id)
      const a = current.tensors[id],
        b = baseline.tensors[id]
      if (!a || !b || a.role === 'parameter' || tensorComparisonIssue(a, b)) {
        skipped++
        return
      }
      let max = 0,
        firstChangedIndex = -1,
        squares = 0,
        changed = 0,
        count = 0
      for (let i = 0; i < a.data.length; i++) {
        if (Object.is(a.data[i], b.data[i]) || a.data[i] === b.data[i]) {
          if (Number.isFinite(a.data[i])) count++
          continue
        }
        if (!Number.isFinite(a.data[i]) || !Number.isFinite(b.data[i])) {
          skipped++
          return
        }
        const d = Math.abs(a.data[i] - b.data[i])
        max = Math.max(max, d)
        squares += d * d
        count++
        if (d > tolerance) {
          changed++
          if (firstChangedIndex < 0) firstChangedIndex = i
        }
      }
      if (changed)
        rows.push({
          tensor: id,
          title: a.name,
          frame: index,
          call: frame.call,
          index: firstChangedIndex,
          changed,
          total: count,
          max,
          rms: Math.sqrt(squares / Math.max(1, count)),
        })
    }),
  )
  const samePrefix = current.settings.inputIds.join() === baseline.settings.inputIds.join()
  return { issue: undefined, rows, skipped, samePrefix, first: rows[0] }
}
