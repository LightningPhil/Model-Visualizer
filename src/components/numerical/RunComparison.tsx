import { useMemo, useState } from 'react'
import type { NumericalTrace } from '../../model/numerics/engine'
import { compareTraces } from '../../model/study/comparison'
import { valueLabel } from '../../model/numerics/explain'
import '../help/study.css'

export function RunComparison({
  trace,
  baseline,
  onInspect,
  piece,
}: {
  trace: NumericalTrace
  baseline: NumericalTrace
  onInspect: (frame: number, tensor: string, index: number) => void
  piece?: (id: number) => string
}) {
  const [target, setTarget] = useState(0),
    [all, setAll] = useState(false)
  const comparison = useMemo(() => compareTraces(trace, baseline), [trace, baseline])
  const p = trace.tensors['call0/probabilities'],
    b = baseline.tensors['call0/probabilities'],
    size = trace.config.vocabSize,
    index = (p?.data.length ?? 0) - size + Math.min(target, size - 1),
    bi = (b?.data.length ?? 0) - size + Math.min(target, size - 1)
  return (
    <details className="run-comparison" open={Boolean(trace.intervention)}>
      <summary>Follow differences through the run</summary>
      {comparison.issue ? (
        <p>{comparison.issue}</p>
      ) : (
        <>
          <p>
            Compare matching tensor IDs, axes and absolute positions, in execution order. A cell
            changes when |Δ| &gt; 10⁻⁶. This is a numerical threshold, not statistical significance.
            RMS uses finite aligned cells; masked infinities are never subtracted.
          </p>
          {trace.intervention && (
            <p>
              <b>
                Controlled {trace.intervention.mode === 'zero' ? 'zero ablation' : 'donor patch'}:
              </b>{' '}
              layer {trace.intervention.layer}, query head {trace.intervention.head}, prompt
              position {trace.intervention.position}. {trace.intervention.source} Recomputed from
              this head output through Wo, residuals and every later layer. Original checkpoint
              weights are unchanged.
            </p>
          )}
          <p>
            {comparison.samePrefix
              ? 'Both runs consume the same prompt IDs.'
              : 'Prompt IDs differ: this is an input comparison, not an isolated internal intervention.'}{' '}
            Later decoder calls can consume different selected IDs; the call-0 comparison precedes
            that divergence. Differences do not by themselves identify a learned feature or prove
            necessity.
          </p>
          <p>
            {comparison.rows.length} changed output tensors · {comparison.skipped} absent or
            incompatible outputs excluded.
          </p>
          {comparison.first ? (
            <button
              onClick={() =>
                onInspect(
                  comparison.first!.frame,
                  comparison.first!.tensor,
                  comparison.first!.index,
                )
              }
            >
              Inspect first changed operation: {comparison.first.title}
            </button>
          ) : (
            <p>No aligned output cells exceeded the threshold.</p>
          )}
          {p && b && (
            <>
              <label>
                Next-token metric: vocabulary ID
                <input
                  aria-label="Comparison target token"
                  type="number"
                  min={0}
                  max={size - 1}
                  step={1}
                  value={Math.min(target, size - 1)}
                  onChange={(e) => {
                    if (e.currentTarget.validity.valid && e.target.value)
                      setTarget(Number(e.target.value))
                  }}
                />
              </label>
              <p>
                Token #{Math.min(target, size - 1)} {piece?.(Math.min(target, size - 1))}: baseline
                P = {valueLabel(b.data[bi], 7)} → current P = {valueLabel(p.data[index], 7)}; ΔP ={' '}
                {valueLabel(p.data[index] - b.data[bi], 7)}. Final prompt row, temperature{' '}
                {trace.settings.temperature}
                {trace.settings.temperature !== baseline.settings.temperature
                  ? ' (temperatures differ)'
                  : ''}
                .
              </p>
            </>
          )}
          <div className="study-table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Operation output</th>
                  <th>Call</th>
                  <th>Changed cells</th>
                  <th>Max |Δ|</th>
                  <th>RMS Δ</th>
                </tr>
              </thead>
              <tbody>
                {comparison.rows.slice(0, all ? undefined : 16).map((row) => (
                  <tr key={row.tensor}>
                    <td>
                      <button onClick={() => onInspect(row.frame, row.tensor, row.index)}>
                        {row.title}
                      </button>
                    </td>
                    <td>{row.call}</td>
                    <td>
                      {row.changed}/{row.total}
                    </td>
                    <td>{valueLabel(row.max)}</td>
                    <td>{valueLabel(row.rms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {comparison.rows.length > 16 && (
            <button onClick={() => setAll(!all)}>
              {all ? 'Show first 16' : 'Show all changed operations'}
            </button>
          )}
          {trace.intervention && (
            <p>
              Interpretation:{' '}
              <a href="https://arxiv.org/abs/2309.16042" target="_blank" rel="noreferrer">
                Zhang & Nanda, activation patching methods
              </a>
              . This is a single head-vector intervention with an explicit next-token probability
              metric, not a circuit-discovery result.
            </p>
          )}
        </>
      )}
    </details>
  )
}
