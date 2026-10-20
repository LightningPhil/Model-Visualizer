import { kvHeadFor } from '../../model/attention'
import type { NumericalTrace } from '../../model/numerics/engine'
import { valueLabel, type OperandCell } from '../../model/numerics/explain'

export function HeadObservatory({
  trace,
  cursor,
  call,
  layer,
  row,
  onInspect,
}: {
  trace: NumericalTrace
  cursor: number
  call: number
  layer: number
  row: number
  onInspect: (cell: OperandCell) => void
}) {
  const available = Array.from({ length: trace.config.nQueryHeads }, (_, h) => ({
    h,
    t: trace.tensors[`call${call}/l${layer}/attention-${h}`],
  })).filter(({ t }) => t && t.createdAt <= cursor)
  if (!available.length) return null
  const sample = available[0].t,
    rows = sample.shape.at(-2)!,
    cols = sample.shape.at(-1)!,
    selected = Math.min(row, rows - 1)
  return (
    <details className="head-observatory">
      <summary>
        Attention head observatory{' '}
        <small>
          L{layer} · call {call}
        </small>
      </summary>
      <p>
        Each query head has its own distribution. Rows below show probabilities for query position{' '}
        <strong>{sample.positions?.T?.[selected] ?? selected}</strong>; columns are absolute key
        positions. Click a value to inspect it.
      </p>
      <div className="observatory-query">
        Query{' '}
        {Array.from({ length: rows }, (_, q) => (
          <button
            key={q}
            aria-pressed={q === selected}
            onClick={() => onInspect({ tensor: sample.id, index: q * cols })}
          >
            {sample.positions?.T?.[q] ?? q}
          </button>
        ))}
      </div>
      <div
        className="observatory-grid"
        style={{ gridTemplateColumns: `73px repeat(${cols},minmax(28px,1fr))` }}
      >
        <span />
        {Array.from({ length: cols }, (_, k) => (
          <span className="tensor-index" key={`key${k}`}>
            {sample.positions?.U?.[k] ?? k}
          </span>
        ))}
        {available.map(({ h, t }) => (
          <div className="tensor-grid-row" key={h}>
            <span className="observatory-head">
              Q{h}
              <small>
                → K{kvHeadFor(trace.config, h)} V{kvHeadFor(trace.config, h)}
              </small>
            </span>
            {Array.from({ length: cols }, (_, k) => {
              const i = selected * cols + k,
                p = t.data[i],
                masked =
                  t.recipe.op === 'softmax' ? trace.tensors[t.recipe.input].recipe : undefined,
                excluded = masked?.op === 'mask' && !masked.allowed[i]
              return (
                <button
                  key={k}
                  aria-label={`Head ${h}, query ${sample.positions?.T?.[selected] ?? selected}, key ${t.positions?.U?.[k] ?? k}: ${valueLabel(p, 7)}`}
                  className={`observatory-cell ${excluded ? 'numeric-masked' : ''}`}
                  style={{
                    backgroundColor: `color-mix(in srgb,var(--cyan) ${5 + p * 65}%,var(--panel))`,
                  }}
                  onClick={() => onInspect({ tensor: t.id, index: i })}
                >
                  {excluded ? '×' : valueLabel(p, 2)}
                </button>
              )
            })}
          </div>
        ))}
      </div>
      {available.length < trace.config.nQueryHeads && (
        <p>
          {trace.config.nQueryHeads - available.length} heads have not reached softmax at this
          playback step.
        </p>
      )}
    </details>
  )
}
