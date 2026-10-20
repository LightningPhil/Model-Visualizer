import type { NumericalTrace } from '../../model/numerics/engine'
import { unravel, type NumericTensor } from '../../model/numerics/tensor'
import { valueLabel, type OperandCell } from '../../model/numerics/explain'

/** Geometry illustrates the recorded operation; all values come from the trace. */
export function OperationSketch({
  trace,
  tensor,
  index,
  onOperand,
}: {
  trace: NumericalTrace
  tensor: NumericTensor
  index: number
  onOperand: (cell: OperandCell) => void
}) {
  const recipe = tensor.recipe
  if (recipe.op === 'matmul') {
    const a = trace.tensors[recipe.a],
      b = trace.tensors[recipe.b],
      [m, k] = a.shape.slice(-2),
      n = b.shape.at(-1)!,
      indices = unravel(index, tensor.shape),
      row = indices.at(-2)!,
      col = indices.at(-1)!,
      batch = Math.floor(index / (m * n))
    const blocks = [
      { x: 12, w: 97, h: 64, rows: m, cols: k, shape: `${m} × ${k}`, label: 'A · select row' },
      { x: 152, w: 67, h: 96, rows: k, cols: n, shape: `${k} × ${n}`, label: 'B · select column' },
      { x: 269, w: 65, h: 64, rows: m, cols: n, shape: `${m} × ${n}`, label: 'C · one cell' },
    ]
    return (
      <div className="operation-sketch" aria-label="Matrix contraction diagram">
        <svg
          viewBox="0 0 350 150"
          role="img"
          aria-label={`Row ${row} of ${m} by column ${col} of ${n}; contract ${k} features`}
        >
          {blocks.map((r, i) => (
            <g key={r.label}>
              <rect className="sketch-matrix" x={r.x} y={18} width={r.w} height={r.h} rx={3} />
              <path
                className="sketch-grid"
                d={[
                  ...Array.from(
                    { length: r.cols <= 32 ? r.cols - 1 : 0 },
                    (_, j) => `M${r.x + (r.w * (j + 1)) / r.cols} 18v${r.h}`,
                  ),
                  ...Array.from(
                    { length: r.rows <= 32 ? r.rows - 1 : 0 },
                    (_, j) => `M${r.x} ${18 + (r.h * (j + 1)) / r.rows}h${r.w}`,
                  ),
                ].join(' ')}
              />
              <text x={r.x + r.w / 2} y={132}>
                {r.shape}
              </text>
              <text className="sketch-label" x={r.x + r.w / 2} y={146}>
                {r.label}
              </text>
              {i === 0 && (
                <rect
                  className="sketch-selection"
                  x={r.x}
                  y={18 + (row / m) * r.h}
                  width={r.w}
                  height={r.h / m}
                />
              )}
              {i === 1 && (
                <rect
                  className="sketch-selection secondary"
                  x={r.x + (col / n) * r.w}
                  y={18}
                  width={r.w / n}
                  height={r.h}
                />
              )}
              {i === 2 && (
                <rect
                  className="sketch-selection"
                  x={r.x + (col / n) * r.w}
                  y={18 + (row / m) * r.h}
                  width={r.w / n}
                  height={r.h / m}
                />
              )}
            </g>
          ))}
          <text className="sketch-symbol" x={130} y={58}>
            ×
          </text>
          <text className="sketch-symbol" x={243} y={58}>
            =
          </text>
        </svg>
        <div>
          <button onClick={() => onOperand({ tensor: a.id, index: batch * m * k + row * k })}>
            Open row {row} ↗
          </button>
          <span>Σ {k} products</span>
          <button
            onClick={() =>
              onOperand({ tensor: b.id, index: (b.shape.length === 2 ? 0 : batch) * k * n + col })
            }
          >
            Open column {col} ↗
          </button>
        </div>
        <p>
          Final two axes shown. Batch axes stay separate. Highlights mark mathematical indices; this
          sketch is not a storage layout.
          {Math.max(m, k, n) > 32 ? ' Grid lines are omitted on axes larger than 32.' : ''}
        </p>
      </div>
    )
  }
  if (recipe.op === 'rope') {
    const indices = unravel(index, tensor.shape),
      d = tensor.shape.at(-1)!,
      feature = indices.at(-1)!,
      pair = feature - (feature % 2),
      baseIndex = index - feature + pair,
      input = trace.tensors[recipe.input],
      x = input.data[baseIndex],
      y = input.data[baseIndex + 1],
      rx = tensor.data[baseIndex],
      ry = tensor.data[baseIndex + 1],
      length = Math.max(1e-9, Math.hypot(x, y)),
      scale = 65 / length,
      position = recipe.positions[indices.at(-2)!],
      angle = position * recipe.base ** (-pair / d)
    return (
      <div className="operation-sketch" aria-label="Rotary position diagram">
        <svg
          viewBox="0 0 350 195"
          role="img"
          aria-label={`Feature pair ${pair}, ${pair + 1} rotated by ${valueLabel(angle, 5)} radians at position ${position}`}
        >
          <circle className="sketch-matrix" cx={175} cy={86} r={65} />
          <path className="sketch-grid" d="M90 86H260 M175 5V168" />
          <path
            className="sketch-vector original"
            d={`M175 86L${175 + x * scale} ${86 - y * scale}`}
          />
          <path className="sketch-vector" d={`M175 86L${175 + rx * scale} ${86 - ry * scale}`} />
          <circle className="sketch-dot original" cx={175 + x * scale} cy={86 - y * scale} r={4} />
          <circle className="sketch-dot" cx={175 + rx * scale} cy={86 - ry * scale} r={4} />
          <text x={175} y={185}>
            position {position} · pair ({pair}, {pair + 1}) · θ {valueLabel(angle, 4)} rad
          </text>
          <text className="sketch-label" x={281} y={91}>
            x
          </text>
          <text className="sketch-label" x={178} y={9}>
            y
          </text>
        </svg>
        <div>
          <button onClick={() => onOperand({ tensor: input.id, index: baseIndex })}>
            ┄ Input pair ↗
          </button>
          <span>━ Rotated pair</span>
        </div>
        <p>
          Direction changes; pair length is preserved up to Float32 rounding. Larger positions can
          rotate more than one turn.
        </p>
      </div>
    )
  }
  return null
}
