import { useMemo, useState } from 'react'
import { ArrowDownUp, ChevronLeft, ChevronRight } from 'lucide-react'
import {
  offset,
  unravel,
  tensorComparisonIssue,
  type NumericTensor,
} from '../../model/numerics/tensor'
import { valueLabel } from '../../model/numerics/explain'

export function TensorView({
  tensor,
  index,
  onCell,
  comparison,
}: {
  tensor: NumericTensor
  index: number
  onCell: (index: number) => void
  comparison?: NumericTensor
}) {
  const [transposed, setTransposed] = useState(false)
  const [delta, setDelta] = useState(false)
  const issue = comparison ? tensorComparisonIssue(tensor, comparison) : undefined,
    compatible = Boolean(comparison && !issue)
  const rank = tensor.shape.length,
    indices = unravel(Math.min(index, Math.max(0, tensor.data.length - 1)), tensor.shape)
  const rowAxis = rank === 1 ? -1 : rank - (transposed ? 1 : 2),
    colAxis = rank === 1 ? 0 : rank - (transposed ? 2 : 1)
  const rows = rowAxis < 0 ? 1 : tensor.shape[rowAxis],
    cols = tensor.shape[colAxis],
    row = rowAxis < 0 ? 0 : indices[rowAxis],
    col = indices[colAxis]
  const pageRow = Math.floor(row / 8) * 8,
    pageCol = Math.floor(col / 8) * 8
  const scale = useMemo(
    () => Math.max(1e-12, ...Array.from(tensor.data).filter(Number.isFinite).map(Math.abs)),
    [tensor],
  )
  const locate = (r: number, c: number) => {
    const next = [...indices]
    if (rowAxis >= 0) next[rowAxis] = r
    next[colAxis] = c
    return offset(next, tensor.shape)
  }
  const label = (axis: number, position: number) =>
    tensor.positions?.[tensor.axes[axis]]?.[position] ?? position
  return (
    <section className="tensor-view" aria-label={`Tensor ${tensor.name}`}>
      <div className="tensor-shape-line">
        <strong>[{tensor.shape.join(' × ')}]</strong>
        <span>
          {tensor.dtype} · {tensor.data.length.toLocaleString()} values
        </span>
      </div>
      <div className="tensor-axis-line">
        {tensor.axes.map((axis, a) => (
          <span key={a}>
            <b>{axis}</b> {tensor.shape[a]}
            {a < rank - 2 && tensor.shape[a] > 1 ? (
              <select
                aria-label={`Tensor ${axis} slice`}
                value={indices[a]}
                onChange={(e) => {
                  const next = [...indices]
                  next[a] = Number(e.target.value)
                  onCell(offset(next, tensor.shape))
                }}
              >
                {Array.from({ length: tensor.shape[a] }, (_, i) => (
                  <option key={i} value={i}>
                    {i}
                  </option>
                ))}
              </select>
            ) : null}
          </span>
        ))}
      </div>
      {tensor.data.length === 0 ? (
        <p className="numeric-empty">An empty prefix: this tensor has no stored positions yet.</p>
      ) : (
        <>
          <div className="matrix-tools">
            <span>
              {rowAxis < 0 ? 'Vector' : tensor.axes[rowAxis]} ↓ · {tensor.axes[colAxis]} →
            </span>
            {rank > 1 && (
              <button aria-pressed={transposed} onClick={() => setTransposed(!transposed)}>
                <ArrowDownUp size={12} /> Transpose view
              </button>
            )}
            {comparison && (
              <button
                disabled={!compatible}
                aria-pressed={delta && compatible}
                onClick={() => setDelta(!delta)}
              >
                Δ baseline
              </button>
            )}
          </div>
          <div className="tensor-grid-wrap">
            <div
              className="tensor-grid"
              role="grid"
              aria-label="Numerical tensor cells"
              style={{
                gridTemplateColumns: `27px repeat(${Math.min(8, cols - pageCol)}, minmax(35px,1fr))`,
              }}
            >
              <div className="tensor-grid-header" role="row">
                <span role="columnheader" aria-label="Row position" />
                {Array.from({ length: Math.min(8, cols - pageCol) }, (_, i) => (
                  <span className="tensor-index" role="columnheader" key={`c${i}`}>
                    {label(colAxis, pageCol + i)}
                  </span>
                ))}
              </div>
              {Array.from({ length: Math.min(8, rows - pageRow) }, (_, i) => {
                const r = pageRow + i
                return (
                  <div className="tensor-grid-row" role="row" key={r}>
                    <span className="tensor-index" role="rowheader">
                      {rowAxis < 0 ? '' : label(rowAxis, r)}
                    </span>
                    {Array.from({ length: Math.min(8, cols - pageCol) }, (_, j) => {
                      const c = pageCol + j,
                        flat = locate(r, c),
                        original = tensor.data[flat],
                        value =
                          delta &&
                          compatible &&
                          Number.isFinite(original) &&
                          Number.isFinite(comparison!.data[flat])
                            ? original - comparison!.data[flat]
                            : original
                      return (
                        <button
                          key={flat}
                          role="gridcell"
                          data-flat-index={flat}
                          aria-selected={flat === index}
                          className={`numeric-cell ${!Number.isFinite(value) ? 'numeric-masked' : ''} ${value < 0 ? 'numeric-negative' : 'numeric-positive'}`}
                          style={
                            Number.isFinite(value)
                              ? {
                                  backgroundColor: `color-mix(in srgb, var(--${value < 0 ? 'violet' : 'cyan'}) ${Math.round(5 + Math.min(1, Math.abs(value) / scale) * 38)}%, var(--panel))`,
                                }
                              : {}
                          }
                          onClick={() => onCell(flat)}
                          data-tooltip-detail={`[${unravel(flat, tensor.shape).join(', ')}] = ${String(value)}`}
                          aria-label={`Cell ${unravel(flat, tensor.shape).join(', ')}: ${valueLabel(value, 7)}`}
                        >
                          {valueLabel(value, 3)}
                        </button>
                      )
                    })}
                  </div>
                )
              })}
            </div>
          </div>
          <div className="matrix-paging">
            <div>
              <button
                aria-label="Previous tensor rows"
                disabled={pageRow === 0}
                onClick={() => onCell(locate(Math.max(0, row - 8), col))}
              >
                <ChevronLeft size={12} />
              </button>
              <span>
                Rows {pageRow}–{Math.min(rows - 1, pageRow + 7)} / {rows}
              </span>
              <button
                aria-label="Next tensor rows"
                disabled={pageRow + 8 >= rows}
                onClick={() => onCell(locate(Math.min(rows - 1, row + 8), col))}
              >
                <ChevronRight size={12} />
              </button>
            </div>
            <div>
              <button
                aria-label="Previous tensor columns"
                disabled={pageCol === 0}
                onClick={() => onCell(locate(row, Math.max(0, col - 8)))}
              >
                <ChevronLeft size={12} />
              </button>
              <span>
                Columns {pageCol}–{Math.min(cols - 1, pageCol + 7)} / {cols}
              </span>
              <button
                aria-label="Next tensor columns"
                disabled={pageCol + 8 >= cols}
                onClick={() => onCell(locate(row, Math.min(cols - 1, col + 8)))}
              >
                <ChevronRight size={12} />
              </button>
            </div>
          </div>
          <div className="tensor-legend">
            <span>− negative</span>
            <i />
            <span>0</span>
            <i />
            <span>+ positive</span>
            <small>× / −∞ excluded by mask</small>
          </div>
          {comparison && !compatible && (
            <p className="numeric-note">
              {issue === 'visibility'
                ? 'Baseline has a different visibility mask. Masked entries encode exclusion, so score deltas are not shown across different masks. Compare the attention probabilities instead.'
                : 'Baseline has different shapes, axes, storage types or absolute position IDs. Values are not subtracted across incompatible slices.'}
            </p>
          )}
          {comparison && delta && compatible && (
            <p className="numeric-note">
              Grid values show current − baseline. The cell calculation below explains the current
              run. Selected: current {valueLabel(tensor.data[index], 7)}, baseline{' '}
              {valueLabel(comparison.data[index], 7)}.
              {tensor.recipe.op === 'mask'
                ? ' Excluded cells retain their mask marker; no subtraction is defined for them.'
                : ''}
            </p>
          )}
        </>
      )}
    </section>
  )
}
