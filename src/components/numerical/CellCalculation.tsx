import { useState } from 'react'
import type { NumericalTrace } from '../../model/numerics/engine'
import { unravel, type NumericTensor } from '../../model/numerics/tensor'
import { explainCell, valueLabel, type OperandCell } from '../../model/numerics/explain'
import { MathBlock } from '../Math'
import { OperationSketch } from './OperationSketch'

export function CellCalculation({
  trace,
  tensor,
  index,
  onOperand,
  sources,
}: {
  trace: NumericalTrace
  tensor: NumericTensor
  index: number
  onOperand: (cell: OperandCell) => void
  sources: { id: string; title: string; url: string }[]
}) {
  const [expanded, setExpanded] = useState(false)
  if (!tensor.data.length) return null
  const proof = explainCell(trace, tensor, index)
  const terms = expanded ? proof.terms : proof.terms.slice(0, 16)
  const max = Math.max(1e-12, ...proof.terms.map((t) => Math.abs(t.result)).filter(Number.isFinite))
  return (
    <section className="cell-calculation" aria-label="Selected cell calculation">
      <div className="numeric-section-heading">
        <span className="eyebrow">THIS CELL, EXPLAINED</span>
        <code data-testid="cell-result" title={String(proof.result)}>
          {valueLabel(proof.result, 8)}
        </code>
      </div>
      <h3>{proof.title}</h3>
      <MathBlock tex={proof.tex} />
      <p>{proof.explanation}</p>
      <OperationSketch trace={trace} tensor={tensor} index={index} onOperand={onOperand} />
      {!!proof.details.length && (
        <dl className="proof-details">
          {proof.details.map((d) => (
            <div key={d.label}>
              <dt>{d.label}</dt>
              <dd>{typeof d.value === 'number' ? valueLabel(d.value, 8) : d.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {!!proof.terms.length && (
        <div className="proof-terms">
          <div className="proof-terms-head">
            <span>Term</span>
            <span>Operand / operands</span>
            <span>Contribution</span>
          </div>
          {terms.map((term, i) => (
            <div className="proof-term" key={i}>
              <small>{term.label}</small>
              <span>
                <button title="Inspect this operand" onClick={() => onOperand(term.source)}>
                  {valueLabel(term.a, 5)}
                </button>
                {tensor.recipe.op === 'rmsnorm' && <sup>2</sup>}
                {term.b !== undefined && (
                  <>
                    {tensor.recipe.op === 'add' ? ' + ' : ' × '}
                    <button
                      disabled={!term.second}
                      title={term.second ? 'Inspect this operand' : 'Rotation coefficient'}
                      onClick={() => term.second && onOperand(term.second)}
                    >
                      {valueLabel(term.b, 5)}
                    </button>
                  </>
                )}
              </span>
              <span className={term.result < 0 ? 'negative-term' : ''}>
                <i style={{ width: `${Math.min(100, (Math.abs(term.result) / max) * 100)}%` }} />
                {valueLabel(term.result, 6)}
              </span>
            </div>
          ))}
          {proof.terms.length > 16 && (
            <button className="numeric-text-button" onClick={() => setExpanded(!expanded)}>
              {expanded ? 'Show first 16 terms' : `Show all ${proof.terms.length} terms`}
            </button>
          )}
        </div>
      )}
      {!!proof.parents.length && (
        <div className="operand-links">
          <span>TRACE BACK TO THE INPUTS</span>
          {proof.parents.map((cell, i) => (
            <button key={`${cell.tensor}/${cell.index}/${i}`} onClick={() => onOperand(cell)}>
              {trace.tensors[cell.tensor].name}
              <code>[{unravel(cell.index, trace.tensors[cell.tensor].shape).join(', ')}]</code>
              <span>↗</span>
            </button>
          ))}
        </div>
      )}
      {!!proof.sourceIds.length && (
        <div className="numeric-proof-sources">
          <span>PRIMARY REFERENCES</span>
          {sources
            .filter((source) => proof.sourceIds.includes(source.id))
            .map((source) => (
              <a key={source.id} href={source.url} target="_blank" rel="noreferrer">
                {source.title} ↗
              </a>
            ))}
        </div>
      )}
    </section>
  )
}
