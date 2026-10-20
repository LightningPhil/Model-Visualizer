import { useState } from 'react'
import type { NumericalTrace } from '../../model/numerics/engine'
import { formatBytes } from '../../model/schema'
import type { OperandCell } from '../../model/numerics/explain'

export function ParameterLibrary({
  trace,
  onInspect,
}: {
  trace: NumericalTrace
  onInspect: (cell: OperandCell) => void
}) {
  const [query, setQuery] = useState(''),
    parameters = Object.values(trace.tensors).filter((t) => t.role === 'parameter'),
    count = parameters.reduce(
      (n, t) => n + (t.recipe.op === 'checkpoint' && t.recipe.tiedTo ? 0 : t.data.length),
      0,
    ),
    bytes = parameters.reduce((n, t) => n + t.data.byteLength, 0)
  return (
    <details className="parameter-library">
      <summary>
        All parameter tensors <small>{parameters.length}</small>
      </summary>
      <p>
        <strong>
          {count.toLocaleString()} values · {formatBytes(bytes)}
        </strong>
        <br />
        {trace.checkpoint
          ? 'Trained checkpoint weights, fixed across decoder calls. Unique values exclude the tied vocabulary alias; displayed bytes include its materialized transpose.'
          : 'FP32 parameter payload. Seeded, untrained and fixed across decoder calls.'}{' '}
        Allocator and trace-history overhead excluded.
      </p>
      <input
        aria-label="Find a numerical parameter"
        placeholder="Find Wq, gain, embedding…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div>
        {parameters
          .filter((t) => t.name.toLowerCase().includes(query.toLowerCase()))
          .map((t) => (
            <button key={t.id} onClick={() => onInspect({ tensor: t.id, index: 0 })}>
              <span>{t.name}</span>
              <code>{t.shape.join(' × ')}</code>
              <small>
                {t.data.length.toLocaleString()} values · {formatBytes(t.data.byteLength)}
              </small>
            </button>
          ))}
      </div>
    </details>
  )
}
